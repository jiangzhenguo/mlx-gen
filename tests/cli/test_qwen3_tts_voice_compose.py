"""Tests for tools/compose_voice.py — the multi-segment composer.

The neural synthesis is stubbed (see test_qwen3_tts_voice_worker.py); the
segment stitching, the per-segment gain and the peak normalisation are real.
"""

import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "tools" / "compose_voice.py"

STUB = '''
import numpy as np


class _Result:
    def __init__(self, audio, sample_rate):
        self.audio = audio
        self.sample_rate = sample_rate
        self.is_final_chunk = False


class _Model:
    sample_rate = 24000

    def generate_voice_design(self, text, instruct, language="auto", stream=False, **kwargs):
        t = np.arange(self.sample_rate * 2, dtype=np.float32) / self.sample_rate
        yield _Result(0.25 * np.sin(2 * np.pi * 440 * t), self.sample_rate)


def load_model(path):
    return _Model()
'''


def _make_stub(tmp_path, monkeypatch):
    package = tmp_path / "stub" / "mlx_audio" / "tts"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "utils.py").write_text(STUB, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "stub"))
    return tmp_path / "stub"


def _make_package(tmp_path):
    model_dir = tmp_path / "pkg" / "model"
    (model_dir / "speech_tokenizer").mkdir(parents=True)
    (model_dir / "config.json").write_text('{"tts_model_type": "voice_design"}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"0" * 16)
    (model_dir / "speech_tokenizer" / "model.safetensors").write_bytes(b"0" * 16)
    return tmp_path / "pkg"


def _manifest(tmp_path, segments):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"language": "chinese", "segments": segments}, ensure_ascii=False), encoding="utf-8")
    return path


def _run(manifest, package, output, tmp_path):
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            str(manifest),
            "--model-path",
            str(package),
            "--output",
            str(output),
            "--temp-dir",
            str(tmp_path / "segments"),
            "--metadata",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_two_segments_are_stitched_with_their_gain(tmp_path, monkeypatch):
    _make_stub(tmp_path, monkeypatch)
    package = _make_package(tmp_path)
    manifest = _manifest(
        tmp_path,
        [
            {"text": "嗯……", "voice": "a low female voice", "gain_db": 0, "gap_after_s": 0.1},
            {"text": "舒服死了！", "voice": "the same voice, shouting", "gain_db": 6, "gap_after_s": 0},
        ],
    )
    output = tmp_path / "composed.wav"
    result = _run(manifest, package, output, tmp_path)
    assert result.returncode == 0, result.stderr

    with wave.open(str(output), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 24000
        frames = handle.readframes(handle.getnframes())
    data = np.frombuffer(frames, dtype=np.int16)

    # 2 s + 0.1 s gap + 2 s, the loud segment pushed by 6 dB
    assert len(data) / 24000 == pytest_approx(4.1)
    assert np.abs(data).max() <= 32767

    sidecar = json.loads((tmp_path / "composed.wav.json").read_text(encoding="utf-8"))
    assert len(sidecar["segments"]) == 2
    assert sidecar["segments"][1]["gain_db"] == 6
    assert sidecar["duration"] == pytest_approx(4.1)


def test_the_loud_segment_is_louder_than_the_quiet_one(tmp_path, monkeypatch):
    _make_stub(tmp_path, monkeypatch)
    package = _make_package(tmp_path)
    manifest = _manifest(
        tmp_path,
        [
            {"text": "quiet", "voice": "a voice", "gain_db": 0, "gap_after_s": 0},
            {"text": "loud", "voice": "a voice", "gain_db": 12, "gap_after_s": 0},
        ],
    )
    output = tmp_path / "composed.wav"
    assert _run(manifest, package, output, tmp_path).returncode == 0

    with wave.open(str(output), "rb") as handle:
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32)
    first = data[: 2 * 24000]
    second = data[2 * 24000 : 4 * 24000]
    assert np.sqrt((second**2).mean()) > np.sqrt((first**2).mean()) * 3


def test_a_segment_without_text_is_rejected(tmp_path, monkeypatch):
    _make_stub(tmp_path, monkeypatch)
    package = _make_package(tmp_path)
    manifest = _manifest(tmp_path, [{"voice": "a voice"}])
    result = _run(manifest, package, tmp_path / "out.wav", tmp_path)
    assert result.returncode != 0
    assert "text" in (result.stderr + result.stdout)


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=0.05)


def test_the_tool_is_syntactically_valid():
    assert shutil.which("python3") or True
    compile(TOOL.read_text(encoding="utf-8"), str(TOOL), "exec")
