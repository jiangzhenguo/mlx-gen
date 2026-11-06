"""End-to-end test for `mlxgen dub --voice-design`.

The neural synthesis is stubbed (see test_qwen3_tts_voice_worker.py), but
everything else is real: the dub CLI, the runner, the worker subprocess, the
ffmpeg mux and the metadata sidecar.
"""

import json
import shutil
import subprocess
import sys

import pytest

from mflux.models.cosyvoice3.cli import dub

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for the mux")

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


def _make_clip(tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=12",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(clip),
        ],
        check=True,
        capture_output=True,
    )
    return clip


def test_dub_with_a_designed_voice_muxes_and_writes_metadata(tmp_path, monkeypatch):
    _make_stub(tmp_path, monkeypatch)
    package = _make_package(tmp_path)
    clip = _make_clip(tmp_path)
    output = tmp_path / "out.mp4"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mlxgen", "dub",
            "--video-path", str(clip),
            "--voice-design", "一位温暖沉稳的女性声音",
            "--text", "晚上好，欢迎回来。",
            "--voice-model-path", str(package),
            "--output", str(output),
            "--metadata",
        ],
    )
    dub.main()

    assert output.is_file()
    metadata = json.loads(output.with_suffix(output.suffix + ".json").read_text(encoding="utf-8"))
    assert metadata["voice_backend"] == "qwen3-tts-voice-design"
    assert metadata["mode"] == "voice_design"
    assert metadata["voice_description"] == "一位温暖沉稳的女性声音"
    assert metadata["ref_audio"] is None
    assert metadata["sample_rate"] == 24000
    assert metadata["package"] == str(package)

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "video" in probe.stdout
    assert "audio" in probe.stdout


def test_dub_still_requires_a_reference_voice_without_voice_design(tmp_path, capsys):
    clip = _make_clip(tmp_path)
    monkey_argv = [
        "mlxgen", "dub",
        "--video-path", str(clip),
        "--text", "旁白内容",
    ]
    original = sys.argv
    sys.argv = monkey_argv
    try:
        with pytest.raises(SystemExit):
            dub.main()
    finally:
        sys.argv = original
    assert "--ref-audio" in capsys.readouterr().err
