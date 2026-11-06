"""Worker-level tests for the Qwen3-TTS VoiceDesign synthesis path.

The 4.5 GB weights are not needed to exercise the worker's contract: a stub
`mlx_audio` package on PYTHONPATH replaces the real inference stack, so the
tests cover the JSONL event contract, the WAV writer and the runner's relay
without downloading a model.
"""

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from mflux.models.qwen3_tts_voice.runner import VoiceDesignPackage, synthesize

WORKER = Path(__file__).resolve().parents[2] / "src" / "mflux" / "models" / "qwen3_tts_voice" / "worker.py"

STUB = '''
import numpy as np


class _Result:
    def __init__(self, audio, sample_rate):
        self.audio = audio
        self.sample_rate = sample_rate


class _Model:
    sample_rate = 24000

    def generate_voice_design(self, text, instruct, language="auto", stream=False, **kwargs):
        assert text, "text must not be empty"
        assert instruct, "voice description must not be empty"
        # Two chunks so the streaming branch is exercised.
        for part in range(2):
            samples = np.zeros(self.sample_rate // 4, dtype=np.float32)
            samples[:10] = 0.5
            yield _Result(samples, self.sample_rate)


def load_model(path):
    return _Model()
'''


@pytest.fixture()
def stub_mlx_audio(tmp_path, monkeypatch):
    """Install a stub `mlx_audio` package and expose it to subprocesses."""
    package = tmp_path / "stub" / "mlx_audio" / "tts"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "utils.py").write_text(STUB, encoding="utf-8")

    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "stub"))
    return tmp_path / "stub"


def test_worker_emits_the_runtime_event_contract(tmp_path, stub_mlx_audio):
    text_file = tmp_path / "text.txt"
    text_file.write_text("我爱你", encoding="utf-8")
    output = tmp_path / "out.wav"

    completed = subprocess.run(
        [
            sys.executable,
            str(WORKER),
            "--model-dir",
            str(tmp_path / "model"),
            "--text-file",
            str(text_file),
            "--output",
            str(output),
            "--voice",
            "一位温暖的女性声音",
            "--language",
            "chinese",
            "--seed",
            "42",
            "--command",
            "mlxgen voice-design",
        ],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    assert completed.returncode == 0, completed.stderr

    events = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    phases = [event["phase"] for event in events]
    assert phases[0] == "loading"
    assert "loaded" in phases
    assert "synthesizing" in phases
    assert phases[-1] == "complete"

    for event in events:
        assert event["type"] == "runtime"
        assert event["command"] == "mlxgen voice-design"
        assert event["model"] == "qwen3-tts-voice-design"
        assert event["seed"] == 42

    complete = events[-1]
    assert complete["sample_rate"] == 24000
    assert complete["audio_seconds"] == pytest.approx(0.5, abs=0.01)
    assert complete["output_path"] == str(output)

    with wave.open(str(output), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 24000
        assert handle.getnframes() == 24000 // 2


def test_worker_reports_failures_as_events(tmp_path, stub_mlx_audio):
    text_file = tmp_path / "text.txt"
    text_file.write_text("", encoding="utf-8")
    output = tmp_path / "out.wav"

    completed = subprocess.run(
        [
            sys.executable,
            str(WORKER),
            "--model-dir",
            str(tmp_path / "model"),
            "--text-file",
            str(text_file),
            "--output",
            str(output),
            "--voice",
            "a voice",
        ],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    assert completed.returncode == 1
    events = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    assert events[-1]["phase"] == "failed"
    assert "empty" in events[-1]["error"]


def test_runner_relays_the_worker_result(tmp_path, stub_mlx_audio):
    package = VoiceDesignPackage(root=tmp_path / "pkg")
    output = tmp_path / "narration.wav"

    result = synthesize(
        package,
        text="我爱你",
        output_path=output,
        voice_description="一位温暖的女性声音",
        language="chinese",
        seed=42,
        json_events=False,
        command="mlxgen voice-design",
    )

    assert result.audio_seconds == pytest.approx(0.5, abs=0.01)
    assert result.sample_rate == 24000
    assert result.rtf is not None
    assert result.synth_seconds is not None
    assert result.worker_event_count >= 4
    assert output.is_file()
