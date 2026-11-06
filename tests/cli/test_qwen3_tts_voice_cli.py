"""Tests for the Qwen3-TTS VoiceDesign command surface.

Everything here runs without the prepared package: argument validation, package
resolution, language mapping and the `mlxgen dub` backend switch are pure logic.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mflux.cli import mlx_gen
from mflux.models.cosyvoice3.cli import dub
from mflux.models.qwen3_tts_voice import constants
from mflux.models.qwen3_tts_voice.cli import voice_design
from mflux.models.qwen3_tts_voice.runner import VoiceDesignPackageError, resolve_package


def test_voice_design_requires_a_voice_description(capsys):
    with patch("sys.argv", ["mlxgen", "voice-design", "--text", "你好"]):
        with pytest.raises(SystemExit):
            voice_design.main()
    err = capsys.readouterr().err
    assert "--voice" in err
    assert "describing the voice to design" in err


def test_voice_design_requires_text(capsys):
    with patch("sys.argv", ["mlxgen", "voice-design", "--voice", "a warm voice"]):
        with pytest.raises(SystemExit):
            voice_design.main()
    assert "Provide --text or --text-file" in capsys.readouterr().err


def test_voice_design_rejects_unknown_model(capsys):
    with patch("sys.argv", ["mlxgen", "voice-design", "-m", "kokoro", "--voice", "v", "--text", "hi"]):
        with pytest.raises(SystemExit):
            voice_design.main()
    assert "qwen3-tts-voice-design" in capsys.readouterr().err


def test_voice_design_unprepared_package_guides_to_prepare(tmp_path, capsys):
    with patch(
        "sys.argv",
        [
            "mlxgen",
            "voice-design",
            "--voice",
            "a warm voice",
            "--text",
            "你好",
            "--model-path",
            str(tmp_path / "pkg"),
        ],
    ):
        with pytest.raises(SystemExit):
            voice_design.main()
    err = capsys.readouterr().err
    assert "mlxgen prepare --model qwen3-tts-voice-design" in err


def test_resolve_package_reports_missing_parts(tmp_path):
    root = tmp_path / "pkg"
    root.mkdir()
    with pytest.raises(VoiceDesignPackageError) as excinfo:
        resolve_package(str(root))
    message = str(excinfo.value)
    assert "config.json" in message
    assert "model.safetensors" in message
    assert "speech_tokenizer" in message


def test_language_normalization_accepts_aliases():
    assert constants.normalize_language("zh") == "chinese"
    assert constants.normalize_language("EN") == "english"
    assert constants.normalize_language(None) == "auto"
    assert constants.normalize_language("auto") == "auto"


def test_language_normalization_rejects_unknown_language():
    with pytest.raises(ValueError) as excinfo:
        constants.normalize_language("klingon")
    assert "Unsupported language" in str(excinfo.value)


def test_alias_detection_accepts_the_documented_spellings():
    assert constants.is_qwen3_tts_voice_alias("qwen3-tts-voice-design")
    assert constants.is_qwen3_tts_voice_alias("Qwen3-TTS")
    assert constants.is_qwen3_tts_voice_alias("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
    assert not constants.is_qwen3_tts_voice_alias("cosyvoice3")


def test_router_dispatches_the_voice_design_command(capsys):
    with patch("sys.argv", ["mlxgen", "voice-design", "--help"]):
        with pytest.raises(SystemExit) as excinfo:
            mlx_gen.main()
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--voice" in out
    assert "--language" in out


def test_dub_voice_design_uses_the_voice_design_backend(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"RIFF")
    seen = {}

    def fake_voice_synthesize(package, *, text, output_path, voice_description, language, seed, json_events, command, temperature=None, debug=False):
        seen["voice_description"] = voice_description
        seen["language"] = language
        seen["text"] = text
        output_path.write_bytes(b"RIFF")
        return SimpleNamespace(
            output_path=output_path,
            audio_seconds=1.0,
            sample_rate=24000,
            rtf=0.5,
            synth_seconds=0.5,
            worker_event_count=1,
        )

    monkeypatch.setattr(dub, "_probe_video", lambda path: (2.0, False))
    monkeypatch.setattr(dub, "_run_mux", lambda *args, **kwargs: None)
    monkeypatch.setattr(dub.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "mflux.models.qwen3_tts_voice.runner.resolve_package",
        lambda path: SimpleNamespace(root=tmp_path / "pkg", model_dir=tmp_path),
    )
    monkeypatch.setattr("mflux.models.qwen3_tts_voice.runner.synthesize", fake_voice_synthesize)

    with patch(
        "sys.argv",
        [
            "mlxgen",
            "dub",
            "--video-path",
            str(video),
            "--text",
            "旁白内容",
            "--voice-design",
            "a warm female voice",
            "--language",
            "zh",
        ],
    ):
        dub.main()

    assert seen["voice_description"] == "a warm female voice"
    assert seen["language"] == "chinese"
    assert seen["text"] == "旁白内容"


def test_dub_without_voice_design_still_requires_a_reference_voice(tmp_path, capsys):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"RIFF")
    with patch(
        "sys.argv",
        ["mlxgen", "dub", "--video-path", str(video), "--text", "旁白内容"],
    ):
        with pytest.raises(SystemExit):
            dub.main()
    err = capsys.readouterr().err
    assert "--ref-audio" in err


def test_prepare_writes_a_manifest_and_validates_the_package(tmp_path, monkeypatch):
    """The download is stubbed; everything around it is the real code path."""
    from mflux.models.qwen3_tts_voice import prepare

    def fake_download(model_dir, *, json_events, source):
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"0" * 16)
        (model_dir / "speech_tokenizer").mkdir(parents=True, exist_ok=True)
        (model_dir / "speech_tokenizer" / "model.safetensors").write_bytes(b"0" * 16)

    monkeypatch.setattr(prepare, "_download_weights", fake_download)
    package = prepare.prepare_package(tmp_path / "pkg")

    manifest = json.loads((package / constants.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["model"] == constants.MODEL_ALIAS
    assert manifest["weights_repo"] == constants.WEIGHTS_REPO_ID
    assert manifest["runtime"].startswith("mlx-audio")
    assert "chinese" in manifest["languages"]
    # The package must pass the same gate the runner uses.
    resolve_package(str(package))
