"""Tests for the CosyVoice3 `mlxgen tts` / `mlxgen dub` command surface.

Everything here runs without the prepared package: mode resolution, argument
validation, and the ffmpeg mux argv builder are pure logic.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.models.cosyvoice3.cli import dub, tts
from mflux.models.cosyvoice3.runner import CosyVoice3PackageError, resolve_package


def _ref_audio(tmp_path: Path) -> Path:
    ref = tmp_path / "voice.wav"
    ref.write_bytes(b"RIFF")
    return ref


def test_tts_requires_text(tmp_path, capsys):
    ref = _ref_audio(tmp_path)
    with patch("sys.argv", ["mlxgen", "tts", "--ref-audio", str(ref)]):
        with pytest.raises(SystemExit):
            tts.main()
    assert "Provide --text or --text-file" in capsys.readouterr().err


def test_tts_requires_ref_audio(tmp_path, capsys):
    with patch("sys.argv", ["mlxgen", "tts", "--text", "你好"]):
        with pytest.raises(SystemExit):
            tts.main()
    err = capsys.readouterr().err
    assert "--ref-audio" in err
    assert "no preset voices" in err


def test_tts_missing_ref_audio_file_is_rejected(tmp_path, capsys):
    with patch("sys.argv", ["mlxgen", "tts", "--text", "你好", "--ref-audio", str(tmp_path / "nope.wav")]):
        with pytest.raises(SystemExit):
            tts.main()
    assert "--ref-audio not found" in capsys.readouterr().err


def test_tts_mode_resolution(tmp_path):
    ref = _ref_audio(tmp_path)
    parser = tts._parser()
    base = ["--text", "你好", "--ref-audio", str(ref)]

    args = parser.parse_args(base)
    assert tts.resolve_mode(args, parser) == "cross_lingual"

    args = parser.parse_args([*base, "--ref-text", "参考文字"])
    assert tts.resolve_mode(args, parser) == "zero_shot"

    args = parser.parse_args([*base, "--instruct", "用温柔的语气说"])
    assert tts.resolve_mode(args, parser) == "instruct2"

    args = parser.parse_args([*base, "--ref-text", "参考文字", "--instruct", "快一点"])
    assert tts.resolve_mode(args, parser) == "instruct2"


def test_tts_rejects_unknown_model(tmp_path, capsys):
    ref = _ref_audio(tmp_path)
    with patch("sys.argv", ["mlxgen", "tts", "-m", "kokoro", "--text", "hi", "--ref-audio", str(ref)]):
        with pytest.raises(SystemExit):
            tts.main()
    assert "cosyvoice3" in capsys.readouterr().err


def test_tts_unprepared_package_guides_to_prepare(tmp_path, capsys):
    ref = _ref_audio(tmp_path)
    with patch(
        "sys.argv",
        ["mlxgen", "tts", "--text", "你好", "--ref-audio", str(ref), "--model-path", str(tmp_path / "pkg")],
    ):
        with pytest.raises(SystemExit):
            tts.main()
    err = capsys.readouterr().err
    assert "mlxgen prepare --model cosyvoice3" in err


def test_resolve_package_reports_missing_parts(tmp_path):
    root = tmp_path / "pkg"
    root.mkdir()
    with pytest.raises(CosyVoice3PackageError) as excinfo:
        resolve_package(str(root))
    message = str(excinfo.value)
    assert "cosyvoice3.yaml" in message
    assert "venv" in message


def test_dub_requires_video(tmp_path, capsys):
    ref = _ref_audio(tmp_path)
    with patch("sys.argv", ["mlxgen", "dub", "--video-path", str(tmp_path / "x.mp4"), "--text", "hi", "--ref-audio", str(ref)]):
        with pytest.raises(SystemExit):
            dub.main()
    assert "--video-path not found" in capsys.readouterr().err


def test_dub_mux_replace_pads_and_clamps_to_video_length(tmp_path):
    argv = dub._capture_mux_argv(
        tmp_path / "video.mp4",
        tmp_path / "narration.wav",
        tmp_path / "out.mp4",
        video_seconds=14.375,
        mix_mode="replace",
        orig_volume=0.15,
        tts_volume=1.0,
    )
    assert argv[0] == "ffmpeg"
    joined = " ".join(argv)
    assert "-c:v copy" in joined
    assert "apad=whole_dur=14.375" in joined
    assert "-t 14.375" in joined
    assert "-map 0:v:0" in joined and "-map 1:a:0" in joined


def test_dub_mux_mix_blends_original_audio(tmp_path):
    argv = dub._capture_mux_argv(
        tmp_path / "video.mp4",
        tmp_path / "narration.wav",
        tmp_path / "out.mp4",
        video_seconds=14.375,
        mix_mode="mix",
        orig_volume=0.15,
        tts_volume=1.2,
    )
    joined = " ".join(argv)
    assert "volume=0.15" in joined
    assert "volume=1.2" in joined
    assert "amix=inputs=2:duration=first" in joined
    assert "-c:a aac" in joined
