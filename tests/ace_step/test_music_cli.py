"""Tests for the ACE-Step 1.5 `mlxgen music` command surface.

Everything here runs without the prepared package: argument parsing, validation
and the metadata sidecar are pure logic. The runner is replaced by a fake module
so no worker, no weights and no network are involved.
"""

from __future__ import annotations

import io
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mflux.models.ace_step import constants
from mflux.models.ace_step.cli import music


@dataclass
class FakePackage:
    root: Path


@dataclass
class FakeResult:
    output_path: Path
    audio_seconds: float
    sample_rate: int
    gen_seconds: float


@dataclass
class FakeRunner:
    resolved_with: str | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def resolve_package(self, model_path: str | None) -> FakePackage:
        self.resolved_with = model_path
        return FakePackage(root=Path(model_path or "/tmp/ace-step-pkg"))

    def synthesize(self, package: FakePackage, **kwargs: Any) -> FakeResult:
        self.calls.append(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fLaC")
        return FakeResult(
            output_path=output_path,
            audio_seconds=61.5,
            sample_rate=44100,
            gen_seconds=12.25,
        )


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    runner = FakeRunner()
    module = types.ModuleType("mflux.models.ace_step.runner")
    module.resolve_package = runner.resolve_package  # type: ignore[attr-defined]
    module.synthesize = runner.synthesize  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mflux.models.ace_step.runner", module)
    return runner


def _run_cli(argv: list[str]) -> None:
    with patch("sys.argv", ["mlxgen", "music", *argv]):
        music.main()


def test_parser_defaults():
    args = music._parser().parse_args([])

    assert args.model == constants.MODEL_ALIAS
    assert args.model_path == constants.default_package_dir()
    assert args.prompt is None
    assert args.lyrics is None
    assert args.lyrics_file is None
    assert args.duration == constants.DEFAULT_DURATION_SECONDS
    assert args.bpm is None
    assert args.keyscale is None
    assert args.timesignature is None
    assert args.instrumental is False
    assert args.vocal_language == "zh"
    assert args.use_lm is True
    assert args.seed == 42
    assert args.output == "song.flac"
    assert args.metadata is False
    assert args.json_events is False
    assert args.debug is False
    assert args.replace is True


def test_prompt_and_caption_are_the_same_option():
    parser = music._parser()

    assert parser.parse_args(["--prompt", "钢琴伴奏"]).prompt == "钢琴伴奏"
    assert parser.parse_args(["--caption", "钢琴伴奏"]).prompt == "钢琴伴奏"


def test_no_use_lm_flag_is_off_by_default():
    parser = music._parser()

    assert parser.parse_args(["--no-use-lm"]).use_lm is False
    assert parser.parse_args(["--use-lm"]).use_lm is True


def test_lyrics_file_reads_stdin(monkeypatch: pytest.MonkeyPatch):
    parser = music._parser()
    args = parser.parse_args(["--lyrics-file", "-"])
    monkeypatch.setattr("sys.stdin", io.StringIO("第一行歌词\n第二行歌词\n"))

    assert music._read_lyrics(args, parser) == "第一行歌词\n第二行歌词"


def test_lyrics_file_reads_a_utf8_file(tmp_path: Path):
    parser = music._parser()
    lyrics_file = tmp_path / "lyrics.txt"
    lyrics_file.write_text("歌词内容\n", encoding="utf-8")
    args = parser.parse_args(["--lyrics-file", str(lyrics_file)])

    assert music._read_lyrics(args, parser) == "歌词内容"


def test_empty_prompt_is_rejected(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        _run_cli(["--prompt", "   "])

    assert "--prompt" in capsys.readouterr().err


def test_missing_prompt_is_rejected(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        _run_cli([])

    assert "--prompt" in capsys.readouterr().err


def test_duration_above_the_maximum_is_rejected(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        _run_cli(["--prompt", "轻音乐", "--duration", str(constants.MAX_DURATION_SECONDS + 1)])

    assert "must not exceed" in capsys.readouterr().err


def test_non_positive_duration_is_rejected(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        _run_cli(["--prompt", "轻音乐", "--duration", "0"])

    assert "--duration must be positive" in capsys.readouterr().err


def test_unknown_model_is_rejected(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        _run_cli(["--model", "suno", "--prompt", "轻音乐"])

    assert "ace-step" in capsys.readouterr().err


def test_no_use_lm_without_lyrics_is_rejected(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit):
        _run_cli(["--prompt", "轻音乐", "--no-use-lm"])

    assert "--lyrics" in capsys.readouterr().err


def test_metadata_sidecar_is_written(tmp_path: Path, fake_runner: FakeRunner):
    output = tmp_path / "song.flac"
    _run_cli(
        [
            "--prompt",
            "一首中文流行歌曲，钢琴伴奏，副歌激昂",
            "--lyrics",
            "晚风吹过城市的屋顶",
            "--duration",
            "30",
            "--bpm",
            "96",
            "--keyscale",
            "C major",
            "--timesignature",
            "4/4",
            "--vocal-language",
            "zh",
            "--seed",
            "7",
            "--output",
            str(output),
            "--metadata",
        ]
    )

    sidecar = output.with_suffix(output.suffix + ".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["task"] == "text-to-music"
    assert metadata["model"] == constants.MODEL_ALIAS
    assert metadata["prompt"] == "一首中文流行歌曲，钢琴伴奏，副歌激昂"
    assert metadata["lyrics"] == "晚风吹过城市的屋顶"
    assert metadata["duration"] == 30.0
    assert metadata["bpm"] == 96
    assert metadata["keyscale"] == "C major"
    assert metadata["timesignature"] == "4/4"
    assert metadata["instrumental"] is False
    assert metadata["vocal_language"] == "zh"
    assert metadata["seed"] == 7
    assert metadata["output_path"] == str(output)
    assert metadata["audio_seconds"] == 61.5
    assert metadata["sample_rate"] == 44100
    assert metadata["gen_seconds"] == 12.25
    assert metadata["package"]


def test_runner_receives_the_requested_parameters(tmp_path: Path, fake_runner: FakeRunner):
    output = tmp_path / "out.flac"
    _run_cli(
        [
            "--prompt",
            "轻音乐",
            "--lyrics",
            "歌词",
            "--duration",
            "12",
            "--bpm",
            "120",
            "--keyscale",
            "A minor",
            "--timesignature",
            "3/4",
            "--instrumental",
            "--vocal-language",
            "en",
            "--no-use-lm",
            "--seed",
            "3",
            "--output",
            str(output),
        ]
    )

    assert len(fake_runner.calls) == 1
    call = fake_runner.calls[0]
    assert call["caption"] == "轻音乐"
    assert call["lyrics"] == "歌词"
    assert call["duration"] == 12.0
    assert call["bpm"] == 120
    assert call["keyscale"] == "A minor"
    assert call["timesignature"] == "3/4"
    assert call["instrumental"] is True
    assert call["vocal_language"] == "en"
    assert call["use_lm"] is False
    assert call["seed"] == 3
    assert call["audio_format"] == "flac"
    assert call["command"] == "mlxgen music"
    assert call["output_path"] == output


def test_lyrics_are_optional_when_the_planner_lm_runs(tmp_path: Path, fake_runner: FakeRunner):
    output = tmp_path / "out.flac"
    _run_cli(["--prompt", "轻音乐", "--output", str(output)])

    assert fake_runner.calls[0]["lyrics"] is None
    assert fake_runner.calls[0]["use_lm"] is True


def test_keep_prompt_keeps_the_caption_verbatim(tmp_path: Path, fake_runner: FakeRunner):
    output = tmp_path / "out.flac"
    _run_cli(["--prompt", "编钟与古筝", "--output", str(output), "--keep-prompt"])

    assert fake_runner.calls[0]["keep_caption"] is True


def test_keep_prompt_is_off_by_default(tmp_path: Path, fake_runner: FakeRunner):
    output = tmp_path / "out.flac"
    _run_cli(["--prompt", "轻音乐", "--output", str(output)])

    assert fake_runner.calls[0]["keep_caption"] is False


def test_no_replace_keeps_the_existing_output(tmp_path: Path, fake_runner: FakeRunner):
    output = tmp_path / "song.flac"
    output.write_bytes(b"old")
    _run_cli(["--prompt", "轻音乐", "--output", str(output), "--no-replace"])

    assert fake_runner.calls[0]["output_path"] == tmp_path / "song_1.flac"
    assert output.read_bytes() == b"old"


def test_model_path_is_forwarded_to_the_runner(tmp_path: Path, fake_runner: FakeRunner):
    package_dir = tmp_path / "pkg"
    _run_cli(["--prompt", "轻音乐", "--model-path", str(package_dir), "--output", str(tmp_path / "o.flac")])

    assert fake_runner.resolved_with == str(package_dir)


def test_leading_music_token_is_accepted(tmp_path: Path, fake_runner: FakeRunner):
    output = tmp_path / "out.flac"
    with patch("sys.argv", ["mlxgen-music", "--prompt", "轻音乐", "--output", str(output)]):
        music.main()

    assert fake_runner.calls[0]["caption"] == "轻音乐"
