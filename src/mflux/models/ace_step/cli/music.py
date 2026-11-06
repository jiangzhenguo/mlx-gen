"""`mlxgen music` — ACE-Step 1.5 text-to-music through the isolated worker.

The music backend runs in its own prepared package (weights + upstream runtime +
isolated venv), so it has its own command surface instead of a route through
`mlxgen generate`. The runner module is imported lazily: the audio stack must
stay off the plain router import path.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

from mflux.cli.output_paths import resolve_collision_free_path
from mflux.cli.runtime_events import cli_print
from mflux.models.ace_step import constants

RUNNER_MODULE = "mflux.models.ace_step.runner"


def main() -> None:
    argv = sys.argv[1:]
    # Allow `mlxgen-music ...` and `mlxgen music ...` (the dispatcher strips "music").
    if argv and argv[0] == "music":
        argv = argv[1:]
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _run(args, parser)
    except _error_types() as exc:
        cli_print(str(exc), json_events=bool(getattr(args, "json_events", False)), error=True)
        raise SystemExit(1) from None


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not constants.is_ace_step_alias(args.model):
        parser.error(f"mlxgen music currently supports --model {constants.MODEL_ALIAS} only. Got: {args.model!r}")
    caption = _read_prompt(args, parser)
    lyrics = _read_lyrics(args, parser)
    duration = _validate_duration(args, parser)
    if lyrics is None and not args.use_lm and not args.instrumental:
        parser.error(
            "--lyrics (or --lyrics-file) is required when the planner LM is disabled (--no-use-lm): "
            "ACE-Step has nothing to sing. Add --instrumental for a track without vocals."
        )

    output_path = resolve_collision_free_path(args.output, overwrite=args.replace)
    runner = _import_runner()
    package = runner.resolve_package(str(args.model_path) if args.model_path else None)
    result = runner.synthesize(
        package,
        caption=caption,
        lyrics=lyrics,
        output_path=output_path,
        duration=duration,
        bpm=args.bpm,
        keyscale=args.keyscale,
        timesignature=args.timesignature,
        instrumental=bool(args.instrumental),
        vocal_language=args.vocal_language,
        seed=args.seed,
        audio_format=_audio_format(output_path),
        use_lm=bool(args.use_lm),
        keep_caption=bool(args.keep_prompt),
        json_events=bool(args.json_events),
        command="mlxgen music",
        debug=bool(args.debug),
    )
    if args.metadata:
        _write_metadata(
            args,
            package,
            result,
            output_path=output_path,
            caption=caption,
            lyrics=lyrics,
            duration=duration,
        )


def _read_prompt(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """The style description ACE-Step conditions on (its upstream `caption`)."""
    prompt = (args.prompt or "").strip()
    if not prompt:
        parser.error('--prompt (style description) is required, e.g. "一首中文流行歌曲，钢琴伴奏，副歌激昂".')
    return prompt


def _read_lyrics(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str | None:
    """Return the lyrics to sing, or None when the planner LM writes them."""
    if args.lyrics is not None:
        lyrics = args.lyrics.strip()
        if not lyrics:
            parser.error("--lyrics is empty.")
        return lyrics
    if args.lyrics_file is None:
        return None
    if args.lyrics_file == "-":
        lyrics = sys.stdin.read().strip()
    else:
        try:
            lyrics = Path(args.lyrics_file).expanduser().read_text(encoding="utf-8").strip()
        except OSError as exc:
            parser.error(f"Could not read --lyrics-file {args.lyrics_file}: {exc}")
    if not lyrics:
        parser.error(f"Lyrics file {args.lyrics_file} is empty.")
    return lyrics


def _validate_duration(args: argparse.Namespace, parser: argparse.ArgumentParser) -> float:
    duration = float(args.duration)
    if duration <= 0:
        parser.error(f"--duration must be positive. Got: {duration}")
    if duration > constants.MAX_DURATION_SECONDS:
        parser.error(f"--duration must not exceed {constants.MAX_DURATION_SECONDS} seconds. Got: {duration}")
    return duration


def _audio_format(output_path: Path) -> str:
    """The container the worker writes: the output suffix, else the default."""
    suffix = output_path.suffix.lower().lstrip(".")
    return suffix or constants.DEFAULT_AUDIO_FORMAT


def _import_runner() -> ModuleType:
    try:
        return importlib.import_module(RUNNER_MODULE)
    except ImportError as exc:
        raise RuntimeError(
            f"The ACE-Step runner is not available in this installation ({RUNNER_MODULE}: {exc}). "
            "Reinstall mlx-gen to restore the music backend."
        ) from exc


def _write_metadata(
    args: argparse.Namespace,
    package: object,
    result: object,
    *,
    output_path: Path,
    caption: str,
    lyrics: str | None,
    duration: float,
) -> None:
    metadata = {
        "task": "text-to-music",
        "model": constants.MODEL_ALIAS,
        "weights_repo": constants.WEIGHTS_REPO_ID,
        "prompt": caption,
        # The lyrics the model actually sang: the ones passed in, or the ones the
        # planner LM wrote from the prompt.
        "lyrics": getattr(result, "lyrics", None) or lyrics,
        "duration": duration,
        "bpm": getattr(result, "bpm", None) or args.bpm,
        "keyscale": getattr(result, "keyscale", None) or args.keyscale,
        "timesignature": getattr(result, "timesignature", None) or args.timesignature,
        "instrumental": bool(args.instrumental),
        "keep_prompt": bool(args.keep_prompt),
        "vocal_language": getattr(result, "vocal_language", None) or args.vocal_language,
        "seed": args.seed,
        "output_path": str(getattr(result, "output_path", output_path)),
        "audio_seconds": getattr(result, "audio_seconds", None),
        "sample_rate": getattr(result, "sample_rate", None),
        "gen_seconds": getattr(result, "gen_seconds", None),
        "package": str(getattr(package, "root", "")),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cli_print(f"Metadata: {metadata_path}", json_events=bool(args.json_events))


def _error_types() -> tuple[type[BaseException], ...]:
    """Exception types the CLI turns into a clean exit-1 message.

    The runner's package error is added when the runner module is loaded, so an
    unprepared package is reported as "run mlxgen prepare --model ace-step"
    instead of a traceback.
    """
    error_types: list[type[BaseException]] = [FileNotFoundError, ValueError, RuntimeError]
    runner = sys.modules.get(RUNNER_MODULE)
    package_error = getattr(runner, "AceStepPackageError", None) if runner is not None else None
    if isinstance(package_error, type) and issubclass(package_error, BaseException):
        error_types.append(package_error)
    return tuple(error_types)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxgen music",
        description=(
            "Generate a song with ACE-Step 1.5 (text-to-music with optional lyrics, vocals or an "
            "instrumental track). Requires a prepared package: mlxgen prepare --model ace-step."
        ),
    )
    parser.add_argument("--model", "-m", default=constants.MODEL_ALIAS, help="Model alias (default: ace-step).")
    parser.add_argument(
        "--model-path",
        default=constants.default_package_dir(),
        help=f"Prepared package directory (default: {constants.default_package_dir()}).",
    )
    parser.add_argument(
        "--prompt",
        "--caption",
        dest="prompt",
        default=None,
        help='Style description of the song, e.g. "一首中文流行歌曲，钢琴伴奏，副歌激昂".',
    )
    lyrics_group = parser.add_mutually_exclusive_group()
    lyrics_group.add_argument(
        "--lyrics",
        default=None,
        help="Lyrics to sing; omit them to let the planner LM write the lyrics from --prompt.",
    )
    lyrics_group.add_argument(
        "--lyrics-file",
        default=None,
        help='Path to a UTF-8 lyrics file; "-" reads stdin.',
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=constants.DEFAULT_DURATION_SECONDS,
        help=f"Song length in seconds (default: {constants.DEFAULT_DURATION_SECONDS}).",
    )
    parser.add_argument("--bpm", type=int, default=None, help="Tempo in beats per minute.")
    parser.add_argument("--keyscale", default=None, help='Key and scale, e.g. "C major".')
    parser.add_argument("--timesignature", default=None, help='Time signature, e.g. "4/4".')
    parser.add_argument("--instrumental", action="store_true", help="Generate an instrumental track (no vocals).")
    parser.add_argument(
        "--vocal-language",
        default="zh",
        help="Language of the sung lyrics (default: zh).",
    )
    parser.add_argument(
        "--use-lm",
        dest="use_lm",
        action="store_true",
        default=True,
        help="Use the planner LM that writes the lyrics and the song blueprint (default).",
    )
    parser.add_argument(
        "--no-use-lm",
        dest="use_lm",
        action="store_false",
        help="Disable the planner LM; --lyrics is then required unless --instrumental is set.",
    )
    parser.add_argument(
        "--keep-prompt",
        dest="keep_prompt",
        action="store_true",
        help="Keep --prompt verbatim instead of letting the planner LM rewrite it into a caption. "
        "Use it when the prompt names instruments the LM tends to drop (e.g. bianzhong/编钟).",
    )
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument("--output", "-o", default="song.flac", help='Output audio path (default "song.flac").')
    parser.add_argument("--metadata", action="store_true", help="Write <output>.json metadata sidecar.")
    parser.add_argument("--json-events", action="store_true", help="Emit machine-readable runtime events on stdout.")
    parser.add_argument("--debug", action="store_true", help="Stream worker logs (stderr) through to the terminal.")
    parser.add_argument(
        "--replace",
        action="store_true",
        default=True,
        help="Replace the output when it exists (default).",
    )
    parser.add_argument("--no-replace", action="store_false", dest="replace")
    return parser


if __name__ == "__main__":
    main()
