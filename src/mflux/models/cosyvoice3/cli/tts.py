"""`mlxgen tts` — CosyVoice3 text-to-speech through the isolated worker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mflux.cli.output_paths import resolve_collision_free_path
from mflux.cli.runtime_events import cli_print
from mflux.models.cosyvoice3 import constants
from mflux.models.cosyvoice3.runner import CosyVoice3PackageError, resolve_package, synthesize


def main() -> None:
    argv = sys.argv[1:]
    # Allow `mlxgen-tts ...` and `mlxgen tts ...` (the dispatcher strips "tts").
    if argv and argv[0] == "tts":
        argv = argv[1:]
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _run(args, parser)
    except (CosyVoice3PackageError, FileNotFoundError, ValueError, RuntimeError) as exc:
        cli_print(str(exc), json_events=bool(args.json_events), error=True)
        raise SystemExit(1) from None


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not constants.is_cosyvoice3_alias(args.model):
        parser.error(f"mlxgen tts currently supports --model {constants.MODEL_ALIAS} only. Got: {args.model!r}")
    text = _read_text(args, parser)
    mode = resolve_mode(args, parser)

    output_path = resolve_collision_free_path(args.output, overwrite=args.replace)
    package = resolve_package(args.model_path)

    result = synthesize(
        package,
        text=text,
        output_path=output_path,
        ref_audio=Path(args.ref_audio).expanduser(),
        ref_text=args.ref_text,
        instruct=args.instruct,
        mode=mode,
        speed=args.speed,
        seed=args.seed,
        json_events=bool(args.json_events),
        command="mlxgen tts",
        debug=bool(args.debug),
    )
    if args.metadata:
        metadata = {
            "task": "text-to-speech",
            "model": constants.MODEL_ALIAS,
            "weights_repo": constants.WEIGHTS_REPO_ID,
            "mode": mode,
            "text": text,
            "ref_audio": str(args.ref_audio),
            "ref_text": args.ref_text,
            "instruct": args.instruct,
            "speed": args.speed,
            "seed": args.seed,
            "output_path": str(result.output_path),
            "sample_rate": result.sample_rate,
            "audio_seconds": result.audio_seconds,
            "synth_seconds": result.synth_seconds,
            "rtf": result.rtf,
            "package": str(package.root),
        }
        metadata_path = output_path.with_suffix(output_path.suffix + ".json")
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        cli_print(f"Metadata: {metadata_path}", json_events=bool(args.json_events))


def resolve_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """CosyVoice3 has no built-in speakers: a reference voice is always required.

    Mode follows the options given: --instruct selects the instructed mode, a
    transcript of the reference selects zero-shot cloning, and a bare reference
    falls back to cross-lingual cloning (which does not need the transcript).
    """
    if args.ref_audio is None:
        parser.error(
            "CosyVoice3 ships no preset voices; --ref-audio <voice.wav> is required (3-30 s of clean speech). "
            "Add --ref-text with the reference transcript for zero-shot cloning, or --instruct for style control."
        )
    if not Path(args.ref_audio).expanduser().is_file():
        parser.error(f"--ref-audio not found: {args.ref_audio}")
    if args.instruct is not None:
        return "instruct2"
    if args.ref_text is not None:
        return "zero_shot"
    return "cross_lingual"


def _read_text(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.text is None and args.text_file is None:
        parser.error("Provide --text or --text-file (use --text - to read from stdin).")
    if args.text == "-":
        text = sys.stdin.read().strip()
    elif args.text is not None:
        text = args.text.strip()
    else:
        try:
            text = Path(args.text_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            parser.error(f"Could not read --text-file {args.text_file}: {exc}")
    if not text:
        parser.error("Synthesis text is empty.")
    return text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxgen tts",
        description=(
            "Synthesize speech with CosyVoice3 (zero-shot voice cloning, instructed style, 9 languages "
            "and 18+ Chinese dialects). Requires a prepared package: mlxgen prepare --model cosyvoice3."
        ),
    )
    parser.add_argument("--model", "-m", default=constants.MODEL_ALIAS, help="Model alias (default: cosyvoice3).")
    parser.add_argument(
        "--model-path",
        default=None,
        help=f"Prepared package directory (default: {constants.default_package_dir()}).",
    )
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument("--text", type=str, help='Text to synthesize; "-" reads stdin.')
    text_group.add_argument("--text-file", type=str, default=None, help="Path to a UTF-8 text file.")
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="Reference voice (3-30 s of clean speech). Required: CosyVoice3 has no preset voices.",
    )
    parser.add_argument(
        "--ref-text",
        default=None,
        help="Exact transcript of --ref-audio; enables zero-shot cloning mode.",
    )
    parser.add_argument(
        "--instruct",
        default=None,
        help='Style instruction, e.g. "用温柔亲切的语气说" or "Speak quickly and excitedly"; selects instructed mode.',
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Speech rate multiplier (default 1.0).")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument("--output", "-o", type=str, default="speech.wav", help='Output wav path (default "speech.wav").')
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
