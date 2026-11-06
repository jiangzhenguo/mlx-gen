"""`mlxgen voice-design` — create a voice from a text description and speak with it.

Unlike `mlxgen tts` (which clones an existing voice from a reference clip), this
command *designs* the voice: a natural-language description such as "a warm
female voice in her thirties, speaking slowly" is turned into a speaker by
Qwen3-TTS VoiceDesign. No reference audio required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mflux.cli.output_paths import resolve_collision_free_path
from mflux.cli.runtime_events import cli_print
from mflux.models.qwen3_tts_voice import constants
from mflux.models.qwen3_tts_voice.profiles import print_profile_list, resolve_selection
from mflux.models.qwen3_tts_voice.runner import VoiceDesignPackageError, resolve_package, synthesize


def main() -> None:
    argv = sys.argv[1:]
    # Allow `mlxgen-voice-design ...` and `mlxgen voice-design ...` (the dispatcher strips the command).
    if argv and argv[0] in {"voice-design", "voice_design"}:
        argv = argv[1:]
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _run(args, parser)
    except (VoiceDesignPackageError, FileNotFoundError, ValueError, RuntimeError) as exc:
        cli_print(str(exc), json_events=bool(args.json_events), error=True)
        raise SystemExit(1) from None


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not constants.is_qwen3_tts_voice_alias(args.model):
        parser.error(
            f"mlxgen voice-design currently supports --model {constants.MODEL_ALIAS} only. Got: {args.model!r}"
        )
    if args.list_voice_profiles:
        print_profile_list(args.voice_profile_file)
        return
    text = _read_text(args, parser)
    selection = resolve_selection(args, parser, required=True)
    voice = selection.description or ""
    language = constants.normalize_language(selection.language)
    seed = selection.seed if selection.seed is not None else 42

    output_path = resolve_collision_free_path(args.output, overwrite=args.replace)
    package = resolve_package(args.model_path)

    result = synthesize(
        package,
        text=text,
        output_path=output_path,
        voice_description=voice,
        language=language,
        seed=seed,
        json_events=bool(args.json_events),
        command="mlxgen voice-design",
        temperature=selection.temperature,
        debug=bool(args.debug),
    )
    if args.metadata:
        metadata = {
            "task": "voice-design",
            "model": constants.MODEL_ALIAS,
            "weights_repo": constants.WEIGHTS_REPO_ID,
            "upstream_repo": constants.UPSTREAM_REPO_ID,
            "voice_description": voice,
            "voice_profile": selection.profile.name if selection.profile else None,
            "voice_profile_source": selection.profile.source if selection.profile else None,
            "language": language,
            "text": text,
            "seed": seed,
            "temperature": selection.temperature,
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
        prog="mlxgen voice-design",
        description=(
            "Design a voice from a natural-language description and speak a text with it "
            "(Qwen3-TTS VoiceDesign, 10 languages, MLX-native). "
            "Requires a prepared package: mlxgen prepare --model qwen3-tts-voice-design."
        ),
    )
    parser.add_argument("--model", "-m", default=constants.MODEL_ALIAS, help="Model alias (default: qwen3-tts-voice-design).")
    parser.add_argument(
        "--model-path",
        default=None,
        help=f"Prepared package directory (default: {constants.default_package_dir()}).",
    )
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument("--text", type=str, help='Text to synthesize; "-" reads stdin.')
    text_group.add_argument("--text-file", type=str, default=None, help="Path to a UTF-8 text file.")
    voice_group = parser.add_mutually_exclusive_group()
    voice_group.add_argument(
        "--voice",
        type=str,
        default=None,
        help='Description of the wanted voice, e.g. "a deep male voice with a warm, raspy timbre".',
    )
    voice_group.add_argument("--voice-file", type=str, default=None, help="Path to a UTF-8 file with the description.")
    parser.add_argument(
        "--language",
        default=None,
        help=f"Speech language: {', '.join(constants.LANGUAGES)} (default: auto).",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed (default 42).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (model default when omitted).",
    )
    parser.add_argument("--output", "-o", type=str, default="speech.wav", help='Output wav path (default "speech.wav").')
    parser.add_argument(
        "--voice-profile",
        type=str,
        default=None,
        help="Named voice profile to use instead of --voice (see --list-voice-profiles).",
    )
    parser.add_argument(
        "--voice-profile-file",
        type=str,
        default=None,
        help="Voice profile JSON file (default: ./mlxgen-voice-profiles.json or ~/.config/mlxgen/voice-profiles.json).",
    )
    parser.add_argument(
        "--list-voice-profiles",
        action="store_true",
        help="Print the available voice profiles and exit.",
    )
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
