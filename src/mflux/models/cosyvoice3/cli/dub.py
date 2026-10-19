"""`mlxgen dub` — narrate an existing video with CosyVoice3 speech.

This is the standalone chaining command for MiniMax-H3 clips: H3 generates the
pictures (and its own short soundtrack), `mlxgen dub` synthesizes arbitrary-length
narration with CosyVoice3 and muxes it into the MP4 with ffmpeg — replacing the
original track or mixing over it. Nothing here re-runs the video model.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from mflux.cli.output_paths import resolve_collision_free_path
from mflux.cli.runtime_events import cli_print
from mflux.models.cosyvoice3 import constants
from mflux.models.cosyvoice3.cli.tts import _read_text, resolve_mode
from mflux.models.cosyvoice3.runner import CosyVoice3PackageError, resolve_package, synthesize


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "dub":
        argv = argv[1:]
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        _run(args, parser)
    except (CosyVoice3PackageError, FileNotFoundError, ValueError, RuntimeError) as exc:
        cli_print(str(exc), json_events=bool(getattr(args, "json_events", False)), error=True)
        raise SystemExit(1) from None


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not constants.is_cosyvoice3_alias(args.model):
        parser.error(f"mlxgen dub currently supports --model {constants.MODEL_ALIAS} only. Got: {args.model!r}")
    video_path = Path(args.video_path).expanduser()
    if not video_path.is_file():
        parser.error(f"--video-path not found: {args.video_path}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("mlxgen dub needs ffmpeg on PATH for the final mux (macOS: brew install ffmpeg).")
    text = _read_text(args, parser)
    mode = resolve_mode(args, parser)

    output_path = Path(args.output) if args.output else video_path.with_name(f"{video_path.stem}_dubbed.mp4")
    output_path = resolve_collision_free_path(output_path, overwrite=args.replace)

    package = resolve_package(args.model_path)
    video_seconds, has_audio = _probe_video(video_path)
    cli_print(
        f"Dubbing {video_path.name} ({video_seconds:.1f}s, {'with' if has_audio else 'no'} original audio track)...",
        json_events=bool(args.json_events),
    )

    wav_dir = Path(tempfile.mkdtemp(prefix="mlxgen_dub_"))
    wav_path = (output_path.parent / f"{output_path.stem}.wav") if args.keep_wav else wav_dir / "narration.wav"
    result = synthesize(
        package,
        text=text,
        output_path=wav_path,
        ref_audio=Path(args.ref_audio).expanduser(),
        ref_text=args.ref_text,
        instruct=args.instruct,
        mode=mode,
        speed=args.speed,
        seed=args.seed,
        json_events=bool(args.json_events),
        command="mlxgen dub",
        debug=bool(args.debug),
    )

    mix_mode = args.mode
    if mix_mode == "mix" and not has_audio:
        cli_print("Source video has no audio track; falling back to --mode replace.", json_events=bool(args.json_events))
        mix_mode = "replace"
    cli_print(f"Muxing narration ({mix_mode})...", json_events=bool(args.json_events))
    _run_mux(
        ffmpeg,
        video_path=video_path,
        wav_path=wav_path,
        output_path=output_path,
        video_seconds=video_seconds,
        mix_mode=mix_mode,
        orig_volume=args.orig_volume,
        tts_volume=args.tts_volume,
    )
    cli_print(f"Saved dubbed video to: {output_path}", json_events=bool(args.json_events))

    if args.metadata:
        metadata = {
            "task": "video-dub",
            "model": constants.MODEL_ALIAS,
            "weights_repo": constants.WEIGHTS_REPO_ID,
            "mode": mode,
            "text": text,
            "ref_audio": str(args.ref_audio),
            "ref_text": args.ref_text,
            "instruct": args.instruct,
            "speed": args.speed,
            "seed": args.seed,
            "video_path": str(video_path),
            "video_seconds": video_seconds,
            "source_audio": has_audio,
            "mix_mode": mix_mode,
            "orig_volume": args.orig_volume if mix_mode == "mix" else None,
            "tts_volume": args.tts_volume,
            "output_path": str(output_path),
            "narration_wav": str(wav_path) if args.keep_wav else None,
            "narration_seconds": result.audio_seconds,
            "sample_rate": result.sample_rate,
            "rtf": result.rtf,
            "package": str(package.root),
        }
        metadata_path = output_path.with_suffix(output_path.suffix + ".json")
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        cli_print(f"Metadata: {metadata_path}", json_events=bool(args.json_events))

    if not args.keep_wav:
        shutil.rmtree(wav_dir, ignore_errors=True)


def _probe_video(video_path: Path) -> tuple[float, bool]:
    """Return (duration_seconds, has_audio) using PyAV (an mlx-gen base dependency)."""
    import av

    with av.open(str(video_path)) as container:
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        has_audio = any(stream.type == "audio" for stream in container.streams)
    return duration, has_audio


def _run_mux(
    ffmpeg: str,
    *,
    video_path: Path,
    wav_path: Path,
    output_path: Path,
    video_seconds: float,
    mix_mode: str,
    orig_volume: float,
    tts_volume: float,
) -> None:
    argv = _capture_mux_argv(
        video_path,
        wav_path,
        output_path,
        video_seconds=video_seconds,
        mix_mode=mix_mode,
        orig_volume=orig_volume,
        tts_volume=tts_volume,
        ffmpeg=ffmpeg,
    )
    completed = subprocess.run(argv, capture_output=True, text=True)
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.splitlines()[-15:])
        raise RuntimeError(f"ffmpeg mux failed with code {completed.returncode}:\n{tail}")


def _capture_mux_argv(
    video_path: Path,
    wav_path: Path,
    output_path: Path,
    *,
    video_seconds: float,
    mix_mode: str,
    orig_volume: float,
    tts_volume: float,
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """Build the ffmpeg mux command (pure; unit-tested without running ffmpeg)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mix_mode == "mix":
        filtergraph = (
            f"[0:a]volume={orig_volume}[a0];"
            f"[1:a]volume={tts_volume},apad=whole_dur={video_seconds:.3f}[a1];"
            f"[a0][a1]amix=inputs=2:duration=first:normalize=0[aout]"
        )
        return [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(wav_path),
            "-filter_complex",
            filtergraph,
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
    # replace: pad the narration with silence up to the video length, then cut
    # at the video end so a longer narration never extends the container.
    return [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(wav_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-af",
        f"volume={tts_volume},apad=whole_dur={video_seconds:.3f}",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{video_seconds:.3f}",
        str(output_path),
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxgen dub",
        description=(
            "Narrate an existing video (e.g. a MiniMax-H3 clip) with CosyVoice3 speech and mux the "
            "result into a new MP4. The video is stream-copied; only the audio is re-encoded. "
            "Requires a prepared package: mlxgen prepare --model cosyvoice3."
        ),
    )
    parser.add_argument("--video-path", required=True, help="Source video (e.g. an H3-generated MP4).")
    parser.add_argument("--model", "-m", default=constants.MODEL_ALIAS, help="Model alias (default: cosyvoice3).")
    parser.add_argument(
        "--model-path",
        default=None,
        help=f"Prepared package directory (default: {constants.default_package_dir()}).",
    )
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument("--text", type=str, help='Narration text; "-" reads stdin.')
    text_group.add_argument("--text-file", type=str, default=None, help="Path to a UTF-8 text file.")
    parser.add_argument("--ref-audio", default=None, help="Reference voice (3-30 s of clean speech). Required.")
    parser.add_argument("--ref-text", default=None, help="Exact transcript of --ref-audio (zero-shot cloning).")
    parser.add_argument("--instruct", default=None, help="Style instruction; selects instructed mode.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech rate multiplier (default 1.0).")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument(
        "--mode",
        choices=["replace", "mix"],
        default="replace",
        help="replace: swap the soundtrack for the narration; mix: blend over the original audio (default replace).",
    )
    parser.add_argument("--orig-volume", type=float, default=0.15, help="Original-audio volume in mix mode (default 0.15).")
    parser.add_argument("--tts-volume", type=float, default=1.0, help="Narration volume (default 1.0).")
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Keep the synthesized narration as <output>.wav next to the video.",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output MP4 (default <video>_dubbed.mp4).")
    parser.add_argument("--metadata", action="store_true", help="Write <output>.json metadata sidecar.")
    parser.add_argument("--json-events", action="store_true", help="Emit machine-readable runtime events on stdout.")
    parser.add_argument("--debug", action="store_true", help="Stream worker logs (stderr) through to the terminal.")
    parser.add_argument("--replace", action="store_true", default=True, help="Replace the output when it exists (default).")
    parser.add_argument("--no-replace", action="store_false", dest="replace")
    return parser


if __name__ == "__main__":
    main()
