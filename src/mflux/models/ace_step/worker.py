"""ACE-Step 1.5 music-generation worker.

This module runs INSIDE the prepared package's isolated venv
(`<package>/venv/bin/python worker.py ...`), not inside MLX-Gen's own environment.
It must stay self-contained: standard library plus the venv's packages only — never
import mflux here.

Protocol: one JSON runtime event per line on stdout (the MLX-Gen JSONL contract);
all library logging stays on stderr. The parent process (`runner.py`) relays or
renders these events.

The upstream project (`ace-step/ACE-Step-1.5`) is imported lazily inside `main()`,
so this module also imports cleanly in MLX-Gen's own environment where the heavy
upstream dependencies (torch 2.9, transformers 4.5x, mlx-lm) are not installed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

TASK = "text-to-music"

# Upstream progress milestones (acestep/core/generation/handler/generate_music_*.py):
#   0.10 / 0.50 -> planner LM (CoT metadata + audio codes)
#   0.51        -> preparing the DiT inputs
#   0.52..0.79  -> diffusion
#   0.80        -> VAE decode
#   0.99        -> saving
DIFFUSION_START = 0.52
DIFFUSION_END = 0.79

# Upstream component names inside the prepared weights bundle.
DIT_CONFIG_PATH = "acestep-v15-turbo"
VAE_VARIANT = "official"  # resolves to <checkpoints>/vae upstream
LM_MODEL_PATH = "acestep-5Hz-lm-1.7B"
REQUIRED_COMPONENTS = ("acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B")

DEVICE = "mps"
LM_BACKEND = "mlx"

DEFAULT_DURATION_SECONDS = 60.0
MIN_DURATION_SECONDS = 10.0
MAX_DURATION_SECONDS = 600.0
DEFAULT_SEED = 42
DEFAULT_AUDIO_FORMAT = "flac"
DEFAULT_VOCAL_LANGUAGE = "unknown"
DEFAULT_MODEL_LABEL = "ace-step"
DEFAULT_COMMAND = "mlxgen music"

AUDIO_EXTENSIONS = ("flac", "wav", "mp3", "opus", "aac", "m4a", "ogg")


class EventEmitter:
    """Writes the JSONL event stream to the worker's real stdout.

    The stream is captured up front on purpose: the worker points `sys.stdout`
    at stderr while the upstream stack runs (a stray upstream `print()` would
    otherwise corrupt the JSONL channel the parent parses), so the emitter must
    hold on to the original stream instead of looking `sys.stdout` up later.
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def emit(self, *, command: str, model: str, seed: int, phase: str, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "type": "runtime",
            "command": command,
            "model": model,
            "seed": seed,
            "phase": phase,
            "task": TASK,
            "step": fields.pop("step", 0),
            "total_steps": fields.pop("total_steps", 0),
            "progress": fields.pop("progress", 0.0),
        }
        payload.update({key: value for key, value in fields.items() if value is not None})
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock:
            self._stream.write(line)
            self._stream.flush()


@contextlib.contextmanager
def _library_stdout():
    """Send everything the upstream stack prints to stderr.

    The worker's stdout is the JSONL event channel; the upstream library talks
    to the user through loguru/tqdm (stderr) but a stray `print()` would still
    land on stdout and break the parent's line-by-line parsing.
    """
    original = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = original


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acestep-worker")
    parser.add_argument(
        "--runtime-dir", required=True, help="Prepared upstream source tree (contains the acestep package)."
    )
    parser.add_argument(
        "--model-dir", required=True, help="Prepared weights bundle (DiT + VAE + text encoder [+ planner LM])."
    )
    parser.add_argument("--output", required=True, help="Exact path the generated audio is written to.")
    parser.add_argument("--caption", required=True, help="Style/prompt description of the wanted music.")
    parser.add_argument("--lyrics", default=None, help="Lyrics; the planner LM writes them when omitted.")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS, help="Target length in seconds.")
    parser.add_argument("--bpm", type=int, default=None, help="Beats per minute; auto-detected when omitted.")
    parser.add_argument("--keyscale", default=None, help="Musical key, e.g. 'C major'; auto-detected when omitted.")
    parser.add_argument("--timesignature", default=None, help="Time signature, e.g. '4'; auto-detected when omitted.")
    parser.add_argument("--instrumental", action="store_true", help="Generate an instrumental track without vocals.")
    parser.add_argument("--vocal-language", default=DEFAULT_VOCAL_LANGUAGE, help="Language code of the vocals.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--audio-format", default=DEFAULT_AUDIO_FORMAT, help="Container of the generated audio.")
    parser.add_argument("--use-lm", action="store_true", help="Use the planner LM (lyrics + blueprint).")
    parser.add_argument(
        "--no-cot-caption",
        action="store_true",
        help="Keep the caller's caption verbatim instead of letting the planner LM rewrite it.",
    )
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL, help="Model label reported in the events.")
    parser.add_argument("--command", default=DEFAULT_COMMAND, help="Command name reported in the events.")
    return parser


def _prepare_environment(*, runtime_dir: str, checkpoints_dir: str) -> None:
    """Point the upstream runtime at the prepared package.

    * `ACESTEP_CHECKPOINTS_DIR` is the upstream's own override for the
      checkpoints directory; the prepared weights live in `<package>/model`.
      It has to be set before `acestep` is imported.
    * `ACESTEP_PROJECT_ROOT` keeps the upstream's small cache files (the
      diffusion progress estimates) inside the package instead of the CWD.
    * The runtime tree goes first on `sys.path` so `import acestep` resolves to
      the prepared upstream source.
    * The package is complete by construction, so the hub stays offline: the
      upstream would otherwise try to re-download ~10 GB of weights.
    """
    os.environ["ACESTEP_CHECKPOINTS_DIR"] = checkpoints_dir
    os.environ["ACESTEP_PROJECT_ROOT"] = runtime_dir
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("ACESTEP_DISABLE_TQDM", "1")
    if runtime_dir not in sys.path:
        sys.path.insert(0, runtime_dir)


def _resolve_checkpoints_dir(model_dir: str) -> tuple[str, str | None]:
    """Return (checkpoints_dir, shadow_dir) for the upstream runtime.

    The upstream presence check spans the whole weights bundle — the planner LM
    included — even for a DiT-only run, so a package prepared with `--no-lm`
    would make the upstream re-download ~10 GB before it ever starts. The
    worker therefore assembles a shadow checkpoints directory that links the
    components the package has and stubs the missing planner LM, which a
    DiT-only run never loads (the LM handler is not even created).
    """
    root = Path(model_dir)
    if (root / LM_MODEL_PATH).is_dir():
        return model_dir, None
    shadow = Path(tempfile.mkdtemp(prefix="acestep_checkpoints_"))
    for component in REQUIRED_COMPONENTS:
        source = root / component
        if source.is_dir():
            os.symlink(source, shadow / component)
    stub = shadow / LM_MODEL_PATH
    stub.mkdir()
    # A valid, empty safetensors file: the upstream only checks that *some*
    # weights file exists, and this stub is never loaded.
    header = b'{"__metadata__":{}}'
    (stub / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    return str(shadow), str(shadow)


def _verify_model_dir(model_dir: str, *, use_lm: bool) -> None:
    """Fail early (and clearly) when the prepared weights are incomplete."""
    root = Path(model_dir)
    required = list(REQUIRED_COMPONENTS)
    if use_lm:
        required.append(LM_MODEL_PATH)
    missing = [name for name in required if not (root / name).is_dir()]
    if missing:
        raise RuntimeError(
            f"The prepared ACE-Step package is incomplete: {', '.join(missing)} missing under {model_dir}. "
            "Run `mlxgen prepare --model ace-step` again to fetch the missing weights."
        )


def _resolve_lyrics(lyrics: str | None, instrumental: bool) -> tuple[str, bool]:
    """Mirror the upstream CLI's lyrics/instrumental normalisation."""
    text = (lyrics or "").strip()
    if text.lower() in {"[inst]", "[instrumental]"}:
        return text, True
    if instrumental and not text:
        return "[Instrumental]", True
    return text, instrumental


def _resolve_duration(duration: float | None) -> float:
    """Clamp the requested length into the range the upstream model supports."""
    if duration is None or duration <= 0:
        return -1.0  # upstream: the model picks the length itself
    return float(min(max(float(duration), MIN_DURATION_SECONDS), MAX_DURATION_SECONDS))


def _diffusion_step(fraction: float, total_steps: int) -> int:
    """Map an upstream progress fraction onto a diffusion step counter."""
    if total_steps <= 0:
        return 0
    if fraction >= DIFFUSION_END:
        return total_steps
    span = DIFFUSION_END - DIFFUSION_START
    ratio = (fraction - DIFFUSION_START) / span
    return max(0, min(total_steps, int(round(ratio * total_steps))))


def _make_progress_callback(
    *,
    emitter: EventEmitter,
    command: str,
    model: str,
    seed: int,
    total_steps: int,
    state: dict[str, int],
):
    """Build the callback the upstream `generate_music` reports progress to.

    The upstream callback is tqdm-shaped: `progress(value, desc=...)` with
    `value` a 0..1 fraction. Values below the diffusion milestone belong to the
    planner LM (already announced with the `planning` phase); values from the
    diffusion milestone on are the DiT's own progress.
    """

    def _progress(value: float, desc: Any = None, *args: Any, **kwargs: Any) -> None:
        try:
            fraction = float(value)
        except (TypeError, ValueError):
            return
        if fraction < DIFFUSION_START:
            return
        step = _diffusion_step(fraction, total_steps)
        state["step"] = step
        emitter.emit(
            command=command,
            model=model,
            seed=seed,
            phase="synthesizing",
            step=step,
            total=total_steps,
            total_steps=total_steps,
            progress=round(fraction, 4),
            desc=desc if isinstance(desc, str) else None,
        )

    return _progress


def _create_sample(
    llm_handler: Any,
    *,
    caption: str,
    vocal_language: str,
    duration: float,
) -> Any | None:
    """Let the planner LM write the song itself (upstream's "Simple Mode").

    The upstream text-to-music path only expands the caption and the metas; the
    lyrics come from this separate call. Returns the upstream `CreateSampleResult`
    or None when the planner LM could not produce one.
    """
    from acestep.inference import create_sample

    processor = getattr(llm_handler, "constrained_processor", None)
    if processor is not None and duration and duration > 0:
        # Keep the written lyrics within the requested song length.
        processor.max_duration = int(min(duration, MAX_DURATION_SECONDS))
    language = (vocal_language or "").strip()
    result = create_sample(
        llm_handler,
        query=caption,
        instrumental=False,
        vocal_language=None if language.lower() in {"", "unknown"} else language,
    )
    if not getattr(result, "success", False):
        return None
    return result


def _locate_audio(*, result: Any, save_dir: str, audio_format: str) -> Path | None:
    """Find the audio file the upstream wrote into `save_dir`."""
    for audio in getattr(result, "audios", None) or []:
        path = audio.get("path") if isinstance(audio, dict) else None
        if path and Path(path).is_file():
            return Path(path)
    root = Path(save_dir)
    extension = "wav" if audio_format == "wav32" else audio_format
    candidates = sorted(root.glob(f"*.{extension}"), key=lambda item: item.stat().st_mtime)
    if candidates:
        return candidates[-1]
    for candidate in AUDIO_EXTENSIONS:
        matches = sorted(root.glob(f"*.{candidate}"), key=lambda item: item.stat().st_mtime)
        if matches:
            return matches[-1]
    return None


def _publish_audio(*, result: Any, save_dir: str, output: str, audio_format: str) -> Path:
    """Copy the produced audio to the exact requested output path."""
    produced = _locate_audio(result=result, save_dir=save_dir, audio_format=audio_format)
    if produced is None:
        status = str(getattr(result, "status_message", "") or "").strip()
        error = str(getattr(result, "error", "") or "").strip()
        detail = " ".join(part for part in (status, error) if part).strip()
        raise RuntimeError(
            "ACE-Step finished without producing an audio file." + (f" Upstream reported: {detail}" if detail else "")
        )
    target = Path(os.path.abspath(os.path.expanduser(output)))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, target)
    return target


def _audio_facts(*, result: Any, produced: Path) -> tuple[float | None, int | None]:
    """Return (audio_seconds, sample_rate) for the generated track."""
    sample_rate: int | None = None
    frames: int | None = None
    for audio in getattr(result, "audios", None) or []:
        if not isinstance(audio, dict):
            continue
        rate = audio.get("sample_rate")
        if isinstance(rate, (int, float)) and rate > 0:
            sample_rate = int(rate)
        tensor = audio.get("tensor")
        if tensor is not None:
            try:
                frames = int(tensor.shape[-1])
            except (AttributeError, IndexError, TypeError):
                frames = None
        if sample_rate and frames:
            break
    if sample_rate and frames:
        return round(frames / sample_rate, 2), sample_rate
    try:
        import soundfile as sf

        info = sf.info(str(produced))
        return round(info.frames / info.samplerate, 2), int(info.samplerate)
    except Exception:  # noqa: BLE001 - the file facts are best-effort metadata
        return None, None


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    command, label, seed = args.command, args.model_label, args.seed
    emitter = EventEmitter(sys.stdout)
    shadow_dir: str | None = None
    try:
        emitter.emit(command=command, model=label, seed=seed, phase="loading")
        runtime_dir = str(Path(args.runtime_dir).expanduser().resolve())
        model_dir = str(Path(args.model_dir).expanduser().resolve())
        _verify_model_dir(model_dir, use_lm=args.use_lm)
        checkpoints_dir, shadow_dir = _resolve_checkpoints_dir(model_dir)
        _prepare_environment(runtime_dir=runtime_dir, checkpoints_dir=checkpoints_dir)

        load_start = time.time()
        with _library_stdout():
            from acestep.handler import AceStepHandler
            from acestep.inference import GenerationConfig, GenerationParams, generate_music
            from acestep.llm_inference import LLMHandler

            dit_handler = AceStepHandler()
            status, ok = dit_handler.initialize_service(
                project_root=runtime_dir,
                config_path=DIT_CONFIG_PATH,
                device=DEVICE,
                use_mlx_dit=True,
                vae_checkpoint=VAE_VARIANT,
            )
            if not ok:
                raise RuntimeError(f"ACE-Step DiT initialization failed: {status}")

            llm_handler = None
            if args.use_lm:
                llm_handler = LLMHandler()
                status, ok = llm_handler.initialize(
                    checkpoint_dir=model_dir,
                    lm_model_path=LM_MODEL_PATH,
                    backend=LM_BACKEND,
                    device=DEVICE,
                )
                if not ok:
                    raise RuntimeError(f"ACE-Step planner LM initialization failed: {status}")
        load_seconds = round(time.time() - load_start, 2)
        emitter.emit(
            command=command,
            model=label,
            seed=seed,
            phase="loaded",
            load_seconds=load_seconds,
            sample_rate=getattr(dit_handler, "sample_rate", None),
        )

        lyrics, instrumental = _resolve_lyrics(args.lyrics, args.instrumental)
        duration = _resolve_duration(args.duration)
        caption = args.caption
        sample = None
        if args.use_lm and not lyrics and not instrumental:
            # Upstream's text-to-music path never writes lyrics: the planner LM only
            # expands the caption and the metas. Writing the lyrics is a separate
            # upstream call ("Simple Mode"), so use it when the caller brought no
            # lyrics of its own.
            emitter.emit(command=command, model=label, seed=seed, phase="planning", progress=0.0)
            with _library_stdout():
                sample = _create_sample(
                    llm_handler,
                    caption=caption,
                    vocal_language=args.vocal_language,
                    duration=duration,
                )
            if sample is not None:
                caption = getattr(sample, "caption", "") or caption
                lyrics = getattr(sample, "lyrics", "") or ""
        params = GenerationParams(
            task_type="text2music",
            caption=caption,
            lyrics=lyrics,
            instrumental=instrumental,
            vocal_language=args.vocal_language,
            bpm=args.bpm if args.bpm else (getattr(sample, "bpm", None) if sample is not None else None),
            keyscale=args.keyscale or (getattr(sample, "keyscale", "") if sample is not None else "") or "",
            timesignature=args.timesignature
            or (getattr(sample, "timesignature", "") if sample is not None else "")
            or "",
            duration=duration,
            seed=args.seed,
            thinking=args.use_lm,
            # With a planner-written sample the caption and the metas are already
            # settled, so the CoT pass must not overwrite them. `--no-cot-caption`
            # keeps the caller's caption verbatim in the same way.
            use_cot_caption=args.use_lm and sample is None and not args.no_cot_caption,
            use_cot_metas=args.use_lm and sample is None,
            use_cot_language=args.use_lm and sample is None,
            use_cot_lyrics=False,
        )
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[args.seed],
            audio_format=args.audio_format,
        )
        total_steps = int(params.inference_steps)
        state = {"step": 0}
        progress = _make_progress_callback(
            emitter=emitter,
            command=command,
            model=label,
            seed=seed,
            total_steps=total_steps,
            state=state,
        )

        if args.use_lm and sample is None:
            emitter.emit(command=command, model=label, seed=seed, phase="planning", progress=0.0)

        gen_start = time.time()
        save_dir = tempfile.mkdtemp(prefix="acestep_save_")
        try:
            with _library_stdout():
                result = generate_music(
                    dit_handler,
                    llm_handler,
                    params,
                    config,
                    save_dir=save_dir,
                    progress=progress,
                )
            output_path = _publish_audio(
                result=result,
                save_dir=save_dir,
                output=args.output,
                audio_format=args.audio_format,
            )
            audio_seconds, sample_rate = _audio_facts(result=result, produced=output_path)
            gen_seconds = round(time.time() - gen_start, 2)
            # The upstream planner LM writes the lyrics, the caption and the song
            # structure into the params object it mutates in place, so the values
            # below are the ones the DiT actually sang.
            effective_lyrics = params.lyrics or getattr(params, "cot_lyrics", "")
            emitter.emit(
                command=command,
                model=label,
                seed=seed,
                phase="complete",
                step=total_steps,
                total=total_steps,
                total_steps=total_steps,
                progress=1.0,
                output_path=str(output_path),
                audio_seconds=audio_seconds,
                sample_rate=sample_rate,
                duration=round(duration, 2) if duration > 0 else audio_seconds,
                gen_seconds=gen_seconds,
                load_seconds=load_seconds,
                lyrics=effective_lyrics or None,
                caption=params.caption or getattr(params, "cot_caption", "") or None,
                bpm=params.bpm or getattr(params, "cot_bpm", None) or None,
                keyscale=params.keyscale or getattr(params, "cot_keyscale", "") or None,
                timesignature=params.timesignature or getattr(params, "cot_timesignature", "") or None,
                vocal_language=params.vocal_language or getattr(params, "cot_vocal_language", "") or None,
            )
            return 0
        finally:
            shutil.rmtree(save_dir, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 - the worker reports every failure as an event
        traceback.print_exc(file=sys.stderr)
        emitter.emit(
            command=command,
            model=label,
            seed=seed,
            phase="failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return 1
    finally:
        if shadow_dir is not None:
            shutil.rmtree(shadow_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
