"""Parent-side runner for the isolated ACE-Step 1.5 worker.

Resolves the prepared package, spawns `<package>/venv/bin/python` on the worker
module, and relays the worker's JSONL runtime events — verbatim when the caller
asked for `--json-events`, otherwise rendered as concise human progress lines.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from mflux.cli.runtime_events import cli_print
from mflux.models.ace_step import constants


class AceStepPackageError(RuntimeError):
    pass


@dataclass(frozen=True)
class AceStepPackage:
    root: Path

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def runtime_dir(self) -> Path:
        return self.root / "runtime"

    @property
    def venv_python(self) -> Path:
        return self.root / "venv" / "bin" / "python"

    @property
    def manifest_path(self) -> Path:
        return self.root / constants.MANIFEST_NAME


def resolve_package(model_path: str | None) -> AceStepPackage:
    """Validate a prepared ACE-Step package and return its layout."""
    root = Path(model_path).expanduser() if model_path else constants.default_package_dir()
    package = AceStepPackage(root=root)
    problems = []
    if not root.is_dir():
        problems.append(f"package directory {root} does not exist")
    else:
        if not (package.model_dir / constants.DIT_DIR).is_dir():
            problems.append(f"{package.model_dir}/{constants.DIT_DIR} is missing (DiT weights not downloaded)")
        if not (package.model_dir / constants.VAE_DIR).is_dir():
            problems.append(f"{package.model_dir}/{constants.VAE_DIR} is missing (VAE weights not downloaded)")
        if not (package.runtime_dir / "acestep" / "__init__.py").is_file():
            problems.append(f"{package.runtime_dir} is missing the acestep inference code")
        if not package.venv_python.is_file():
            problems.append(f"{package.venv_python} is missing (isolated environment not created)")
        if not package.manifest_path.is_file():
            problems.append(f"{package.manifest_path} is missing (package manifest not written)")
    if problems:
        raise AceStepPackageError(
            "ACE-Step 1.5 is not prepared: " + "; ".join(problems) + ".\n"
            "Run `mlxgen prepare --model ace-step`" + (f" --path {root}" if model_path else "") + " first, then retry."
        )
    return package


@dataclass
class GenerationResult:
    output_path: Path
    audio_seconds: float | None
    sample_rate: int | None
    seed: int | None
    duration: float | None
    load_seconds: float | None
    gen_seconds: float | None
    worker_event_count: int
    lyrics: str | None = None
    caption: str | None = None
    bpm: int | None = None
    keyscale: str | None = None
    timesignature: str | None = None
    vocal_language: str | None = None


def synthesize(
    package: AceStepPackage,
    *,
    caption: str = "",
    lyrics: str | None = None,
    output_path: Path,
    duration: float = constants.DEFAULT_DURATION_SECONDS,
    bpm: int | None = None,
    keyscale: str | None = None,
    timesignature: str | None = None,
    instrumental: bool = False,
    vocal_language: str = "unknown",
    seed: int = 42,
    audio_format: str = constants.DEFAULT_AUDIO_FORMAT,
    use_lm: bool = True,
    keep_caption: bool = False,
    json_events: bool = False,
    command: str = "mlxgen music",
    debug: bool = False,
) -> GenerationResult:
    """Run one generation job in the isolated worker and return its result facts."""
    worker = Path(__file__).with_name("worker.py")
    argv = [
        str(package.venv_python),
        str(worker),
        "--runtime-dir",
        str(package.runtime_dir),
        "--model-dir",
        str(package.model_dir),
        "--output",
        str(output_path),
        "--caption",
        caption,
        "--duration",
        str(duration),
        "--seed",
        str(seed),
        "--vocal-language",
        vocal_language,
        "--audio-format",
        audio_format,
        "--command",
        command,
        "--model-label",
        constants.MODEL_ALIAS,
    ]
    if lyrics is not None:
        argv.extend(["--lyrics", lyrics])
    if bpm is not None:
        argv.extend(["--bpm", str(bpm)])
    if keyscale is not None:
        argv.extend(["--keyscale", keyscale])
    if timesignature is not None:
        argv.extend(["--timesignature", timesignature])
    if instrumental:
        argv.append("--instrumental")
    if use_lm:
        argv.append("--use-lm")
    if keep_caption:
        argv.append("--no-cot-caption")

    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if not debug else None,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    result = GenerationResult(
        output_path=output_path,
        audio_seconds=None,
        sample_rate=None,
        seed=seed,
        duration=duration,
        load_seconds=None,
        gen_seconds=None,
        worker_event_count=0,
    )
    last_human_phase = None
    worker_error: str | None = None
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # A library leaked onto stdout; surface it like any other progress text.
            cli_print(line, json_events=json_events)
            continue
        result.worker_event_count += 1
        if json_events:
            sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        else:
            last_human_phase = _render_human_event(event, last_human_phase)
        phase = event.get("phase")
        if phase == "complete":
            result.output_path = Path(event.get("output_path") or output_path)
            result.audio_seconds = event.get("audio_seconds")
            result.sample_rate = event.get("sample_rate")
            result.seed = event.get("seed", result.seed)
            result.duration = event.get("duration", result.duration)
            result.load_seconds = event.get("load_seconds")
            result.gen_seconds = event.get("gen_seconds")
            result.lyrics = event.get("lyrics") or result.lyrics
            result.caption = event.get("caption") or result.caption
            result.bpm = event.get("bpm") or result.bpm
            result.keyscale = event.get("keyscale") or result.keyscale
            result.timesignature = event.get("timesignature") or result.timesignature
            result.vocal_language = event.get("vocal_language") or result.vocal_language
        elif phase == "failed":
            worker_error = event.get("error") or "unknown error"
    stderr_tail = ""
    if not debug and process.stderr is not None:
        stderr_tail = "\n".join(process.stderr.read().splitlines()[-30:])
    return_code = process.wait()
    if return_code != 0 or worker_error is not None:
        raise AceStepPackageError(
            f"ACE-Step worker failed with exit code {return_code}."
            + (f"\nWorker error: {worker_error}" if worker_error else "")
            + (f"\nWorker stderr (tail):\n{stderr_tail}" if stderr_tail else " Re-run with --debug to see worker logs.")
        )
    return result


def _render_human_event(event: dict, last_phase: str | None) -> str | None:
    phase = event.get("phase")
    if phase == "loading":
        cli_print("Loading ACE-Step 1.5 (isolated worker)...", json_events=False)
    elif phase == "loaded":
        cli_print(f"Model loaded in {event.get('load_seconds')}s; composing...", json_events=False)
    elif phase == "planning":
        cli_print("Planner LM is writing the blueprint and the lyrics...", json_events=False)
    elif phase == "synthesizing":
        step = event.get("step")
        total = event.get("total_steps") or event.get("total")
        if step is not None and total:
            print(f"\rSynthesizing step {step}/{total}...", end="", file=sys.stderr, flush=True)
    elif phase == "complete":
        if last_phase == "synthesizing":
            print(file=sys.stderr)
        cli_print(
            f"Done: {event.get('audio_seconds')}s audio in {event.get('gen_seconds')}s -> {event.get('output_path')}",
            json_events=False,
        )
    elif phase == "failed":
        cli_print(f"Worker failed: {event.get('error')}", json_events=False, error=True)
    return phase
