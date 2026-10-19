"""Parent-side runner for the isolated CosyVoice3 worker.

Resolves the prepared package, spawns `<package>/venv/bin/python` on the worker
module, and relays the worker's JSONL runtime events — verbatim when the caller
asked for `--json-events`, otherwise rendered as concise human progress lines.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mflux.cli.runtime_events import cli_print
from mflux.models.cosyvoice3 import constants


class CosyVoice3PackageError(RuntimeError):
    pass


@dataclass(frozen=True)
class CosyVoice3Package:
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


def resolve_package(model_path: str | None) -> CosyVoice3Package:
    root = Path(model_path).expanduser() if model_path else constants.default_package_dir()
    package = CosyVoice3Package(root=root)
    problems = []
    if not root.is_dir():
        problems.append(f"package directory {root} does not exist")
    else:
        if not (package.model_dir / "cosyvoice3.yaml").is_file():
            problems.append(f"{package.model_dir}/cosyvoice3.yaml is missing (weights not downloaded)")
        if not (package.runtime_dir / "cosyvoice" / "cli" / "cosyvoice.py").is_file():
            problems.append(f"{package.runtime_dir} is missing the cosyvoice inference code")
        if not package.venv_python.is_file():
            problems.append(f"{package.venv_python} is missing (isolated environment not created)")
    if problems:
        raise CosyVoice3PackageError(
            "CosyVoice3 is not prepared: " + "; ".join(problems) + ".\n"
            "Run `mlxgen prepare --model cosyvoice3`"
            + (f" --path {root}" if model_path else "")
            + " first, then retry."
        )
    return package


@dataclass
class SynthesisResult:
    output_path: Path
    audio_seconds: float | None
    sample_rate: int | None
    rtf: float | None
    synth_seconds: float | None
    worker_event_count: int


def synthesize(
    package: CosyVoice3Package,
    *,
    text: str,
    output_path: Path,
    ref_audio: Path,
    ref_text: str | None,
    instruct: str | None,
    mode: str,
    speed: float,
    seed: int,
    json_events: bool,
    command: str,
    debug: bool = False,
) -> SynthesisResult:
    """Run one synthesis job in the isolated worker and return its result facts."""
    worker = Path(__file__).with_name("worker.py")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", prefix="cosyvoice3_text_", delete=False, encoding="utf-8") as tf:
        tf.write(text)
        text_file = Path(tf.name)

    argv = [
        str(package.venv_python),
        str(worker),
        "--runtime-dir",
        str(package.runtime_dir),
        "--model-dir",
        str(package.model_dir),
        "--text-file",
        str(text_file),
        "--output",
        str(output_path),
        "--ref-audio",
        str(ref_audio),
        "--mode",
        mode,
        "--speed",
        str(speed),
        "--seed",
        str(seed),
        "--command",
        command,
        "--model-label",
        constants.MODEL_ALIAS,
    ]
    if ref_text is not None:
        argv.extend(["--ref-text", ref_text])
    if instruct is not None:
        argv.extend(["--instruct", instruct])

    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if not debug else None,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    result = SynthesisResult(
        output_path=output_path,
        audio_seconds=None,
        sample_rate=None,
        rtf=None,
        synth_seconds=None,
        worker_event_count=0,
    )
    last_human_phase = None
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
        if event.get("phase") == "complete":
            result.audio_seconds = event.get("audio_seconds")
            result.sample_rate = event.get("sample_rate")
            result.rtf = event.get("rtf")
            result.synth_seconds = event.get("synth_seconds")
    stderr_tail = ""
    if not debug and process.stderr is not None:
        stderr_tail = "\n".join(process.stderr.read().splitlines()[-30:])
    return_code = process.wait()
    text_file.unlink(missing_ok=True)
    if return_code != 0:
        raise CosyVoice3PackageError(
            f"CosyVoice3 worker failed with exit code {return_code}."
            + (f"\nWorker stderr (tail):\n{stderr_tail}" if stderr_tail else " Re-run with --debug to see worker logs.")
        )
    return result


def _render_human_event(event: dict, last_phase: str | None) -> str | None:
    phase = event.get("phase")
    if phase == "loading":
        cli_print("Loading CosyVoice3 (isolated worker)...", json_events=False)
    elif phase == "loaded":
        cli_print(f"Model loaded in {event.get('load_seconds')}s; synthesizing...", json_events=False)
    elif phase == "synthesizing":
        audio_seconds = event.get("audio_seconds")
        if audio_seconds is not None:
            print(f"\rSynthesized {audio_seconds:.1f}s of audio...", end="", file=sys.stderr, flush=True)
    elif phase == "complete":
        if last_phase == "synthesizing":
            print(file=sys.stderr)
        cli_print(
            f"Done: {event.get('audio_seconds')}s audio in {event.get('synth_seconds')}s "
            f"(rtf {event.get('rtf')}) -> {event.get('output_path')}",
            json_events=False,
        )
    elif phase == "failed":
        cli_print(f"Worker failed: {event.get('error')}", json_events=False, error=True)
    return phase
