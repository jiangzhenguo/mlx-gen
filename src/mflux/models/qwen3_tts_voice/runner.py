"""Parent-side runner for the Qwen3-TTS VoiceDesign worker.

Resolves the prepared package, spawns the worker module with the *current*
interpreter (Qwen3-TTS is MLX-native, so no isolated environment is needed), and
relays the worker's JSONL runtime events — verbatim when the caller asked for
`--json-events`, otherwise rendered as concise human progress lines.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from mflux.cli.runtime_events import cli_print
from mflux.models.qwen3_tts_voice import constants


class VoiceDesignPackageError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceDesignPackage:
    root: Path

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def manifest_path(self) -> Path:
        return self.root / constants.MANIFEST_NAME


def resolve_package(model_path: str | None) -> VoiceDesignPackage:
    root = Path(model_path).expanduser() if model_path else constants.default_package_dir()
    package = VoiceDesignPackage(root=root)
    problems = []
    if not root.is_dir():
        problems.append(f"package directory {root} does not exist")
    else:
        if not (package.model_dir / "config.json").is_file():
            problems.append(f"{package.model_dir}/config.json is missing (weights not downloaded)")
        if not (package.model_dir / "model.safetensors").is_file():
            problems.append(f"{package.model_dir}/model.safetensors is missing")
        if not (package.model_dir / "speech_tokenizer" / "model.safetensors").is_file():
            problems.append(f"{package.model_dir}/speech_tokenizer/model.safetensors is missing")
    if problems:
        raise VoiceDesignPackageError(
            "Qwen3-TTS VoiceDesign is not prepared: " + "; ".join(problems) + ".\n"
            "Run `mlxgen prepare --model qwen3-tts-voice-design`"
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
    package: VoiceDesignPackage,
    *,
    text: str,
    output_path: Path,
    voice_description: str,
    language: str,
    seed: int,
    json_events: bool,
    command: str,
    temperature: float | None = None,
    debug: bool = False,
) -> SynthesisResult:
    """Run one synthesis job in the worker and return its result facts."""
    worker = Path(__file__).with_name("worker.py")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", prefix="qwen3tts_text_", delete=False, encoding="utf-8") as tf:
        tf.write(text)
        text_file = Path(tf.name)

    argv = [
        sys.executable,
        str(worker),
        "--model-dir",
        str(package.model_dir),
        "--text-file",
        str(text_file),
        "--output",
        str(output_path),
        "--voice",
        voice_description,
        "--language",
        language,
        "--seed",
        str(seed),
        "--command",
        command,
        "--model-label",
        constants.MODEL_ALIAS,
    ]
    if temperature is not None:
        argv.extend(["--temperature", str(temperature)])

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
        raise VoiceDesignPackageError(
            f"Qwen3-TTS VoiceDesign worker failed with exit code {return_code}."
            + (f"\nWorker stderr (tail):\n{stderr_tail}" if stderr_tail else " Re-run with --debug to see worker logs.")
        )
    return result


def _render_human_event(event: dict, last_phase: str | None) -> str | None:
    phase = event.get("phase")
    if phase == "loading":
        cli_print("Loading Qwen3-TTS VoiceDesign (MLX)...", json_events=False)
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
