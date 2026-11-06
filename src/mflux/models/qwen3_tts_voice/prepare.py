"""Prepare the Qwen3-TTS VoiceDesign package (MLX weights + manifest).

The weights are the `mlx-community` conversion of Qwen3-TTS VoiceDesign: a 3.8 GB
talker plus a 0.7 GB 12 Hz speech-tokenizer decoder. Nothing else is needed at
runtime — `mlx-audio` provides the MLX inference code and is a dependency of
MLX-Gen itself (it is pure MLX, so unlike CosyVoice3 there is no isolated torch
environment to build).
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from mflux.cli.runtime_events import cli_print
from mflux.models.qwen3_tts_voice import constants
from mflux.models.qwen3_tts_voice.download import chunked_download, mirror_urls, remote_size
from mflux.models.qwen3_tts_voice.runner import resolve_package


def prepare_package(
    path: Path,
    *,
    force: bool = False,
    json_events: bool = False,
    source: str = "auto",
) -> Path:
    path = path.expanduser().resolve()
    if force and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    mlx_audio_version = _require_mlx_audio(json_events=json_events)
    _download_weights(path / "model", json_events=json_events, source=source)

    manifest = {
        "model": constants.MODEL_ALIAS,
        "weights_repo": constants.WEIGHTS_REPO_ID,
        "upstream_repo": constants.UPSTREAM_REPO_ID,
        "runtime": "mlx-audio (in-process MLX inference)",
        "mlx_audio_version": mlx_audio_version,
        "python": sys.version.split()[0],
        "languages": list(constants.LANGUAGES),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (path / constants.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Validate the finished package through the same gate the runner uses.
    resolve_package(str(path))
    cli_print(f"Qwen3-TTS VoiceDesign package ready at {path}", json_events=json_events)
    cli_print(
        'Try it: mlxgen voice-design --voice "a warm female voice in her thirties" --text "你好，世界" --output hello.wav',
        json_events=json_events,
    )
    return path


def _require_mlx_audio(*, json_events: bool) -> str:
    """Fail early and clearly when the MLX inference stack is missing."""
    try:
        import mlx_audio
    except ImportError as exc:
        raise RuntimeError(
            "mlx-audio is required for Qwen3-TTS VoiceDesign but is not installed. "
            f"Install it with: pip install '{constants.MLX_AUDIO_REQUIREMENT}'"
        ) from exc
    version = _mlx_audio_version(mlx_audio)
    cli_print(f"Using mlx-audio {version}.", json_events=json_events)
    return version


def _mlx_audio_version(mlx_audio_module) -> str:
    """mlx-audio does not expose __version__; the installed distribution does."""
    from importlib import metadata

    try:
        return metadata.version("mlx-audio")
    except metadata.PackageNotFoundError:
        return str(getattr(mlx_audio_module, "__version__", "unknown"))


def _download_weights(model_dir: Path, *, json_events: bool, source: str = "auto") -> None:
    """Download every weight file in resumable chunks, from two mirrors.

    The talker is 3.8 GB in a single file: a plain streamed download either
    completes or has to start over. Chunks with explicit Range requests survive
    connection resets and can be retried on the other mirror.
    """
    cli_print(
        f"Downloading Qwen3-TTS VoiceDesign weights from {constants.WEIGHTS_REPO_ID} (~4.5 GB)...",
        json_events=json_events,
    )

    for name in constants.WEIGHT_FILES:
        urls = mirror_urls(constants.WEIGHTS_REPO_ID, name, source=source)
        size = _first_known_size(urls)
        if size is None:
            cli_print(f"Skipping {name} (not present on the mirror).", json_events=json_events)
            continue
        target = model_dir / name
        marker = target.with_name(target.name + ".complete")
        if marker.is_file() and target.is_file() and target.stat().st_size == size:
            cli_print(f"Already downloaded: {name}.", json_events=json_events)
            continue

        def progress(file_name: str, done: int, total: int) -> None:
            cli_print(
                f"  {file_name}: {done / 1024**2:.0f}/{total / 1024**2:.0f} MB",
                json_events=json_events,
            )

        completed = chunked_download(urls, target, expected_size=size, on_progress=progress)
        if not completed:
            raise RuntimeError(
                f"Could not complete {name} ({size / 1024**2:.0f} MB) from any mirror. "
                "Re-run `mlxgen prepare --model qwen3-tts-voice-design` to continue where it stopped."
            )
        cli_print(f"Downloaded {name} ({size / 1024**2:.0f} MB).", json_events=json_events)


def _first_known_size(urls: list[str]) -> int | None:
    for url in urls:
        size = remote_size(url)
        if size:
            return size
    return None
