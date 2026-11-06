"""Fetch the Qwen3-TTS VoiceDesign MLX weights into the workspace.

Uses the same resumable chunked downloader as `mlxgen prepare`, but keeps
retrying in a loop so a throttled or flaky link eventually completes instead of
failing the whole run.

The destination defaults to `out/qwen3-tts-voice-design/model` next to this
repository and can be overridden with `$QWEN3_TTS_VOICE_DEST`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mflux.models.qwen3_tts_voice import constants  # noqa: E402
from mflux.models.qwen3_tts_voice.download import chunked_download, mirror_urls, probe_mirrors  # noqa: E402

DEST = Path(
    os.environ.get(
        "QWEN3_TTS_VOICE_DEST",
        Path(__file__).resolve().parents[1] / "out" / "qwen3-tts-voice-design" / "model",
    )
)


def _size(urls: list[str]) -> int | None:
    for _, size, _ in probe_mirrors(urls):
        if size:
            return size
    return None


def main() -> int:
    for name in constants.WEIGHT_FILES:
        urls = mirror_urls(constants.WEIGHTS_REPO_ID, name, source="auto")
        size = _size(urls)
        if size is None:
            print(f"skip {name}: not on any mirror", flush=True)
            continue
        target = DEST / name
        marker = target.with_name(target.name + ".complete")
        if marker.is_file() and target.is_file() and target.stat().st_size == size:
            print(f"have {name} ({size / 1024**2:.0f} MB)", flush=True)
            continue
        for attempt in range(1, 100):
            done = chunked_download(
                urls,
                target,
                expected_size=size,
                on_progress=lambda file_name, done_bytes, total: print(
                    f"  {file_name}: {done_bytes / 1024**2:.0f}/{total / 1024**2:.0f} MB", flush=True
                ),
            )
            if done:
                print(f"done {name} ({size / 1024**2:.0f} MB)", flush=True)
                break
            print(f"[{name}] attempt {attempt} incomplete; retrying in 15s", flush=True)
            time.sleep(15)
        else:
            print(f"giving up on {name}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
