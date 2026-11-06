"""Resumable, chunked downloads for the Qwen3-TTS VoiceDesign weights.

The voice-design package is 4.5 GB in two large files. A single long-lived HTTP
stream is fragile on the networks this project is used on: connections reset
mid-file, and a resumed transfer restarts the whole file. Downloading in fixed
chunks with explicit Range requests and writing each chunk straight to its
offset in the target file makes every attempt cheap to retry and lets a
half-finished download continue where it stopped — across process restarts, too
(the chunk map lives next to the file).

Not every mirror honours Range requests: the Hugging Face CDN behind
`hf-mirror.com` answers a range request with the *whole* file, which would
silently turn a chunked download into N full downloads. Every mirror is probed
once before it is used, and mirrors that ignore ranges are dropped.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Sequence

CHUNK_SIZE = 8 * 1024 * 1024

# Parallel range requests reach a link's real bandwidth; a single stream on a
# throttled connection gets a fraction of it.
DEFAULT_WORKERS = int(os.environ.get("MLXGEN_DOWNLOAD_WORKERS", "16"))

# Files below this size are fetched in one request instead of in chunks.
WHOLE_FILE_LIMIT = 64 * 1024 * 1024

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
HF_MIRROR = "https://hf-mirror.com"
MODELSCOPE_ENDPOINT = "https://modelscope.cn/models"


def mirror_urls(repo_id: str, name: str, *, source: str = "auto") -> list[str]:
    """Return the candidate URLs for one file, most preferred first."""
    urls: list[str] = []
    if source in {"auto", "hf"}:
        urls.append(f"{HF_ENDPOINT}/{repo_id}/resolve/main/{name}")
        if HF_ENDPOINT != HF_MIRROR:
            urls.append(f"{HF_MIRROR}/{repo_id}/resolve/main/{name}")
    if source in {"auto", "modelscope"}:
        urls.append(f"{MODELSCOPE_ENDPOINT}/{repo_id}/resolve/master/{name}")
    return urls


def probe_mirror(url: str, *, timeout: tuple[int, int] = (10, 20)) -> tuple[int | None, bool]:
    """Ask a mirror for its first byte in one request.

    Returns (size, honours_ranges). The size comes from the Content-Range
    header when the mirror answers with 206, otherwise from Content-Length.
    """
    import requests

    try:
        response = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=timeout)
    except (OSError, requests.RequestException):
        return None, False
    try:
        if response.status_code not in {200, 206}:
            return None, False
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            total = content_range.rsplit("/", 1)[1].strip()
            if total.isdigit():
                return int(total), response.status_code == 206
        content_length = response.headers.get("Content-Length", "")
        size = int(content_length) if content_length.isdigit() else None
        return size, response.status_code == 206
    finally:
        response.close()


def remote_size(url: str, *, timeout: tuple[int, int] = (10, 20)) -> int | None:
    return probe_mirror(url, timeout=timeout)[0]


def probe_mirrors(urls: Sequence[str]) -> list[tuple[str, int | None, bool]]:
    """Probe every mirror concurrently: (url, size, honours_ranges).

    Probing serially costs the sum of the per-mirror timeouts, and the first
    mirror is often an unreachable endpoint — the probes have to overlap.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
        results = list(pool.map(probe_mirror, urls))
    return [(url, size, honours) for url, (size, honours) in zip(urls, results)]


def supports_range(url: str, *, timeout: tuple[int, int] = (10, 20)) -> bool:
    return probe_mirror(url, timeout=timeout)[1]


def chunked_download(
    urls: Sequence[str],
    target: Path,
    *,
    expected_size: int | None = None,
    chunk_size: int = CHUNK_SIZE,
    attempts: int = 40,
    workers: int | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> bool:
    """Download `target` in chunks, resuming whatever is already on disk.

    Returns True when the file is complete. `expected_size` is the size the
    chunk map is built from; when it is None the size is probed from the first
    mirror that answers.

    Chunks are fetched by several connections at once: one stream on a
    throttled link is limited to a fraction of what the link can carry, while
    parallel range requests reach the link's real bandwidth.
    """
    if not urls:
        raise ValueError("No download URL candidates were provided.")

    probes = probe_mirrors(urls)
    size = expected_size
    if size is None:
        for _, probed_size, _ in probes:
            if probed_size:
                size = probed_size
                break
    if not size:
        raise RuntimeError(f"Could not determine the size of {target.name} from any mirror.")

    target.parent.mkdir(parents=True, exist_ok=True)
    state_path = target.with_name(target.name + ".chunks.json")
    marker_path = target.with_name(target.name + ".complete")
    # A file that only has the right *size* is not a finished download: the
    # chunked path pre-allocates the target, so an interrupted run leaves a
    # zero-filled file that looks complete. The marker file is the only proof
    # that the bytes are really there.
    if marker_path.is_file() and target.is_file() and target.stat().st_size == size:
        return True
    done = _load_state(state_path)

    if size <= WHOLE_FILE_LIMIT:
        if _download_whole(urls, target, size, on_progress=on_progress):
            marker_path.touch()
            return True
        return False

    ranged = [url for url, _, honours in probes if honours]
    if not ranged:
        raise RuntimeError(
            f"No mirror of {target.name} honours range requests; refusing to stream "
            f"{size / 1024**3:.1f} GB in one response."
        )

    with open(target, "r+b" if target.exists() else "wb") as handle:
        handle.truncate(size)

    missing = [
        (index, offset, length)
        for index, (offset, length) in enumerate(_chunk_ranges(size, chunk_size))
        if done.get(str(index)) != length
    ]
    if not missing:
        marker_path.touch()
        state_path.unlink(missing_ok=True)
        return True

    lock = threading.Lock()
    worker_count = workers or DEFAULT_WORKERS
    failed: list[int] = []

    def worker(index: int, offset: int, length: int) -> bool:
        if not _fetch_chunk(ranged, target, offset, length, attempts=attempts):
            return False
        with lock:
            done[str(index)] = length
            # Persist after every chunk: a 3.8 GB file downloaded at a few
            # hundred KB/s runs for hours, and an interrupted run has to
            # continue, not restart.
            _save_state(state_path, done)
            if on_progress is not None:
                on_progress(target.name, sum(done.values()), size)
        return True

    with ThreadPoolExecutor(max_workers=max(1, min(worker_count, len(missing)))) as pool:
        futures = {pool.submit(worker, index, offset, length): index for index, offset, length in missing}
        for future in as_completed(futures):
            if not future.result():
                failed.append(futures[future])
                for pending in futures:
                    pending.cancel()

    if failed:
        _save_state(state_path, done)
        return False
    if target.stat().st_size != size:
        return False
    marker_path.touch()
    state_path.unlink(missing_ok=True)
    return True


def _download_whole(
    urls: Sequence[str],
    target: Path,
    size: int,
    *,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> bool:
    """Fetch a small file in one request, retrying on the other mirrors."""
    import requests

    for attempt in range(1, 6):
        url = urls[(attempt - 1) % len(urls)]
        try:
            response = requests.get(url, stream=True, timeout=(30, 300))
            response.raise_for_status()
            payload = response.content
            response.close()
        except (OSError, requests.RequestException):
            time.sleep(min(20, 2 * attempt))
            continue
        if len(payload) != size:
            time.sleep(min(20, 2 * attempt))
            continue
        target.write_bytes(payload)
        if on_progress is not None:
            on_progress(target.name, size, size)
        return True
    return False


def _chunk_count(size: int, chunk_size: int) -> int:
    return (size + chunk_size - 1) // chunk_size


def _chunk_ranges(size: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    offset = 0
    while offset < size:
        length = min(chunk_size, size - offset)
        yield offset, length
        offset += length


def _fetch_chunk(urls: Sequence[str], target: Path, offset: int, length: int, *, attempts: int) -> bool:
    """Fetch one range and write it at its offset. Each worker uses its own handle."""
    import requests

    end = offset + length - 1
    for attempt in range(1, attempts + 1):
        url = urls[(attempt - 1) % len(urls)]
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={offset}-{end}"},
                stream=True,
                timeout=(30, 300),
            )
            status = response.status_code
            if status != 206:
                response.close()
                time.sleep(min(20, 2 * attempt))
                continue
            payload = response.content
            response.close()
        except (OSError, requests.RequestException):
            time.sleep(min(20, 2 * attempt))
            continue
        if len(payload) != length:
            time.sleep(min(20, 2 * attempt))
            continue
        with open(target, "r+b") as handle:
            handle.seek(offset)
            handle.write(payload)
            handle.flush()
        return True
    return False


def _load_state(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(key): int(value) for key, value in data.items()}


def _save_state(path: Path, done: dict[str, int]) -> None:
    try:
        path.write_text(json.dumps(done), encoding="utf-8")
    except OSError:
        pass
