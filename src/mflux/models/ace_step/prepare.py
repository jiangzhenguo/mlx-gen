"""Prepare a self-contained ACE-Step 1.5 package for `mlxgen music`.

Steps: download the official weights, fetch the pinned upstream runtime tree,
apply the documented mlx-gen patches, create the isolated worker venv, install
its requirements, and write a manifest the runner validates against.

The package layout is:

    <path>/
      model/      official ACE-Step/Ace-Step1.5 weights
      runtime/    official ace-step/ACE-Step-1.5 source tree (pinned commit)
      venv/       isolated Python environment the generation worker runs in
      manifest.json
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

from mflux.cli.runtime_events import cli_print
from mflux.models.ace_step import constants

# The upstream runtime imports modelscope for a download fallback mlx-gen never
# uses (the worker always gets a local model dir). The import is replaced by a stub
# that only fails when it is actually called: importing modelscope for real is slow
# and touches ~/.modelscope, which breaks inside sandboxes and read-only home dirs.
_MODELSCOPE_IMPORT = "from modelscope import snapshot_download"
_MODELSCOPE_STUB = """def snapshot_download(*args, **kwargs):
    # mlx-gen patch: modelscope is disabled, the worker always resolves a local model dir.
    raise RuntimeError('modelscope is not installed; pass a local model_dir')"""
_MODELSCOPE_STUB_MARKER = "mlx-gen patch: modelscope is disabled"

# Runtime files that import modelscope. The first one is required (it is the module
# the upstream CLI uses), the second one only exists in the pinned commit.
_REQUIRED_PATCH_TARGETS = ("acestep/model_downloader.py",)
_OPTIONAL_PATCH_TARGETS = ("acestep/api/model_download.py",)

# ModelScope endpoints. The resolve endpoint serves concrete files only - a
# `directory/**` pattern has to be expanded against the repository listing first.
_MODELSCOPE_API = "https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Root={root}"
_MODELSCOPE_RESOLVE = "https://modelscope.cn/models/{repo}/resolve/master/"

# Upstream pins `requires-python = ">=3.11,<3.13"`; the isolated worker venv is
# created from the interpreter that runs mlx-gen, so it has to match.
_MIN_PYTHON = (3, 11)
_MAX_PYTHON = (3, 13)


def prepare_package(
    path: Path,
    *,
    with_lm: bool = True,
    force: bool = False,
    json_events: bool = False,
    source: str = "auto",
) -> Path:
    """Build the ACE-Step 1.5 package and return its root directory.

    Args:
        path: Package directory (created when missing).
        with_lm: Also fetch the planner LM weights (~3.7 GB). The LM writes the
            lyrics and the song blueprint, so it is needed for prompt-only runs.
        force: Delete an existing package at `path` before building it.
        json_events: Emit progress on stderr instead of stdout.
        source: Weight source - "hf", "modelscope" or "auto" (HF with a
            ModelScope fallback).
    """
    if source not in {"auto", "hf", "modelscope"}:
        raise ValueError(f"Unknown weight source: {source!r}. Use 'auto', 'hf' or 'modelscope'.")
    _check_interpreter_version()

    path = path.expanduser().resolve()
    if force and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    _download_weights(path / "model", with_lm=with_lm, json_events=json_events, source=source)
    _download_code(path / "runtime", json_events=json_events)
    _apply_patches(path / "runtime", json_events=json_events)
    python = _create_venv(path / "venv", json_events=json_events)
    _install_requirements(python, json_events=json_events)

    manifest = {
        "model": constants.MODEL_ALIAS,
        "model_name": constants.MODEL_NAME,
        "weights_repo": constants.WEIGHTS_REPO_ID,
        "modelscope_repo": constants.MODELSCOPE_REPO_ID,
        "code_repo": constants.CODE_REPO,
        "code_commit": constants.CODE_COMMIT,
        "code_tarball": constants.CODE_TARBALL_URL,
        "python": sys.version.split()[0],
        "requirements": list(constants.WORKER_REQUIREMENTS),
        "with_lm": with_lm,
        "patches": ["modelscope-import-optional"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (path / constants.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Validate the finished package through the same gate the runner uses.
    _validate_package(path, json_events=json_events)
    cli_print(f"ACE-Step 1.5 package ready at {path}", json_events=json_events)
    cli_print(
        "Try it: mlxgen music --prompt '一首中文流行歌曲，钢琴伴奏，副歌激昂' --output song.flac",
        json_events=json_events,
    )
    return path


def _check_interpreter_version() -> None:
    version = sys.version_info[:2]
    if version < _MIN_PYTHON or version >= _MAX_PYTHON:
        raise RuntimeError(
            f"ACE-Step 1.5 needs Python >=3.11,<3.13 (upstream pins that range), but this "
            f"interpreter is {sys.version.split()[0]}. Re-run mlx-gen from a Python 3.11 or "
            "3.12 environment."
        )


def _download_weights(model_dir: Path, *, with_lm: bool, json_events: bool, source: str = "auto") -> None:
    patterns = list(constants.REQUIRED_WEIGHT_PATTERNS)
    if with_lm:
        patterns.extend(constants.OPTIONAL_WEIGHT_PATTERNS["lm"])
    cli_print(
        f"Downloading ACE-Step 1.5 weights from {constants.WEIGHTS_REPO_ID} ({constants.WEIGHTS_SIZE_HINT})...",
        json_events=json_events,
    )

    use_hf = source in {"auto", "hf"}
    use_modelscope = source in {"auto", "modelscope"}
    if use_hf:
        try:
            _download_weights_hf(model_dir, patterns=patterns, json_events=json_events)
            return
        except Exception as exc:  # noqa: BLE001 - the ModelScope mirror is the fallback
            if not use_modelscope:
                raise
            cli_print(
                f"Hugging Face download failed ({type(exc).__name__}); "
                f"falling back to the ModelScope mirror of {constants.WEIGHTS_REPO_ID}.",
                json_events=json_events,
            )
    _download_weights_modelscope(model_dir, patterns=patterns, json_events=json_events)


def _download_weights_hf(model_dir: Path, *, patterns: list[str], json_events: bool) -> None:
    from huggingface_hub import snapshot_download

    def attempt() -> None:
        snapshot_download(constants.WEIGHTS_REPO_ID, allow_patterns=patterns, local_dir=str(model_dir))

    _retrying(attempt, what="weights", json_events=json_events, attempts=3)


def _download_weights_modelscope(model_dir: Path, *, patterns: list[str], json_events: bool) -> None:
    """File-by-file download from the ModelScope mirror with resume support.

    ModelScope's resolve endpoint answers Range requests, so interrupted large
    files (the DiT checkpoint is 4.8 GB) continue where they stopped. Plain
    stdlib + requests, no modelscope SDK dependency.
    """
    listing = _modelscope_listing(json_events=json_events)
    base = _MODELSCOPE_RESOLVE.format(repo=constants.MODELSCOPE_REPO_ID)
    for name in _expand_patterns(patterns, listing.keys()):
        target = model_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        remote_size = listing.get(name)
        if remote_size is None:
            remote_size = _modelscope_remote_size(base + name)
        if remote_size is None:
            cli_print(f"Skipping {name} (not present on the mirror).", json_events=json_events)
            continue
        if _file_size(target) == remote_size:
            continue
        _download_modelscope_file(base + name, target, remote_size, json_events=json_events)
        cli_print(f"Downloaded {name} ({remote_size / 1024**2:.0f} MB).", json_events=json_events)


def _modelscope_listing(*, json_events: bool = False) -> dict[str, int]:
    """List every file of the ModelScope mirror with its size in bytes."""
    import requests

    listing: dict[str, int] = {}
    pending = [""]
    while pending:
        root = pending.pop(0)
        url = _MODELSCOPE_API.format(repo=constants.MODELSCOPE_REPO_ID, root=root)
        response = requests.get(url, timeout=(30, 120))
        response.raise_for_status()
        payload = response.json()
        data = payload.get("Data") if isinstance(payload, dict) else None
        files = data.get("Files") if isinstance(data, dict) else None
        for entry in files or []:
            path = entry.get("Path") or ""
            if not path:
                continue
            if entry.get("Type") == "tree":
                pending.append(path)
            else:
                listing[path] = int(entry.get("Size") or 0)
    return listing


def _expand_patterns(patterns: list[str], paths: Iterable[str]) -> list[str]:
    """Expand the Hugging Face style patterns into concrete file paths.

    The weights bundle is a multi-model bundle with sub-directories and the
    ModelScope resolve endpoint serves concrete files only, so a `dir/**` pattern
    has to be expanded against the repository listing first.
    """
    available = list(paths)
    files: list[str] = []
    for pattern in patterns:
        if pattern.endswith("/**"):
            prefix = pattern[:-3]
            matches = available if not prefix else [path for path in available if path.startswith(f"{prefix}/")]
        else:
            matches = [pattern]
        for match in matches:
            if match not in files:
                files.append(match)
    return files


def _download_modelscope_file(url: str, target: Path, remote_size: int, *, json_events: bool) -> None:
    if remote_size < 10 * 1024 * 1024 and target.exists():
        # Small responses do not honour Range; restart them from scratch.
        target.unlink()
    for attempt in range(1, 31):
        if _file_size(target) == remote_size:
            break
        error = _fetch_modelscope_chunk(url, target, remote_size)
        if error is not None:
            cli_print(
                f"Downloading {target.name} interrupted ({type(error).__name__}); retry {attempt}/30...",
                json_events=json_events,
            )
        if _file_size(target) != remote_size:
            time.sleep(min(30, 5 * attempt))
    if _file_size(target) != remote_size:
        raise RuntimeError(
            f"Could not complete {target.name} from the ModelScope mirror ({_file_size(target)}/{remote_size} bytes)."
        )


def _fetch_modelscope_chunk(url: str, target: Path, remote_size: int) -> Exception | None:
    """Fetch the next chunk of a ModelScope file, resuming when possible."""
    import requests

    headers = {}
    mode = "wb"
    current = _file_size(target)
    if 0 < current < remote_size:
        headers["Range"] = f"bytes={current}-"
        mode = "ab"
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
            if response.status_code == 416:  # already complete despite the size mismatch
                return None
            response.raise_for_status()
            if mode == "ab" and response.status_code != 206:
                mode = "wb"  # server ignored the Range header
            with target.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)
    except Exception as exc:  # noqa: BLE001 - the caller retries the download
        return exc
    return None


def _modelscope_remote_size(url: str) -> int | None:
    import requests

    try:
        response = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(30, 60))
    except Exception:  # noqa: BLE001 - the mirror may be unreachable
        return None
    if response.status_code not in {200, 206}:
        return None
    content_range = response.headers.get("Content-Range", "")
    if "/" in content_range:
        total = content_range.rsplit("/", 1)[1].strip()
        if total.isdigit():
            return int(total)
    content_length = response.headers.get("Content-Length", "")
    return int(content_length) if content_length.isdigit() else None


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def _download_code(runtime_dir: Path, *, json_events: bool) -> None:
    cli_print(
        f"Fetching the official ACE-Step 1.5 runtime from {constants.CODE_REPO} @ {constants.CODE_COMMIT[:12]}...",
        json_events=json_events,
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    parent = runtime_dir.parent if runtime_dir.parent.is_dir() else None
    with tempfile.TemporaryDirectory(prefix=".acestep-source-", dir=str(parent) if parent else None) as tmp:
        archive = Path(tmp) / "acestep.tar.gz"
        _retrying(
            lambda: _fetch_tarball(constants.CODE_TARBALL_URL, archive),
            what="ACE-Step runtime",
            json_events=json_events,
            attempts=3,
        )
        _extract_runtime(archive, runtime_dir)
    if not (runtime_dir / "acestep" / "__init__.py").is_file():
        raise RuntimeError(
            f"The fetched ACE-Step runtime does not contain acestep/__init__.py (looked in "
            f"{runtime_dir}). The upstream repository layout changed."
        )


def _fetch_tarball(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "mlx-gen/ace-step"})
    with urllib.request.urlopen(request) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)


def _extract_runtime(archive: Path, runtime_dir: Path) -> None:
    """Extract the tarball, stripping its single top-level directory component."""
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        root = _tarball_root(members)
        rewritten = []
        for member in members:
            relative = _relative_member_path(member.name, root)
            if relative is None:
                continue
            member.name = relative
            rewritten.append(member)
        if hasattr(tarfile, "data_filter"):
            tar.extractall(runtime_dir, members=rewritten, filter="data")
        else:  # pragma: no cover - Python < 3.10.12 has no extraction filters
            tar.extractall(runtime_dir, members=rewritten)


def _tarball_root(members: list[tarfile.TarInfo]) -> str:
    roots = set()
    for member in members:
        name = member.name[2:] if member.name.startswith("./") else member.name
        if not name or name == ".":
            continue
        roots.add(name.split("/", 1)[0])
    if len(roots) != 1:
        raise RuntimeError(
            f"Unexpected ACE-Step tarball layout: expected a single top-level directory, got {sorted(roots)}."
        )
    return roots.pop()


def _relative_member_path(name: str, root: str) -> str | None:
    """Map a tarball member name to a path relative to the runtime directory."""
    normalized = name[2:] if name.startswith("./") else name
    if normalized in {"", ".", root, f"{root}/"}:
        return None
    prefix = f"{root}/"
    if not normalized.startswith(prefix):
        raise RuntimeError(f"Unexpected entry {name!r} in the ACE-Step tarball (expected it below {root!r}).")
    relative = normalized[len(prefix) :]
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise RuntimeError(f"Refusing to extract the unsafe path {name!r} from the ACE-Step tarball.")
    return relative


def _apply_patches(runtime_dir: Path, *, json_events: bool) -> None:
    patched = []
    for relative in _REQUIRED_PATCH_TARGETS:
        target = runtime_dir / relative
        if not target.is_file():
            raise RuntimeError(
                f"ACE-Step patch no longer applies: {relative} is missing from the runtime tree. "
                "The upstream repository layout changed."
            )
        if _patch_modelscope_import(target):
            patched.append(relative)
    for relative in _OPTIONAL_PATCH_TARGETS:
        target = runtime_dir / relative
        if not target.is_file():
            continue
        if _patch_modelscope_import(target):
            patched.append(relative)
    cli_print(f"Patched: modelscope optional ({len(patched)} file(s)).", json_events=json_events)


def _patch_modelscope_import(target: Path) -> bool:
    """Replace the modelscope import with a raising stub. True when the file changed."""
    source = target.read_text(encoding="utf-8")
    if _MODELSCOPE_STUB_MARKER in source:
        return False  # already patched (idempotent re-prepare)
    match = re.search(rf"^(?P<indent>[ \t]*){re.escape(_MODELSCOPE_IMPORT)}$", source, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(
            f"ACE-Step patch no longer applies: {target.name} does not contain "
            f"{_MODELSCOPE_IMPORT!r}. The upstream runtime has drifted; the patch table in "
            "mflux/models/ace_step/prepare.py needs a refresh."
        )
    indent = match.group("indent")
    stub = "\n".join(f"{indent}{line}" if line else line for line in _MODELSCOPE_STUB.splitlines())
    target.write_text(source[: match.start()] + stub + source[match.end() :], encoding="utf-8")
    return True


def _create_venv(venv_dir: Path, *, json_events: bool) -> Path:
    python = venv_dir / "bin" / "python"
    if python.is_file():
        return python
    cli_print(f"Creating the isolated worker environment at {venv_dir}...", json_events=json_events)
    uv = shutil.which("uv")
    if uv is not None:
        subprocess.run([uv, "venv", str(venv_dir), "--python", sys.executable, "-q"], check=True)
    else:
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    return python


def _install_requirements(python: Path, *, json_events: bool) -> None:
    cli_print("Installing worker requirements (torch, transformers, mlx, ...)...", json_events=json_events)
    uv = shutil.which("uv")

    def attempt() -> None:
        if uv is not None:
            subprocess.run(
                [uv, "pip", "install", "--python", str(python), *constants.WORKER_REQUIREMENTS],
                check=True,
                capture_output=not json_events,
            )
        else:
            subprocess.run([str(python), "-m", "ensurepip"], check=True, capture_output=True)
            subprocess.run(
                [str(python), "-m", "pip", "install", *constants.WORKER_REQUIREMENTS],
                check=True,
                capture_output=not json_events,
            )

    _retrying(attempt, what="worker requirements", json_events=json_events)


def _validate_package(path: Path, *, json_events: bool) -> None:
    """Validate the finished package through the same gate the runner uses."""
    try:
        from mflux.models.ace_step.runner import resolve_package
    except ImportError:
        cli_print(
            "The ACE-Step runner is not importable yet; skipping the package self-check.",
            json_events=json_events,
        )
        return
    resolve_package(str(path))


def _retrying(operation: Callable[[], None], *, what: str, json_events: bool, attempts: int = 8) -> None:
    for attempt in range(1, attempts + 1):
        error = _attempt(operation)
        if error is None:
            return
        if attempt == attempts:
            raise error
        delay = min(30, 5 * attempt)
        cli_print(
            f"Downloading {what} failed ({type(error).__name__}: {str(error)[:120]}); "
            f"retry {attempt}/{attempts - 1} in {delay}s...",
            json_events=json_events,
        )
        time.sleep(delay)


def _attempt(operation: Callable[[], None]) -> Exception | None:
    try:
        operation()
    except Exception as exc:  # noqa: BLE001 - the caller retries and re-raises
        return exc
    return None
