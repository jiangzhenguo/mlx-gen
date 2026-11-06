"""Tests for the ACE-Step 1.5 prepare pipeline.

Nothing here touches the network: the weight download, the runtime fetch, the
venv creation and the pip install are all mocked. The upstream runtime tree is
replaced by a tiny local tarball that mimics the pinned GitHub layout.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import huggingface_hub
import pytest

from mflux.models.ace_step import constants, prepare

# The upstream snippet the prepare pipeline has to neutralise: the runtime
# imports modelscope for a download path mlx-gen never uses.
UPSTREAM_DOWNLOADER = '''"""ACE-Step Model Downloader."""


def _download_from_modelscope_internal(repo_id, local_dir):
    from modelscope import snapshot_download

    snapshot_download(model_id=repo_id, local_dir=str(local_dir))
'''

FAKE_UV = "/opt/homebrew/bin/uv"


def _write_upstream_runtime(root: Path) -> Path:
    """Write a minimal stand-in for the upstream ace-step package."""
    runtime = root / "runtime"
    (runtime / "acestep").mkdir(parents=True)
    (runtime / "acestep" / "__init__.py").write_text('"""ACE-Step package."""\n', encoding="utf-8")
    (runtime / "acestep" / "model_downloader.py").write_text(UPSTREAM_DOWNLOADER, encoding="utf-8")
    return runtime


def _write_fake_tarball(tmp_path: Path) -> Path:
    """Build a tarball with the same layout as the pinned GitHub archive."""
    root = tmp_path / "source" / "ACE-Step-1.5-testcommit"
    (root / "acestep").mkdir(parents=True)
    (root / "acestep" / "__init__.py").write_text('"""ACE-Step package."""\n', encoding="utf-8")
    (root / "acestep" / "model_downloader.py").write_text(UPSTREAM_DOWNLOADER, encoding="utf-8")
    archive = tmp_path / "acestep.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(root, arcname=root.name)
    return archive


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """The retry backoff must not slow the test suite down."""
    monkeypatch.setattr(prepare.time, "sleep", lambda *args, **kwargs: None)


@pytest.fixture
def prepare_env(tmp_path, monkeypatch):
    """Mock every network and process boundary of the prepare pipeline."""
    calls: dict[str, list] = {"snapshot_download": [], "subprocess": [], "resolve_package": []}

    def fake_snapshot_download(repo_id, **kwargs):
        calls["snapshot_download"].append({"repo_id": repo_id, **kwargs})

    def fake_run(argv, **kwargs):
        calls["subprocess"].append(list(argv))
        return subprocess.CompletedProcess(argv, 0)

    def fake_which(name, *args, **kwargs):
        return FAKE_UV if name == "uv" else None

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", fake_which)

    runner_module = types.ModuleType("mflux.models.ace_step.runner")

    def fake_resolve_package(model_path):
        calls["resolve_package"].append(model_path)
        return model_path

    runner_module.resolve_package = fake_resolve_package
    monkeypatch.setitem(sys.modules, "mflux.models.ace_step.runner", runner_module)

    archive = _write_fake_tarball(tmp_path)
    monkeypatch.setattr(constants, "CODE_TARBALL_URL", archive.as_uri())

    return calls


def test_expand_patterns_expands_directory_patterns():
    listing = [
        "acestep-v15-turbo/config.json",
        "acestep-v15-turbo/model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "Qwen3-Embedding-0.6B/config.json",
        "acestep-5Hz-lm-1.7B/model.safetensors",
    ]

    files = prepare._expand_patterns(constants.REQUIRED_WEIGHT_PATTERNS, listing)

    assert files == [
        "acestep-v15-turbo/config.json",
        "acestep-v15-turbo/model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "Qwen3-Embedding-0.6B/config.json",
    ]


def test_expand_patterns_keeps_the_planner_lm_optional():
    listing = ["acestep-v15-turbo/config.json", "acestep-5Hz-lm-1.7B/model.safetensors"]

    without_lm = prepare._expand_patterns(constants.REQUIRED_WEIGHT_PATTERNS, listing)
    with_lm = prepare._expand_patterns(
        constants.REQUIRED_WEIGHT_PATTERNS + constants.OPTIONAL_WEIGHT_PATTERNS["lm"],
        listing,
    )

    assert without_lm == ["acestep-v15-turbo/config.json"]
    assert with_lm == ["acestep-v15-turbo/config.json", "acestep-5Hz-lm-1.7B/model.safetensors"]


def test_download_weights_hf_passes_the_expected_allow_patterns(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append({"repo_id": repo_id, **kwargs})

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    patterns = list(constants.REQUIRED_WEIGHT_PATTERNS)

    prepare._download_weights_hf(tmp_path, patterns=patterns, json_events=False)

    assert len(calls) == 1
    assert calls[0]["repo_id"] == constants.WEIGHTS_REPO_ID
    assert calls[0]["allow_patterns"] == patterns
    assert calls[0]["local_dir"] == str(tmp_path)


def test_download_weights_falls_back_to_modelscope(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("hugging face is unreachable")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    seen = {}

    def fake_modelscope(model_dir, *, patterns, json_events):
        seen.update({"dir": model_dir, "patterns": patterns})

    monkeypatch.setattr(prepare, "_download_weights_modelscope", fake_modelscope)

    prepare._download_weights(tmp_path, with_lm=True, json_events=False, source="auto")

    assert seen["dir"] == tmp_path
    assert seen["patterns"] == constants.REQUIRED_WEIGHT_PATTERNS + constants.OPTIONAL_WEIGHT_PATTERNS["lm"]


def test_download_weights_without_the_planner_lm_drops_its_patterns(tmp_path, monkeypatch):
    seen = {}

    def fake_modelscope(model_dir, *, patterns, json_events):
        seen.update({"patterns": patterns})

    monkeypatch.setattr(prepare, "_download_weights_modelscope", fake_modelscope)

    prepare._download_weights(tmp_path, with_lm=False, json_events=False, source="modelscope")

    assert seen["patterns"] == list(constants.REQUIRED_WEIGHT_PATTERNS)


def test_download_weights_hf_source_never_falls_back(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("hugging face is unreachable")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)
    monkeypatch.setattr(
        prepare,
        "_download_weights_modelscope",
        lambda *args, **kwargs: pytest.fail("the modelscope mirror must not be used"),
    )

    with pytest.raises(RuntimeError, match="hugging face is unreachable"):
        prepare._download_weights(tmp_path, with_lm=False, json_events=False, source="hf")


def test_download_weights_modelscope_expands_directory_patterns(tmp_path, monkeypatch):
    listing = {
        "acestep-v15-turbo/config.json": 12,
        "acestep-v15-turbo/model.safetensors": 4096,
        "vae/config.json": 7,
        "vae/diffusion_pytorch_model.safetensors": 2048,
        "acestep-5Hz-lm-1.7B/model.safetensors": 8192,
    }
    monkeypatch.setattr(prepare, "_modelscope_listing", lambda **kwargs: listing)
    downloaded = []

    def fake_download(url, target, remote_size, *, json_events):
        downloaded.append((url, target, remote_size))

    monkeypatch.setattr(prepare, "_download_modelscope_file", fake_download)

    prepare._download_weights_modelscope(tmp_path, patterns=list(constants.REQUIRED_WEIGHT_PATTERNS), json_events=False)

    names = [target.relative_to(tmp_path).as_posix() for _, target, _ in downloaded]
    assert names == [
        "acestep-v15-turbo/config.json",
        "acestep-v15-turbo/model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    # The mirror only serves concrete files - a directory URL is never requested.
    assert [url.rsplit("/", 1)[-1] for url, _, _ in downloaded] == [
        "config.json",
        "model.safetensors",
        "config.json",
        "diffusion_pytorch_model.safetensors",
    ]
    assert all(
        url.startswith("https://modelscope.cn/models/ACE-Step/Ace-Step1.5/resolve/master/") for url, _, _ in downloaded
    )


def test_modelscope_patch_is_idempotent(tmp_path):
    runtime = _write_upstream_runtime(tmp_path)
    target = runtime / "acestep" / "model_downloader.py"

    prepare._apply_patches(runtime, json_events=False)
    patched = target.read_text(encoding="utf-8")

    assert "from modelscope import snapshot_download" not in patched
    assert "mlx-gen patch" in patched
    compile(patched, str(target), "exec")

    # A second prepare on an already patched tree must succeed and change nothing.
    prepare._apply_patches(runtime, json_events=False)
    assert target.read_text(encoding="utf-8") == patched


def test_modelscope_stub_only_raises_when_it_is_called(tmp_path):
    runtime = _write_upstream_runtime(tmp_path)
    target = runtime / "acestep" / "model_downloader.py"

    prepare._apply_patches(runtime, json_events=False)

    namespace: dict = {}
    exec(compile(target.read_text(encoding="utf-8"), str(target), "exec"), namespace)
    with pytest.raises(RuntimeError, match="modelscope is not installed"):
        namespace["_download_from_modelscope_internal"]("ACE-Step/Ace-Step1.5", str(tmp_path))


def test_modelscope_patch_fails_loudly_when_the_upstream_drifted(tmp_path):
    runtime = tmp_path / "runtime"
    (runtime / "acestep").mkdir(parents=True)
    (runtime / "acestep" / "model_downloader.py").write_text("import os\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no longer applies"):
        prepare._apply_patches(runtime, json_events=False)


def test_modelscope_patch_requires_the_downloader_module(tmp_path):
    runtime = tmp_path / "runtime"
    (runtime / "acestep").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="no longer applies"):
        prepare._apply_patches(runtime, json_events=False)


def test_prepare_package_writes_the_manifest_and_builds_the_package(tmp_path, prepare_env):
    package = tmp_path / "pkg"

    result = prepare.prepare_package(package, with_lm=True, json_events=False)

    assert result == package.resolve()
    manifest = json.loads((package / constants.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["model"] == constants.MODEL_ALIAS
    assert manifest["weights_repo"] == constants.WEIGHTS_REPO_ID
    assert manifest["modelscope_repo"] == constants.MODELSCOPE_REPO_ID
    assert manifest["code_repo"] == constants.CODE_REPO
    assert manifest["code_commit"] == constants.CODE_COMMIT
    assert manifest["python"] == sys.version.split()[0]
    assert manifest["requirements"] == list(constants.WORKER_REQUIREMENTS)
    assert manifest["with_lm"] is True
    assert manifest["created_at"]

    # The runtime tree is extracted without its top-level directory component.
    assert (package / "runtime" / "acestep" / "__init__.py").is_file()
    downloader = (package / "runtime" / "acestep" / "model_downloader.py").read_text(encoding="utf-8")
    assert "from modelscope import snapshot_download" not in downloader

    # The weights are fetched with the expected allow_patterns.
    call = prepare_env["snapshot_download"][0]
    assert call["repo_id"] == constants.WEIGHTS_REPO_ID
    assert call["allow_patterns"] == constants.REQUIRED_WEIGHT_PATTERNS + constants.OPTIONAL_WEIGHT_PATTERNS["lm"]
    assert call["local_dir"] == str(package / "model")

    # The isolated venv is created from the interpreter that runs mlx-gen.
    commands = prepare_env["subprocess"]
    venv_command = commands[0]
    assert venv_command[:3] == [FAKE_UV, "venv", str(package / "venv")]
    assert venv_command[venv_command.index("--python") + 1] == sys.executable
    install_command = commands[1]
    assert install_command[:4] == [FAKE_UV, "pip", "install", "--python"]
    assert install_command[4] == str(package / "venv" / "bin" / "python")
    assert install_command[5:] == list(constants.WORKER_REQUIREMENTS)

    # The finished package is validated through the runner gate.
    assert prepare_env["resolve_package"] == [str(package.resolve())]


def test_prepare_package_without_the_planner_lm(tmp_path, prepare_env):
    package = tmp_path / "pkg"

    prepare.prepare_package(package, with_lm=False, json_events=False)

    manifest = json.loads((package / constants.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["with_lm"] is False
    call = prepare_env["snapshot_download"][0]
    assert call["allow_patterns"] == list(constants.REQUIRED_WEIGHT_PATTERNS)


def test_prepare_package_force_wipes_the_existing_package(tmp_path, prepare_env):
    package = tmp_path / "pkg"
    package.mkdir()
    stale = package / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    prepare.prepare_package(package, force=True, json_events=False)

    assert not stale.exists()
    assert (package / constants.MANIFEST_NAME).is_file()


def test_prepare_package_keeps_existing_files_without_force(tmp_path, prepare_env):
    package = tmp_path / "pkg"
    package.mkdir()
    stale = package / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    prepare.prepare_package(package, force=False, json_events=False)

    assert stale.exists()


def test_prepare_package_rejects_an_unknown_source(tmp_path, prepare_env):
    with pytest.raises(ValueError, match="Unknown weight source"):
        prepare.prepare_package(tmp_path / "pkg", source="ftp")


def test_prepare_package_rejects_an_unsupported_interpreter(tmp_path, prepare_env, monkeypatch):
    monkeypatch.setattr(prepare, "_MAX_PYTHON", (3, 11))

    with pytest.raises(RuntimeError, match="needs Python"):
        prepare.prepare_package(tmp_path / "pkg")
