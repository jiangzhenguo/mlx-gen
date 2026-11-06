"""Tests for the ACE-Step 1.5 worker + runner pair.

Nothing here touches the network, the upstream runtime or a model: the runner is
exercised against a fake worker process that replays a scripted JSONL stream, and
the worker module is imported in MLX-Gen's own environment where the upstream
`acestep` package (and its heavy torch/transformers stack) is not installed.
"""

from __future__ import annotations

import ast
import io
import json
import shutil
import sys
from pathlib import Path

import pytest

from mflux.models.ace_step import (
    constants,
    runner as ace_runner,
    worker as ace_worker,
)
from mflux.models.ace_step.runner import (
    AceStepPackage,
    AceStepPackageError,
    resolve_package,
    synthesize,
)


class _FakeProcess:
    """A `subprocess.Popen` stand-in that replays a scripted stdout/stderr."""

    def __init__(self, lines: list[str], *, returncode: int = 0, stderr: str = "") -> None:
        self.stdout = io.StringIO("".join(f"{line}\n" for line in lines))
        self.stderr = io.StringIO(stderr)
        self._returncode = returncode

    def wait(self) -> int:
        return self._returncode


def _install_fake_process(
    monkeypatch: pytest.MonkeyPatch,
    lines: list[str],
    *,
    returncode: int = 0,
    stderr: str = "",
) -> dict:
    captured: dict = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        return _FakeProcess(lines, returncode=returncode, stderr=stderr)

    monkeypatch.setattr(ace_runner.subprocess, "Popen", fake_popen)
    return captured


def _make_package(tmp_path: Path) -> Path:
    """Build the package layout `mlxgen prepare --model ace-step` produces."""
    root = tmp_path / "pkg"
    for component in (constants.DIT_DIR, constants.VAE_DIR, constants.TEXT_ENCODER_DIR, constants.LM_DIR):
        (root / "model" / component).mkdir(parents=True)
    (root / "runtime" / "acestep").mkdir(parents=True)
    (root / "runtime" / "acestep" / "__init__.py").write_text("", encoding="utf-8")
    (root / "venv" / "bin").mkdir(parents=True)
    (root / "venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / constants.MANIFEST_NAME).write_text(json.dumps({"model": constants.MODEL_ALIAS}), encoding="utf-8")
    return root


def _events() -> list[str]:
    return [
        json.dumps({"phase": "loading", "command": "mlxgen music", "model": "ace-step", "seed": 42}),
        json.dumps({"phase": "loaded", "load_seconds": 12.5, "sample_rate": 48000}),
        json.dumps({"phase": "synthesizing", "step": 3, "total": 8, "total_steps": 8, "progress": 0.6}),
        json.dumps(
            {
                "phase": "complete",
                "output_path": "/tmp/song.flac",
                "audio_seconds": 60.0,
                "sample_rate": 48000,
                "seed": 42,
                "duration": 60.0,
                "gen_seconds": 31.25,
                "load_seconds": 12.5,
            }
        ),
    ]


# --------------------------------------------------------------------------------------
# runner: package resolution
# --------------------------------------------------------------------------------------


def test_resolve_package_accepts_a_complete_package(tmp_path):
    root = _make_package(tmp_path)
    package = resolve_package(str(root))
    assert package.root == root
    assert package.model_dir == root / "model"
    assert package.runtime_dir == root / "runtime"
    assert package.venv_python == root / "venv" / "bin" / "python"
    assert package.manifest_path == root / constants.MANIFEST_NAME


def test_resolve_package_reports_missing_parts(tmp_path):
    root = tmp_path / "pkg"
    root.mkdir()
    with pytest.raises(AceStepPackageError) as excinfo:
        resolve_package(str(root))
    message = str(excinfo.value)
    assert "mlxgen prepare --model ace-step" in message
    assert constants.DIT_DIR in message
    assert constants.VAE_DIR in message
    assert "venv" in message
    assert "manifest.json" in message


def test_resolve_package_reports_a_missing_package_directory(tmp_path):
    with pytest.raises(AceStepPackageError) as excinfo:
        resolve_package(str(tmp_path / "nope"))
    message = str(excinfo.value)
    assert "does not exist" in message
    assert "mlxgen prepare --model ace-step" in message


# --------------------------------------------------------------------------------------
# runner: the worker subprocess
# --------------------------------------------------------------------------------------


def test_synthesize_relays_json_events_verbatim(tmp_path, monkeypatch, capsys):
    root = _make_package(tmp_path)
    package = resolve_package(str(root))
    captured = _install_fake_process(monkeypatch, _events())

    result = synthesize(
        package,
        caption="a cheerful pop song",
        output_path=tmp_path / "song.flac",
        duration=60.0,
        seed=42,
        json_events=True,
        command="mlxgen music",
    )

    out = capsys.readouterr().out
    relayed = [json.loads(line) for line in out.strip().splitlines()]
    assert [event["phase"] for event in relayed] == ["loading", "loaded", "synthesizing", "complete"]
    assert relayed[0]["command"] == "mlxgen music"
    assert result.worker_event_count == 4
    assert result.audio_seconds == 60.0
    assert result.sample_rate == 48000
    assert result.seed == 42
    assert result.duration == 60.0
    assert result.gen_seconds == 31.25
    assert result.load_seconds == 12.5
    assert result.output_path == Path("/tmp/song.flac")
    assert captured["argv"][0] == str(package.venv_python)
    assert captured["argv"][1].endswith("worker.py")


def test_synthesize_builds_the_worker_argv(tmp_path, monkeypatch):
    root = _make_package(tmp_path)
    package = resolve_package(str(root))
    captured = _install_fake_process(monkeypatch, _events())

    synthesize(
        package,
        caption="a cheerful pop song",
        lyrics="la la la",
        output_path=tmp_path / "song.flac",
        duration=30.0,
        bpm=120,
        keyscale="C major",
        timesignature="4",
        instrumental=True,
        vocal_language="zh",
        seed=7,
        audio_format="wav",
        use_lm=True,
        json_events=True,
        command="mlxgen music",
    )

    argv = captured["argv"]
    assert "--runtime-dir" in argv and str(package.runtime_dir) in argv
    assert "--model-dir" in argv and str(package.model_dir) in argv
    assert "--output" in argv and str(tmp_path / "song.flac") in argv
    assert "--caption" in argv and "a cheerful pop song" in argv
    assert "--lyrics" in argv and "la la la" in argv
    assert "--bpm" in argv and "120" in argv
    assert "--keyscale" in argv and "C major" in argv
    assert "--timesignature" in argv and "4" in argv
    assert "--vocal-language" in argv and "zh" in argv
    assert "--audio-format" in argv and "wav" in argv
    assert "--seed" in argv and "7" in argv
    assert "--duration" in argv and "30.0" in argv
    assert "--instrumental" in argv
    assert "--use-lm" in argv
    assert "--model-label" in argv and constants.MODEL_ALIAS in argv
    assert "--command" in argv and "mlxgen music" in argv


def test_synthesize_omits_optional_flags(tmp_path, monkeypatch):
    root = _make_package(tmp_path)
    package = resolve_package(str(root))
    captured = _install_fake_process(monkeypatch, _events())

    synthesize(
        package,
        caption="a cheerful pop song",
        output_path=tmp_path / "song.flac",
        use_lm=False,
        json_events=True,
    )

    argv = captured["argv"]
    assert "--lyrics" not in argv
    assert "--bpm" not in argv
    assert "--keyscale" not in argv
    assert "--timesignature" not in argv
    assert "--instrumental" not in argv
    assert "--use-lm" not in argv


def test_synthesize_renders_human_progress(tmp_path, monkeypatch, capsys):
    root = _make_package(tmp_path)
    package = resolve_package(str(root))
    _install_fake_process(monkeypatch, _events())

    synthesize(
        package,
        caption="a cheerful pop song",
        output_path=tmp_path / "song.flac",
        json_events=False,
    )

    out = capsys.readouterr().out
    assert "Loading ACE-Step 1.5" in out
    assert "Model loaded in 12.5s" in out
    assert "Done:" in out
    assert "song.flac" in out
    # The human rendering must not leak the raw JSONL stream.
    assert '"phase"' not in out


def test_synthesize_raises_on_a_non_zero_exit(tmp_path, monkeypatch):
    root = _make_package(tmp_path)
    package = resolve_package(str(root))
    _install_fake_process(
        monkeypatch,
        [json.dumps({"phase": "failed", "error": "boom"})],
        returncode=1,
        stderr="Traceback (most recent call last):\nRuntimeError: boom",
    )

    with pytest.raises(RuntimeError) as excinfo:
        synthesize(
            package,
            caption="a cheerful pop song",
            output_path=tmp_path / "song.flac",
            json_events=True,
        )
    message = str(excinfo.value)
    assert "exit code 1" in message
    assert "boom" in message
    assert "Worker stderr (tail)" in message
    assert "RuntimeError: boom" in message


def test_synthesize_relays_a_leaked_stdout_line(tmp_path, monkeypatch, capsys):
    root = _make_package(tmp_path)
    package = resolve_package(str(root))
    _install_fake_process(monkeypatch, ["this is not json", *_events()])

    synthesize(
        package,
        caption="a cheerful pop song",
        output_path=tmp_path / "song.flac",
        json_events=False,
    )

    assert "this is not json" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# worker: module shape (importable without the upstream package)
# --------------------------------------------------------------------------------------


def test_worker_imports_the_upstream_lazily():
    source = Path(ace_worker.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    module_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    for node in module_level_imports:
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("acestep")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("acestep")

    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    nested = [
        node
        for statement in main.body
        for node in ast.walk(statement)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("acestep")
    ]
    assert {node.module for node in nested} >= {
        "acestep.handler",
        "acestep.inference",
        "acestep.llm_inference",
    }


def test_worker_reports_a_missing_upstream_as_a_failed_event(tmp_path, capsys, monkeypatch):
    """The upstream package is not installed in MLX-Gen's own environment."""
    model_dir = tmp_path / "model"
    for component in ace_worker.REQUIRED_COMPONENTS:
        (model_dir / component).mkdir(parents=True)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    for key in ("ACESTEP_CHECKPOINTS_DIR", "ACESTEP_PROJECT_ROOT", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))

    exit_code = ace_worker.main(
        [
            "--runtime-dir",
            str(runtime_dir),
            "--model-dir",
            str(model_dir),
            "--output",
            str(tmp_path / "song.flac"),
            "--caption",
            "a cheerful pop song",
        ]
    )

    assert exit_code == 1
    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert events[0]["phase"] == "loading"
    assert events[-1]["phase"] == "failed"
    assert events[-1]["error"]
    assert events[-1]["error_type"]


def test_worker_parser_accepts_the_documented_argv():
    args = ace_worker._build_parser().parse_args(
        [
            "--runtime-dir",
            "/pkg/runtime",
            "--model-dir",
            "/pkg/model",
            "--output",
            "/tmp/song.flac",
            "--caption",
            "a cheerful pop song",
            "--lyrics",
            "la la la",
            "--duration",
            "45",
            "--bpm",
            "120",
            "--keyscale",
            "C major",
            "--timesignature",
            "4",
            "--instrumental",
            "--vocal-language",
            "zh",
            "--seed",
            "7",
            "--audio-format",
            "wav",
            "--use-lm",
            "--model-label",
            "ace-step",
            "--command",
            "mlxgen music",
        ]
    )
    assert args.runtime_dir == "/pkg/runtime"
    assert args.model_dir == "/pkg/model"
    assert args.output == "/tmp/song.flac"
    assert args.caption == "a cheerful pop song"
    assert args.lyrics == "la la la"
    assert args.duration == 45.0
    assert args.bpm == 120
    assert args.keyscale == "C major"
    assert args.timesignature == "4"
    assert args.instrumental is True
    assert args.vocal_language == "zh"
    assert args.seed == 7
    assert args.audio_format == "wav"
    assert args.use_lm is True
    assert args.model_label == "ace-step"
    assert args.command == "mlxgen music"


def test_worker_defaults_are_sane():
    args = ace_worker._build_parser().parse_args(
        [
            "--runtime-dir",
            "/pkg/runtime",
            "--model-dir",
            "/pkg/model",
            "--output",
            "/tmp/song.flac",
            "--caption",
            "a cheerful pop song",
        ]
    )
    assert args.lyrics is None
    assert args.bpm is None
    assert args.keyscale is None
    assert args.timesignature is None
    assert args.instrumental is False
    assert args.use_lm is False
    assert args.seed == 42
    assert args.audio_format == "flac"


def test_worker_emitter_writes_one_json_object_per_line():
    stream = io.StringIO()
    emitter = ace_worker.EventEmitter(stream)
    emitter.emit(command="mlxgen music", model="ace-step", seed=42, phase="loading")
    emitter.emit(
        command="mlxgen music",
        model="ace-step",
        seed=42,
        phase="complete",
        output_path="/tmp/song.flac",
        audio_seconds=60.0,
        sample_rate=48000,
        duration=60.0,
        gen_seconds=31.25,
    )

    lines = stream.getvalue().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["phase"] == "loading"
    assert first["type"] == "runtime"
    assert first["command"] == "mlxgen music"
    assert first["model"] == "ace-step"
    assert first["seed"] == 42
    assert first["task"] == "text-to-music"

    second = json.loads(lines[1])
    assert second["phase"] == "complete"
    assert second["output_path"] == "/tmp/song.flac"
    assert second["audio_seconds"] == 60.0
    assert second["sample_rate"] == 48000
    assert second["duration"] == 60.0
    assert second["gen_seconds"] == 31.25


def test_worker_emitter_drops_empty_fields():
    stream = io.StringIO()
    ace_worker.EventEmitter(stream).emit(
        command="mlxgen music",
        model="ace-step",
        seed=42,
        phase="complete",
        output_path=None,
        error="boom",
    )
    payload = json.loads(stream.getvalue().strip())
    assert "output_path" not in payload
    assert payload["error"] == "boom"


def test_worker_lyrics_and_instrumental_normalisation():
    assert ace_worker._resolve_lyrics(None, True) == ("[Instrumental]", True)
    assert ace_worker._resolve_lyrics("[instrumental]", False) == ("[instrumental]", True)
    assert ace_worker._resolve_lyrics("la la la", False) == ("la la la", False)
    assert ace_worker._resolve_lyrics("la la la", True) == ("la la la", True)


def test_worker_duration_is_clamped_to_the_upstream_range():
    assert ace_worker._resolve_duration(5.0) == 10.0
    assert ace_worker._resolve_duration(120.0) == 120.0
    assert ace_worker._resolve_duration(5000.0) == 600.0
    assert ace_worker._resolve_duration(0.0) == -1.0


def test_worker_progress_maps_onto_diffusion_steps():
    assert ace_worker._diffusion_step(0.52, 8) == 0
    assert ace_worker._diffusion_step(0.655, 8) == 4
    assert ace_worker._diffusion_step(0.79, 8) == 8
    assert ace_worker._diffusion_step(0.99, 8) == 8
    assert ace_worker._diffusion_step(0.6, 0) == 0


def test_worker_progress_callback_emits_synthesizing_events():
    stream = io.StringIO()
    emitter = ace_worker.EventEmitter(stream)
    state = {"step": 0}
    callback = ace_worker._make_progress_callback(
        emitter=emitter,
        command="mlxgen music",
        model="ace-step",
        seed=42,
        total_steps=8,
        state=state,
    )
    callback(0.1, "Phase 1: Generating CoT metadata")
    callback(0.655, desc="Generating music")
    callback(0.99, desc="Preparing audio data...")

    events = [json.loads(line) for line in stream.getvalue().strip().splitlines()]
    # The planner-LM milestone is not a diffusion step, so it is not relayed.
    assert [event["phase"] for event in events] == ["synthesizing", "synthesizing"]
    assert events[0]["step"] == 4
    assert events[0]["total"] == 8
    assert events[0]["total_steps"] == 8
    assert state["step"] == 8


def test_worker_publish_audio_copies_to_the_exact_output_path(tmp_path):
    save_dir = tmp_path / "save"
    save_dir.mkdir()
    produced = save_dir / "0123abcd.flac"
    produced.write_bytes(b"flac-bytes")

    class _Result:
        audios = [{"path": str(produced), "sample_rate": 48000}]

    target = tmp_path / "nested" / "song.flac"
    published = ace_worker._publish_audio(
        result=_Result(),
        save_dir=str(save_dir),
        output=str(target),
        audio_format="flac",
    )
    assert published == target
    assert published.read_bytes() == b"flac-bytes"


def test_worker_publish_audio_reports_a_missing_file(tmp_path):
    save_dir = tmp_path / "save"
    save_dir.mkdir()

    class _Result:
        audios: list = []

    with pytest.raises(RuntimeError, match="without producing an audio file"):
        ace_worker._publish_audio(
            result=_Result(),
            save_dir=str(save_dir),
            output=str(tmp_path / "song.flac"),
            audio_format="flac",
        )


def test_worker_verify_model_dir_reports_missing_components(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        ace_worker._verify_model_dir(str(model_dir), use_lm=False)

    for component in ace_worker.REQUIRED_COMPONENTS:
        (model_dir / component).mkdir()
    ace_worker._verify_model_dir(str(model_dir), use_lm=False)
    with pytest.raises(RuntimeError, match=ace_worker.LM_MODEL_PATH):
        ace_worker._verify_model_dir(str(model_dir), use_lm=True)

    (model_dir / ace_worker.LM_MODEL_PATH).mkdir()
    ace_worker._verify_model_dir(str(model_dir), use_lm=True)


def test_worker_prepare_environment_points_at_the_package(tmp_path, monkeypatch):
    runtime_dir = str(tmp_path / "runtime")
    checkpoints_dir = str(tmp_path / "model")
    for key in ("ACESTEP_CHECKPOINTS_DIR", "ACESTEP_PROJECT_ROOT", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    ace_worker._prepare_environment(runtime_dir=runtime_dir, checkpoints_dir=checkpoints_dir)

    import os

    assert os.environ["ACESTEP_CHECKPOINTS_DIR"] == checkpoints_dir
    assert os.environ["ACESTEP_PROJECT_ROOT"] == runtime_dir
    assert sys.path[0] == runtime_dir


def test_worker_resolve_checkpoints_dir_uses_the_model_dir_when_the_lm_is_present(tmp_path):
    model_dir = tmp_path / "model"
    for component in (*ace_worker.REQUIRED_COMPONENTS, ace_worker.LM_MODEL_PATH):
        (model_dir / component).mkdir(parents=True)
    checkpoints_dir, shadow = ace_worker._resolve_checkpoints_dir(str(model_dir))
    assert checkpoints_dir == str(model_dir)
    assert shadow is None


def test_worker_resolve_checkpoints_dir_stubs_a_missing_planner_lm(tmp_path):
    model_dir = tmp_path / "model"
    for component in ace_worker.REQUIRED_COMPONENTS:
        (model_dir / component).mkdir(parents=True)
    checkpoints_dir, shadow = ace_worker._resolve_checkpoints_dir(str(model_dir))
    assert shadow == checkpoints_dir
    try:
        root = Path(checkpoints_dir)
        assert (root / ace_worker.LM_MODEL_PATH / "model.safetensors").is_file()
        for component in ace_worker.REQUIRED_COMPONENTS:
            assert (root / component).is_dir()
    finally:
        shutil.rmtree(shadow, ignore_errors=True)


def _write_fake_upstream(runtime_dir: Path) -> None:
    """Write a stand-in for the upstream `acestep` package.

    The fake mimics the upstream API surface the worker uses and — like the
    real thing — is chatty on stdout, which is exactly what the worker's
    stdout guard has to keep away from the JSONL event channel.
    """
    package = runtime_dir / "acestep"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "handler.py").write_text(
        "print('upstream handler noise')\n"
        "\n"
        "\n"
        "class AceStepHandler:\n"
        "    sample_rate = 48000\n"
        "\n"
        "    def initialize_service(self, **kwargs):\n"
        "        print('upstream init noise')\n"
        "        return 'ok', True\n",
        encoding="utf-8",
    )
    (package / "llm_inference.py").write_text(
        "print('upstream llm noise')\n"
        "\n"
        "\n"
        "class LLMHandler:\n"
        "    def initialize(self, **kwargs):\n"
        "        return 'ok', True\n",
        encoding="utf-8",
    )
    (package / "inference.py").write_text(
        "import os\n"
        "\n"
        "print('upstream inference noise')\n"
        "\n"
        "\n"
        "class GenerationParams:\n"
        "    inference_steps = 8\n"
        "\n"
        "    def __init__(self, **kwargs):\n"
        "        self.__dict__.update(kwargs)\n"
        "\n"
        "\n"
        "class GenerationConfig:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.__dict__.update(kwargs)\n"
        "\n"
        "\n"
        "class _Tensor:\n"
        "    shape = (2, 48000)\n"
        "\n"
        "\n"
        "class _Result:\n"
        "    status_message = 'ok'\n"
        "    error = None\n"
        "\n"
        "\n"
        "class _Sample:\n"
        "    success = True\n"
        "    caption = 'an intimate folk song'\n"
        "    lyrics = '[Verse]\\nwritten by the planner'\n"
        "    bpm = 88\n"
        "    keyscale = 'C major'\n"
        "    timesignature = '4'\n"
        "\n"
        "\n"
        "def create_sample(llm_handler, query, instrumental=False, vocal_language=None, **kwargs):\n"
        "    print('upstream create sample noise')\n"
        "    return _Sample()\n"
        "\n"
        "\n"
        "def generate_music(dit_handler, llm_handler, params, config, save_dir=None, progress=None):\n"
        "    print('upstream generate noise')\n"
        "    if progress is not None:\n"
        "        progress(0.6, desc='Generating music')\n"
        "    path = os.path.join(save_dir, 'abc123.flac')\n"
        "    with open(path, 'wb') as handle:\n"
        "        handle.write(b'fake-flac')\n"
        "    result = _Result()\n"
        "    result.audios = [{'path': path, 'sample_rate': 48000, 'tensor': _Tensor()}]\n"
        "    return result\n",
        encoding="utf-8",
    )


def test_worker_keeps_stdout_free_of_library_output(tmp_path, capsys, monkeypatch):
    runtime_dir = tmp_path / "runtime"
    _write_fake_upstream(runtime_dir)
    model_dir = tmp_path / "model"
    for component in (*ace_worker.REQUIRED_COMPONENTS, ace_worker.LM_MODEL_PATH):
        (model_dir / component).mkdir(parents=True)
    output = tmp_path / "song.flac"
    for key in ("ACESTEP_CHECKPOINTS_DIR", "ACESTEP_PROJECT_ROOT", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))

    try:
        exit_code = ace_worker.main(
            [
                "--runtime-dir",
                str(runtime_dir),
                "--model-dir",
                str(model_dir),
                "--output",
                str(output),
                "--caption",
                "a cheerful pop song",
                "--use-lm",
            ]
        )
    finally:
        for name in list(sys.modules):
            if name == "acestep" or name.startswith("acestep."):
                del sys.modules[name]

    captured = capsys.readouterr()
    assert exit_code == 0
    # stdout carries the JSONL event channel and nothing else.
    events = [json.loads(line) for line in captured.out.strip().splitlines()]
    assert [event["phase"] for event in events] == ["loading", "loaded", "planning", "synthesizing", "complete"]
    assert events[-1]["output_path"] == str(output)
    assert events[-1]["audio_seconds"] == 1.0
    assert events[-1]["sample_rate"] == 48000
    assert "upstream generate noise" in captured.err
    assert output.read_bytes() == b"fake-flac"


def test_worker_sings_the_planner_written_lyrics_when_none_are_given(tmp_path, capsys, monkeypatch):
    """Without lyrics the worker asks the planner LM to write them (upstream "Simple Mode")."""
    runtime_dir = tmp_path / "runtime"
    _write_fake_upstream(runtime_dir)
    model_dir = tmp_path / "model"
    for component in (*ace_worker.REQUIRED_COMPONENTS, ace_worker.LM_MODEL_PATH):
        (model_dir / component).mkdir(parents=True)
    output = tmp_path / "song.flac"
    for key in ("ACESTEP_CHECKPOINTS_DIR", "ACESTEP_PROJECT_ROOT", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))

    try:
        exit_code = ace_worker.main(
            [
                "--runtime-dir",
                str(runtime_dir),
                "--model-dir",
                str(model_dir),
                "--output",
                str(output),
                "--caption",
                "a cheerful pop song",
                "--use-lm",
            ]
        )
    finally:
        for name in list(sys.modules):
            if name == "acestep" or name.startswith("acestep."):
                del sys.modules[name]

    captured = capsys.readouterr()
    assert exit_code == 0
    events = [json.loads(line) for line in captured.out.strip().splitlines()]
    assert events[-1]["phase"] == "complete"
    assert events[-1]["lyrics"] == "[Verse]\nwritten by the planner"
    assert events[-1]["bpm"] == 88
    assert events[-1]["keyscale"] == "C major"


def test_package_dataclass_exposes_the_layout(tmp_path):
    package = AceStepPackage(root=tmp_path)
    assert package.model_dir == tmp_path / "model"
    assert package.runtime_dir == tmp_path / "runtime"
    assert package.venv_python == tmp_path / "venv" / "bin" / "python"
    assert package.manifest_path == tmp_path / constants.MANIFEST_NAME
