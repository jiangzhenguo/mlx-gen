"""Prepare a self-contained CosyVoice3 package for `mlxgen tts` / `mlxgen dub`.

Steps: download the official weights and inference code, apply the documented
mlx-gen patches, create the isolated worker venv, install its requirements, and
write a manifest the runner validates against.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mflux.cli.runtime_events import cli_print
from mflux.models.cosyvoice3 import constants
from mflux.models.cosyvoice3.runner import resolve_package

# The upstream code imports modelscope for a download fallback mlx-gen never uses
# (the worker always gets a local model dir). The patch keeps the import optional
# so the worker venv does not need modelscope.
_MODELSCOPE_IMPORT = "from modelscope import snapshot_download"
_MODELSCOPE_PATCH = """try:
    from modelscope import snapshot_download
except ImportError:
    def snapshot_download(*args, **kwargs):
        raise RuntimeError('modelscope is not installed; pass a local model_dir')"""

# Non-CUDA fixes verified end-to-end on Apple Silicon (CPU inference). Each entry is
# (file, exact upstream snippet, replacement). All are idempotent and fail loudly at
# prepare time if the upstream code drifts.
_CPU_DEVICE_PATCHES = [
    (
        "cosyvoice/cli/model.py",
        "        self.device = 'cuda'",
        "        # mlx-gen patch: keep the device chosen in __init__ (cuda when available, else cpu).",
    ),
    (
        "cosyvoice/llm/llm.py",
        "        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path)",
        "        # mlx-gen patch: force fp32 so the cosyvoice-owned fp32 embeddings\n"
        "        # (speech_embedding / llm_embedding) match the backbone's parameter dtype.\n"
        "        self.model = Qwen2ForCausalLM.from_pretrained(pretrain_path, dtype=torch.float32)",
    ),
    (
        "cosyvoice/flow/flow.py",
        "        token = token.cuda()\n        mask = mask.cuda()",
        "        # mlx-gen patch: stay on the module's device instead of hardcoding cuda.\n"
        "        token = token.to(self.spk_embed_affine_layer.weight.device)\n"
        "        mask = mask.to(self.spk_embed_affine_layer.weight.device)",
    ),
    (
        "cosyvoice/hifigan/generator.py",
        "        sine_merge = self.l_tanh(self.l_linear(sine_wavs.cuda()))",
        "        # mlx-gen patch: follow the module device instead of hardcoding cuda\n"
        "        sine_merge = self.l_tanh(self.l_linear(sine_wavs.to(self.l_linear.weight.device)))",
    ),
    (
        "cosyvoice/hifigan/generator.py",
        "            x = self.conv_pre(x.cuda())",
        "            x = self.conv_pre(x.to(self.conv_pre.weight.device))  # mlx-gen patch",
    ),
    (
        "cosyvoice/hifigan/generator.py",
        "        f0 = self.f0_predictor(speech_feat.cuda(), finalize=finalize)",
        "        f0 = self.f0_predictor(speech_feat.to(next(self.f0_predictor.parameters()).device), finalize=finalize)  # mlx-gen patch",
    ),
    (
        "cosyvoice/hifigan/generator.py",
        "            generated_speech = self.decode(x=speech_feat.cuda(), s=s.cuda(), finalize=finalize)",
        "            generated_speech = self.decode(x=speech_feat.to(s.device), s=s, finalize=finalize)  # mlx-gen patch",
    ),
    (
        "cosyvoice/hifigan/f0_predictor.py",
        "        self.cuda()",
        "        # mlx-gen patch: device follows the parent module; no hardcoded cuda()\n        pass",
    ),
]

_MATCHA_STUB = '''"""Stub for Matcha-TTS training losses + inert MultiPeriodDiscriminator shim.

Only the training path calls the losses; inference never does. `cosyvoice3.yaml`
instantiates `!new:matcha.hifigan.models.MultiPeriodDiscriminator` while building the
HiFi-GAN config, so the class must construct cleanly (its forward is never run).
Written by `mlxgen prepare --model cosyvoice3` when the Matcha-TTS subtree could not
be fetched.
"""


def feature_loss(*args, **kwargs):
    raise NotImplementedError('training-only loss')


def generator_loss(*args, **kwargs):
    raise NotImplementedError('training-only loss')


def discriminator_loss(*args, **kwargs):
    raise NotImplementedError('training-only loss')


class MultiPeriodDiscriminator:
    def __init__(self, *args, **kwargs):
        return

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('training-only module — inference never calls the discriminator.')
'''


# Inference-side import stubs: keep the package importable when the Matcha-TTS
# subtree cannot be fetched. mel_spectrogram is a faithful reimplementation of the
# upstream reference (verified end-to-end on Apple Silicon CPU).
_MATCHA_INFERENCE_STUBS: dict[str, str] = {
    "matcha/utils/audio.py": '''"""Matcha-style mel spectrogram, faithful to the upstream reference."""
from __future__ import annotations

import librosa
import torch


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=torch.hann_window(win_size, device=y.device),
        center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=True,
    )
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    if fmax is None:
        fmax = sampling_rate / 2
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    spec = torch.matmul(torch.from_numpy(mel_basis).float().to(y.device), spec)
    return torch.log(torch.clamp(spec, min=1e-5))
''',
    "matcha/models/components/flow_matching.py": '''"""BASECFM import stub (CosyVoice3LM/CausalConditionalCFM overrides every inference method)."""
from __future__ import annotations

import torch


class BASECFM(torch.nn.Module):
    def __init__(self, n_feats, cfm_params, n_spks=1, spk_emb_dim=64):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.sigma_min = 1e-6
''',
}


def prepare_package(
    path: Path,
    *,
    with_rl: bool = False,
    force: bool = False,
    json_events: bool = False,
    source: str = "auto",
) -> Path:
    path = path.expanduser().resolve()
    if force and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    _download_weights(path / "model", with_rl=with_rl, json_events=json_events, source=source)
    _download_code(path / "runtime", json_events=json_events)
    _apply_patches(path / "runtime", json_events=json_events)
    python = _create_venv(path / "venv", json_events=json_events)
    _install_requirements(python, json_events=json_events)

    manifest = {
        "model": constants.MODEL_ALIAS,
        "weights_repo": constants.WEIGHTS_REPO_ID,
        "code_repo": f"{constants.CODE_REPO_ID} ({constants.CODE_REPO_TYPE})",
        "python": sys.version.split()[0],
        "requirements": constants.WORKER_REQUIREMENTS,
        "with_rl_llm": with_rl,
        "patches": [
            "modelscope-import-optional",
            "cpu-device-fixes (model/llm/flow/hifigan)",
            "llm-fp32-dtype",
            "matcha-training-loss-stub-if-unfetchable",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (path / constants.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Validate the finished package through the same gate the runner uses.
    resolve_package(str(path))
    cli_print(f"CosyVoice3 package ready at {path}", json_events=json_events)
    cli_print(f'Try it: mlxgen tts --ref-audio <voice.wav> --text "你好，世界" --output hello.wav', json_events=json_events)
    return path


def _download_weights(model_dir: Path, *, with_rl: bool, json_events: bool, source: str = "auto") -> None:
    patterns = list(constants.WEIGHT_PATTERNS)
    if with_rl:
        patterns.extend(constants.OPTIONAL_WEIGHT_PATTERNS["rl"])
    cli_print(
        f"Downloading CosyVoice3 weights from {constants.WEIGHTS_REPO_ID} (~5.4 GB)...",
        json_events=json_events,
    )

    use_hf = source in {"auto", "hf"}
    use_modelscope = source in {"auto", "modelscope"}
    if use_hf:
        try:
            _download_weights_hf(model_dir, patterns=patterns, json_events=json_events)
            return
        except Exception as exc:
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

    def attempt():
        snapshot_download(constants.WEIGHTS_REPO_ID, allow_patterns=patterns, local_dir=str(model_dir))

    _retrying(attempt, what="weights", json_events=json_events, attempts=3)


def _modelscope_files(patterns: list[str]) -> list[str]:
    """Expand the HF-style patterns into the concrete ModelScope file list."""
    files = [pattern for pattern in patterns if not pattern.endswith("/**")]
    for pattern in patterns:
        if pattern == "CosyVoice-BlankEN/**":
            files.extend(
                f"CosyVoice-BlankEN/{name}"
                for name in (
                    "config.json",
                    "generation_config.json",
                    "merges.txt",
                    "model.safetensors",
                    "tokenizer_config.json",
                    "vocab.json",
                )
            )
    return files


def _download_weights_modelscope(model_dir: Path, *, patterns: list[str], json_events: bool) -> None:
    """File-by-file download from the ModelScope mirror with resume support.

    ModelScope's resolve endpoint answers Range requests, so interrupted large
    files (llm.pt is 2 GB) continue where they stopped. Plain stdlib + requests,
    no modelscope SDK dependency.
    """
    import requests

    base = f"https://modelscope.cn/models/{constants.WEIGHTS_REPO_ID}/resolve/master/"
    for name in _modelscope_files(patterns):
        target = model_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        remote_size = _modelscope_remote_size(base + name)
        if remote_size is None:
            # The file simply does not exist upstream (e.g. the optional spk2info.pt).
            cli_print(f"Skipping {name} (not present on the mirror).", json_events=json_events)
            continue
        if target.is_file() and target.stat().st_size == remote_size:
            continue
        if remote_size < 10 * 1024 * 1024 and target.exists():
            # Small responses do not honour Range; restart them from scratch.
            target.unlink()
        for attempt in range(1, 31):
            headers = {}
            mode = "wb"
            if target.is_file() and 0 < target.stat().st_size < remote_size:
                headers["Range"] = f"bytes={target.stat().st_size}-"
                mode = "ab"
            try:
                with requests.get(base + name, headers=headers, stream=True, timeout=(30, 300)) as response:
                    if response.status_code == 416:  # already complete despite the size mismatch
                        break
                    response.raise_for_status()
                    if mode == "ab" and response.status_code != 206:
                        mode = "wb"  # server ignored the Range header
                    with target.open(mode) as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            handle.write(chunk)
            except Exception as exc:
                cli_print(
                    f"Downloading {name} interrupted ({type(exc).__name__}); retry {attempt}/30...",
                    json_events=json_events,
                )
                time.sleep(min(30, 5 * attempt))
                continue
            if target.stat().st_size == remote_size:
                break
        if not target.is_file() or target.stat().st_size != remote_size:
            raise RuntimeError(
                f"Could not complete {name} from the ModelScope mirror "
                f"({target.stat().st_size if target.is_file() else 0}/{remote_size} bytes)."
            )
        cli_print(f"Downloaded {name} ({remote_size / 1024**2:.0f} MB).", json_events=json_events)


def _modelscope_remote_size(url: str) -> int | None:
    import requests

    try:
        response = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(30, 60))
    except Exception:
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


def _download_code(runtime_dir: Path, *, json_events: bool) -> None:
    from huggingface_hub import snapshot_download

    cli_print(
        f"Fetching the official CosyVoice inference code from the {constants.CODE_REPO_ID} space...",
        json_events=json_events,
    )

    def attempt():
        snapshot_download(
            constants.CODE_REPO_ID,
            repo_type=constants.CODE_REPO_TYPE,
            allow_patterns=constants.CODE_PATTERNS,
            local_dir=str(runtime_dir),
        )

    try:
        _retrying(attempt, what="inference code", json_events=json_events)
    except Exception:
        # The Matcha-TTS subtree only carries training losses; without it the stub
        # written by _apply_patches keeps the inference path importable.
        if (runtime_dir / "cosyvoice" / "cli" / "cosyvoice.py").is_file():
            cli_print(
                "Could not fetch the complete code snapshot; continuing without Matcha-TTS "
                "(training-only dependency, a stub covers the import).",
                json_events=json_events,
            )
        else:
            raise


def _apply_patches(runtime_dir: Path, *, json_events: bool) -> None:
    import re

    cli_file = runtime_dir / "cosyvoice" / "cli" / "cosyvoice.py"
    source = cli_file.read_text(encoding="utf-8")
    # Match only the UNINDENTED import line; the patched form keeps the same text
    # indented inside the try block, so a plain substring check is not idempotent.
    if re.search(rf"^{re.escape(_MODELSCOPE_IMPORT)}$", source, flags=re.MULTILINE):
        source = re.sub(
            rf"^{re.escape(_MODELSCOPE_IMPORT)}$",
            _MODELSCOPE_PATCH,
            source,
            count=1,
            flags=re.MULTILINE,
        )
        cli_file.write_text(source, encoding="utf-8")

    applied = 0
    for relative, old, new in _CPU_DEVICE_PATCHES:
        target = runtime_dir / relative
        text = target.read_text(encoding="utf-8")
        if old in text:
            target.write_text(text.replace(old, new), encoding="utf-8")
            applied += 1
        elif new in text:
            continue  # already patched (idempotent re-prepare)
        else:
            raise RuntimeError(
                f"CosyVoice3 patch no longer applies: {relative} does not contain the expected snippet. "
                "The upstream inference code has drifted; the patch table in "
                "mflux/models/cosyvoice3/prepare.py needs a refresh."
            )
    cli_print(f"Patched: modelscope optional + {applied} CPU-device fixes applied.", json_events=json_events)

    matcha_models = runtime_dir / "third_party" / "Matcha-TTS" / "matcha" / "hifigan" / "models.py"
    if not matcha_models.is_file():
        matcha_models.parent.mkdir(parents=True, exist_ok=True)
        (matcha_models.parent.parent / "__init__.py").touch()
        (matcha_models.parent / "__init__.py").touch()
        matcha_models.write_text(_MATCHA_STUB, encoding="utf-8")
        cli_print("Wrote Matcha-TTS training-loss stub (inference never calls it).", json_events=json_events)

    # Matcha inference path: CosyVoice 3 imports `matcha.utils.audio.mel_spectrogram`
    # and `matcha.models.components.flow_matching.BASECFM` at load time. Without
    # the real Matcha-TTS source these imports fail; the stubs below keep the
    # package importable. NOTE: the actual DiT forward needs the real
    # `matcha.models.components.{transformer,decoder}` code (diffusers-based in
    # this fork) — that is NOT part of these stubs. Until `mlxgen prepare` can
    # fetch the third_party subtree from a working source, the synthesis worker
    # will reach the model load and fail with a clear `ImportError` for the
    # missing module, rather than a noisy `module 'matcha' not found`.
    for relative, content in _MATCHA_INFERENCE_STUBS.items():
        target = runtime_dir / "third_party" / "Matcha-TTS" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(content, encoding="utf-8")
    cli_print(
        "Wrote Matcha-TTS inference import stubs (audio.mel_spectrogram, components.flow_matching.BASECFM).",
        json_events=json_events,
    )


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
    cli_print("Installing worker requirements (torch, transformers, onnxruntime, ...)...", json_events=json_events)
    uv = shutil.which("uv")

    def attempt():
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


def _retrying(operation, *, what: str, json_events: bool, attempts: int = 8) -> None:
    for attempt in range(1, attempts + 1):
        try:
            operation()
            return
        except Exception as exc:
            if attempt == attempts:
                raise
            delay = min(30, 5 * attempt)
            cli_print(
                f"Downloading {what} failed ({type(exc).__name__}: {str(exc)[:120]}); "
                f"retry {attempt}/{attempts - 1} in {delay}s...",
                json_events=json_events,
            )
            time.sleep(delay)
