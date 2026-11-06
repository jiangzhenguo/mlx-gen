"""CosyVoice3 text-to-speech integration constants.

CosyVoice3 runs in an isolated Python environment: the official inference stack pins
older torch/transformers combinations that cannot share MLX-Gen's own environment
(mlx-gen requires transformers>=5, the CosyVoice runtime is validated against 4.x).
`mlxgen prepare --model cosyvoice3` builds a self-contained package:

    <path>/
      model/      official FunAudioLLM/Fun-CosyVoice3-0.5B-2512 weights
      runtime/    official cosyvoice inference code (HF space snapshot) + mlx-gen patches
      venv/       isolated Python environment the synthesis worker runs in
      manifest.json

`mlxgen tts` and `mlxgen dub` spawn `venv/bin/python` on the worker module and speak
the standard MLX-Gen JSONL runtime-event contract over its stdout.
"""

from pathlib import Path

from mflux.cli.defaults.defaults import MFLUX_CACHE_DIR

MODEL_ALIAS = "cosyvoice3"
MODEL_NAME = "Fun-CosyVoice3-0.5B-2512"
WEIGHTS_REPO_ID = f"FunAudioLLM/{MODEL_NAME}"
CODE_REPO_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B"
CODE_REPO_TYPE = "space"

# The inference subset of the weight repository: llm.rl.pt (RL variant),
# flow.decoder.estimator.fp32.onnx (TensorRT export) and the batched speech tokenizer
# stay server-side; nothing in the local synthesis path reads them.
WEIGHT_PATTERNS = [
    "CosyVoice-BlankEN/**",
    "campplus.onnx",
    "speech_tokenizer_v3.onnx",
    "cosyvoice3.yaml",
    "config.json",
    "configuration.json",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "spk2info.pt",
]
OPTIONAL_WEIGHT_PATTERNS = {
    # The RL-tuned LLM variant; `mlxgen tts --llm-variant rl` swaps it in.
    "rl": ["llm.rl.pt"],
}

# Official inference code, mirrored from the HF demo space (GitHub is the upstream,
# but the space snapshot is complete and reachable through the same HF transport as
# the weights). third_party/Matcha-TTS only contributes training losses imported at
# module load; prepare falls back to a documented stub when it cannot be fetched.
CODE_PATTERNS = [
    "cosyvoice/**",
    "third_party/Matcha-TTS/matcha/**",
    "zero_shot_prompt.wav",
]

# Worker venv dependencies.
#
# The versions matter — they are the ones the upstream CosyVoice 3 Space pins, and
# deviating from them breaks the model in ways that are hard to see:
#   * transformers MUST stay on 4.51.x. On 5.x the Qwen2 backbone silently computes
#     different attention/mask semantics, the LLM emits garbage speech tokens, and
#     the audio comes out as unintelligible noise instead of an error.
#   * numpy MUST stay < 2. onnxruntime 1.18 is compiled against numpy 1.x and
#     fails to import under numpy 2 ("_ARRAY_API not found").
#   * wetext MUST be 0.1.x. 0.0.4 downloads its FST models from ModelScope during
#     Normalizer() construction, which breaks offline use and read-only home dirs.
#   * setuptools<80 keeps pkg_resources, which cosyvoice's dataset processor imports.
#   * torch is capped below 2.8: newer releases reorganised torch.mps and break the
#     map_location path the worker relies on.
WORKER_REQUIREMENTS = [
    "torch>=2.7,<2.8",
    "torchaudio>=2.7,<2.8",
    "transformers==4.51.3",
    "numpy<2",
    "diffusers==0.29.0",
    "conformer==0.3.2",
    "einops",
    "x-transformers>=2.3.5",
    "hyperpyyaml==1.2.2",
    "ruamel.yaml<0.18",
    "onnxruntime==1.18.0",
    "openai-whisper",
    "tiktoken",
    "inflect==7.3.1",
    "wetext==0.1.8",
    "omegaconf",
    "soundfile==0.12.1",
    "librosa==0.10.2",
    "pyarrow==18.1.0",
    "pyworld==0.3.4",
    "setuptools<80",
]

MANIFEST_NAME = "manifest.json"


def default_package_dir() -> Path:
    return MFLUX_CACHE_DIR / "cosyvoice3"


def is_cosyvoice3_alias(model: str | None) -> bool:
    if model is None:
        return False
    normalized = model.strip().lower().replace("_", "-")
    return normalized in {MODEL_ALIAS, MODEL_NAME.lower(), WEIGHTS_REPO_ID.lower()}
