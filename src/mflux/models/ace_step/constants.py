"""ACE-Step 1.5 music-generation integration constants.

ACE-Step runs in an isolated Python environment: the official inference stack pins
`transformers>=4.51,<4.58` and `torch>=2.9.1`, which cannot share MLX-Gen's own
environment (mlx-gen requires transformers>=5). `mlxgen prepare --model ace-step`
builds a self-contained package:

    <path>/
      model/      official ACE-Step/Ace-Step1.5 weights (DiT + VAE + text encoder + planner LM)
      runtime/    official ace-step/ACE-Step-1.5 source tree (pinned commit)
      venv/       isolated Python environment the generation worker runs in
      manifest.json

`mlxgen music` spawns `venv/bin/python` on the worker module and speaks the
standard MLX-Gen JSONL runtime-event contract over its stdout.

The official upstream ships an MLX backend for the DiT decoder, the audio VAE and
the planner LM, so on Apple Silicon the isolated worker still runs natively on
MLX (`--backend mlx`, `use_mlx_dit=True`).
"""

from pathlib import Path

from mflux.cli.defaults.defaults import MFLUX_CACHE_DIR

MODEL_ALIAS = "ace-step"
MODEL_NAME = "ACE-Step 1.5"

# Weights. The Hugging Face repository is a multi-model bundle: the DiT decoder,
# the audio VAE, the Qwen3 text encoder and the 5 Hz planner LM live in separate
# sub-directories that the upstream runtime resolves by name.
WEIGHTS_REPO_ID = "ACE-Step/Ace-Step1.5"
MODELSCOPE_REPO_ID = "ACE-Step/Ace-Step1.5"

# Upstream source. The official project is published on GitHub (not as a Hugging
# Face space), so the runtime tree is fetched as a pinned commit tarball.
CODE_REPO = "ace-step/ACE-Step-1.5"
CODE_COMMIT = "ca1e85fe9430179831e6bc6be790c332190a3866"
CODE_TARBALL_URL = f"https://codeload.github.com/ace-step/ACE-Step-1.5/tar.gz/{CODE_COMMIT}"

# Component directories inside the weights bundle.
DIT_DIR = "acestep-v15-turbo"
VAE_DIR = "vae"
TEXT_ENCODER_DIR = "Qwen3-Embedding-0.6B"
LM_DIR = "acestep-5Hz-lm-1.7B"

# The DiT decoder, the VAE and the text encoder are required for every run; the
# planner LM is optional and only needed when the user wants ACE-Step to write
# lyrics and the song blueprint from a theme instead of using supplied lyrics.
REQUIRED_WEIGHT_PATTERNS = [
    f"{DIT_DIR}/**",
    f"{VAE_DIR}/**",
    f"{TEXT_ENCODER_DIR}/**",
]
OPTIONAL_WEIGHT_PATTERNS = {
    "lm": [
        f"{LM_DIR}/**",
    ],
}

# The weights are ~10 GB in total; the required subset is ~6.3 GB.
WEIGHTS_SIZE_HINT = "~10 GB (DiT 4.8 GB + VAE 0.3 GB + text encoder 1.2 GB + planner LM 3.7 GB)"

# Worker venv dependencies.
#
# These are the upstream inference requirements, narrowed to what the local
# generation path actually imports. Deviations that matter:
#   * transformers MUST stay on 4.51.x-4.57.x. On 5.x the Qwen3 text encoder and
#     the planner LM change attention/mask semantics and the conditioning breaks.
#   * torch is the CPU/MPS build on macOS: the upstream runtime loads the DiT
#     checkpoint with torch and then converts it to the MLX decoder.
#   * mlx / mlx-lm provide the native Apple Silicon path for the DiT, the VAE and
#     the planner LM.
#   * gradio, the API server, the training extras and nano-vllm are deliberately
#     absent: the worker only calls the inference API, never the UI.
WORKER_REQUIREMENTS = [
    "torch>=2.9.1",
    "torchaudio>=2.9.1",
    "torchvision",
    "transformers>=4.51.0,<4.58.0",
    "diffusers>=0.37.0",
    "mlx>=0.25.2",
    "mlx-lm>=0.20.0",
    "numpy",
    "scipy>=1.10.1",
    "soundfile>=0.13.1",
    "einops>=0.8.1",
    "accelerate>=1.12.0",
    "loguru>=0.7.3",
    "tqdm",
    "vector-quantize-pytorch>=1.27.15,<1.28.0",
    "torchao>=0.16.0,<0.17.0",
    "pytorch-wavelets>=1.3.0",
    "pywavelets>=1.9.0",
    "numba>=0.63.1",
    "toml",
    "setuptools<72",
]

MANIFEST_NAME = "manifest.json"

# Audio container written by the worker. Upstream saves FLAC by default; the
# worker normalizes to a single file the CLI can report on.
DEFAULT_AUDIO_FORMAT = "flac"
DEFAULT_DURATION_SECONDS = 60.0
MAX_DURATION_SECONDS = 600.0


def default_package_dir() -> Path:
    return MFLUX_CACHE_DIR / "ace-step"


def is_ace_step_alias(model: str | None) -> bool:
    if model is None:
        return False
    normalized = model.strip().lower().replace("_", "-")
    return normalized in {
        MODEL_ALIAS,
        MODEL_NAME.lower(),
        WEIGHTS_REPO_ID.lower(),
        "ace-step-1.5",
        "acestep-v15-turbo",
    }
