"""Qwen3-TTS VoiceDesign integration constants.

CosyVoice3 can only imitate a voice that already exists as a reference clip.
Qwen3-TTS VoiceDesign closes that gap: it turns a *description* of a voice
("a warm female voice in her thirties, speaking slowly") into a speaker, so no
reference audio is needed at all.

Qwen3-TTS is also the first speech model MLX-Gen runs natively in MLX — the
weights ship as an `mlx-community` conversion and the inference code lives in
`mlx-audio`, which is a pure-MLX dependency. There is therefore no isolated
torch environment here (unlike CosyVoice3); the synthesis worker simply runs as
a subprocess of the current interpreter so the audio stack stays off the CLI's
import path.

`mlxgen prepare --model qwen3-tts-voice-design` builds:

    <path>/
      model/      mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 weights
      manifest.json

`mlxgen voice-design` and `mlxgen dub --voice-design ...` spawn the worker and
speak the standard MLX-Gen JSONL runtime-event contract over its stdout.
"""

from pathlib import Path

from mflux.cli.defaults.defaults import MFLUX_CACHE_DIR

MODEL_ALIAS = "qwen3-tts-voice-design"

# The upstream (PyTorch) release, and the MLX conversion MLX-Gen actually runs.
UPSTREAM_REPO_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
WEIGHTS_REPO_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"

# The MLX inference implementation. Pinned because the voice-design code path
# (`generate_voice_design`) only exists from 0.6.x onwards.
MLX_AUDIO_REQUIREMENT = "mlx-audio>=0.6.4"

# Everything the model needs: the talker (3.8 GB), the 12 Hz speech-tokenizer
# decoder (0.7 GB) and the text tokenizer.
WEIGHT_PATTERNS = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
    "speech_tokenizer/**",
]

# Concrete file list for the ModelScope mirror (the glob above has to be
# expanded because the mirror exposes one file per request).
WEIGHT_FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/model.safetensors",
    "speech_tokenizer/preprocessor_config.json",
]

MANIFEST_NAME = "manifest.json"

# The languages the model's codec carries an explicit id for. "auto" lets the
# model pick the language from the text (the default).
LANGUAGES = (
    "auto",
    "chinese",
    "english",
    "german",
    "italian",
    "portuguese",
    "spanish",
    "japanese",
    "korean",
    "french",
    "russian",
)


def default_package_dir() -> Path:
    return MFLUX_CACHE_DIR / "qwen3-tts-voice-design"


def is_qwen3_tts_voice_alias(model: str | None) -> bool:
    if model is None:
        return False
    normalized = model.strip().lower().replace("_", "-")
    return normalized in {
        MODEL_ALIAS,
        "qwen3-tts",
        "qwen3-tts-voicedesign",
        "qwen3-tts-voice-design",
        "qwen3-tts-12hz-1.7b-voicedesign",
        WEIGHTS_REPO_ID.lower(),
        UPSTREAM_REPO_ID.lower(),
    }


def normalize_language(language: str | None) -> str:
    """Map a user-supplied language name onto the model's language id."""
    if language is None:
        return "auto"
    normalized = language.strip().lower()
    aliases = {
        "zh": "chinese",
        "cn": "chinese",
        "en": "english",
        "ja": "japanese",
        "ko": "korean",
        "de": "german",
        "fr": "french",
        "ru": "russian",
        "pt": "portuguese",
        "es": "spanish",
        "it": "italian",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in LANGUAGES:
        raise ValueError(f"Unsupported language {language!r}. Choose one of: {', '.join(LANGUAGES)}.")
    return normalized
