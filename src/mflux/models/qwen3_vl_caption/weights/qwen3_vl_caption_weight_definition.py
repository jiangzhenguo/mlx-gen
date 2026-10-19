from typing import List

from mflux.models.common.tokenizer import VisionLanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.qwen3_vl_caption.tokenizer.qwen3vl_processor import Qwen3VLProcessor

# MLX-converted Qwen3-VL checkpoints (mlx-lm / mlx-vlm / LM Studio) use these top-level prefixes.
DECODER_KEY_PREFIX = "language_model.model."
LM_HEAD_KEY_PREFIX = "language_model.lm_head."
VISION_KEY_PREFIX = "vision_tower."


class Qwen3VLCaptionWeightMapping:
    """Key translation for MLX-format Qwen3-VL checkpoints.

    Those repos already publish MLX tensor layouts - `nn.Linear` weights are (out, in) and
    `patch_embed.proj.weight` is MLX `nn.Conv3d` (out, kD, kH, kW, in) - so the translation is a
    pure prefix strip onto mlx-gen's module tree, with no transpose step.
    """

    @staticmethod
    def decoder_key(key: str) -> str | None:
        if key.startswith(DECODER_KEY_PREFIX):
            return key[len(DECODER_KEY_PREFIX) :]
        if key.startswith(LM_HEAD_KEY_PREFIX):
            return f"lm_head.{key[len(LM_HEAD_KEY_PREFIX) :]}"
        return None

    @staticmethod
    def visual_key(key: str) -> str | None:
        if key.startswith(VISION_KEY_PREFIX):
            return key[len(VISION_KEY_PREFIX) :]
        return None


class Qwen3VLCaptionWeightDefinition:
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        # No mapping_getter on purpose: the key_transform above already produces module-tree paths,
        # so both components take the passthrough branch and keep their packed `.scales`/`.biases`
        # entries exactly as published (the text tower is 4-bit, the vision tower stays bf16).
        return [
            ComponentDefinition(
                name="decoder",
                hf_subdir="",
                loading_mode="mlx_native",
                weight_prefix_filters=[DECODER_KEY_PREFIX, LM_HEAD_KEY_PREFIX],
                key_transform=Qwen3VLCaptionWeightMapping.decoder_key,
            ),
            ComponentDefinition(
                name="visual",
                hf_subdir="",
                loading_mode="mlx_native",
                weight_prefix_filters=[VISION_KEY_PREFIX],
                key_transform=Qwen3VLCaptionWeightMapping.visual_key,
                skip_quantization=True,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="qwen3_vl",
                hf_subdir=".",
                tokenizer_class="Qwen2Tokenizer",
                encoder_class=VisionLanguageTokenizer,
                processor_class=Qwen3VLProcessor,
                max_length=4096,
                # Qwen3-VL repos ship chat_template.jinja instead of an embedded template, so leave
                # `chat_template` unset and let the tokenizer keep the one it loads from disk.
                download_patterns=[
                    "vocab.json",
                    "merges.txt",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                    "special_tokens_map.json",
                    "added_tokens.json",
                ],
            ),
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "*.safetensors",
            "*.json",
            "chat_template.jinja",
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        # The vision tower is a separate bf16 component; the decoder module still nests it, so the
        # predicate has to exclude that subtree explicitly.
        if path.startswith("visual."):
            return False
        return hasattr(module, "to_quantized")
