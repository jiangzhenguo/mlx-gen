import json
from pathlib import Path

import mlx.nn as nn
from mlx.utils import tree_flatten

from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.resolution.quantization_resolution import QuantizationResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder import Qwen3VLDecoder
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel
from mflux.models.qwen3_vl_caption.tokenizer.qwen3vl_processor import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
)
from mflux.models.qwen3_vl_caption.weights.qwen3_vl_caption_weight_definition import (
    Qwen3VLCaptionWeightDefinition,
)


class Qwen3VLCaptionInitializer:
    """Build a caption route from an MLX-format Qwen3-VL checkpoint.

    Supports the mlx-lm / mlx-vlm / LM Studio layout (`language_model.model.*`,
    `language_model.lm_head.*`, `vision_tower.*`) for any dense Qwen3-VL size and any published
    quantization level, because every shape and the quantization triplet come from `config.json`.
    """

    @staticmethod
    def init(
        model,
        model_path: str,
        quantize: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        root_path = PathResolution.resolve(
            path=model_path,
            patterns=Qwen3VLCaptionWeightDefinition.get_download_patterns(),
        )
        config = Qwen3VLCaptionInitializer._load_config(root_path)
        config_pixels = Qwen3VLCaptionInitializer._resolve_pixel_budget(root_path)

        weights = WeightLoader.load(
            weight_definition=Qwen3VLCaptionWeightDefinition,
            model_path=model_path,
        )
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Qwen3VLCaptionWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )
        model.processor.set_pixel_budget(
            min_pixels=config_pixels[0] if min_pixels is None else min_pixels,
            max_pixels=config_pixels[1] if max_pixels is None else max_pixels,
        )

        model.decoder = Qwen3VLCaptionInitializer._build_decoder(config)
        model.bits = Qwen3VLCaptionInitializer._apply_weights(
            decoder=model.decoder,
            weights=weights,
            config=config,
            quantize=quantize,
        )

    @staticmethod
    def _load_config(root_path: Path) -> dict:
        config_path = root_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Qwen3-VL checkpoint is missing config.json: {config_path}")
        return json.loads(config_path.read_text())

    @staticmethod
    def _resolve_pixel_budget(root_path: Path) -> tuple[int, int]:
        """Read the checkpoint's own image-token budget.

        `processor_config.json` carries the effective values; Qwen3-VL repos that only ship
        `preprocessor_config.json` put the same budget under `size.shortest_edge/longest_edge`
        (their top-level `min_pixels`/`max_pixels` are null).
        """

        def _read(path: Path) -> dict:
            if not path.exists():
                return {}
            data = json.loads(path.read_text())
            nested = data.get("image_processor")
            return nested if isinstance(nested, dict) else data

        processor_config = _read(root_path / "processor_config.json")
        preprocessor_config = _read(root_path / "preprocessor_config.json")

        min_pixels = processor_config.get("min_pixels") or preprocessor_config.get("min_pixels")
        max_pixels = processor_config.get("max_pixels") or preprocessor_config.get("max_pixels")
        size = preprocessor_config.get("size") or processor_config.get("size") or {}
        if min_pixels is None:
            min_pixels = size.get("shortest_edge")
        if max_pixels is None:
            max_pixels = size.get("longest_edge")

        return (
            int(min_pixels) if min_pixels else DEFAULT_MIN_PIXELS,
            int(max_pixels) if max_pixels else DEFAULT_MAX_PIXELS,
        )

    @staticmethod
    def _build_decoder(config: dict) -> Qwen3VLDecoder:
        text_config = config["text_config"]
        vision_config = config["vision_config"]
        rope_scaling = text_config.get("rope_scaling") or {}

        visual = Qwen3VLVisionModel(
            patch_size=vision_config["patch_size"],
            temporal_patch_size=vision_config["temporal_patch_size"],
            in_channels=vision_config["in_channels"],
            hidden_size=vision_config["hidden_size"],
            num_heads=vision_config["num_heads"],
            intermediate_size=vision_config["intermediate_size"],
            depth=vision_config["depth"],
            spatial_merge_size=vision_config["spatial_merge_size"],
            num_position_embeddings=vision_config["num_position_embeddings"],
            out_hidden_size=vision_config["out_hidden_size"],
            deepstack_visual_indexes=vision_config["deepstack_visual_indexes"],
            hidden_act=vision_config.get("hidden_act", "gelu_pytorch_tanh"),
        )
        return Qwen3VLDecoder(
            vocab_size=text_config["vocab_size"],
            hidden_size=text_config["hidden_size"],
            num_hidden_layers=text_config["num_hidden_layers"],
            num_attention_heads=text_config["num_attention_heads"],
            num_key_value_heads=text_config["num_key_value_heads"],
            intermediate_size=text_config["intermediate_size"],
            max_position_embeddings=text_config["max_position_embeddings"],
            rope_theta=text_config["rope_theta"],
            rms_norm_eps=text_config["rms_norm_eps"],
            head_dim=text_config["head_dim"],
            attention_bias=text_config.get("attention_bias", False),
            mrope_section=rope_scaling.get("mrope_section"),
            image_token_id=config["image_token_id"],
            visual=visual,
            # Qwen3-VL is trained with (t, h, w) image positions and deepstack re-injection; the
            # shared decoder keeps both off by default so FIBO-VLM / Qwen-Image-2.1 are untouched.
            multimodal_position_ids=True,
            deepstack=True,
        )

    @staticmethod
    def _apply_weights(
        decoder: Qwen3VLDecoder,
        weights: LoadedWeights,
        config: dict,
        quantize: int | None,
    ) -> int | None:
        decoder_weights = weights.components.get("decoder")
        visual_weights = weights.components.get("visual")
        if decoder_weights is None or visual_weights is None:
            raise ValueError("Qwen3-VL checkpoint is missing its decoder or vision tower weights.")

        quantization = config.get("quantization") or config.get("quantization_config") or {}
        stored_bits = int(quantization["bits"]) if quantization.get("bits") else None
        group_size = int(quantization.get("group_size") or 64)
        mode = str(quantization.get("mode") or "affine")

        bits, warning = QuantizationResolution.resolve(stored=stored_bits, requested=quantize)
        if warning:
            print(f"⚠️  {warning}")

        # Key checks run against the live module tree, so quantization has to happen first: only
        # then does the tree expose the `.scales`/`.biases` entries a 4-bit checkpoint publishes.
        if stored_bits is not None and bits is not None:
            # Already-quantized checkpoint: build matching quantized modules so the packed
            # weight/scales/biases triple applies directly, using the checkpoint's own group size
            # and mode (mxfp4/nvfp4 repos differ from the affine default).
            nn.quantize(
                decoder,
                group_size=group_size,
                bits=bits,
                mode=mode,
                class_predicate=Qwen3VLCaptionWeightDefinition.quantization_predicate,
            )

        Qwen3VLCaptionInitializer._assert_known_keys(decoder, decoder_weights, "decoder")
        Qwen3VLCaptionInitializer._assert_known_keys(decoder.visual, visual_weights, "visual")
        decoder.update(decoder_weights, strict=False)
        decoder.visual.update(visual_weights, strict=False)

        if stored_bits is None and bits is not None:
            # Plain bf16 checkpoint with a requested level: load first, then quantize in place.
            nn.quantize(
                decoder,
                group_size=group_size,
                bits=bits,
                mode=mode,
                class_predicate=Qwen3VLCaptionWeightDefinition.quantization_predicate,
            )
        else:
            Qwen3VLCaptionInitializer._assert_filled(decoder, decoder_weights, "decoder")
        return bits

    @staticmethod
    def _assert_known_keys(module, component_weights: dict, name: str) -> None:
        expected = set(dict(tree_flatten(module.parameters())))
        unknown = sorted(key for key in dict(tree_flatten(component_weights)) if key not in expected)
        if unknown:
            preview = ", ".join(unknown[:5])
            raise ValueError(
                f"Unsupported Qwen3-VL {name} checkpoint: {len(unknown)} weight(s) do not map onto "
                f"the module tree (e.g. {preview}). This route expects an MLX-format Qwen3-VL repo "
                "with `language_model.model.*` / `language_model.lm_head.*` / `vision_tower.*` keys."
            )

    @staticmethod
    def _assert_filled(decoder: Qwen3VLDecoder, component_weights: dict, name: str) -> None:
        provided = set(dict(tree_flatten(component_weights)))
        missing = sorted(
            key
            for key in dict(tree_flatten(decoder.parameters()))
            if key not in provided and not key.startswith("visual.") and not key.endswith("inv_freq")
        )
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Incomplete Qwen3-VL {name} weights: {len(missing)} parameter(s) were not filled "
                f"by the checkpoint (e.g. {preview})."
            )
