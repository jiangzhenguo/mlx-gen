import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder import Qwen3VLDecoder
from mflux.models.common_models.qwen3_vl.qwen3_vl_util import Qwen3VLUtil
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel
from mflux.models.qwen3_vl_caption.cli.caption import build_parser
from mflux.models.qwen3_vl_caption.model.qwen3_vl_caption import (
    BACKEND_MLX_VLM,
    BACKEND_MLXGEN,
    DEFAULT_MAX_PIXELS,
    Qwen3VLCaption,
)
from mflux.models.qwen3_vl_caption.model.util import DEFAULT_CAPTION_PROMPT
from mflux.models.qwen3_vl_caption.qwen3_vl_caption_initializer import Qwen3VLCaptionInitializer
from mflux.models.qwen3_vl_caption.tokenizer.qwen3vl_processor import (
    DEFAULT_MAX_PIXELS as PROCESSOR_DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS as PROCESSOR_DEFAULT_MIN_PIXELS,
)
from mflux.models.qwen3_vl_caption.weights.qwen3_vl_caption_weight_definition import (
    Qwen3VLCaptionWeightDefinition,
    Qwen3VLCaptionWeightMapping,
)


def _tiny_decoder(**overrides) -> Qwen3VLDecoder:
    visual = Qwen3VLVisionModel(
        patch_size=16,
        temporal_patch_size=2,
        in_channels=3,
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        depth=2,
        spatial_merge_size=2,
        num_position_embeddings=64,
        out_hidden_size=64,
        deepstack_visual_indexes=[0, 1],
    )
    kwargs = {
        "vocab_size": 128,
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 128,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "head_dim": 32,
        "mrope_section": [8, 4, 4],
        "image_token_id": 7,
        "visual": visual,
        "multimodal_position_ids": True,
        "deepstack": True,
    }
    kwargs.update(overrides)
    return Qwen3VLDecoder(**kwargs)


@pytest.mark.fast
def test_key_mapping_strips_mlx_prefixes_and_rejects_hf_layout():
    mapping = Qwen3VLCaptionWeightMapping

    assert mapping.decoder_key("language_model.model.layers.0.mlp.down_proj.scales") == (
        "layers.0.mlp.down_proj.scales"
    )
    assert mapping.decoder_key("language_model.model.norm.weight") == "norm.weight"
    assert mapping.decoder_key("language_model.lm_head.biases") == "lm_head.biases"
    assert mapping.visual_key("vision_tower.blocks.3.attn.qkv.weight") == "blocks.3.attn.qkv.weight"
    assert mapping.visual_key("vision_tower.patch_embed.proj.weight") == "patch_embed.proj.weight"

    # HF/source layout (torch) keys must not silently map onto the MLX module tree.
    assert mapping.decoder_key("model.language_model.layers.0.mlp.down_proj.weight") is None
    assert mapping.visual_key("model.visual.blocks.0.attn.qkv.weight") is None


@pytest.mark.fast
def test_quantization_predicate_keeps_the_vision_tower_in_bf16():
    predicate = Qwen3VLCaptionWeightDefinition.quantization_predicate
    linear = nn.Linear(64, 64)

    assert predicate("layers.0.self_attn.q_proj", linear) is True
    assert predicate("lm_head", linear) is True
    # The decoder module nests the vision tower, which already ships unquantized in the checkpoint.
    assert predicate("visual.blocks.0.attn.qkv", linear) is False


@pytest.mark.fast
def test_pixel_budget_prefers_processor_config_then_preprocessor_size(tmp_path: Path):
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"image_processor": {"min_pixels": 65536, "max_pixels": 16777216}})
    )
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "min_pixels": None,
                "max_pixels": None,
                "size": {"shortest_edge": 4096, "longest_edge": 1048576},
            }
        )
    )
    assert Qwen3VLCaptionInitializer._resolve_pixel_budget(tmp_path) == (65536, 16777216)

    # Repos that only ship preprocessor_config.json keep the budget under `size`.
    (tmp_path / "processor_config.json").unlink()
    assert Qwen3VLCaptionInitializer._resolve_pixel_budget(tmp_path) == (4096, 1048576)

    # Nothing published -> Qwen3-VL processor defaults, never Qwen2-VL's 256*28*28 budget.
    (tmp_path / "preprocessor_config.json").unlink()
    assert Qwen3VLCaptionInitializer._resolve_pixel_budget(tmp_path) == (
        PROCESSOR_DEFAULT_MIN_PIXELS,
        PROCESSOR_DEFAULT_MAX_PIXELS,
    )


@pytest.mark.fast
def test_multimodal_position_ids_follow_the_image_grid_coordinates():
    # One image token group of 4 (grid 1x4x4, merge 2 -> 2x2 merged patches) between two text tokens.
    input_ids = mx.array([[1, 7, 7, 7, 7, 3]])
    image_grid_thw = mx.array([[1, 4, 4]])

    positions = Qwen3VLUtil.build_multimodal_position_ids(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=2,
        image_token_id=7,
    )

    assert positions.shape == (3, 1, 6)
    # Text keeps the running position; image rows use (t, h, w); text then resumes at 3.
    np.testing.assert_array_equal(np.array(positions[0, 0, :]), np.array([0, 1, 1, 1, 1, 3]))
    np.testing.assert_array_equal(np.array(positions[1, 0, :]), np.array([0, 1, 1, 2, 2, 3]))
    np.testing.assert_array_equal(np.array(positions[2, 0, :]), np.array([0, 1, 2, 1, 2, 3]))


@pytest.mark.fast
def test_multimodal_position_ids_reject_a_grid_that_does_not_match_the_token_count():
    with pytest.raises(ValueError, match="does not match image_grid_thw"):
        Qwen3VLUtil.build_multimodal_position_ids(
            input_ids=mx.array([[1, 7, 7, 7, 3]]),
            image_grid_thw=mx.array([[1, 4, 4]]),
            spatial_merge_size=2,
            image_token_id=7,
        )


@pytest.mark.fast
def test_greedy_sampling_returns_the_argmax_without_dividing_by_zero():
    logits = mx.array([0.0, 10.0, 1.0], dtype=mx.float32)

    assert int(Qwen3VLUtil.sample_top_p(logits, top_p=0.9, temperature=0.0).item()) == 1


@pytest.mark.fast
def test_deepstack_injection_touches_image_rows_only():
    hidden = mx.zeros((1, 4, 2), dtype=mx.float32)
    flat_positions = mx.array([False, True, True, False])
    offsets = mx.array([0, 0, 1, 1], dtype=mx.int32)
    deepstack = mx.array([[5.0, 5.0], [9.0, 9.0]], dtype=mx.float32)

    result = Qwen3VLDecoder._add_deepstack_embeddings(
        hidden_states=hidden,
        flat_positions=flat_positions,
        offsets=offsets,
        deepstack_embeds=deepstack,
    )

    np.testing.assert_array_equal(
        np.array(result).reshape(4, 2),
        np.array([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0], [0.0, 0.0]], dtype=np.float32),
    )


@pytest.mark.fast
def test_quantized_decoder_runs_the_image_route_with_multimodal_rope_and_deepstack():
    decoder = _tiny_decoder()
    nn.quantize(
        decoder,
        group_size=64,
        bits=4,
        class_predicate=Qwen3VLCaptionWeightDefinition.quantization_predicate,
    )

    # A 4-bit lm_head is a packed QuantizedLinear; the decoder must route logits through the module.
    assert isinstance(decoder.lm_head, nn.QuantizedLinear)

    input_ids = mx.array([[1, 2, 7, 7, 7, 7, 3]])
    pixel_values = mx.zeros((16, 3 * 2 * 16 * 16), dtype=mx.float32)
    image_grid_thw = mx.array([[1, 4, 4]])

    logits, past_key_values = decoder(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        use_cache=True,
        max_cache_length=16,
    )
    assert logits.shape == (1, 7, 128)

    # A cached step reuses the prompt positions; images are only part of the first pass.
    step_logits, _ = decoder(
        input_ids=mx.array([[5]]),
        attention_mask=mx.ones((1, 1), dtype=mx.int32),
        use_cache=True,
        past_key_values=past_key_values,
        max_cache_length=16,
    )
    assert step_logits.shape == (1, 1, 128)


@pytest.mark.fast
def test_multimodal_rope_delta_continues_cached_positions_from_the_prompt():
    decoder = _tiny_decoder()
    input_ids = mx.array([[1, 2, 7, 7, 7, 7, 3]])
    pixel_values = mx.zeros((16, 3 * 2 * 16 * 16), dtype=mx.float32)

    decoder(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=mx.array([[1, 4, 4]]),
        use_cache=True,
        max_cache_length=16,
    )

    # The prompt's multimodal positions stop at 4 while it holds 7 flat tokens, so a cached decode
    # step must resume at position 5 instead of 7 (HF's mrope_position_deltas).
    np.testing.assert_array_equal(np.array(decoder.mrope_position_delta), np.array([-2], dtype=np.int32))

    # A text-only first pass clears the offset again, so mixed calls cannot inherit a stale shift.
    decoder(input_ids=mx.array([[1, 2, 3]]), use_cache=True, max_cache_length=8)
    assert decoder.mrope_position_delta is None


@pytest.mark.fast
def test_caption_defaults_match_the_caption_model_card(monkeypatch):
    from mflux.models.qwen3_vl_caption.cli.caption import PROMPT_CHOICES

    monkeypatch.setattr(sys, "argv", ["mlxgen-caption", "--image-path", "img.png", "--model", "some/model"])
    args = build_parser().parse_args()

    assert args.temperature == 0.0
    assert args.max_pixels == DEFAULT_MAX_PIXELS
    assert args.backend == BACKEND_MLXGEN
    assert args.prompt is None and args.prompt_preset is None
    # The Chinese long-form instruction is what the V4.5 caption fine-tune was trained with.
    assert DEFAULT_CAPTION_PROMPT.startswith("请用自然流畅的中文")
    assert PROMPT_CHOICES["zh"] == DEFAULT_CAPTION_PROMPT
    assert PROMPT_CHOICES["en"] != DEFAULT_CAPTION_PROMPT
    assert DEFAULT_MAX_PIXELS == 640_000


@pytest.mark.fast
def test_mlx_vlm_backend_reports_the_missing_optional_dependency(monkeypatch):
    from mflux.models.qwen3_vl_caption.backend import mlx_vlm_backend

    # `None` in sys.modules makes `import mlx_vlm` raise ImportError, matching a host without it.
    monkeypatch.setitem(sys.modules, "mlx_vlm", None)

    with pytest.raises(ImportError, match="pip install 'mlx-vlm"):
        mlx_vlm_backend._import_mlx_vlm()


@pytest.mark.fast
def test_caption_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        Qwen3VLCaption(model_path="some/model", backend=BACKEND_MLXGEN + "-typo")

    assert BACKEND_MLX_VLM == "mlx-vlm"
