"""Convert a community mflux/prepared Qwen-Image-2.1 text encoder into the
HuggingFace-style layout that both Qwen-Image-2.1 routes in this fork expect.

Why this exists
---------------
`vanch007/Qwen-Image-2.1-Turbo-Heretic-mlx-bf16` ships a text encoder whose keys are

    language_model.embed_tokens.weight
    language_model.layers.0.mlp.down_proj.weight
    visual.patch_embed.proj.weight          (Qwen3-VL vision tower, used by the edit route)
    lm_head.weight                          (in generation_head.safetensors)

and whose shards carry `mflux_version` / `quantization_level` metadata. That metadata puts the
loader on the "prepared package" fast path, which loads the module tree verbatim and therefore
drops every key here.

This script rewrites the shards without that metadata into the layout the two routes read:

- `model.language_model.*` — the text-to-image route maps exactly these keys
  (`Qwen21WeightMapping.get_text_encoder_mapping`); the edit route's `text_key` strips the
  `model.` prefix, so the same keys serve both.
- `model.visual.*` — the edit route's text encoder is a full Qwen3-VL model, so the vision tower
  is required there and ignored by text-to-image. `visible.patch_embed.proj.weight` is written
  back in the HF layout `[out, in, T, H, W]` because the edit route transposes it itself.
- `config.json` is copied alongside, because the edit route reads component geometry from it.

Usage
-----
    python tools/convert_heretic_text_encoder.py \
        --src ~/ai/models/vanch-heretic/text_encoder \
        --dst ~/ai/models/Qwen-Image-2.1-Turbo-Heretic/text_encoder
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

LANGUAGE_PREFIX = "language_model."
LANGUAGE_NEW_PREFIX = "model.language_model."
VISION_PREFIX = "visual."
VISION_NEW_PREFIX = "model.visual."
PATCH_EMBED_KEY = "model.visual.patch_embed.proj.weight"

EXPECTED_LANGUAGE_KEYS = 399  # 36 layers x 11 tensors + embed_tokens + norm + rotary inv_freq
EXPECTED_VISION_KEYS = 352  # Qwen3-VL vision tower incl. deepstack mergers and rotary inv_freq
SHARD_BYTES = 2 << 30


def to_hf_patch_embed(key: str, value: mx.array) -> mx.array:
    """Return the vision patch-embedding conv in the HF `[out, in, T, H, W]` layout."""
    if value.ndim != 5:
        raise SystemExit(f"{key}: expected a 5-D conv weight, got shape {value.shape}")
    if value.shape[-1] == 3:  # MLX `[out, T, H, W, in]`
        return mx.transpose(value, (0, 4, 1, 2, 3))
    if value.shape[1] == 3:  # already HF `[out, in, T, H, W]`
        return value
    raise SystemExit(f"{key}: cannot tell the layout of shape {value.shape}")


def rename(key: str) -> tuple[str, bool] | None:
    """Map a source key to its output key; None means drop it."""
    if key.startswith(VISION_PREFIX):
        return VISION_NEW_PREFIX + key[len(VISION_PREFIX) :], True
    if key.startswith(LANGUAGE_PREFIX):
        return LANGUAGE_NEW_PREFIX + key[len(LANGUAGE_PREFIX) :], False
    return None  # lm_head and friends: not part of either route


def normalize_config(config: dict) -> dict:
    """Rewrite transformers-5.x `rope_parameters` into the `rope_scaling` + `rope_theta` pair.

    The reference-editing initializer builds its RoPE from `text_config["rope_theta"]` and
    `text_config["rope_scaling"]["mrope_section"]`, which is the older transformers layout that
    the official Qwen-Image-2.1 checkpoint uses. Community exports come from transformers 5.x,
    where both moved into a single `rope_parameters` dict.
    """
    text_config = config.get("text_config", {})
    rope = text_config.get("rope_parameters")
    if rope and "rope_scaling" not in text_config:
        text_config["rope_scaling"] = {key: value for key, value in rope.items() if key != "rope_theta"}
        text_config["rope_theta"] = rope.get("rope_theta", text_config.get("rope_theta"))
        text_config.pop("rope_parameters", None)
        print("normalized text_config rope: rope_parameters -> rope_scaling + rope_theta")
    return config


def convert(src: Path, dst: Path, shard_bytes: int = SHARD_BYTES) -> dict:
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"no model.safetensors.index.json in {src}")
    weight_map = json.loads(index_path.read_text())["weight_map"]

    shards = sorted(set(weight_map.values()))
    dst.mkdir(parents=True, exist_ok=True)

    # Stream shard by shard so peak RAM stays near one input shard instead of the whole encoder.
    new_map: dict[str, str] = {}
    dropped: set[str] = set()
    counts = {"language": 0, "vision": 0}
    shard_i = 0
    for shard_file in shards:
        batch: dict[str, mx.array] = {}
        for key, value in mx.load(str(src / shard_file)).items():
            mapped = rename(key)
            if mapped is None:
                dropped.add(key.split(".")[0])
                continue
            new_key, is_vision = mapped
            if new_key == PATCH_EMBED_KEY:
                value = to_hf_patch_embed(key, value)
            batch[new_key] = value
            counts["vision" if is_vision else "language"] += 1
        if not batch:
            continue
        name = f"{shard_i}.safetensors"
        mx.save_safetensors(str(dst / name), batch)  # no metadata on purpose
        new_map.update(dict.fromkeys(batch, name))
        shard_i += 1

    expected = {"language": EXPECTED_LANGUAGE_KEYS, "vision": EXPECTED_VISION_KEYS}
    if counts != expected:
        raise SystemExit(f"expected {expected} tensors, converted {counts}")
    print(f"kept {sum(counts.values())} tensors {counts}; dropped top-level groups: {sorted(dropped)}")

    (dst / "model.safetensors.index.json").write_text(json.dumps({"weight_map": new_map}, indent=2))

    # The edit route reads component geometry from <component>/config.json.
    config = src / "config.json"
    if config.exists():
        payload = normalize_config(json.loads(config.read_text()))
        (dst / "config.json").write_text(json.dumps(payload, indent=2))
        print("copied and normalized config.json (needed by the reference-editing initializer)")
    else:
        print("WARNING: no config.json in the source; the editing route will refuse to load")
    return new_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="source text_encoder directory")
    parser.add_argument("--dst", type=Path, required=True, help="output text_encoder directory")
    parser.add_argument("--shard-bytes", type=int, default=SHARD_BYTES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dst.exists() and any(args.dst.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{args.dst} is not empty; pass --overwrite to replace it")
        shutil.rmtree(args.dst)

    new_map = convert(args.src, args.dst, args.shard_bytes)
    print(f"wrote {len(set(new_map.values()))} shard(s) and an index with {len(new_map)} keys to {args.dst}")


if __name__ == "__main__":
    main()
