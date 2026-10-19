"""Convert a community MLX Qwen-Image-2.1 checkpoint into an mflux "prepared package".

Why this exists
---------------
Community Qwen-Image-2.1 MLX conversions (e.g. abenzerps/Qwen-Image-2.1-Uncensored-GGUF)
ship a single flat safetensors file whose keys are already in *module-tree* form
(`transformer_blocks.0.attn.to_q.weight`) rather than the diffusers/HF form the
declarative mapper expects, and they carry no mflux metadata.

mflux has two load paths:

1. `WeightLoader._try_load_mflux_format` - reads an mflux "prepared package"
   (`model.safetensors.index.json` + shards carrying `quantization_level` /
   `mflux_version` metadata) and `tree_unflatten`s the keys verbatim. No mapping,
   no re-quantization. This is what pre-quantized weights need.
2. The HF path - applies the declarative mapping and *drops* `*.scales` / `*.biases`
   auxiliary keys, which silently destroys a pre-quantized checkpoint.

So the fix is to reshape the community file into layout (1): keep every tensor
verbatim (weight + scales + biases), shard it, and write the metadata + index.

It also renames the two key families whose names differ between the two conventions,
so the resulting package matches mflux's module tree exactly:

    modulation.0.*                     -> modulation.layers.1.*
    time_text_embed.linear_{1,2}.*     -> time_text_embed.timestep_embedder.linear_{1,2}.*
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx

# checkpoint key prefix -> mflux module-tree prefix
KEY_REWRITES = (
    ("modulation.0.", "modulation.layers.1."),
    ("time_text_embed.linear_1.", "time_text_embed.timestep_embedder.linear_1."),
    ("time_text_embed.linear_2.", "time_text_embed.timestep_embedder.linear_2."),
)

MAX_SHARD_BYTES = 2 << 30  # 2 GiB, matching ModelSaver._split_weights


def rewrite_key(key: str) -> str:
    for src, dst in KEY_REWRITES:
        if key.startswith(src):
            return dst + key[len(src) :]
    return key


# --- text encoder -------------------------------------------------------------------
# The community Qwen3-VL export keys the language tower as plain "model.<x>"
# (model.layers.N.*, model.embed_tokens.*, model.norm.*) whereas mlx-gen's mapping wants
# "model.language_model.<x>". Everything under "model.visual.*" belongs to the vision tower,
# which mlx-gen's Qwen21TextEncoder does not build (the mapping references zero visual
# keys), so those are dropped to keep the package small.
TE_LANGUAGE_PREFIX = "model.language_model."
TE_COMMUNITY_PREFIX = "model."
TE_VISUAL_PREFIX = "model.visual."


def rewrite_te_key(key: str) -> str | None:
    """Return the mflux-side key, or None to drop the tensor."""
    if key.startswith(TE_VISUAL_PREFIX):
        return None  # vision tower: unused by mflux's text encoder
    if key == "lm_head.weight":
        return None  # tied head: unused, and mlx-gen's mapping never binds it
    if key.startswith(TE_COMMUNITY_PREFIX):
        return TE_LANGUAGE_PREFIX + key[len(TE_COMMUNITY_PREFIX) :]
    return key


def convert_text_encoder(src: Path, dst_dir: Path, dry_run: bool = False) -> dict:
    if not src.is_file():
        raise SystemExit(f"text encoder not found: {src}")

    raw = dict(mx.load(str(src)))
    print(f"loaded {len(raw)} tensors from {src.name}")

    weights, kept, dropped = {}, 0, 0
    for key, tensor in raw.items():
        new_key = rewrite_te_key(key)
        if new_key is None:
            dropped += 1
            continue
        if new_key in weights:
            raise SystemExit(f"key collision after rewrite: {new_key}")
        weights[new_key] = tensor
        kept += 1
    print(f"kept {kept} tensors, dropped {dropped} (vision tower / tied head)")
    # Free the full-file dict before sharding: the dropped vision tower (~1 GiB here)
    # is otherwise kept alive by `raw` for the entire write phase.
    del raw
    if dry_run:
        return {"kept": kept, "dropped": dropped}

    dst_dir.mkdir(parents=True, exist_ok=True)
    for stale in dst_dir.glob("*.safetensors"):
        stale.unlink()

    # Shard like the transformer path so large encoders stay split.
    shards: list[dict] = []
    shard: dict = {}
    shard_bytes = 0
    for key, tensor in weights.items():
        if shard and shard_bytes + tensor.nbytes > MAX_SHARD_BYTES:
            shards.append(shard)
            shard, shard_bytes = {}, 0
        shard[key] = tensor
        shard_bytes += tensor.nbytes
    if shard:
        shards.append(shard)

    # IMPORTANT: write NO safetensors-level metadata here.
    #
    # WeightLoader._try_load_mflux_format() runs BEFORE the normal mapped path and claims a
    # component directory as soon as it finds `model.safetensors.index.json` plus
    # `mflux_version` / `quantization_level` metadata in the shards. That path returns
    # `tree_unflatten(raw keys)` with NO key mapping applied.
    #
    # The text encoder is a MAPPED component (its checkpoint keys are
    # model.language_model.*, its module tree is layers.*), so tripping that shortcut means
    # every key misses and `model.update(strict=False)` silently applies nothing - the model
    # loads, generates, and produces garbage. Verified: this produced noise output.
    weight_map: dict[str, str] = {}
    total = len(shards)
    for i, part in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
        mx.save_safetensors(str(dst_dir / name), part)
        for key in part:
            weight_map[key] = name
        print(f"  wrote {name}: {len(part)} tensors")

    index = {
        "metadata": {"total_size": sum(t.nbytes for t in weights.values())},
        "weight_map": weight_map,
    }
    (dst_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote model.safetensors.index.json ({len(weight_map)} entries)")
    return {"kept": kept, "dropped": dropped, "shards": len(shards)}


def convert(src: Path, dst_dir: Path, bits: int, mflux_version: str, dry_run: bool = False) -> dict:
    if not src.is_file():
        raise SystemExit(f"source checkpoint not found: {src}")

    raw = dict(mx.load(str(src)))
    print(f"loaded {len(raw)} tensors from {src.name}")

    renamed, weights = 0, {}
    for key, tensor in raw.items():
        new_key = rewrite_key(key)
        if new_key != key:
            renamed += 1
        if new_key in weights:
            raise SystemExit(f"key collision after rewrite: {new_key}")
        weights[new_key] = tensor
    print(f"rewrote {renamed} keys into mflux module-tree names")
    # Re-keying only rebinds references, so `weights` now shares storage with `raw`.
    # Drop `raw` before sharding so the dropped tensors (e.g. the vision tower) can be
    # freed instead of staying alive for the whole write phase.
    del raw

    base = sum(1 for k in weights if not k.endswith((".scales", ".biases")))
    aux = len(weights) - base
    print(f"tensors: {base} base + {aux} quantization aux (scales/biases)")
    if aux == 0:
        print("WARNING: no scales/biases found - is this really a quantized checkpoint?")

    if dry_run:
        print("dry run: nothing written")
        return {"renamed": renamed, "base": base, "aux": aux}

    # Shard in the same near-flat order ModelSaver uses; the loader's first
    # materialization then walks the file close to sequentially.
    dst_dir.mkdir(parents=True, exist_ok=True)
    for stale in dst_dir.glob("*.safetensors"):
        stale.unlink()
    index_path = dst_dir / "model.safetensors.index.json"
    if index_path.exists():
        index_path.unlink()

    metadata = {"quantization_level": str(bits), "mflux_version": mflux_version}

    shards: list[dict] = []
    shard: dict = {}
    shard_bytes = 0
    for key, tensor in weights.items():
        nbytes = tensor.nbytes
        if shard and shard_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(shard)
            shard, shard_bytes = {}, 0
        shard[key] = tensor
        shard_bytes += nbytes
    if shard:
        shards.append(shard)

    weight_map: dict[str, str] = {}
    for i, part in enumerate(shards):
        name = f"{i}.safetensors"
        mx.save_safetensors(str(dst_dir / name), part, metadata)
        for key in part:
            weight_map[key] = name
        print(f"  wrote {name}: {len(part)} tensors, {sum(t.nbytes for t in part.values()) / 2**30:.2f} GiB")

    index_path.write_text(json.dumps({"metadata": metadata, "weight_map": weight_map}, indent=2))
    print(f"wrote {index_path.name} ({len(weight_map)} entries)")
    return {"renamed": renamed, "base": base, "aux": aux, "shards": len(shards)}


def _detect_mflux_version() -> str:
    """Prefer mflux's own resolver so the stamp matches what ModelSaver would write."""
    try:
        from mflux.utils.version_util import VersionUtil

        return VersionUtil.get_mflux_version()
    except Exception:
        # Tool must stay usable (e.g. dry runs) even if mflux is not importable.
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, type=Path, help="community .safetensors checkpoint")
    parser.add_argument("--dst", required=True, type=Path, help="output directory (becomes the component dir)")
    parser.add_argument("--bits", type=int, help="stored quantization bits (required for --component transformer)")
    parser.add_argument(
        "--component",
        choices=("transformer", "text_encoder"),
        default="transformer",
        help="which component of the community release is being converted",
    )
    parser.add_argument("--mflux-version", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    version = args.mflux_version or _detect_mflux_version()
    if args.component == "transformer":
        if args.bits is None:
            parser.error("--bits is required for --component transformer")
        convert(
            src=args.src,
            dst_dir=args.dst,
            bits=args.bits,
            mflux_version=version,
            dry_run=args.dry_run,
        )
    else:
        # NOTE: no mflux_version is written for the text encoder on purpose - see the
        # comment inside convert_text_encoder about the prepared-format shortcut.
        convert_text_encoder(
            src=args.src,
            dst_dir=args.dst,
            dry_run=args.dry_run,
        )
    print("done")


if __name__ == "__main__":
    main()
