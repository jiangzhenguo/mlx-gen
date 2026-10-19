#!/bin/bash
# Build an mflux-loadable local package for Qwen-Image-2.1 Uncensored (MLX) from the
# community release at abenzerps/Qwen-Image-2.1-Uncensored-GGUF.
#
# Layout produced (what Qwen21WeightDefinition.get_download_patterns expects):
#
#   <ROOT>/
#   ├── transformer/            <- mflux prepared package (metadata + sharded, verbatim keys)
#   │   ├── 0.safetensors ...
#   │   └── model.safetensors.index.json
#   ├── text_encoder/           <- Qwen3-VL 8B, bf16, installed as a "multi_json" component
#   │   ├── model-0000N-of-0000N.safetensors
#   │   ├── model.safetensors.index.json
#   │   └── config.json
#   ├── vae/                    <- community causal VAE
#   │   ├── diffusion_pytorch_model.safetensors
#   │   └── config.json
#   └── processor/              <- tokenizer, reused from the official bf16 snapshot
#
# Usage: ./build_package.sh [--bits 8] [--dry-run]
set -euo pipefail

MLXGEN="/Users/jiangzhenguo/ai/mlx-gen"
SRC_ROOT="/Users/jiangzhenguo/ai/models/Qwen-Image-2.1-Uncensored-MLX-8bit"
OFFICIAL="/Users/jiangzhenguo/ai/models/Qwen-Image-2.1-bf16"
OUT="/Users/jiangzhenguo/ai/models/Qwen-Image-2.1-Uncensored-MLX"

BITS=8
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bits) BITS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

PY="$MLXGEN/.venv/bin/python"
CKPT="$SRC_ROOT/qwen-image-2.1-UC-MLX-${BITS}bit.safetensors"

# A still-downloading file is sparse: apparent size reaches the final value early while only
# the fetched blocks are actually allocated. An aria2c .aria2 control file means "unfinished",
# and comparing allocated vs apparent bytes catches the same thing for other downloaders.
check_download_complete() {
  local f="$1" label="$2"
  [[ -f "$f" ]] || { echo "ERROR: $label not found: $f" >&2; return 1; }
  if [[ -f "$f.aria2" ]]; then
    echo "ERROR: $label is still downloading (found $f.aria2)" >&2
    return 1
  fi
  local apparent allocated tol
  apparent=$(stat -f%z "$f")
  allocated=$(( $(stat -f%b "$f") * 512 ))
  # Tolerance for filesystem block rounding, scaled to the file so it is valid for both
  # tiny and multi-GiB inputs (a flat 1 MiB slack would accept a fully-sparse 1 MiB file).
  tol=$(( apparent / 100 ))
  (( tol > 65536 )) && tol=65536
  if (( allocated < apparent - tol )); then
    echo "ERROR: $label looks incomplete: $((allocated/1048576)) MiB allocated of $((apparent/1048576)) MiB" >&2
    echo "       (still downloading? check $SRC_ROOT/CHECK_STATUS.sh)" >&2
    return 1
  fi
  return 0
}

echo "=== 1. transformer: community ckpt -> mflux prepared package ==="
if [[ "$DRY_RUN" == "0" ]]; then
  check_download_complete "$CKPT" "transformer checkpoint" || exit 1
elif [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  echo "(is the download finished? check $SRC_ROOT/CHECK_STATUS.sh)" >&2
  exit 1
fi
if [[ "$DRY_RUN" == "1" ]]; then
  "$PY" "$MLXGEN/tools/qwen21_community_to_mflux.py" --src "$CKPT" --dst "$OUT/transformer" --bits "$BITS" --dry-run
else
  "$PY" "$MLXGEN/tools/qwen21_community_to_mflux.py" --src "$CKPT" --dst "$OUT/transformer" --bits "$BITS"
fi

echo
echo "=== 2. text_encoder: install + write index/config ==="
TE_SRC="$SRC_ROOT/text_encoders/qwen3vl_8b_bf16.safetensors"
if [[ ! -f "$TE_SRC" ]]; then
  echo "WARNING: text encoder not downloaded yet: $TE_SRC" >&2
else
  # The community Qwen3-VL export keys the language tower as plain "model.*" and ships the
  # vision tower, which mflux's Qwen21TextEncoder does not build; convert_text_encoder
  # rewrites the prefix and drops model.visual.*.
  #
  # CRITICAL: the text encoder is a MAPPED component. Its shards must NOT carry
  # mflux_version/quantization_level metadata, or WeightLoader._try_load_mflux_format()
  # claims the directory and returns unmapped keys, so model.update(strict=False) silently
  # applies nothing and generation degrades to noise. The converter already handles this;
  # the assertion below makes a regression fail loudly instead of producing a bad package.
  if [[ "$DRY_RUN" == "1" ]]; then
    "$PY" "$MLXGEN/tools/qwen21_community_to_mflux.py" --component text_encoder \
      --src "$TE_SRC" --dst "$OUT/text_encoder" --dry-run
  else
    check_download_complete "$TE_SRC" "text encoder" || exit 1
    "$PY" "$MLXGEN/tools/qwen21_community_to_mflux.py" --component text_encoder \
      --src "$TE_SRC" --dst "$OUT/text_encoder"
    if [[ -f "$OFFICIAL/text_encoder/config.json" ]]; then
      cp "$OFFICIAL/text_encoder/config.json" "$OUT/text_encoder/config.json"
    fi
    # Fail fast if the prepared-format shortcut would claim the mapped text encoder.
    "$PY" - "$OUT/text_encoder" <<'PYEOF'
import sys
from pathlib import Path
from mflux.models.common.weights.loading.weight_loader import WeightLoader
w, _, _ = WeightLoader._try_load_mflux_format(Path(sys.argv[1]))
if w is not None:
    print("ERROR: text_encoder tripped the mflux prepared-format shortcut; its weights "
          "would be loaded WITHOUT key mapping (generation would degrade to noise).", file=sys.stderr)
    sys.exit(1)
print("  text_encoder: mapped-load path confirmed (no prepared-format shortcut)")
PYEOF
    [[ $? -eq 0 ]] || exit 1
  fi
fi

echo
echo "=== 3. vae: reuse the OFFICIAL diffusers VAE from the local bf16 snapshot ==="
# The community repo's vae/qwen_image_2.1_vae_bf16.safetensors is NOT usable here: it is a
# ComfyUI-format file (keys conv1/conv2/encoder.downsamples/decoder.upsamples, 3D-conv
# layout) and shares ZERO key names with the diffusers VAE that Qwen21WeightMapping
# expects (encoder.conv_in, encoder.down_blocks.N.resnets.M.norm{1,2}.gamma, ...).
# The VAE is not part of the "uncensored" fine-tune anyway (only the transformer is), so we
# install the official one that is already on disk.
VAE_OFFICIAL="$OFFICIAL/vae/diffusion_pytorch_model.safetensors"
if [[ ! -f "$VAE_OFFICIAL" ]]; then
  echo "ERROR: official VAE not found: $VAE_OFFICIAL" >&2
  echo "Run: mlxgen download --model Qwen/Qwen-Image-2.1" >&2
  exit 1
fi
if [[ "$DRY_RUN" == "0" ]]; then
  mkdir -p "$OUT/vae"
  # vae uses loading_mode="single", which globs *.safetensors and takes the FIRST file.
  # Clear stale shards from an earlier run so exactly one file is ever present.
  for stale in "$OUT/vae"/*.safetensors; do
    [[ -e "$stale" ]] && rm -f "$stale"
  done
  cp "$VAE_OFFICIAL" "$OUT/vae/diffusion_pytorch_model.safetensors"
  [[ -f "$OFFICIAL/vae/config.json" ]] && cp "$OFFICIAL/vae/config.json" "$OUT/vae/config.json"
  count=$(ls "$OUT/vae"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$count" != "1" ]]; then
    echo "ERROR: expected exactly 1 vae safetensors file, found $count" >&2
    exit 1
  fi
  echo "  vae: official diffusers VAE installed ($(du -h "$OUT/vae/diffusion_pytorch_model.safetensors" | cut -f1))"
fi

echo
echo "=== 4. processor: reuse tokenizer from the official bf16 snapshot ==="
if [[ -d "$OFFICIAL/processor" ]]; then
  if [[ "$DRY_RUN" == "0" ]]; then
    mkdir -p "$OUT/processor"
    cp "$OFFICIAL"/processor/* "$OUT/processor"/
    echo "  processor: $(ls "$OUT/processor" | wc -l | tr -d ' ') files"
  fi
else
  echo "ERROR: official processor dir missing: $OFFICIAL/processor" >&2
  echo "The tokenizer cannot be synthesised; run: mlxgen download --model Qwen/Qwen-Image-2.1" >&2
  exit 1
fi

echo
echo "=== result: $OUT ==="
[[ "$DRY_RUN" == "0" ]] && find "$OUT" -maxdepth 2 -type f | sed "s|$OUT/||" | sort | head -40
echo "total: $(du -sh "$OUT" 2>/dev/null | cut -f1)"
