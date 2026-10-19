#!/bin/bash
# Post-download smoke test for the converted Qwen-Image-2.1 Uncensored MLX-8bit package.
#
# Verifies, in order:
#   1. the package directory has all four components
#   2. the transformer package carries the 8-bit quantization metadata (i.e. it will be
#      loaded as pre-quantized, NOT re-quantized from a mapping that would drop scales/biases)
#   3. the model loads and actually generates an image (real end-to-end)
#
# Usage: ./smoke_test.sh [extra CLI args...]
set -uo pipefail

MLXGEN="/Users/jiangzhenguo/ai/mlx-gen"
PKG="/Users/jiangzhenguo/ai/models/Qwen-Image-2.1-Uncensored-MLX"
OUTDIR="$MLXGEN/out/qwen21_test"
PY="$MLXGEN/.venv/bin/python"

echo "=== 1. package structure ==="
ok=1
for comp in transformer text_encoder vae processor; do
  if [[ -d "$PKG/$comp" ]]; then
    n=$(ls "$PKG/$comp" 2>/dev/null | wc -l | tr -d ' ')
    echo "  OK   $comp/  ($n entries)"
  else
    echo "  MISSING $comp/"
    ok=0
  fi
done
[[ $ok -eq 1 ]] || { echo "FAIL: incomplete package"; exit 1; }

echo
echo "=== 2. transformer metadata (must be pre-quantized 8-bit) ==="
"$PY" - "$PKG/transformer" <<'PYEOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
idx = json.loads((p / "model.safetensors.index.json").read_text())
md = idx.get("metadata", {})
print(f"  quantization_level = {md.get('quantization_level')!r}")
print(f"  mflux_version      = {md.get('mflux_version')!r}")
print(f"  tensors in index   = {len(idx.get('weight_map', {}))}")
ql = md.get("quantization_level")
if ql != "8":
    print("FAIL: expected quantization_level '8'")
    sys.exit(1)
print("  OK: loads as pre-quantized (scales/biases preserved)")
PYEOF
[[ $? -eq 0 ]] || exit 1

echo
echo "=== 3. end-to-end generation ==="
mkdir -p "$OUTDIR"
OUT="$OUTDIR/uncensored_8bit_smoke.png"
rm -f "$OUT"
start=$(date +%s)
"$MLXGEN/.venv/bin/mlxgen-generate-qwen-2.1" \
  --model "$PKG" \
  --prompt "A red apple on a wooden table, soft daylight, photorealistic" \
  --steps 8 --seed 42 \
  --output "$OUT" "$@"
rc=$?
elapsed=$(( $(date +%s) - start ))

echo
if [[ $rc -ne 0 || ! -f "$OUT" ]]; then
  echo "FAIL: generation failed (rc=$rc)"
  exit 1
fi

echo "=== 4. image sanity ==="
"$PY" - "$OUT" <<'PYEOF'
import sys
import numpy as np
from PIL import Image

# A flat std check is NOT sufficient: a fully corrupted (noise) render also has a high std.
# Real renders are spatially smooth, so a low mean gradient is required too. Observed:
#   good render  : std ~65, grad_dx ~1.7
#   noise render : std ~73, grad_dx ~11.3  (text encoder weights silently not applied)
#   blank render : std ~10
im = Image.open(sys.argv[1])
a = np.asarray(im.convert("RGB")).astype(float)
std = a.std()
grad = np.abs(np.diff(a, axis=1)).mean()
print(f"  size={im.size} mode={im.mode}")
print(f"  mean RGB={a.reshape(-1,3).mean(0).round(1)}")
print(f"  std={std:.2f}  grad_dx={grad:.2f}")

ok = True
if std < 15:
    print("  FAIL: image is nearly flat (weights likely not applied)")
    ok = False
if grad > 8:
    print("  FAIL: image is noise-like (a component's weights were not applied)")
    ok = False
if ok:
    print("  OK: real, spatially coherent image content")
sys.exit(0 if ok else 1)
PYEOF
rc2=$?

echo
echo "=== smoke test $( [[ $rc2 -eq 0 ]] && echo PASSED || echo FAILED ) in ${elapsed}s ==="
echo "artifact: $OUT"
exit $rc2
