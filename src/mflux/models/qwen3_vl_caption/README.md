# Qwen3-VL Caption (image → text)

`mlxgen-caption` describes an image with a **local MLX Qwen3-VL checkpoint**. It runs on mlx-gen's
own Qwen3-VL decoder and vision tower by default, and can hand the same checkpoint to the optional
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) runtime for comparison.

This is the inverse of the generation routes: instead of turning a prompt into an image, it turns an
image into a prompt/caption (提示词反推).

## Quick start

```bash
# Describe an image with a local LM Studio / mlx-lm / mlx-vlm Qwen3-VL checkpoint.
mlxgen-caption \
  --model "/Users/you/.lmstudio/models/gavinmroy/Qwen3-VL-8B-NSFW-Caption-V4.5-affine-4bit" \
  --image-path photo.jpg \
  --output caption.txt
```

Python:

```python
from mflux import Qwen3VLCaption

captioner = Qwen3VLCaption(model_path="/path/to/qwen3-vl-mlx-checkpoint", max_pixels=640_000)
print(captioner.caption("photo.jpg", max_tokens=512))
```

## Supported checkpoints

Any **MLX-format dense Qwen3-VL** repo that uses the mlx-lm / mlx-vlm key layout
(`language_model.model.*`, `language_model.lm_head.*`, `vision_tower.*`): 2B/4B/8B/32B, and either
bf16 or already quantized (affine, mxfp4, nvfp4 — the level, group size and mode all come from the
checkpoint's `config.json`). The vision tower stays in its published dtype; only the text tower is
quantized.

The route refuses to run on a checkpoint whose weights do not map onto the module tree instead of
silently producing a broken model, so torch/HF-layout repos (or MoE `qwen3_vl_moe` checkpoints) fail
with a message naming the first offending keys.

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--image-path` | required | Image to describe |
| `--model` | required | Local path or Hugging Face repo id |
| `--backend` | `mlxgen` | `mlxgen` (built-in) or `mlx-vlm` (optional dependency) |
| `--prompt` | V4.5 model-card prompt | Caption instruction; `--prompt-preset zh\|en` selects a built-in |
| `--max-tokens` | `1024` | Generated text budget |
| `--temperature` | `0.0` | `0` decodes greedily (recommended for captions) |
| `--top-p` | `0.9` | Ignored at temperature 0 |
| `--max-pixels` | `640000` | Image resize ceiling (~800x800, the caption fine-tune's training size) |
| `--min-pixels` | checkpoint | Image resize floor |
| `-q/--quantize` | none | Quantize a bf16 checkpoint; ignored for already-quantized ones |
| `--output` | stdout | Write the caption to a file |

## Cost model

Image tokens are `resized_h * resized_w / 1024`, and prefill cost grows with their square. On an M5
Pro, one 1920x1080 photo with the 8B/4-bit caption checkpoint measured:

| Image budget | Image tokens | Prompt tokens | Time (prompt + 16 tokens) | Peak memory |
| --- | --- | --- | --- | --- |
| `--max-pixels 640000` (default) | 594 | 670 | 4.2 s (cold process) | 6.3 GB |
| `--max-pixels 802816` | 777 | 853 | 2.6 s | 6.5 GB |
| checkpoint ceiling | 2040 | 2116 | 6.7 s | 7.6 GB |

Greedy throughput at the default budget, prefill included: 64 tokens in 6.9 s, 256 tokens in 16.1 s
(about 9-16 tok/s).

## Fidelity notes

The built-in route turns on the two Qwen3-VL mechanisms the shared decoder keeps off by default:

- **multimodal RoPE** — image tokens get `(t, h, w)` grid positions instead of one flat `arange`,
  which is what the checkpoint was trained with;
- **deepstack** — the three extra vision mergers are injected back onto the image token rows after
  the first decoder layers.

Both are opt-in per model instance (`multimodal_position_ids=`, `deepstack=` on `Qwen3VLDecoder`),
so the FIBO-VLM and Qwen-Image-2.1 text-encoder routes keep their previous numerics.

## Backend comparison

`--backend mlx-vlm` delegates to the optional package (`pip install 'mlx-vlm>=0.5.0'`) and is the
cross-check when a caption looks wrong: it loads the same on-disk checkpoint with no conversion step,
and both runtimes produce long Chinese captions in the same order of magnitude on this hardware. Use
it to separate "the checkpoint disagrees" from "this route mis-reads the image"; the built-in route
keeps the quantized `lm_head` in packed form, which is why it decodes at ~9-16 tok/s rather than at
the slower dequantized rate.

## Model notes (V4.5 NSFW caption fine-tune)

- Lineage: `Qwen/Qwen3-VL-8B-Instruct` → LoRA fine-tune (`thesby/...V4.5`, ~2M image-text pairs) →
  reupload (`Disty0/...`) → MLX 4-bit (`gavinmroy/...-affine-4bit`).
- The model card's instruction (used as the default prompt) asks for a long, detailed Chinese
  description; the checkpoint was trained on images up to 800x800.
- No license is published with the community conversions: check the upstream card before
  redistributing weights or captions.
