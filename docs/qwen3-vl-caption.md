# Qwen3-VL caption (image → text)

`mlxgen-caption` turns an image into text with a **local MLX Qwen3-VL checkpoint**: it is the inverse
of the generation routes, and the command people reach for when they want a prompt out of a picture
(提示词反推) or a description of what an image contains.

It runs on mlx-gen's own Qwen3-VL decoder and vision tower, and can hand the same on-disk checkpoint
to the optional [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) runtime for a parity check.

## What it does

- loads any **MLX-format dense Qwen3-VL** repo (mlx-lm / mlx-vlm / LM Studio layout:
  `language_model.model.*`, `language_model.lm_head.*`, `vision_tower.*`), 2B/4B/8B/32B;
- accepts bf16 checkpoints and already-quantized ones (affine, mxfp4, nvfp4): the level, group size
  and mode come from the checkpoint's `config.json`, and the vision tower stays in its published
  dtype while only the text tower is quantized;
- enables the two Qwen3-VL mechanisms the shared decoder keeps off by default — multimodal RoPE
  (image tokens carry `(t, h, w)` grid positions) and deepstack (the extra vision mergers are
  injected back onto the image token rows) — so the route matches how the checkpoint was trained;
- refuses to run rather than silently degrading when a checkpoint's keys do not map onto the module
  tree (torch/HF-layout repos, MoE `qwen3_vl_moe` checkpoints).

## Quick start

```bash
mlxgen-caption \
  --model "/Users/you/.lmstudio/models/gavinmroy/Qwen3-VL-8B-NSFW-Caption-V4.5-affine-4bit" \
  --image-path photo.jpg \
  --output caption.txt
```

`--model` takes a local directory or a Hugging Face repo id. MLX-Gen never downloads during a run:
point it at files that are already on disk, or fetch them first (`mlxgen download`).

Python:

```python
from mflux import Qwen3VLCaption

captioner = Qwen3VLCaption(model_path="/path/to/qwen3-vl-mlx-checkpoint", max_pixels=640_000)
print(captioner.caption("photo.jpg", max_tokens=512))
```

## Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--image-path` | required | Image to describe |
| `--model` / `-m` | required | Local path or Hugging Face repo id |
| `--backend` | `mlxgen` | `mlxgen` (built-in route) or `mlx-vlm` (optional dependency) |
| `--prompt` | V4.5 model-card prompt | Caption instruction |
| `--prompt-preset` | none | `zh` or `en` built-in instructions |
| `--max-tokens` | `1024` | Generated text budget |
| `--temperature` | `0.0` | `0` decodes greedily, which is what captions want |
| `--top-p` | `0.9` | Ignored at temperature 0 |
| `--seed` | none | Sampling seed |
| `--max-pixels` | `640000` | Image resize ceiling (about 800x800) |
| `--min-pixels` | checkpoint value | Image resize floor |
| `-q` / `--quantize` | none | Quantize a bf16 checkpoint; ignored for quantized ones |
| `--output` | stdout | Write the caption to a file |

## What it costs

The image becomes `resized_h * resized_w / 1024` image tokens, and prefill cost grows with the square
of that count, so `--max-pixels` is the main speed knob. Measured on an M5 Pro with the 8B/4-bit
caption checkpoint and one 1920x1080 photo:

| Image budget | Image tokens | Prompt tokens | Time (prompt + 16 tokens) | Peak memory |
| --- | --- | --- | --- | --- |
| `--max-pixels 640000` (default, ~800x800) | 594 | 670 | 4.2 s (cold process) | 6.3 GB |
| `--max-pixels 802816` | 777 | 853 | 2.6 s | 6.5 GB |
| checkpoint ceiling (16.7M pixels) | 2040 | 2116 | 6.7 s | 7.6 GB |

Greedy throughput at the default budget, per-token length included: 64 tokens in 6.9 s and 256 tokens
in 16.1 s (about 9-16 tok/s, prefill included). Peak memory holds at 6-8 GB because the text tower
stays 4-bit and only the vision tower runs bf16.

The default matches the size the V4.5 caption fine-tune reports training on, and keeps one caption
interactive. The optional `mlx-vlm` backend loads the same directory and is the cross-check when a
caption looks wrong rather than the fast path: both runtimes finish a long Chinese caption in the
same order of magnitude on this hardware.

## Fidelity switches

The built-in route enables the two Qwen3-VL mechanisms the shared decoder keeps off by default:

- **multimodal RoPE** — image tokens carry `(t, h, w)` grid positions instead of one flat `arange`,
  and cached decode steps resume from the prompt's multimodal numbering (the `mrope_position_deltas`
  behaviour: after a 594-token image block the prompt's positions only advance by `max(llm_h, llm_w)`,
  so continuing with flat indices jumps hundreds of positions and collapses into repetition);
- **deepstack** — the three extra vision mergers are injected back onto the image token rows after
  the first decoder layers.

Both are opt-in per model instance (`multimodal_position_ids=`, `deepstack=` on `Qwen3VLDecoder`), so
the FIBO-VLM and Qwen-Image-2.1 text-encoder routes keep their previous numerics. On the 8B/4-bit
caption checkpoint, same image/prompt/seed, the switches on produce a longer, section-structured
description (layout, subject, garment attributes) where the flat/no-deepstack configuration produces
one flat paragraph; a short prompt that degenerated into a "好的好的…" loop without the position
delta reads normally with it.

## Backend comparison

```bash
# Built-in route (default): mlx-gen's Qwen3-VL decoder + vision tower.
mlxgen-caption --model ./models/qwen3-vl-8b-4bit --image-path photo.jpg

# Optional mlx-vlm parity backend (pip install 'mlx-vlm>=0.5.0').
mlxgen-caption --model ./models/qwen3-vl-8b-4bit --image-path photo.jpg --backend mlx-vlm
```

Both backends load the same directory; no conversion step is involved. Without the optional package
installed, `--backend mlx-vlm` fails with an install hint instead of a stack trace.

## Model notes

- The default prompt is the Chinese long-form instruction from the
  `Qwen3-VL-8B-NSFW-Caption-V4.5` model card; stock Qwen3-VL instruct checkpoints answer in the
  prompt's language, so `--prompt-preset en` or your own `--prompt` works for other uses.
- Lineage of that checkpoint: `Qwen/Qwen3-VL-8B-Instruct` → LoRA fine-tune
  (`thesby/...V4.5`, about 2M image-text pairs) → reupload (`Disty0/...`) → MLX 4-bit
  (`gavinmroy/...-affine-4bit`).
- Community conversions publish no license for the fine-tuned weights: check the upstream card
  before redistributing weights or generated captions.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Unsupported Qwen3-VL decoder checkpoint: N weight(s) do not map …` | The repo is not an MLX-format dense Qwen3-VL checkpoint (for example a raw torch/HF repo or a MoE variant). Use an `mlx-community` / `mlx-vlm`-converted repo, or `--backend mlx-vlm`. |
| `MLX-Gen will not download model files during generation` | `--model` pointed at a repo id that is not in the local cache. Download it first (`mlxgen download --model ...`). |
| `The 'mlx-vlm' backend needs the optional mlx-vlm package` | Install it: `pip install 'mlx-vlm>=0.5.0'`. |
| Captions are truncated mid-sentence | Raise `--max-tokens` (the caption model card recommends 2048). |
| Runs are slow or memory-heavy | Lower `--max-pixels` (the default is already 640000); see the cost table above. |
