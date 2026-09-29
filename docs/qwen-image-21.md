# Qwen-Image-2.1 Guide

Qwen-Image-2.1 is a second-generation Qwen image model that unifies text-to-image generation and
instruction-driven editing in one 7B stack. This page covers what each route does, what checkpoint
you need, and how to run it.

For the editing route's internals and its relationship to the original Qwen Image edit line, see
[Qwen-Image-2.1 Editing](qwen-image-21-editing.md). For every Qwen route side by side, see the
[Qwen route matrix](qwen-route-matrix.md).

## What 2.1 Is

| Component | Detail |
| --- | --- |
| Transformer | 7B single-stream, block-causal DiT (32 layers, SwiGLU MLP) |
| Text/vision encoder | Qwen3-VL-8B |
| VAE | 64-channel causal, RGBA (16x spatial compression) |
| Native output | 2K |
| Defaults | 40 steps, guidance 1.0 (no guidance) |

2.1 shares nothing with the original Qwen Image line — different transformer, encoder and VAE — so
it is registered as its own family (`qwen21.*`) and inherits none of the older sidecars
(ControlNet, inpaint, reframe, outpaint).

Two properties matter in practice:

- **Native transparency.** The VAE carries an alpha channel, so transparency is generated rather
  than matted afterwards.
- **Multiple reference images.** The editing route accepts up to ten, using a prefix KV cache so
  the text instruction and reference images are encoded once and reused across denoising steps.

## Route Overview

| Capability | What it does | Public inputs |
| --- | --- | --- |
| `qwen21.text` | Text-to-image | `--prompt` |
| `qwen21.latent` | Whole-image restyle at a given strength (latent img2img, **not** instruction editing) | `--image-path + --image-strength + --prompt` |
| `qwen21.edit` | Instruction-driven editing from 1-10 reference images, RGBA output | `--image-paths + --prompt` |

`qwen21.text` and `qwen21.latent` share one single-image pipeline. `qwen21.edit` is a separate
reference implementation — see [why](qwen-image-21-editing.md#why-21-is-not-qwen-edit-with-a-new-checkpoint).

### How The Route Is Selected

There is **one model handle for all three routes** — `qwen-image-2.1` (aliases `qwen-2.1`,
`qwen-image-21`). There is no separate `qwen-image-2.1-edit` handle. What you pass decides the
route:

| You pass | Route taken |
| --- | --- |
| `--prompt` only | `qwen21.text` — text-to-image |
| `--image-path` + `--image-strength` | `qwen21.latent` — latent img2img |
| `--image-paths` (one or more) | `qwen21.edit` — reference editing |

`qwen21.edit` is the default image-to-image row, so a bare `--image-paths` reaches editing. The
`--task` flag can state the intent explicitly (`--task edit` or `--task img2img`), which is useful
when a command needs to be unambiguous.

Note that `--image-path` (singular) and `--image-paths` (plural) are **different flags**: the
former is the latent-img2img source and requires `--image-strength`, the latter is the reference
list for editing.

## Choosing A Checkpoint

This is the part that trips people up, so it comes before the examples.

**The editing route needs the vision tower.** Every reference image is encoded through Qwen3-VL's
vision tower, so the checkpoint's `text_encoder` must ship the `visual.*` tensors alongside the
language model, and each component directory must carry its `config.json`.

Weight-only exports made for **text-to-image** omit the tower entirely — they only ever ran the
language model on text. Loading one into `qwen21.edit` fails on the missing tower, not on routing.

| Handle | Size | Vision tower | Works with `qwen21.edit` |
| --- | --- | --- | --- |
| `Qwen/Qwen-Image-2.1` (official, bf16) | 33.13 GB | Yes | **Yes** |
| `mlx-community/Qwen-Image-2.1-MLX-4bit` | ~6 GB | Yes, named `vision_tower.*` | Needs key remap |
| `toxicdog/Qwen-Image-2.1-MLX` | 10.72 GB | Yes, named `model.visual.*` | Needs prefix fix |
| `JoyFusionAI/Qwen-Image-2.1-MLX-8bit` | 24 GB | **No** | No |
| `ddalcu/Qwen-Image-2.1-MLX-Serve-8bit` | 17.65 GB | **No** | No |

**No published 8-bit build carries the vision tower.** The two 8-bit repos above were built on
2026-09-20/21, six days before editing landed upstream (2026-09-27), when the save path still
exported the text-only implementation. They are valid text-to-image packages; they simply predate
the tower. The builds that do carry it are 4-bit and come from other converters, which is why
their key names differ.

Weight-only packages remain fine for `qwen21.text` and `qwen21.latent`, which never touch the
tower.

### Downloading The Official Checkpoint

```sh
mlxgen download --model Qwen/Qwen-Image-2.1
```

The checkpoint is about 33 GB, so budget accordingly. In regions where `huggingface.co` is
unreachable, ModelScope mirrors the identical revision:

```sh
export HF_ENDPOINT=https://hf-mirror.com   # or use a faster regional mirror
```

If you export a package yourself, use the reference model — `mlxgen save --model qwen-image-2.1`
exports the **text-only** implementation and drops the tower:

```python
from mflux.models.qwen21.reference import QwenImage21Edit

model = QwenImage21Edit(quantize=8)
model.save_model("./qwen21-8bit")
```

## Text To Image

```sh
mlxgen generate --model qwen-image-2.1 \
  --prompt "A red fox in a snowy forest holding a wooden sign that says \"Hello\"" \
  --steps 40 \
  --output fox.png
```

Defaults are 40 steps and no guidance. Passing a negative prompt together with `--guidance > 1`
enables classifier-free guidance; with the default guidance of 1 the negative pass is skipped
entirely.

### Transparency

Ask for it in the prompt, and write to a format that keeps alpha:

```sh
mlxgen generate --model qwen-image-2.1 \
  --prompt "This is an RGBA image with transparency. A cute cartoon red panda sticker. The image has alpha channel and the background is transparent." \
  --steps 40 \
  --output sticker.png
```

All outputs carry four channels; the prompt decides whether the generated alpha is transparent or
opaque. **PNG, WebP and TIFF retain alpha** — writing a transparent result to JPEG silently flattens
it.

## Editing With Reference Images

```sh
mlxgen generate --model qwen-image-2.1 \
  --image-paths portrait.png \
  --prompt "Replace the background with a snowy mountain range. Preserve the subject's face and pose." \
  --output edited.png
```

The `qwen-image-2.1` handle also reaches this route; `--image-paths` selects it. Multiple
references are supported (at most ten), and **their order is significant**:

```sh
mlxgen generate --model qwen-image-2.1 \
  --image-paths subject.png scene.png \
  --prompt "Place the subject from image 1 in the setting of image 2." \
  --output combined.png
```

### Options specific to editing

| Option | Default | Meaning |
| --- | --- | --- |
| `--image-paths` | — | One or more reference images (max 10). Required, unless a `--config-from-metadata` sidecar supplies them. |
| `--output-resolution` | `1024` | Pixel-area budget each reference is resized to, and the source of the automatic output dimensions. |
| `--use-kv-cache` / `--no-use-kv-cache` | on | Reuse the text/reference attention prefix across steps. Turning it off costs speed, not correctness. |
| `--width` / `--height` | derived | Multiples of 32, or a reference-image scale factor. When omitted, derived from `--output-resolution` and the last reference's aspect ratio. |

### Constraints enforced before weights load

- **Output format must retain alpha** (`.png`, `.webp`, `.tif`, `.tiff`). Other suffixes error out
  rather than silently flattening the result.
- **The scheduler is fixed** to the default linear Euler scheduler.
- **`--negative-prompt` only applies with `--guidance > 1`**; at the default it is reported as
  ignored rather than dropped silently.
- **At most ten reference images**, and reference areas must land inside the vision processor's
  range — lower `--output-resolution` if that trips.

### Replaying a run

`--config-from-metadata` restores `use_kv_cache` and `output_resolution` along with the usual
prompt/seed/step fields, so a replay reproduces the same output:

```sh
mlxgen generate --model qwen-image-2.1 \
  --image-paths portrait.png --prompt "..." --metadata --output run.png

mlxgen generate --model qwen-image-2.1 \
  --config-from-metadata run.metadata.json --output replay.png
```

Explicit flags always win over sidecar values. Keep `use_kv_cache` fixed when comparing samples —
cached and uncached execution can differ in reduced precision.

## Measured Performance

Numbers from an Apple Silicon run against the official bf16 checkpoint at `--quantize 8`:

| Run | Configuration | Time |
| --- | --- | --- |
| Load | 8-bit | 8.2 s |
| Edit | 12 steps, 1 reference, `--output-resolution 512` | 16.3 s (672x384 RGBA) |
| Text-to-image | 12 steps, 512x512 | 11.1 s (60% of pixels transparent) |

The first denoising step carries warm-up (~2.9 s in the edit run); steady state is about 0.7 s per
step at this size. The trained defaults of 40 steps at 1024 cost considerably more, and
high-resolution runs are not linear in area because of VAE decode.

`--low-ram` releases the text encoder after prompt encoding and tiles the VAE decode.

## Python

Both routes are reachable directly:

```python
from mflux.models.common.config import ModelConfig
from mflux.models.qwen21.reference import QwenImage21Edit

model = QwenImage21Edit(quantize=8, model_config=ModelConfig.qwen_image_21())

sticker = model.generate_image(
    seed=42,
    prompt="A cute red panda sticker, transparent background",
    width=1024,
    height=1024,
)
sticker.save("panda.png")

edited = model.generate_image(
    seed=42,
    prompt="Give the panda a blue scarf.",
    image_paths=["panda.png"],
    output_resolution=512,
)
edited.save("panda-scarf.png")
```

## Limitations

- **No LoRA and no training** for this model.
- **No mask, ControlNet, reframe or outpaint** on the 2.1 routes. Use the original Qwen Image edit
  line for masked work — see [Masked editing](masked-editing.md).
- **Editing output quality is not yet characterised.** The route runs end to end, but there is no
  published contact sheet, and the upstream validation notes that some 1024x1024 edits missed the
  requested change in both MLX and the Diffusers reference. Treat results as unproven until a
  proof row lands.
- **MLX and PyTorch use different random-number generators**, so matching seeds do not imply
  matching images.
- **Licence.** The weights are under the Qwen Research License: non-commercial research and
  evaluation, with commercial use requiring a separate licence from the Qwen team. Check the
  upstream licence before deploying.

## Related Docs

- [Qwen-Image-2.1 Editing](qwen-image-21-editing.md) — the editing route in depth.
- [Qwen route matrix](qwen-route-matrix.md) — every Qwen route and its proof surface.
- [Quantization](quantization.md) — what `--quantize` does and which packages exist.
- [Model management](model-management.md) — `mlxgen download` / `prepare` details.
