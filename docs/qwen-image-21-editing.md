# Qwen-Image-2.1 Reference Editing

MLX-Gen ships two different Qwen-Image-2.1 workflows that are easy to confuse:

- **`qwen21.text` / `qwen21.latent`** — the second-generation text-to-image route, plus
  strength-based latent image-to-image through the *same* single-image pipeline;
- **`qwen21.edit`** — instruction-driven reference editing, up to ten reference images, with RGBA
  output and prefix KV caching.

They are separate implementations on purpose. If you want the broader Qwen surface (the original
Qwen Image family and its edit/inpaint/control routes), see the
[Qwen route matrix](qwen-route-matrix.md) — `qwen21.*` shares nothing with it.

## Why 2.1 Is Not "Qwen Edit With A New Checkpoint"

Qwen-Image-2.1 is a different stack: a single-stream block-causal 7.1B transformer (SwiGLU MLP,
one global `modulation` pair, `axes_dims_rope [16, 56, 56]`), a 4-channel causal VAE, and a
Qwen3-VL text encoder. Text tokens are conditioned at `t = 0` (`causal_condition`), which is what
makes their activations independent of the denoising step — and what makes the prefix cache
possible.

The editing half lives in its own package (`mflux/models/qwen21/reference/`) rather than extending
the txt2img model, because three things do not fit the single-image flow:

- **multi-image prefix cache** — the text instruction and every reference image are encoded once
  and their attention prefix reused across denoising steps, instead of being recomputed per step;
- **RGBA VAE path** — the 64-channel causal VAE carries an alpha channel, so transparency is
  generated rather than post-processed;
- **Qwen3-VL vision changes** — the vision tower gained a `preserve_input_dtype` switch so
  bfloat16 activations survive position-embedding addition.

## Current Status

| Capability | Upstream parity | Public inputs | Status |
| --- | --- | --- | --- |
| `qwen21.text` | `QwenImage21Pipeline` | `--prompt` | Shipped. 40 steps, no guidance by default. |
| `qwen21.latent` | `QwenImage21Pipeline` (img2img) | `--image-path + --image-strength + --prompt` | Shipped. Whole-image restyle, **not** instruction editing. |
| `qwen21.edit` | `QwenImage21EditPipeline` | repeated `--image-paths + --prompt` | Shipped from upstream `acdfc98ba` (#741). Verified end to end; **not yet in the validated proof matrices.** |

`qwen21.edit` has been run end to end against `Qwen/Qwen-Image-2.1` (33.13 GB bf16) on Apple
Silicon, at `-q 8`:

| Check | Observed |
| --- | --- |
| Load (8-bit) | 8.2 s |
| Single-reference edit, 12 steps, `--output-resolution 512` | 16.3 s, 672x384 RGBA |
| Native transparent output, 12 steps, 512x512 | 60% of pixels transparent, alpha spans 0-255 |
| Checkpoint integrity | 29/29 files byte-exact; `transformer` 297 tensors, `text_encoder` 750 |

The 750 counts `lm_head.weight` and `rotary_emb.inv_freq`, which this route drops; excluding them
gives the 748 upstream reports.

These numbers confirm the plumbing, not output quality — there is still no published contact
sheet for this route, so it stays out of the validated proof matrices until one lands.

## What The Checkpoint Must Contain

`qwen21.edit` encodes every reference image with the **Qwen3-VL vision tower**, so the checkpoint's
`text_encoder` must ship the `visual.*` tensors (`patch_embed`, `blocks`, `merger`,
`deepstack_merger_list`) alongside the language model. Each component directory must also carry its
`config.json`, which supplies the block geometry.

This matters because weight-only exports made for **text-to-image** omit the vision tower entirely:
they only ever ran the language model on text. A published 8-bit 2.1 export of that kind contains
399 `text_encoder` tensors covering `embed_tokens`, `layers`, `norm` and `rotary_emb` — and zero
`visual.*` keys. Loading it into this route fails, and no amount of routing fixes that; the tensors
are not there.

Use a full checkpoint (`Qwen/Qwen-Image-2.1`, or a prepared export that includes the vision tower
and per-component `config.json`) for `qwen21.edit`. Weight-only packages remain fine for
`qwen21.text` and `qwen21.latent`, which never touch the vision tower.

### Published quantizations, checked

Every quantized 2.1 build commonly mirrored today was inspected (file headers only, no weights
downloaded). The pattern is not random — it follows the release timeline:

| Handle | Size | Text-encoder tensors | Vision tower | Usable here |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen-Image-2.1` (official bf16) | 33.13 GB | 750 | `model.visual.*` | **Yes** |
| `mlx-community/Qwen-Image-2.1-MLX-4bit` | ~6 GB | 1438 | `vision_tower.*` | Needs key remap |
| `toxicdog/Qwen-Image-2.1-MLX` | 10.72 GB | 1436 | `model.visual.*` | Needs prefix fix |
| `themindstudio/Qwen-Image-2.1-MLX-4bit` | 11.59 GB | 1434 | `vision_tower.*` | Needs key remap |
| `JoyFusionAI/Qwen-Image-2.1-MLX-8bit` | 24 GB | 399 | **none** | No |
| `ddalcu/Qwen-Image-2.1-MLX-Serve-8bit` | 17.65 GB | 902 | **none** | No |

**No 8-bit build carries the vision tower.** Both 8-bit repos above were published 2026-09-20/21,
six days before #741 landed on 2026-09-27, when `mflux-save` still exported the text-only
implementation. They are perfectly good text-to-image packages; they just predate the tower. The
builds that do carry it are 4-bit and come from other converters, which is why their key names
differ (`vision_tower.*` instead of `visual.*`).

So the practical choice today is the official checkpoint, or adapting one of the 4-bit builds by
mapping its names onto this route's module tree.

## Running An Edit

```sh
mlxgen generate --model qwen-image-2.1 \
  --image-paths reference.png \
  --prompt "Replace the background with a snowy mountain range." \
  --output edited.png
```

Multiple references are supported (up to ten):

```sh
mlxgen generate --model qwen-image-2.1 \
  --image-paths person.png jacket.png shoes.png \
  --prompt "Dress the person in the jacket and shoes." \
  --steps 40 \
  --output outfit.png
```

### Options specific to this route

| Option | Default | Meaning |
| --- | --- | --- |
| `--image-paths` | — | One or more reference images. At most 10. Required (may also come from a `--config-from-metadata` sidecar). |
| `--output-resolution` | `1024` | Pixel-area budget reference images are resized to, and the source of the automatic output dimensions. |
| `--use-kv-cache` / `--no-use-kv-cache` | on | Reuse the text/reference attention prefix across denoising steps. Turning it off costs speed, not correctness. |
| `--width` / `--height` | derived | A multiple of 32, or a reference-image scale factor. When omitted, derived from `--output-resolution` and the last reference's aspect ratio. |

### Constraints the route enforces before loading weights

- **Output format must retain alpha.** RGBA is the point of this route, so the output suffix has
  to be `.png`, `.webp`, `.tif` or `.tiff`; anything else is rejected rather than silently
  flattened.
- **`--scheduler` is fixed to the default linear Euler scheduler.** Any other value errors out.
- **`--negative-prompt` only matters with `--guidance > 1`.** At the default guidance of 1 there
  is no negative branch, so the option is reported as ignored rather than dropped silently.
- **At most 10 reference images**, and reference areas must fall inside the vision processor's
  range — adjust `--output-resolution` if that trips.

### Replaying a run

`--config-from-metadata` restores `use_kv_cache` and `output_resolution` alongside the usual
prompt/seed/step fields, so a replay reproduces the same output. Explicit flags still win over
sidecar values. (Canvas geometry is replayed through `canvas_policy`/`resize_mode` rather than by
restoring `width`/`height`, matching the rest of MLX-Gen.)

## Relationship To `qwen-image-edit`

| | `qwen-image-edit-2511` | 2.1 editing (`qwen21.edit`) |
| --- | --- | --- |
| Family | original Qwen Image edit line | second-generation 2.1 stack |
| Reference images | 1 (plus the multi-reference `2509`/`2511` rows) | **up to 10** |
| Transparency | no | **native RGBA** |
| Prefix KV cache | no | **yes** |
| Mask/inpaint | yes (`qwen.inpaint`) | not exposed on this route |
| Validated proof rows | yes, several | **not yet** |

If you need a proven edit route today, use `AbstractFramework/qwen-image-edit-2511-8bit`. Use
`qwen21.edit` when you specifically want multi-reference composition or native transparency, and
treat the output as not-yet-matrix-validated.

## Related Docs

- [Qwen route matrix](qwen-route-matrix.md) — every Qwen route side by side.
- [Image edit capabilities](edit-capabilities.md) — the cross-family edit validation matrix.
- [Masked editing](masked-editing.md) — masked-edit contracts for routes that support masks.
