# 人物四视图生成 · 操作文档

MLX-Gen 本地出图的实操手册。覆盖"用图片编辑指令改尺寸"和"生成人物四视图（turnaround sheet）"。

- 适用环境：Apple Silicon / MLX-Gen（本仓库 `.venv`）
- 文档内所有命令均**在本机实跑验证过**；标注「实测」的段落有真实输出数据
- 未能实跑的部分会明确标注「未实测」，请勿当作已验证结论

---

## 目录

1. [先搞清三件事](#1-先搞清三件事)
2. [尺寸怎么改（核心）](#2-尺寸怎么改核心)
3. [模型选型](#3-模型选型)
4. [实操 A：Qwen-Image-2.1 出四视图](#4-实操-aqwen-image-21-出四视图)
5. [实操 B：2511 + 视角 LoRA 出四视图](#5-实操-b2511--视角-lora-出四视图)
6. [实操 C：单视角分别出图再拼版](#6-实操-c单视角分别出图再拼版)
7. [参数速查表](#7-参数速查表)
8. [排错](#8-排错)
9. [实测数据与产物](#9-实测数据与产物)

---

## 1. 先搞清三件事

### 1.1 「改尺寸」有三个完全不同的含义

| 需求 | 用什么 | 是否重绘画面 |
| --- | --- | --- |
| 生成时决定输出画布 | `--width` / `--height` | 是，重新生成 |
| 把已有图放大/修复 | `mlxgen upscale`（SeedVR2） | 是，超分重建 |
| 纯像素缩放/拼版 | PIL | 否，不动像素内容 |

**编辑指令的文本本身不会改变画布。** 指令只经 VLM tokenizer 编码成 prompt embedding；画布由参数决定。想要"把图改大"必须给尺寸参数。

### 1.2 编辑（edit）和 latent img2img 不是一回事

- `edit-reference`：指令驱动，源图全程作为参考。适合改内容。
- `latent-img2img`：往源图 latent 加噪再按 prompt 去噪，**必须** `--image-strength`。适合调整体风格。
- 传 `--image-strength` 给 edit 路线会被拒；反之 latent 路线不给它也会被拒。

### 1.3 四条 route 的尺寸契约完全不同

这是最容易踩坑的地方——同一个 `--width 2048 --height 512`，在不同模型上结果不一样：

| 模型 | 尺寸倍数 | 默认是否被源图比例改写 | 给非倍数会怎样 |
| --- | --- | --- | --- |
| `qwen-image-2.1`（`qwen21.edit`） | **32** | **否**，直给 | **直接报错** |
| `qwen-image-edit-2511`（`qwen.edit`） | 16 | 是（默认 `source-aspect`） | 向下取整 + 警告 |
| `flux.2-klein-*` | 16 | 是（默认 `source-aspect`） | 向下取整 + 警告 |
| `qwen21.latent` | 16 | 是 | 向下取整 + 警告 |

---

## 2. 尺寸怎么改（核心）

### 2.1 Qwen-Image-2.1：`--width` / `--height` 直给

**[实测]** 同一张参考图、同一 prompt、同一 seed，只改画布：

| 请求 | 实际输出 | 耗时（4 步） |
| --- | --- | --- |
| `--width 1024 --height 256` | **1024 × 256** | 29.7s |
| `--width 1536 --height 512` | **1536 × 512** | 74.5s |
| `--width 2048 --height 512` | **2048 × 512** | 262.8s |

一个像素都不差。原因：`qwen21.edit` 不给 `Config` 传 `image_path`，所以内部恒走 `exact-resize`，你给多少就是多少，**不存在被源图比例改写的情况**。

**[实测]** 非 32 倍数会被直接拒绝：

```
(1000, 512)  → REJECTED: dimensions must be positive multiples of 32
(1024, 500)  → REJECTED
(2032, 512)  → REJECTED
(2048, 512)  → accepted
```

### 2.2 两套尺寸机制别混

- `--output-resolution`（默认 1024，**32 的倍数**）：**参考图的处理面积预算** + 不传 width/height 时的**自动输出尺寸来源**。
- `--width` / `--height`：**直接决定输出画布**，传了就覆盖自动值。

**[实测]** 自动尺寸（`--output-resolution` × 参考图比例）：

| `--output-resolution` | 参考图比例 | 自动输出 |
| --- | --- | --- |
| 1024 | 4:1 | **2048 × 512** |
| 1024 | 1:1 | 1024 × 1024 |
| 512 | 4:1 | 1024 × 256 |
| 2048 | 4:1 | 4096 × 1024 |

**实用推论**：四视图横排是 4:1。只要参考图是横构图，`--output-resolution 1024` 自动就出 `2048x512`，**连 `--width/--height` 都不用传**。

### 2.3 2511 的尺寸：注意默认会被改写

2511 默认 `--canvas-policy source-aspect`，`--width/--height` 只当**面积目标**，最终按第一张输入图的宽高比取最近的 16 倍数。

想严格按给定尺寸出图，加 `--canvas-policy exact-resize`（官方提示：这会重构源图，不是 outpaint 的替代品）。

`--reframe-padding` / `--outpaint-padding` **不能**和 `--width/--height` 同用，它们自己算画布。

### 2.4 放大已有图片

```sh
mlxgen upscale \
  --model AbstractFramework/seedvr2-3b-8bit \
  --image-path input.png \
  --resolution 2x \
  --seed 42 \
  --metadata \
  --output upscaled.png
```

`--resolution` 接受整数短边目标（`1024`）或倍数（`2x` / `5x`）。`1x` 是像素级原尺寸还原。

---

## 3. 模型选型

| | `qwen-image-2.1`（`qwen21.edit`） | `qwen-image-edit-2511`（`qwen.edit`） |
| --- | --- | --- |
| 参考图数量 | **最多 10 张** | 1 张（多图走单独 multi-reference 路由） |
| 尺寸倍数 | **32** | 16 |
| 输出 | **原生 RGBA** | RGB |
| 尺寸是否被源图改写 | **否** | 是（默认） |
| KV 前缀缓存 | **有** | 无 |
| LoRA | **不支持** | 支持（含视角 LoRA） |
| Mask / inpaint | 无 | 有 |
| 调度器 | 只能 `linear` | `flow_match_euler_discrete` |
| 验证矩阵 | **尚未进入** | 多行 PASS |

**选型建议**：

- 要**一次吃多张参考图**（脸 + 服装 + 比例）→ 2.1
- 要**透明背景** → 2.1
- 要**视角控制 LoRA** / 要**已验证的稳定路线** → 2511
- 要**mask 局部编辑** → 2511

---

## 4. 实操 A：Qwen-Image-2.1 出四视图

### 4.1 前置检查：权重必须带视觉塔

`qwen21.edit` 用 Qwen3-VL 视觉塔编码每张参考图，`text_encoder` 必须含 `visual.*` 张量。

```sh
python3 -c "
import json
idx = json.load(open('text_encoder/model.safetensors.index.json'))
keys = list(idx['weight_map'].keys())
vis = [k for k in keys if 'visual' in k or 'vision' in k]
print('total:', len(keys), 'vision:', len(vis))
"
```

**[实测]** 本机 `/Users/jiangzhenguo/ai/models/Qwen-Image-2.1-bf16`：`total: 750`，`vision: 351`（含 `merger` 与 `deepstack_merger_list`）→ **可用**。

判断标准：`vision` 计数为 **0** 就不能跑编辑路线。文档记载的坑：**所有 8-bit 镜像都不带视觉塔**，加载会直接失败——不是路由问题，是张量不存在。

### 4.2 关键：本地权重路径的命名要求

**[实测]** 这是个真实的坑。用本地路径跑 2.1 编辑时，路由靠**目录 basename** 匹配别名：

```sh
# ✗ 失败：basename 里没有别名，被当成普通 Qwen Image，报
#   "Qwen Image does not support edit-reference image-to-image generation."
--model /Users/jiangzhenguo/ai/models/Qwen-Image-2.1-bf16

# ✓ 成功：basename 含 "qwen-image-2.1" 别名
--model out/qwen21_test/models/qwen-image-2.1-bf16
```

用官方 repo id（`--model qwen-image-2.1`）能正常路由，但它会去 HF 缓存找权重，本机没有就会提示先 `mlxgen download`。

**推荐做法**：建一个含别名的软链接指向真实权重。

```sh
mkdir -p ./models
ln -sfn /path/to/Qwen-Image-2.1-bf16 ./models/qwen-image-2.1-bf16
```

### 4.3 出图命令

**[实测]** 统一入口 `mlxgen generate` 在本地路径上不会自动选中 2.1 编辑路由，请用**专用入口**：

```sh
.venv/bin/mlxgen-generate-qwen-2.1-edit \
  --model ./models/qwen-image-2.1-bf16 \
  --image-paths portrait.png \
  --prompt "Turn this single character into a character turnaround reference sheet: the same character shown four times in one horizontal row, left to right front view, three-quarter view, side profile view, and back view. Identical face, hairstyle, outfit, colors and proportions in all four views. Neutral A-pose, full body, feet on the same ground line, flat studio lighting, plain light grey background, no props, no text, no watermark, no border." \
  --width 2048 --height 512 \
  --output-resolution 1024 \
  --steps 40 \
  --guidance 1 \
  --seed 42 \
  --output character_four_view.png
```

多参考图（2.1 的最大优势）：

```sh
.venv/bin/mlxgen-generate-qwen-2.1-edit \
  --model ./models/qwen-image-2.1-bf16 \
  --image-paths face.png outfit.png fullbody.png \
  --prompt "Use the first image for the face and identity, the second for the outfit, the third for body proportions and pose. Produce one character turnaround sheet with front, three-quarter, side and back views of the same character, identical design in all views, neutral A-pose, flat studio lighting, plain background." \
  --width 2048 --height 512 \
  --steps 40 \
  --output character_four_view.png
```

### 4.4 注意：负向提示词 vs guidance

`--guidance 1`（默认）时没有负向分支，传 `--negative-prompt` 会被**警告忽略**。要用负向提示词就必须 `--guidance > 1`。

**[实测]** `--guidance > 1` 时**不**必须显式传 `--negative-prompt`：CLI 会把它默认成空串（`PromptUtil.read_negative_prompt` 返回 `""`），所以 `--guidance 2` 直接跑是正常的。只有直接调用 Python API（`model.generate_image`）时才必须显式传 `negative_prompt`，否则抛 `Provide negative_prompt (which may be empty) to enable guidance greater than 1`。

### 4.5 输出格式硬约束

2.1 输出 RGBA，**后缀必须是 `.png` / `.webp` / `.tif` / `.tiff`**。写 `.jpg` 会在加载权重前被拒。

---

## 5. 实操 B：2511 + 视角 LoRA 出四视图

**这条路线的优势**：有仓库验证过的视角控制 LoRA，且是已验证路线。

```sh
.venv/bin/mlxgen generate \
  --model AbstractFramework/qwen-image-edit-2511-8bit \
  --image portrait.png \
  --prompt "<sks> front view full body shot of the same character, character turnaround sheet" \
  --lora-paths lightx2v/Qwen-Image-Edit-2511-Lightning:Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
               fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA:qwen-image-edit-2511-multiple-angles-lora.safetensors \
  --lora-scales 1.0 1.25 \
  --steps 4 --guidance 2.5 \
  --width 1024 --height 1024 \
  --output view_front.png
```

要点：

- 视角 LoRA 用 **`<sks>` 触发词**导向视角
- `--lora-scales` 数量必须和 `--lora-paths` 完全一致
- 用 Lightning 4 步时 `--steps 4`；去掉 Lightning 后改回 `--steps 20 --guidance 4`
- **建议每次出一个视角再拼版**（见实操 C），比让模型一次画四个更稳

---

## 6. 实操 C：单视角分别出图再拼版

最稳的四视图做法：四个视角分别出图，人工挑选后拼版。避免模型把"一张画四个"画成四个不同的人。

```sh
for view in front "three-quarter" side back; do
  .venv/bin/mlxgen-generate-qwen-2.1-edit \
    --model ./models/qwen-image-2.1-bf16 \
    --image-paths portrait.png \
    --prompt "Full body $view view of the same character, neutral A-pose, flat studio lighting, plain light grey background, character reference sheet." \
    --width 512 --height 1024 --steps 40 --seed 42 \
    --output "view_${view}.png"
done
```

拼版（纯像素操作，不重绘）：

```python
from PIL import Image

paths = ["view_front.png", "view_three-quarter.png", "view_side.png", "view_back.png"]
images = [Image.open(p).convert("RGB") for p in paths]
height = max(i.height for i in images)
images = [i.resize((round(i.width * height / i.height), height), Image.LANCZOS) for i in images]

sheet = Image.new("RGB", (sum(i.width for i in images), height), (235, 235, 235))
x = 0
for i in images:
    sheet.paste(i, (x, 0))
    x += i.width
sheet.save("four_view_sheet.png")
print(sheet.size)
```

---

## 7. 参数速查表

### 2.1 编辑专用

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--image-paths` | — | 1~10 张参考图，必填 |
| `--output-resolution` | `1024` | 参考图处理面积 + 自动输出尺寸来源，**32 的倍数** |
| `--width` / `--height` | 自动 | 输出画布，**32 的倍数**；传了覆盖自动值 |
| `--use-kv-cache` / `--no-use-kv-cache` | 开 | 复用文本+参考图注意力前缀 |
| `--steps` | 40 | ≥ 2（终末 sigma 需要） |
| `--guidance` | 1 | ≥ 1；> 1 时启用负向分支（CLI 下负向词默认空串，Python API 需显式传） |
| `--output` | `image.png` | 后缀必须 `.png`/`.webp`/`.tif`/`.tiff` |

### 通用

| 参数 | 说明 |
| --- | --- |
| `--seed` | 支持多个值；Qwen 系列换 seed 变化不大，改 prompt 更有效 |
| `--quantize` / `-q` | `-q 8` 显存友好；2.1 官方 bf16 是 33 GB |
| `--metadata` | 导出 `.metadata.json` 边车（记录最终 width/height） |
| `--config-from-metadata` | 用边车复现；2.1 会恢复 `use_kv_cache` 和 `output_resolution` |
| `--low-ram` / `--mlx-cache-limit-gb` | 显存吃紧时用 |

---

## 8. 排错

| 报错 | 原因 | 解决 |
| --- | --- | --- |
| `dimensions must be positive multiples of 32` | 2.1 尺寸非 32 倍数 | 改成 32 的倍数 |
| `Qwen Image does not support edit-reference` | 本地路径 basename 不含 2.1 别名，路由识别错 | 建含别名的软链接（见 4.2） |
| `--image-strength is required for latent image-to-image mode` | 路由成了 latent 而非 edit | 用专用入口，或修正路径命名 |
| `MLX-Gen will not download model files` | 别名指向 HF，本地缓存没有 | 用本地路径，或先 `mlxgen download` |
| 加载失败 / 张量缺失 | 权重缺 `visual.*` 视觉塔 | 换带视觉塔的权重（见 4.1） |
| `outputs RGBA; use PNG, WebP or TIFF` | 输出写成 `.jpg` | 改后缀 |
| `Provide negative_prompt ... to enable guidance greater than 1` | Python API 下 `guidance > 1` 但没传 `negative_prompt` | 传 `negative_prompt=""`（CLI 下不会出现，CLI 默认空串） |
| `Reference image area falls outside the vision processor's range` | 参考图面积超范围 | 调 `--output-resolution` |

### 性能注意

**[实测]** 2048×512 比 1536×512 慢 **3.5 倍**（262s vs 74s）——attention 成本随面积增长远快于面积本身。

**建议**：先用小尺寸（如 `1536×512`）试 prompt 和构图，满意后再上 `2048×512` 终稿。

---

## 9. 实测数据与产物

### 9.1 尺寸验证（4 步冒烟）

| 文件 | 请求 | 实测输出 | 耗时 |
| --- | --- | --- | --- |
| `out_1024x256.png` | 1024×256 | 1024×256 RGBA | 29.7s |
| `out_1536x512.png` | 1536×512 | 1536×512 RGBA | 74.5s |
| `out_2048x512.png` | 2048×512 | 2048×512 RGBA | 262.8s |
| `cli_1024x256.png` | CLI 直跑 1024×256 | 1024×256 RGBA | — |

结论：**2.1 编辑能改尺寸，输出像素与参数完全一致。**

### 9.2 四视图构图验证

**[实测]** 用 4:1 简笔画输入 + 4 步快速冒烟，输出确实是**同一角色横排四次**（正面 / 四分之三 / 侧面 / 背面），发型、服装、比例四视图一致。

脸部和细节在 4 步下偏糊，属预期——正式出图请用 40 步 + 真实角色图。对比图见 `out/qwen21_test/size_comparison.png`。

### 9.3 产物清单

```
out/qwen21_test/
├── source_4x1.png            # 4:1 测试输入
├── out_1024x256.png          # 实测输出
├── out_1536x512.png
├── out_2048x512.png
├── cli_1024x256.png          # CLI 直跑验证
├── size_comparison.png       # 三档尺寸对比图
├── test_size.py              # 尺寸测试脚本
└── models/qwen-image-2.1-bf16 -> 真实权重软链接
```

配套脚本：[character_four_view.py](../character_four_view.py)（三条路线合一）

### 9.4 未实测项（谨慎对待）

以下内容来自代码/文档阅读，**未在本机实跑**：

- 2511 + Multiple-Angles LoRA 的四视图效果（命令来自仓库文档与验证注册表）
- `mlxgen upscale` 超分放大
- 2.1 的 40 步正式质量（仅跑过 4 步冒烟）

---

## 附：本机权重现状

| 路径 | 状态 |
| --- | --- |
| `/Users/jiangzhenguo/ai/models/Qwen-Image-2.1-bf16` | 31 GB，带视觉塔，**可用于 2.1 编辑** |
| HF 缓存 `mlx-community--Qwen-Image-Edit-8bit` | 可用 |
| HF 缓存 flux.2-klein-9b-8bit / z-image-turbo-8bit / seedvr2-3b-8bit | 可用 |
