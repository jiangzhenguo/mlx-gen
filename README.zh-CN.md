# MLX-Gen

[English](README.md)

[![mlx-gen-jzg](https://img.shields.io/pypi/v/mlx-gen-jzg?label=mlx-gen-jzg&logo=pypi&logoColor=white)](https://pypi.org/project/mlx-gen-jzg/)
[![MLX](https://img.shields.io/pypi/v/mlx?label=MLX&logo=pypi&logoColor=white)](https://pypi.org/project/mlx/)
[![CI](https://github.com/lpalbou/mlx-gen/actions/workflows/tests.yml/badge.svg)](https://github.com/lpalbou/mlx-gen/actions/workflows/tests.yml)

MLX-Gen 是面向 Apple Silicon 与 MLX 的本地图像、视频生成运行时。它通过 `mlxgen` 提供文生图、图生图、文生视频、图生视频、模型下载、模型准备、SeedVR2 与 SwiftVR 图像/视频修复与超分、面向 MLX-Gen 优化的量化模型，以及给应用使用的进度回调。

> [!NOTE]
> **本仓库是 [lpalbou/mlx-gen](https://github.com/lpalbou/mlx-gen) 的个人 fork**，在上游之外增加了面向本地 Apple Silicon 工作流的能力，这些改动目前还没有合回上游。相对上游 `main` 的差异见下方 [本 fork 的新增内容](#本-fork-的新增内容)。

> [!IMPORTANT]
> 上游 MLX-Gen 本身 fork 自 [mflux](https://github.com/filipstrand/mflux)。当前代码库的主要功劳属于 Filip Strand 与 mflux 的原贡献者。MLX-Gen 保留这套基础，并专注于让本地生成对真实应用可预期：用户先下载或准备模型，再用一条 `mlxgen` 命令生成、查看支持的模式，并对照示例与实测结果。MLX-Gen 的贡献包括经过测试的文生图 / 图生图路由、Qwen Image Edit 2509/2511 的路由与对齐修复、Bonsai Image、Wan2.2 文生视频 / 图生视频 / 视频生视频、Bernini-R 参考图引导的视频生成与编辑、按模型区分的混合量化策略、为 MLX-Gen 优化并发布到 Hugging Face 的量化仓库，以及给应用用的进度回调。上游 fork 的目的，是让 AbstractVision 及相关 AbstractFramework 项目可以更快迭代，同时仍保留在有价值时把改动合回 mflux 社区的可能。

![MLX-Gen 工作流示例](https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/assets/examples/spaceship-snow/mlx-gen-example.png)

这张截图来自 [AbstractFlow](https://github.com/lpalbou/abstractflow)，它是 [AbstractFramework](https://abstractframework.ai/) 生态里的可视化工作流编写工具。AbstractFlow 可以编排经 AbstractVision 与 AbstractCore 暴露的图像 / 视频生成能力；下面的命令行示例也可以直接跑同样的 MLX-Gen 步骤。

## 本 fork 的新增内容

相对上游 [lpalbou/mlx-gen](https://github.com/lpalbou/mlx-gen) `main`（0.38.0），本 fork 增加了：

| 新增内容 | 命令 / 模型句柄 | 文档 |
| --- | --- | --- |
| **Qwen-Image-2.1** 文生图、latent 图生图、参考图编辑（最多 10 张参考图，原生 RGBA，前缀 KV cache） | `mlxgen generate --model qwen-image-2.1`、`mlxgen-generate-qwen-2.1`、`mlxgen-generate-qwen-2.1-edit` | [docs/qwen-image-21.md](docs/qwen-image-21.md)、[docs/qwen-image-21-editing.md](docs/qwen-image-21-editing.md) |
| **Qwen3-VL caption**（图像 → 文本 / 提示词反推），含多模态 RoPE 与 deepstack；可选 `mlx-vlm` 对齐后端 | `mlxgen-caption` / `mflux-caption` | [docs/qwen3-vl-caption.md](docs/qwen3-vl-caption.md) |
| **CosyVoice 3** 零样本 TTS + 视频旁白混流（独立的 Python 3.11 worker 虚拟环境；Hugging Face / ModelScope 镜像） | `mlxgen tts`、`mlxgen dub`、`mlxgen prepare --model cosyvoice3` | [docs/cosyvoice3-tts.md](docs/cosyvoice3-tts.md) |
| 角色 **三视图 / 四视图** 实操手册（画布、编辑、超分的取舍，各路由的尺寸约定，以及实测配方） | — | [docs/character-four-view-operations.md](docs/character-four-view-operations.md) |
| MLX 下限提高到 **0.32.3**，避免 Qwen3-VL 参考编码静默回退 | `pyproject.toml` | — |

上游 PyPI 包不含上表这些能力。请从本仓库的本地检出安装并运行：

```sh
uv sync
uv run mlxgen --help
uv run mlxgen-caption --help
uv run mlxgen tts --help
```

## 它能做什么

MLX-Gen 在你把模型下载到本地之后，运行受支持的 Hugging Face 源模型，也会运行专为 MLX-Gen 发布在 Hugging Face 上的量化变体。其中一部分变体不会把每一层都量化，而是把对精度敏感的层留在 8-bit 或 BF16，因此它们以 MLX-Gen 专用仓库的形式发布。你需要先显式下载或准备模型，生成时只使用磁盘上已有的文件。这适合桌面应用、工作流引擎和长时间本地任务。

主要能力：

- 文生图：Qwen Image、**Qwen-Image-2.1**、FLUX.2 Klein、Z-Image、ERNIE Image Turbo、Bonsai Image、FIBO，以及已提供的优化量化变体；
- 图生图：latent 图生图、指令 / 参考图编辑、多参考图编辑；所选模型支持 `--mask-path` 时的蒙版编辑 / 局部重绘（Qwen 编辑模型；基础 Qwen 模型可用可调的 `--mask-strength` 热启动，或走已验证的 control-inpaint 旁路；Z-Image Turbo；FLUX.2 Klein 蒸馏版与基础版。Klein 后端路由还可选蒙版区域参考图，见 `docs/masked-editing.md`）；所选模型支持 `--controlnet-image-path` 时的 Qwen 结构化控制；以及所选模型支持的按路由区分的重构图 / 扩图。扩图可以只扩一边、扩一条轴的两边，或一次扩四边；可选扩图条件画布（`--outpaint-fill`）；对很深的双轴扩图可自动拆成两遍（`--outpaint-passes`）。所有 FLUX.2 Klein 模型（蒸馏 4B/9B 与基础 4B/9B）都支持源图锁定扩图，并把原裁切区域还原回去（蒸馏 4B q8 是测得最快的路由，Apple M5 Max 上 54.6 秒，见 `docs/reframe-outpaint.md`）；
- 反向的图像理解：用本地 MLX Qwen3-VL 权重描述图像（`mlxgen-caption`）。它运行 mlx-gen 自己的 Qwen3-VL 解码器与视觉塔，并启用多模态 RoPE 与 deepstack；接受任意已发布量化档位的 MLX 格式稠密 Qwen3-VL 仓库，也可以把同一份权重交给可选的 `mlx-vlm` 运行时做对齐检查，见 `docs/qwen3-vl-caption.md`；
- Wan2.2 文生视频、图生视频，以及提示词引导的视频生视频（普通模式，或用 `--video-mask-path` 做蒙版模式，蒙版外的内容锁定为源视频）。包含 TI2V-5B 的 BF16/q8 包，以及 A14B T2V/I2V 的 BF16 与混合 q8/BF16 包。视频生视频会把源视频重采样到请求的 fps 时间轴（输出保持实时速度），并尽力把源音轨拷到输出上。普通 / 蒙版视频生视频仅限 `Wan2.2-T2V-A14B`，使用一段源视频加一条提示词，并且必须 `--solver unipc`。Wan 图生视频按源图宽高比决定输出尺寸，避免把输入拉伸到不匹配的画布上；
- Wan2.1-VACE-1.3B 原生参考图与蒙版条件（`wan-vace`）：用一张或多张 `--reference-image` 把画面中的物体注入新场景，或对源视频做学习到的蒙版编辑。该路由已对照 diffusers 参考实现逐阶段验证，并附带证明材料；
- MiniMax-H3 文生视频与图生视频，**带同步立体声**（`minimax-h3`、`minimax-h3-turbo`、`minimax-h3-turbo-544p`）：一条提示词，可选一张 `--image-path` 关键帧，生成 24 fps 片段及其音轨，合成单个 MP4（AAC），时长 5 到 15 秒。lightx2v 的 8 步 Turbo 适配器会自动挂上，社区 LoRA（civitai、ai-toolkit、ComfyUI、kohya 键布局）可以叠在上面。音轨覆盖环境声、配乐，以及口型对齐的对白。在 128 GB Mac 上用 `--quantize 8`，或从准备好的 75 GB q8 包运行（`mlxgen prepare`），见 `docs/minimax-h3.md`；
- CosyVoice 3 **零样本音色克隆 TTS** 与视频配音（`mlxgen tts` / `mlxgen dub`）：先用 `mlxgen prepare --model cosyvoice3` 准备隔离的包，再根据参考音频合成 WAV，或把旁白替换 / 混入已有 MP4（通常是 MiniMax-H3 的输出），见 `docs/cosyvoice3-tts.md`；
- 字节跳动 Bernini-R 1.3B 渲染器（`bernini-r-1.3b`）：用 1 到 8 张有序的普通参考图生成视频，用这些参考图编辑一段源视频，或只根据源视频做提示词引导的视频编辑。统一 CLI 与 Python 运行时都支持。推荐下载 BF16 包（`bernini-r-1.3b-bf16`，约 15.6 GiB，输出与源包逐位一致），或 FP32 源包（约 27 GiB）。该路由以 BF16 运行，不支持 `--quantize`。**实验性：** 7 条官方公开 1.3B 示例，加上中档配置的 `ads2v`，已通过定性验收并提交了证明（完整提示词与对照图在 `docs/assets/validation/bernini-r-1.3b-2026-08-11/`）；其余条目的调参配方见 `docs/bernini.md`；
- 通过 `mlxgen upscale` 做图像与视频修复。用 `--model` 选择两个家族，两者都出现在 `mlxgen capabilities` 的 `restoration` 数组里：
  - SeedVR2：图像与视频超分 / 修复。覆盖官方 3B/7B 源模型（含专用的 `seedvr2-7b-sharp` 路由）、已发布的 q8/q4 包、按短边设定目标尺寸、`2x` 与 `3x` 这类显式缩放倍数、长视频的流式修复、保留源帧率、偏保守的主机安全默认视频配置，以及已发布的 3B/7B 视频路径上五秒 `1x` 与 `2x` 验证包；
  - SwiftVR：在 Wan2.2-TI2V-5B 骨干上、按源分辨率做一步视频修复。同一段素材上吞吐大约是 SeedVR2 的 40 倍，峰值内存大约是其五分之一，并发布了五秒的三方对比包；
- 面向本地 MLX-Gen 模型包的显式 `download` 与 `prepare` 流程；
- 在重任务开始前，用 JSON 查看模型能力；
- 按路由严格检查 LoRA 路由与适配器应用。缓存里有适配器元数据时，会先做模型卡兼容性预检。已有公开证明的路由包括：Qwen Image Edit 原版 / 2509 / 2511，基础 Qwen Image 的文生图与 latent，Qwen Image 2512 文生图，基础 Qwen Image q8 的结构化控制与 control-inpaint，精确的 `AbstractFramework/z-image-turbo-8bit` 文生图与 latent 图生图，FLUX.2 Klein 9B 编辑与多参考，FLUX.2 Klein base 4B 扩图，ERNIE Image Turbo 文生图与 latent 图生图，以及所有受支持的 Wan q8 视频路由（含 Lightning 快速视频生视频）。LoRA 指南记录了每条路由上已验证的适配器、严格的 scale 匹配、可直接复制的 `owner/repo:subdir/file` 适配器写法，以及每条证明背后的验证材料（图像与 T2V/I2V 用同种子 A/B 对照图，视频生视频用有界的多种子矩阵）。那些 Lightning 示例是对照 MLX-Gen q8 包记录的，不是任意外部 FP8 权重。Bonsai LoRA 保持失败即关闭；
- 给嵌入 MLX-Gen 的应用使用的共享进度事件；
- 逐步生成预览：为 FLUX.1 与 FLUX.2 潜空间发布了微型自编码器，让交互应用可以做连续实时预览（见 [生成预览](docs/previews.md)）。

长时间的图像编辑之前，先运行 `mlxgen capabilities --model ...`。能力输出描述的是可用路由；验证报告与对照图描述的是某个精确的源模型句柄或 MLX-Gen 优化包是否通过了视觉发布门槛。发布证据应使用真实句柄，例如 `briaai/Fibo-Edit` 或 `AbstractFramework/flux.2-klein-9b-8bit`，而不是短别名。

## 安装

用 `uv` 安装：

```sh
uv tool install --upgrade mlx-gen
```

或装进当前环境：

```sh
python -m pip install -U mlx-gen
```

查看命令面：

```sh
mlxgen --help
```

应用集成和脚本调用请使用 `mlxgen --help` 里列出的 `mlxgen` 命令。为了兼容上游代码库，安装包仍会提供一些 `mflux-generate-*` 入口，但新工具不建议把它们当作集成接口。

本 fork 的新增命令不在上游 PyPI 包里。需要 Qwen-Image-2.1、Qwen3-VL caption 或 CosyVoice 3 时，请从本仓库检出后执行 `uv sync`，再用 `uv run mlxgen ...`。

## 起步命令

先显式下载模型文件：

```sh
mlxgen download --model AbstractFramework/flux.2-klein-9b-8bit
```

生成一张图：

```sh
mlxgen generate \
  --model AbstractFramework/flux.2-klein-9b-8bit \
  --prompt "A cinematic wide shot of a compact sci-fi spaceship resting in deep snow on a frozen alien planet" \
  --width 768 \
  --height 432 \
  --steps 24 \
  --guidance 1.0 \
  --seed 6107 \
  --output spaceship.png
```

一次传入多个种子，可在同一次运行里生成多张图或多段视频：

```sh
mlxgen generate \
  --model qwen-image \
  --prompt "A clean studio product photo" \
  --seed 101 202 303 \
  --output product.png
```

MLX-Gen 为每个种子保存一份结果，并自动把种子追加到输出文件名主干上，例如 `product_seed_101.png`、`product_seed_202.png`、`product_seed_303.png`。`mlxgen generate` 和 `mlxgen upscale` 也支持 `--auto-seeds N`，用一条命令得到若干随机变体。重复的显式种子会被拒绝；`--auto-seeds` 必须大于 0；`--output` 支持 `{seed}`。SeedVR2 多源文件运行还支持 `{input_name}`，并兼容旧写法 `{image_name}`。

用 SeedVR2 放大图像：

```sh
mlxgen download --model AbstractFramework/seedvr2-3b-8bit

mlxgen upscale \
  --model AbstractFramework/seedvr2-3b-8bit \
  --image-path input.png \
  --resolution 2x \
  --softness 0.25 \
  --metadata \
  --output input_2x.png
```

对 SeedVR2，整数 `--resolution` 表示目标短边，`2x`、`3x` 这类值表示缩放倍数。两种模式都保留源图宽高比。源图平滑区域有可见颗粒时，把 `--softness` 设在 `0.25` 到 `0.5`。小图走非分块 VAE；大图会自动使用分块 VAE 解码；`--vae-tiling` 只对图像运行强制分块 VAE 编码 / 解码。可复现的 5 倍 SeedVR2 图像对比，以及已验收的五秒埃菲尔铁塔 `1x` / `2x`、`3B` / `7B` 修复证明，见 [docs/upscaling.md](docs/upscaling.md)。

用 SeedVR2 修复一段真实的五秒验证视频：

```sh
mlxgen upscale \
  --model ByteDance-Seed/SeedVR2-3B \
  --video-path input.mp4 \
  --start-seconds 70 \
  --max-frames 149 \
  --resolution 1x \
  --softness 0.0 \
  --color-correction wavelet \
  --temporal-chunk-size 29 \
  --temporal-chunk-overlap 8 \
  --low-ram \
  --mlx-cache-limit-gb 8 \
  --metadata \
  --output restored.mp4
```

对视频输入，SeedVR2 默认保留源帧率，也默认保留对应的源音频片段。如果 MLX-Gen 不能证明拷贝过去的音频仍然安全对齐，这次运行会失败，而不会意外给出一段无声输出。只有你明确想要无声的修复版 MP4 时，才使用 `--drop-audio`。（这比 Wan 视频生视频的尽力拷贝音轨更严格：修复是保真约定；生成任务里混流失败不应丢掉已经生成好的视频。）公开的安全视频配置默认是 `1x`，会自动开启 `--low-ram`，并且拒绝放大后的视频输出，除非你显式传入 `--force-unsafe-video-memory`。已验收的五秒埃菲尔铁塔质量证明，以及已发布的法航音频拷贝证明，见 [docs/upscaling.md](docs/upscaling.md)。

`mlxgen upscale` 在图像和视频修复上都接受一个或多个显式种子，以及 `--auto-seeds N`。处理多个种子时，MLX-Gen 会自动追加 `_seed_<seed>`。一次 SeedVR2 调用处理多个源文件时，还会追加源文件名主干，让每份结果都有自己的路径。如果两个源文件的基本名相同，请保持 `--replace false` 或重命名输入；`--replace true` 时，可能互相覆盖的 SeedVR2 批次会被拒绝。

运行前查看模型能力：

```sh
mlxgen capabilities --model AbstractFramework/flux.2-klein-9b-8bit
```

能力描述的是路由约定：所选模型能分发哪些任务、图生图模式、图像数量和选项。要对精确的包做发布质量验收，使用：

```sh
mlxgen validation --model AbstractFramework/qwen-image-edit-2509-8bit
mlxgen validation --model bernini-r-1.3b
```

LoRA 支持按路由区分。做 LoRA 时，先在 `mlxgen capabilities` 里看 `supports_lora`、`lora_status` 和 `lora_validation_profile`，再用 `mlxgen download` 显式下载适配器，并使用为所选模型家族和路由训练的适配器。当前已有精确证明的包括：原版 Qwen Image Edit、Qwen Image Edit 2509/2511、基础 Qwen Image 文生图与 latent、Qwen Image 2512 文生图、精确的 `AbstractFramework/z-image-turbo-8bit` 文生图与 latent、FLUX.2 Klein 9B 编辑与多参考、FLUX.2 Klein base 4B 扩图、ERNIE Image Turbo 文生图与 latent，以及当前 Wan q8 公开视频路由。LoRA 指南还包含：当前 LightX2V `4` 步 A14B 与实用的原版 Wan 配置的耗时对比、同种子的无 LoRA 对照 Lightning 的 A/B 图、推荐的 `4` 步 Qwen 2512 与 Qwen Edit 2511 Lightning 示例、`240p` 对照 `480p` 的文生视频扫描，以及 `lightx2v/Wan2.2-Lightning` 可直接复制的下载与 T2V/I2V 命令（含稳定的 `repo:subdir/file.safetensors` 写法，以及下载后等价的本地绝对路径用法）。例如，FLUX.2-dev 的 LoRA 不会被 FLUX.2 Klein 接受。A/B 验证方法见 [docs/lora.md](docs/lora.md)。

创建本地 MLX-Gen 模型包，例如 8-bit 的 Qwen Image 包：

```sh
mlxgen prepare \
  --model Qwen/Qwen-Image \
  --path ./models/qwen-image-8bit \
  --quantize 8
```

`mlxgen generate` 不会下载缺失文件。缓存里没有时，MLX-Gen 会抛出明确的 `DownloadRequiredError`，并给出要执行的命令。`./models/<repo-name>` 下一份完整的本地 MLX-Gen 包，也可以满足对应的 Hugging Face 句柄，例如 `AbstractFramework/qwen-image-edit-2511-8bit`。

## 可复现示例

文档里有一套完整的、基于真实模型的飞船工作流：

- 文生图：生成雪地里的飞船。
- 图生图编辑：把它变成铅笔素描。
- 图生图编辑：让同一艘飞船在雪地里坠毁。
- 图生图多参考：把坠毁构图和铅笔素描风格组合起来。
- A14B 文生视频：生成飞船从冰雪星球起飞。
- A14B 图生视频：让生成出的飞船从源图中起飞。

精确命令和附带素材见 [docs/examples/spaceship-snow.md](docs/examples/spaceship-snow.md)。

![飞船各模式对照图](https://raw.githubusercontent.com/lpalbou/mlx-gen/main/docs/assets/examples/spaceship-snow/spaceship_modes_real_generation_contact_sheet.png)

Qwen Image Edit、Qwen Image Edit 2509/2511、FLUX.2 Klein 以及 latent 图生图模型的图像编辑对照图、命令日志和模型 / 包状态，见 [docs/edit-capabilities.md](docs/edit-capabilities.md)。其中也包括精确的基础 Qwen q8 结构化控制证明、精确的基础 Qwen q8 control-inpaint 证明，以及精确的 Z-Image Turbo q8 原生局部重绘证明。

完整的 Qwen 路由对照（`mlxgen` 能力 id 如何对应上游 Diffusers 的 Qwen 管线，以及已发布的证明在哪里），见 [docs/qwen-route-matrix.md](docs/qwen-route-matrix.md)。

latent 图生图、指令编辑、蒙版编辑 / 局部重绘、多参考合成、Qwen 结构化控制、生成式重构图和扩图的通俗说明与示例，见 [docs/image-edit-modes.md](docs/image-edit-modes.md)。

## 已发布模型

为 MLX-Gen 优化的量化与 BF16 变体发布在 [Hugging Face 上的 AbstractFramework 组织](https://huggingface.co/AbstractFramework)。当前已发布的包包括：

FLUX.2 Klein：

- `AbstractFramework/flux.2-klein-4b-4bit`
- `AbstractFramework/flux.2-klein-4b-8bit`
- `AbstractFramework/flux.2-klein-9b-4bit`
- `AbstractFramework/flux.2-klein-9b-8bit`
- `AbstractFramework/flux.2-klein-base-4b-4bit`
- `AbstractFramework/flux.2-klein-base-4b-8bit`
- `AbstractFramework/flux.2-klein-base-9b-4bit`
- `AbstractFramework/flux.2-klein-base-9b-8bit`

Qwen Image 与 Qwen Image Edit：

- `AbstractFramework/qwen-image-4bit`
- `AbstractFramework/qwen-image-8bit`
- `AbstractFramework/qwen-image-2512-4bit`
- `AbstractFramework/qwen-image-2512-8bit`
- `AbstractFramework/qwen-image-edit-4bit`
- `AbstractFramework/qwen-image-edit-8bit`
- `AbstractFramework/qwen-image-edit-2509-4bit`
- `AbstractFramework/qwen-image-edit-2509-8bit`
- `AbstractFramework/qwen-image-edit-2511-4bit`
- `AbstractFramework/qwen-image-edit-2511-8bit`

Z-Image、ERNIE 与 FIBO：

- `AbstractFramework/z-image-4bit`
- `AbstractFramework/z-image-8bit`
- `AbstractFramework/z-image-turbo-4bit`
- `AbstractFramework/z-image-turbo-8bit`
- `AbstractFramework/ernie-image-turbo-4bit`
- `AbstractFramework/ernie-image-turbo-8bit`
- `AbstractFramework/fibo-4bit`
- `AbstractFramework/fibo-8bit`

SeedVR2 超分：

- `AbstractFramework/seedvr2-3b-4bit`
- `AbstractFramework/seedvr2-3b-8bit`
- `AbstractFramework/seedvr2-7b-4bit`
- `AbstractFramework/seedvr2-7b-8bit`

SeedVR2 7B 也可以直接跑官方源模型 `ByteDance-Seed/SeedVR2-7B`（含 `seedvr2-7b-sharp` 检查点别名），或使用上面已发布的 q8/q4 包。已验证的 7B 源模型 / q8 / q4 配置见 [docs/upscaling.md](docs/upscaling.md) 与 [docs/quantization.md](docs/quantization.md)。

Wan2.2 视频：

- `AbstractFramework/wan2.2-ti2v-5b-diffusers-bf16`
- `AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit`
- `AbstractFramework/wan2.2-t2v-a14b-diffusers-bf16`
- `AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit`
- `AbstractFramework/wan2.2-i2v-a14b-diffusers-bf16`
- `AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit`

Bernini-R 使用官方渲染器 `ByteDance/Bernini-R-1.3B-Diffusers`，以及钉定版本的 Wan2.1 基础组件。目前没有已发布的 MLX-Gen 量化包：使用 `mlxgen download --model bernini-r-1.3b`，不要传 `--quantize`，详见 [Bernini-R 1.3B](docs/bernini.md)。

用 `mlxgen download --model <repo-id>` 缓存已发布模型，再把仓库 id 传给对应命令：图像 / 视频生成用 `mlxgen generate`，SeedVR2 超分用 `mlxgen upscale`。完整的当前包矩阵（源体积、优化包体积、任务覆盖和量化说明）见 [docs/quantization.md](docs/quantization.md)。

对 Wan2.2 TI2V-5B，已发布的 BF16 MLX-Gen 包是 21.2 GiB，上游源快照是 31.9 GiB。它主要是一份更小、可复用、与源等价的包，因为 MLX-Gen 加载 Wan transformer / VAE 权重时本来就用 BF16 运行精度。已发布的 q8 包是 16.9 GiB。自 2026-06-12 的运行精度修复之后，Wan q8 包会在加载时把所有 transformer block 的线性层反量化到 BF16，因此 q8 只节省存储和下载，运行时内存与 BF16 包相同。Wan TI2V-5B 的 q4 或混合 q4/q8 没有作为受支持的包发布。精确的基准配置和带日期的更正见 [docs/quantization.md](docs/quantization.md)。

## Wan A14B 实测

Wan A14B 在 128 GB 统一内存的 Apple M5 Max 上测量。发布卡片上的基准使用可重复的小型低内存运行，记录整进程的 Darwin 物理占用峰值、RSS、MLX 分配器峰值和生成时间。这些数字只对应表中的配置；内存和耗时会随分辨率、帧数、步数、缓存设置和图生视频条件而变化。

> 更正（2026-07-05）：下面的 q8 行早于 2026-06-12 的运行精度修复。自该修复起，Wan q8 包会在加载时把所有 transformer block 的线性层反量化到 BF16，因此运行时内存与 BF16 行相同（同一文生视频配置下重新测得 MLX 峰值为 27.8 GiB）。规划内存时按 BF16 包估算；q8 只节省磁盘和下载。

| 模型 | 包 | 磁盘 | 物理峰值 | 最大 RSS | MLX 峰值 | 耗时 | 配置 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Wan2.2 T2V-A14B | BF16 | 64.1 GiB | 33.0 GiB | 31.8 GiB | 27.7 GiB | 152.7 s | 384x224，33 帧，12 步，8 fps |
| Wan2.2 T2V-A14B | 混合 q8/BF16 | 39.5 GiB | 20.7 GiB | 19.5 GiB | 15.5 GiB | 154.8 s | 384x224，33 帧，12 步，8 fps |
| Wan2.2 I2V-A14B | BF16 | 64.1 GiB | 33.7 GiB | 31.8 GiB | 28.2 GiB | 228.2 s | 384x384，33 帧，12 步，8 fps |
| Wan2.2 I2V-A14B | 混合 q8/BF16 | 39.5 GiB | 21.5 GiB | 19.6 GiB | 15.9 GiB | 242.2 s | 384x384，33 帧，12 步，8 fps |

在这些运行里，混合 q8/BF16 相对 BF16 MLX-Gen 包大约少用 38% 的磁盘；自 2026-06-12 起，它不再降低运行时内存，也不再更快。各模型家族的量化细节和指标 JSON 见 [docs/quantization.md](docs/quantization.md)。0.18.11 也在 41 帧、15 步、目标 480x240 的配置上验证了已发布的 A14B q8 T2V/I2V 句柄，量化文档里保存了 MP4 与对照图。

## 生态

MLX-Gen 被用作本地 Apple Silicon 生成后端，服务于：

- [AbstractVision](https://github.com/lpalbou/abstractvision)：把本地与托管提供方的图像、视频生成能力抽象出来的直接集成层；
- [AbstractCore](https://abstractcore.ai/)：可以暴露由 AbstractVision 提供方支撑的 OpenAI 兼容端点，包括图像与视频能力；
- [AbstractFlow](https://github.com/lpalbou/abstractflow)：可视化工作流编写层，可以把生成式图像 / 视频节点和其他媒体、文本、智能体工作流组合在一起。

MLX-Gen 本身也是可用的独立命令行包，并且是为应用设计的：任务只使用磁盘上已有的模型文件，应用可以在加载权重之前查看支持的模式，进度回调让长时间运行可观察。

## 文档

- [入门](docs/getting-started.md)：安装、第一次运行、SeedVR2 超分、Wan 视频。
- [API 与 CLI](docs/api.md)：命令面、路由器行为、图生图模式、生成式重构图、按后端区分的扩图、SeedVR2 尺寸、Wan 视频尺寸、能力查询，以及 Python 入口。
- [Qwen-Image-2.1](docs/qwen-image-21.md) / [编辑](docs/qwen-image-21-editing.md)：本 fork 增加的第二代 Qwen 图像路由（文生图、latent 图生图、参考图编辑）。
- [Qwen3-VL caption](docs/qwen3-vl-caption.md)：本 fork 增加的图像到文本 / 提示词反推，使用本地 MLX Qwen3-VL 权重。
- [CosyVoice 3 TTS 与配音](docs/cosyvoice3-tts.md)：本 fork 增加的 `mlxgen tts` / `mlxgen dub`，以及隔离的准备包。
- [角色四视图实操](docs/character-four-view-operations.md)：实测的三视图 / 四视图配方，以及各路由的尺寸约定。
- [图像编辑模式](docs/image-edit-modes.md)：latent 图生图、参考编辑、多参考、生成式重构图和扩图在实际中的含义与示例。
- [Wan 视频](docs/wan-video.md)：实用的 Wan2.2 T2V/I2V 尺寸；带证明材料的普通与蒙版提示词引导 A14B 视频生视频；原生移植的 Wan2.1-VACE-1.3B 路由（参考图物体注入与学习到的蒙版条件）；实测的运动保真阶梯（强度与姿态保持）；更宽的 A14B 目标尺寸族；以及 M5 Max 上的 5 秒对比片段。
- [Bernini-R 1.3B](docs/bernini.md)：有序参考图生视频、参考图引导 / 仅源视频的视频编辑、分件下载、仅 BF16 的内存建议，以及可播放的证明材料。
- [示例工作流](docs/examples/spaceship-snow.md)：可复现的图像与视频命令。
- [图像超分](docs/upscaling.md)：SeedVR2 尺寸、已发布的 3B/7B q8/q4 包用法、主机安全的视频修复配置、已发布的五秒埃菲尔铁塔 `1x` 与 `2x` 3B/7B 验证包、可读的色调校正标签，以及 5 倍源图 / 输出对比。
- [图像编辑能力](docs/edit-capabilities.md)：图像编辑对照图、精确的模型 / 包状态，以及命令日志。
- [重构图与扩图](docs/reframe-outpaint.md)：`--reframe-padding` 与 `--outpaint-padding` 路由，任意单边或组合边的按边填充，`--outpaint-fill` 条件画布模式与填充建议，蒸馏权重与基础权重上严格的 FLUX.2 Klein 扩图，以及精确的 Qwen 与 FLUX 路由证明链接。
- [模型管理](docs/model-management.md)：下载、准备，并从本地模型文件运行。
- [量化](docs/quantization.md)：q8 / q4 / BF16 策略与测量。
- [Python 集成](docs/python-integration.md)：按路由解析的运行时加载、统一 `mlxgen generate` 家族的串行多输出复用、SeedVR2 的直接模型边界、进度回调，以及 AbstractVision / AbstractCore 说明。
- [常见问题](docs/faq.md)：反复出现的问题、图生图模式选择、SeedVR2 尺寸、Qwen 编辑变体、负向提示词、重构图 / 扩图、Wan 分辨率，以及使用限制。
- [故障排查](docs/troubleshooting.md)：常见的安装与运行失败。
- [致谢（英文原文）](ACKNOWLEDGEMENTS.md)：上游 mflux 与模型社区的署名。

## 许可证

MLX-Gen 使用 MIT 许可证。模型权重仍受其原始许可证与访问条款约束。

## 致谢

本仓库（[jiangzhenguo/mlx-gen](https://github.com/jiangzhenguo/mlx-gen)）是个人 fork，建立在已有开源工作之上。日常开发与合并只面向本仓库的 [`main`](https://github.com/jiangzhenguo/mlx-gen/tree/main)。

- **[lpalbou/mlx-gen](https://github.com/lpalbou/mlx-gen)**（Laurent-Philippe Albou / AbstractVision）：本仓库直接 fork 自该项目。统一的 `mlxgen` 命令面、先下载再生成的模型准备流程、按模型区分的量化策略，以及大量图像 / 视频路由，都来自上游维护。没有这份基础，就没有这里的 Qwen-Image-2.1、Qwen3-VL caption 与 CosyVoice 3 等本地扩展。
- **[mflux](https://github.com/filipstrand/mflux)**（Filip Strand 与原贡献者）：上游 MLX-Gen 本身 fork 自 mflux。当前代码库的主要功劳属于原作者；`mlx-gen` 包名和 `mlxgen` 命令面是在这份基础上演化出来的。fork 之后的 MLX-Gen 改动由 Laurent-Philippe Albou / AbstractVision 维护。
- **[MLX](https://github.com/ml-explore/mlx)**、Hugging Face Hub，以及 Diffusers / Transformers 模型格式和更广泛的开源 Python、Apple Silicon 生态：提供了本项目所依赖的运行时与分发方式。

本项目路由或准备的模型来自这些上游社区与模型所有者，权重、许可证、访问限制和署名要求仍以原模型所有者及源仓库为准：

- Black Forest Labs：FLUX.1 与 FLUX.2 Klein 模型家族，包括受支持的 fill、depth、Redux、Kontext、ControlNet 与 Klein 路由。
- 通义千问 / 阿里巴巴：Qwen Image、Qwen Image Edit、Qwen Image Edit 2509 与 Qwen Image Edit 2511。
- Wan-AI / 阿里巴巴通义：Wan2.2 TI2V-5B 与 Wan2.2 A14B T2V/I2V 视频模型。
- 通义实验室（Tongyi-MAI）：Z-Image 与 Z-Image Turbo。
- 百度：ERNIE Image Turbo。
- Bria：FIBO 及相关 FIBO 图像生成权重。当前版本没有把 FIBO Edit 作为受支持的统一编辑路由暴露出来。
- Prism ML：Bonsai Image 的三值 / 二值 MLX 权重。
- 字节跳动 / SeedVR2：官方 SeedVR2 图像超分模型家族。
- InstantX、JasperAI、CatVTON 与 Apple Depth Pro：继承自 mflux 路由的可选控制、超分、填充、虚拟试衣与深度相关组件。

AbstractFramework Hugging Face 组织下发布的优化变体，仍是各自源仓库的 MLX-Gen / mflux 保存权重布局的派生作品。

更完整的英文署名见 [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)。
