# CosyVoice3 text-to-speech and video dubbing

`mlxgen tts` and `mlxgen dub` add the [FunAudioLLM CosyVoice 3 (2512)](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) zero-shot voice-cloning TTS to mlx-gen, plus a standalone command that chains it with MiniMax-H3 video output. CosyVoice 3 has no preset voices — you always pass a reference clip; it is multilingual (9 languages) and supports instruction-conditioned synthesis for style control.

## Why a separate command surface

`mlxgen tts` and `mlxgen dub` are first-class top-level commands, not routes through `mlxgen generate`. They need their own argument shape (reference voice, transcript, instruction) and the audio model has its own env and stack (see below), so it would have contaminated the image/video router. The router continues to dispatch everything else; only the CosyVoice 3 path goes through the audio commands.

## The isolated worker

CosyVoice 3's official inference code is a separate package on its own Python 3.11 PyTorch stack:

- `transformers>=4.51` for the Qwen2-derived speech-token LLM (mlx-gen itself pins `transformers>=5.0` for the image/video models, so the two would clash).
- `torchaudio`, `onnxruntime`, `openai-whisper`, `tiktoken`, `inflect`, `wetext` for the speech tokenizer, ONNX decoder, English number normalization, and Chinese text normalization.

`mlxgen prepare --model cosyvoice3` therefore builds a self-contained package at the path you choose:

```
<path>/
  model/        official FunAudioLLM/Fun-CosyVoice3-0.5B-2512 weights (~5.4 GB)
  runtime/      official cosyvoice inference code (vendored from the HF space)
  venv/         a fresh Python 3.11 environment with the inference deps
  manifest.json
```

`mlxgen tts` and `mlxgen dub` spawn `<path>/venv/bin/python` on a tiny worker module that lives in the mlx-gen package; the worker speaks the standard mlx-gen JSONL runtime-event contract over its stdout, so `--json-events` works exactly as it does for `mlxgen generate`. mlx-gen's own environment never loads torch or CosyVoice 3 code.

## Preparation

```bash
mlxgen prepare --model cosyvoice3 --path ./models/cosyvoice3
# or
mlxgen prepare --model cosyvoice3         # default: $MFLUX_CACHE_DIR/cosyvoice3
```

The first run downloads the weights and inference code, creates the isolated venv, and installs the worker requirements. By default the download tries the Hugging Face mirror first and falls back to the ModelScope mirror (CosyVoice 3 is mirrored at `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` on both, and the ModelScope resolve endpoint answers Range requests so an interrupted large file resumes from where it stopped). Use `--source hf`, `--source modelscope`, or `--source auto` to control the choice; `--with-rl` also downloads the RL-tuned LLM variant (`llm.rl.pt`, +2 GB); `--force` recreates an existing package.

After the first successful prepare, every `mlxgen tts` and `mlxgen dub` invocation runs fully offline.

## `mlxgen tts`

Synthesizes a single WAV.

```bash
# zero-shot voice clone (reference transcript required)
mlxgen tts \
  --ref-audio voice.wav --ref-text "参考音频的文字" \
  --text "你好，世界" --output hello.wav

# cross-lingual clone (no transcript of the reference)
mlxgen tts --ref-audio voice.wav --text "Hello world" --output hello.wav

# instructed style — requires a reference voice
mlxgen tts \
  --ref-audio voice.wav --instruct "用温柔亲切的语气说" \
  --text "今天天气真好" --output hello.wav
```

Options of note:

- `--model-path <path>` — prepared package (default the platformdirs cache).
- `--text` / `--text-file` / `--text -` — text to speak.
- `--ref-audio <wav>` — required; 3–30 s of clean speech.
- `--ref-text <txt>` — exact transcript of `--ref-audio` for zero-shot cloning.
- `--instruct <txt>` — style instruction; selects the instructed mode (Chinese examples like "用温柔亲切的语气说", English like "Speak quickly and excitedly").
- `--speed <float>` — speech rate multiplier (default `1.0`).
- `--seed <int>` — random seed (default `42`).
- `--output <wav>` — output path (default `speech.wav`).
- `--metadata` — write a `<output>.json` sidecar with mode, ref paths, durations, and RTF.
- `--json-events` — emit the standard mlx-gen JSONL runtime events on stdout.

The `cross_lingual` mode is selected automatically when only `--ref-audio` is given; it does not need the reference transcript and lets CosyVoice 3 imitate the speaker in a different language.

## `mlxgen dub` — H3 video + CosyVoice 3 narration

The chaining command you asked for: take an existing MP4 (typically a MiniMax-H3 output), synthesize narration with CosyVoice 3, and mux it in. The video is stream-copied — only the audio is re-encoded.

```bash
# replace the (silent or H3-generated) soundtrack
mlxgen dub \
  --video-path fox.mp4 \
  --ref-audio voice.wav --ref-text "参考文字" \
  --text "清晨，一只红狐在雪地里悄悄走过。" \
  --output fox_narrated.mp4

# mix narration over an H3 clip that has a soundscape
mlxgen dub \
  --video-path city.mp4 \
  --ref-audio voice.wav --instruct "用平静的旁白语气" \
  --text "城市在夜风中沉睡。" \
  --mode mix --orig-volume 0.15 \
  --output city_narrated.mp4
```

`mlxgen dub` accepts every `mlxgen tts` argument (text, reference, instruction, speed, seed) plus:

- `--video-path <mp4>` — source video (required).
- `--mode replace|mix` — `replace` swaps the audio track, `mix` blends the narration over the original audio (default `replace`).
- `--orig-volume <float>` — original-audio volume in mix mode (default `0.15`).
- `--tts-volume <float>` — narration volume (default `1.0`).
- `--keep-wav` — keep the synthesized narration as `<output>.wav` next to the video.
- `--output <mp4>` — output (default `<video>_dubbed.mp4`).
- `--metadata` / `--json-events` — same sidecar and event contract as `mlxgen tts`.

The mux is performed with `ffmpeg` (must be on PATH; install with `brew install ffmpeg`). The narration is padded with silence to the video duration and the container is clipped at the video end, so a longer narration never extends the video. The ffmpeg argv builder is unit-tested.

## Hardware and runtime

The CosyVoice 3 inference runs on CPU — the official code does not move tensors to CUDA explicitly, and the PyTorch + ONNX path is what mlx-gen ships. Measured on an Apple M5 Max (torch 2.7.1, transformers 5.x, CPU):

| text length | audio out | wall time | RTF |
|---|---|---|---|
| 4 chars ("大家好。") | 2.8 s | 5.5 s | 1.95 |
| ~20 chars | 8.6 s | 15.3 s | 1.78 |

Model load is ~2 s once the package is prepared. The on-disk footprint is ~5.4 GB (LLM 2 GB fp32, flow DiT 1.3 GB, speech tokenizer 969 MB, campplus 28 MB, CosyVoice-BlankEN 988 MB); the resident working set is well within a 128 GB Mac's budget. CPU speed varies with system load — we observed RTF between 1.3 and 2.0 across runs.

Verified end-to-end on macOS arm64: `mlxgen tts` zero-shot cloning (Chinese), `mlxgen dub` on a 3 s MP4 (video stream-copied, narration padded/clipped to the video length, metadata sidecar written). The upstream code needed eight small patches to run without CUDA (hardcoded `.cuda()` / `'cuda'` device strings and a bf16/fp32 dtype mix in the LLM); `mlxgen prepare --model cosyvoice3` applies them automatically and idempotently — the patch table fails loudly with a refresh hint if the upstream code drifts.

## Limitations

- CosyVoice 3 ships no preset speakers; a `--ref-audio` is always required.
- Synthesis is offline. The venv dependency install and the weight download are the only network steps; after `mlxgen prepare` every call is local.
- The audio is always the worker's output — `mlxgen dub` does not call MiniMax-H3 itself. Generate the video with `mlxgen generate --model minimax-h3-turbo-544p ...`, then dub it with `mlxgen dub --video-path ...`. Pipeline these with shell `&&` or a small `workflow` script when you want a one-shot "video + voiceover" run.
- The reference voice is consumed once per call. To reuse the same voice across many calls, point `--ref-audio` at the same file; the worker is single-threaded per call but multiple `mlxgen tts` runs can be parallelized at the shell level.
- The `matcha` training losses vendored with the official CosyVoice space are stubbed by `mlxgen prepare` when the GitHub-mirrored `third_party/Matcha-TTS/matcha` subtree cannot be fetched (it is training-only code, never called at inference; this just keeps the import graph clean). If the real subtree is present, the prepare step keeps it untouched.
