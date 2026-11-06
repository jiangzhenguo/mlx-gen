# CosyVoice3 text-to-speech and video dubbing

`mlxgen tts` and `mlxgen dub` add the [FunAudioLLM CosyVoice 3 (2512)](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) zero-shot voice-cloning TTS to mlx-gen, plus a standalone command that chains it with MiniMax-H3 video output. CosyVoice 3 has no preset voices — you always pass a reference clip; it is multilingual (9 languages) and supports instruction-conditioned synthesis for style control.

## Why a separate command surface

`mlxgen tts` and `mlxgen dub` are first-class top-level commands, not routes through `mlxgen generate`. They need their own argument shape (reference voice, transcript, instruction) and the audio model has its own env and stack (see below), so it would have contaminated the image/video router. The router continues to dispatch everything else; only the CosyVoice 3 path goes through the audio commands.

## The isolated worker

CosyVoice 3's official inference code is a separate package on its own Python 3.11 PyTorch stack:

- `transformers==4.51.3` for the Qwen2-derived speech-token LLM (mlx-gen itself pins `transformers>=5.0` for the image/video models, so the two would clash — and 5.x would additionally break CosyVoice 3's inference, see the version notes below).
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

## `--ref-text` must be the reference clip's real transcript

This is the one mistake that produces *plausible-looking but unintelligible* audio instead of an error, so it is worth stating explicitly.

`--ref-text` is the transcript of **the audio in `--ref-audio`** — not the sentence you want synthesized. CosyVoice 3 conditions the speech-token LLM on `sos → [reference transcript + target text] → task_id → reference speech tokens`; when the transcript does not match the clip, the model is asked to continue audio that says something else. It then never reaches the end of its utterance: the LLM keeps sampling until it hits the `max_token_text_ratio × text length` cap, so the result is roughly 20× too long, drifts, and is unusable.

The symptom is a clip whose length is pinned to a round number for short inputs (3 characters → 60 tokens → 2.8 s) instead of the ~1 s you expect.

Worked example using the reference clip that ships with the official demo space:

```bash
mlxgen tts \
  --ref-audio zero_shot_prompt.wav \
  --ref-text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
  --text "我爱你" --output iloveyou.wav
# -> 0.96 s of audio, RTF ~3.9
```

Swap the two arguments and you get 2.8 s of noise instead.

If you do not have the reference transcript, use `--ref-audio` alone — the command then selects the `cross_lingual` mode, which needs no transcript.

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

### Dubbing with a designed voice instead of a cloned one

`mlxgen dub` can narrate with a *designed* voice rather than a cloned one — no reference clip, no reference transcript:

```bash
mlxgen dub \
  --video-path fox.mp4 \
  --voice-design "一位沉稳低沉的男性旁白，语速平稳" \
  --text "清晨，一只红狐在雪地里悄悄走过。" \
  --output fox_narrated.mp4
```

That switches the narrator to Qwen3-TTS VoiceDesign, which runs on MLX in the current environment instead of the isolated CosyVoice 3 venv — see [Qwen3-TTS voice design](qwen3-tts-voice-design.md) for the model, the language list and the measured numbers. `--voice-profile <name>` picks a named profile so every clip in a series shares one narrator. CosyVoice 3 remains the default, so existing pipelines keep their voice.

## Hardware and runtime

The CosyVoice 3 inference runs on CPU — the official code hardcodes CUDA in a few places (mlx-gen patches those out) and the PyTorch + ONNX path is what mlx-gen ships. Measured on an Apple M5 Max (torch 2.7.1, transformers 4.51.3, CPU):

| text | audio out | wall time | RTF |
|---|---|---|---|
| "我爱你" (3 chars) | 0.96 s | 3.7 s | 3.9 |
| "我爱你，今天也要开开心心的。" (14 chars) | 3.28 s | 6.4 s | 1.9 |
| "你好呀，很高兴认识你。" (11 chars) | 2.76 s | 6.2 s | 2.3 |

Model load is ~2 s once the package is prepared; the first call in a session pays for it, later calls in the same process reuse the loaded model. The on-disk footprint is ~5.4 GB (LLM 2 GB fp32, flow DiT 1.3 GB, speech tokenizer 969 MB, campplus 28 MB, CosyVoice-BlankEN 988 MB); the resident working set is well within a 128 GB Mac's budget. RTF scales with the utterance length — short clips are dominated by the fixed model-load and first-token overhead, longer ones approach 1.3–2.0.

Verified end-to-end on macOS arm64: `mlxgen tts` zero-shot cloning (Chinese, transcripts cross-checked with Whisper), `mlxgen dub` on a 3 s MP4 (video stream-copied, narration padded/clipped to the video length, metadata sidecar written).

The upstream code needed eight small patches to run without CUDA (hardcoded `.cuda()` / `'cuda'` device strings); `mlxgen prepare --model cosyvoice3` applies them automatically and idempotently — the patch table fails loudly with a refresh hint if the upstream code drifts.

## Dependency versions are pinned on purpose

The worker venv follows the upstream Space's `requirements.txt` closely, because the deviations fail *silently* rather than loudly:

- **`transformers==4.51.3`** — on 5.x the Qwen2 speech-token backbone computes different attention/mask semantics, the LLM emits garbage speech tokens, and `mlxgen tts` still exits 0 with a WAV file full of noise. This is the single most important pin.
- **`numpy<2`** — `onnxruntime==1.18.0` is compiled against numpy 1.x and fails to import under numpy 2.
- **`wetext==0.1.8`** — 0.0.4 downloads its FST models from ModelScope while constructing `Normalizer()`, which breaks offline use and read-only home directories. 0.1.x ships them in the wheel.
- **`diffusers==0.29.0`**, `librosa==0.10.2`, `soundfile==0.12.1`, `onnxruntime==1.18.0`, `conformer==0.3.2`, `setuptools<80` — the versions the upstream Space pins. `setuptools<80` keeps `pkg_resources`, which cosyvoice's dataset processor imports transitively.

The worker also points `MODELSCOPE_HOME`/`MODELSCOPE_CACHE` at `<package>/cache/modelscope` so that any model that does want a ModelScope download lands inside the package instead of the user's home directory.

## Limitations

- CosyVoice 3 ships no preset speakers; a `--ref-audio` is always required.
- Synthesis is offline. The venv dependency install and the weight download are the only network steps; after `mlxgen prepare` every call is local.
- The audio is always the worker's output — `mlxgen dub` does not call MiniMax-H3 itself. Generate the video with `mlxgen generate --model minimax-h3-turbo-544p ...`, then dub it with `mlxgen dub --video-path ...`. Pipeline these with shell `&&` or a small `workflow` script when you want a one-shot "video + voiceover" run.
- The reference voice is consumed once per call. To reuse the same voice across many calls, point `--ref-audio` at the same file; the worker is single-threaded per call but multiple `mlxgen tts` runs can be parallelized at the shell level.
- The `matcha` package vendored with the official CosyVoice space is fetched from the same HF snapshot. If the `third_party/Matcha-TTS/matcha` subtree cannot be fetched, `mlxgen prepare` writes import stubs for the two names the inference path needs (`matcha.utils.audio.mel_spectrogram`, `matcha.models.components.flow_matching.BASECFM`) plus the training-only loss shims; the real diffusers-based `transformer`/`decoder` modules are still required for the flow DiT to load its state dict.
