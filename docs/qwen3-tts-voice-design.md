# Qwen3-TTS voice design (text → timbre)

`mlxgen voice-design` adds **Qwen3-TTS VoiceDesign** to mlx-gen: you describe a voice in plain language and the model speaks with it. No reference audio, no speaker list, no cloning step.

```bash
mlxgen prepare --model qwen3-tts-voice-design
mlxgen voice-design \
  --voice "一位三十多岁的女性，声音温暖沉稳，语速偏慢" \
  --text "我爱你" --output iloveyou.wav
```

## Why this is a separate command

CosyVoice 3 (see [CosyVoice3 TTS](cosyvoice3-tts.md)) can only imitate a voice that already exists as a recording: `--ref-audio` is mandatory. Its `instruct2` mode adds style control (emotion, pace, accent) on top of a cloned voice, but the *timbre itself* still comes from the clip.

Voice design is the other half of the problem: there is no clip, only a description. Qwen3-TTS VoiceDesign takes the description as its speaker condition and generates both the timbre and the delivery.

| | CosyVoice 3 (`mlxgen tts`) | Qwen3-TTS VoiceDesign (`mlxgen voice-design`) |
|---|---|---|
| Where the voice comes from | a reference clip | a text description |
| Style control | `--instruct` on top of the clone | part of the description |
| Languages | 9 + 18 Chinese dialects | 10 (`--language`, `auto` by default) |
| Output | 24 kHz | 24 kHz |
| Runtime | isolated torch venv (CPU) | MLX, in-process (Metal) |
| Size | ~5.4 GB | ~4.5 GB |

## The model

- Weights: [`mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16) — the MLX conversion of the upstream [`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) release (Apache-2.0).
- Structure: a 1.7B talker (Qwen3 backbone) plus a 682 MB 12 Hz speech-tokenizer decoder; output is 24 kHz mono.
- Inference code: [`mlx-audio`](https://github.com/Blaizzy/mlx-audio) ≥ 0.6.4, which mlx-gen depends on directly.

Unlike CosyVoice 3 there is **no isolated environment**: `mlx-audio` is pure MLX, so the synthesis worker runs as a subprocess of the current interpreter and never touches torch. That also means no version trap — mlx-gen's `transformers>=5` and CosyVoice 3's `transformers==4.51.3` do not collide here, because the voice-design path does not use transformers at all.

## Preparation

```bash
mlxgen prepare --model qwen3-tts-voice-design --path ./models/qwen3-tts-voice-design
# or
mlxgen prepare --model qwen3-tts-voice-design   # default: $MFLUX_CACHE_DIR/qwen3-tts-voice-design
```

The package is deliberately thin — there is no vendored inference code to keep in sync:

```
<path>/
  model/      MLX weights (talker + 12 Hz speech tokenizer)
  manifest.json
```

`prepare` downloads the weights in resumable 8 MB chunks, fetched by several connections at once, from two mirrors (Hugging Face and the ModelScope mirror of the same repository). Two details make this work on restricted networks:

- **Mirrors are probed, not assumed.** Only the ModelScope mirror answers range requests on the Hugging Face CDN path — the CDN answers a range request with the *whole* file, which would silently turn the chunked download into 3.8 GB re-fetched per chunk. Every mirror is probed once and the ones that ignore ranges are dropped.
- **Parallel chunks.** A single stream on a throttled link gets a fraction of the link's bandwidth; parallel range requests reach the real bandwidth (measured here: 80 KB/s on one connection versus 1.1 MB/s with 20 connections on the same link).

An interrupted download continues where it stopped — the chunk map lives next to the file — so re-running the same `prepare` command is cheap. `MLXGEN_DOWNLOAD_WORKERS` overrides the connection count.

Use `--source hf`, `--source modelscope` or `--source auto` to pick the mirror order; `--force` recreates the package.

After the first successful prepare, `mlxgen voice-design` runs fully offline.

## Writing a good voice description

The description is the speaker condition, so the more concrete it is, the closer the result. The model reads ordinary descriptive language — it is not a keyword list:

```bash
mlxgen voice-design \
  --voice "a warm female voice in her thirties, slightly breathy, speaking slowly and calmly" \
  --text "晚上好，欢迎回来。" --language chinese --output evening.wav
```

Attributes that reliably move the output: age range, gender, pitch (high/low), pace, emotional tone, and vocal texture (husky, clear, breathy, bright). Chinese descriptions work well too, and the upstream model card's own examples are Chinese.

Two behaviours worth knowing:

- **`--language` defaults to `auto`**, which lets the model pick the language from the text. Set it explicitly (`chinese`, `english`, `japanese`, `korean`, `german`, `french`, `russian`, `portuguese`, `spanish`, `italian`) when the text is short or ambiguous. `auto` was verified to detect English from an English sentence and speak it in English.
- **The description shapes the delivery as well as the timbre.** "speaking angrily" is not just a voice colour; the model will deliver the line angrily. Keep the emotion in a separate sentence if you want a neutral voice with an emotional read: `"a calm, low female voice"` for the voice and add the emotion to the text itself.

## Voice profiles: one narrator, many calls

A description is a long sentence that a project has to keep stable — episode 3's narrator has to sound like episode 1's. A **voice profile** gives that sentence a name:

```json
// mlxgen-voice-profiles.json
{
  "narrator-male": {
    "description": "一位沉稳低沉的男性旁白，语速平稳，不带情绪",
    "language": "chinese",
    "seed": 42
  },
  "episode-3": "a bright young female voice, energetic and clear"
}
```

```bash
mlxgen voice-design --voice-profile narrator-male --text "第三集开始了。" -o ep3.wav
mlxgen dub --video-path clip.mp4 --voice-profile narrator-male --text "第三集开始了。" -o ep3.mp4
mlxgen voice-design --list-voice-profiles
```

- `description` is the only required field. `language`, `temperature` and `seed` are optional, and **a value given on the command line always wins over the profile's** — a profile can pin the voice while one call overrides the language.
- Profile files are searched in this order, first hit wins: `--voice-profile-file PATH`, `$MLXGEN_VOICE_PROFILES`, `./mlxgen-voice-profiles.json`, `~/.config/mlxgen/voice-profiles.json`.
- Six starters ship built in (`narrator-male`, `narrator-female`, `documentary`, `news-anchor`, `warm-female`, `energetic-young`); any profile file overrides or extends them.
- `--voice-profile` cannot be combined with `--voice` / `--voice-file` — the profile *is* the description.
- The metadata sidecar records `voice_profile` and `voice_profile_source`, so a rendered clip can be traced back to the profile it came from.

## Video dubbing with a designed voice

`mlxgen dub` keeps CosyVoice 3 as its default narrator and accepts a designed voice as an opt-in:

```bash
mlxgen dub \
  --video-path clip.mp4 \
  --voice-design "a deep, warm male narrator voice, steady and unhurried" \
  --text "红狐在雪地里慢慢走过，脚步很轻。" \
  --output narrated.mp4 --metadata
```

With `--voice-design` the reference-voice options are not needed (`--ref-audio`, `--ref-text`, `--instruct` are ignored); `--language` and `--temperature` apply. The mux contract is unchanged: the video stream is copied, only the audio is re-encoded, and `--mode mix` still blends the narration over the original soundtrack.

The metadata sidecar records which narrator produced the audio:

```json
{
  "task": "video-dub",
  "voice_backend": "qwen3-tts-voice-design",
  "voice_description": "a deep, warm male narrator voice, steady and unhurried",
  "language": "auto"
}
```

## Command surface

```bash
mlxgen voice-design --help

  --voice TEXT             description of the wanted voice (required unless --voice-profile)
  --voice-file PATH        read the description from a file
  --voice-profile NAME     use a named voice profile instead of --voice
  --voice-profile-file PATH  profile JSON file (default: ./mlxgen-voice-profiles.json)
  --list-voice-profiles    print the available profiles and exit
  --text TEXT | --text-file PATH   what to say ("-" reads stdin)
  --language NAME          auto | chinese | english | ... (default: auto)
  --seed N                 random seed (default 42)
  --temperature FLOAT      sampling temperature (model default when omitted)
  --output PATH            output wav (default speech.wav)
  --model-path PATH        prepared package directory
  --metadata               write <output>.json sidecar
  --json-events            machine-readable runtime events on stdout
```

## Long audio: compose several segments

One generation with a long run of "嗯……啊……" drifts into a babble loop and swallows the words at the end — the tail becomes a hum instead of the line you wrote. So build longer audio from separate segments, each with its own description and gain:

```json
// manifest.json
{
  "language": "chinese",
  "seed": 42,
  "peak": 0.92,
  "segments": [
    {
      "text": "嗯……啊……嗯……啊……嗯……啊……",
      "voice": "一位三十五岁左右的女性，声音低沉磁性；慵懒缓慢地哼吟，语速很慢，气息感很重",
      "gain_db": 0
    },
    {
      "text": "舒服死了！",
      "voice": "同一位女性，情绪爆发，用尽全力大喊出来，音量很大",
      "gain_db": 7
    }
  ]
}
```

```bash
python tools/compose_voice.py manifest.json \
  --model-path out/qwen3-tts-voice-design \
  --output out/final.wav --metadata
```

Each segment is synthesized on its own, the leading/trailing silence is trimmed, `gain_db` sets the level relative to the others, a short gap is inserted between segments and the result is peak-normalised (no clipping). Per-segment durations and RTF land in the metadata sidecar.

Per-segment keys: `text` and `voice` (required), `gain_db` (default 0), `gap_after_s` (default 0.12), plus optional `temperature`, `seed` and `language`.

## Measured

Apple Silicon, 64 GB (M-series, MLX on Metal), `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`.

| What | Value |
| --- | --- |
| Package size | 4515 MB (3833 MB talker + 682 MB 12 Hz speech tokenizer + tokenizer files) |
| Model load | 2.0–3.1 s (5.6 s on the first run, when the decoder graph is still cold) |
| Synthesis | RTF **0.90** (7.44 s of English) and **1.00** (6.56 s of Chinese) — roughly real time |
| Short clip | RTF 5.4 for a 0.96 s clip: the fixed cost of the first decode dominates, so measure RTF on a sentence, not a word |
| Output | 24 kHz, mono, 16-bit PCM WAV |
| Peak memory | well inside 64 GB; the weights are 4.5 GB and the talker is loaded in bf16 |

Content check (Whisper `tiny` as the ASR oracle, Chinese):

```
request   晚上好，欢迎回来。今天的天气很好，我们一起去公园散步吧。
transcript 晚上好 欢迎回来今天的天气很好我们一起去公园散步吧
```

English, verbatim:

```
request   The lighthouse keeper watched the storm roll in over the northern bay, and counted
          the seconds between the lightning and the thunder.
transcript The lighthouse keeper watched the storm roll in over the northern bay, and counted
          the seconds between the lightning and the thunder.
```

The description moves the timbre, not just the wording — median fundamental frequency of the
generated audio:

| Description | Median F0 |
| --- | --- |
| "一位三十多岁的女性，声音温暖沉稳，语速偏慢" | 198 Hz |
| "a deep, warm male narrator voice, steady and unhurried" | 86 Hz |

Download throughput on a throttled link (the reason the downloader is built the way it is):

| Connections | Throughput |
| --- | --- |
| 1 | ~80 KB/s |
| 6 | 792 KB/s |
| 20 | 1161 KB/s |
| 32 | 956 KB/s |

Determinism, checked on the same machine:

- The same description with the same `--seed` produces **byte-identical** WAV output (same MD5), so a pinned profile reproduces a narrator exactly.
- A different `--seed` produces different audio, and `--temperature` changes it too (0.7 and 1.4 gave 3.84 s and 3.52 s of audio for the same sentence).

## Limits

- The model speaks the 10 languages listed above; a description in a language the model does not know will still produce audio, but the timbre will drift towards the closest match.
- Voice design is *sampling*: the same description with the same seed is reproducible, but two different descriptions that mean the same thing give noticeably different voices. Keep the description as a stable config value if a project needs a consistent narrator.
- The dub command still never calls the video model — `mlxgen dub` only narrates an existing MP4 (see [MiniMax-H3](minimax-h3.md) for the H3 side of the pipeline).
