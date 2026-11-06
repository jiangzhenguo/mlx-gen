# ACE-Step 1.5 music generation

`mlxgen music` adds [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) text-to-music to mlx-gen. You describe a song in a prompt ("一首中文流行歌曲，钢琴伴奏，副歌激昂"), optionally supply the lyrics, and the model renders a finished track with vocals — or an instrumental when you ask for one.

ACE-Step 1.5 is a Chinese-first music model: the prompt, the lyrics and the sung voice are all first-class, and `--vocal-language` (default `zh`) selects the language of the vocal line.

## Why a separate command surface

`mlxgen music` is a first-class top-level command, not a route through `mlxgen generate`. Music generation has its own argument shape (prompt, lyrics, duration, bpm, key scale, instrumental) and its own runtime stack (see below), so routing it through the image/video router would have contaminated it. The router keeps dispatching everything else; only the ACE-Step path goes through the music command.

The standalone entry point `mlxgen-music` is equivalent to `mlxgen music` and is handy in scripts.

## The isolated worker

ACE-Step's official inference stack pins `transformers>=4.51,<4.58` and `torch>=2.9.1`, which cannot share mlx-gen's own environment (mlx-gen requires `transformers>=5` for the image/video models). The upstream project ships an MLX backend for the DiT decoder, the audio VAE and the planner LM, so on Apple Silicon the isolated worker still runs natively on MLX.

`mlxgen prepare --model ace-step` therefore builds a self-contained package at the path you choose:

```
<path>/
  model/        official ACE-Step/Ace-Step1.5 weights (~10 GB)
  runtime/      official ace-step/ACE-Step-1.5 source tree (pinned commit)
  venv/         a fresh Python 3.11 environment with the inference deps
  manifest.json
```

`mlxgen music` spawns `<path>/venv/bin/python` on a tiny worker module that lives in the mlx-gen package; the worker speaks the standard mlx-gen JSONL runtime-event contract over its stdout, so `--json-events` works exactly as it does for `mlxgen generate`. mlx-gen's own environment never loads torch or the ACE-Step code.

## Preparation

```bash
mlxgen prepare --model ace-step --path ./models/ace-step
# or
mlxgen prepare --model ace-step         # default: $MFLUX_CACHE_DIR/ace-step
```

The first run downloads the weights and the upstream runtime tree, creates the isolated venv and installs the worker requirements. By default the download tries the Hugging Face mirror first and falls back to the ModelScope mirror (the weights bundle is mirrored at `ACE-Step/Ace-Step1.5` on both). Use `--source hf`, `--source modelscope` or `--source auto` to control the choice; `--force` recreates an existing package.

The weights bundle is ~10 GB in total and ships four components in separate sub-directories: the DiT decoder (`acestep-v15-turbo`, 4.8 GB), the audio VAE (`vae`, 0.3 GB), the Qwen3 text encoder (`Qwen3-Embedding-0.6B`, 1.2 GB) and the 5 Hz planner LM (`acestep-5Hz-lm-1.7B`, 3.7 GB). The first three are required; the planner LM is optional.

`--no-lm` skips the planner LM weights (~3.7 GB) and the isolated environment installs a smaller dependency set. The trade-off is described in the next section.

After the first successful prepare, every `mlxgen music` invocation runs fully offline.

## Lyrics: the planner LM writes them, or you do

The planner LM is the part of ACE-Step that turns a theme into a song: it writes the lyrics and the song blueprint (structure, section tags, style tags) that the DiT decoder renders. Two modes:

- **Default (`--use-lm`)** — you pass a prompt only. The planner LM writes the Chinese lyrics from the prompt and lays out the song (upstream's "Simple Mode"). This is the "describe the song and get a song" path, and it is the only mode that can work without lyrics.
- **`--no-use-lm`** — the planner LM stays out of the loop. You must supply the lyrics yourself with `--lyrics` or `--lyrics-file`; otherwise there is nothing to sing and the command fails before the model is loaded. `--instrumental` is the exception: an instrumental track has no vocals, so it needs no lyrics.

The blueprint still improves a song even when the lyrics are yours, so `--use-lm` is the default even with `--lyrics` given.

Internally the two lyrics paths differ: upstream's text-to-music call only expands the caption and the song metadata, while the lyrics come from the upstream planner call `create_sample`. The worker uses `create_sample` when you bring no lyrics of your own, and passes your lyrics straight into the DiT conditioning otherwise. Either way the LM's tempo, key and meter end up in the metadata sidecar.

## The prompt is rewritten by the planner LM

With `--use-lm` (the default) the planner LM rewrites your prompt into a detailed English caption, and that rewritten caption is what the DiT actually renders. The rewrite is usually an improvement — it names instruments, structure and dynamics that a short prompt leaves implicit — but it also **drops what it does not know**: `编钟` / `bianzhong` / `bronze bells` consistently disappear from the rewritten caption, so a prompt that names them alone does not get them.

`--keep-prompt` stops the rewrite and conditions the DiT on your prompt verbatim while the planner LM still supplies the audio codes, the tempo, the key and the meter. Use it whenever the prompt names a specific instrument.

```bash
mlxgen music \
  --prompt "战国编钟（bianzhong）深沉主旋律，古筝（guzheng）织体，交响乐团铺底，纯器乐" \
  --instrumental --keep-prompt --duration 90 --output bgm.flac
```

## `mlxgen music`

```bash
# the describe-a-song path: the planner LM writes the lyrics
mlxgen music --prompt "一首中文流行歌曲，钢琴伴奏，副歌激昂" --output song.flac

# your own lyrics
mlxgen music \
  --prompt "民谣，小吉他，慢速，副歌温暖" \
  --lyrics "晚风吹过城市的屋顶" \
  --output song.flac

# lyrics from a file (or from stdin with "-")
mlxgen music --prompt "民谣" --lyrics-file lyrics.txt --output song.flac
cat lyrics.txt | mlxgen music --prompt "民谣" --lyrics-file - --output song.flac

# instrumental, no vocals, no planner LM (the smallest package)
mlxgen music --prompt "电影配乐，弦乐，紧张" --instrumental --no-use-lm --output score.flac

# a controlled song: fixed tempo, key and meter
mlxgen music \
  --prompt "中文流行，副歌激昂" \
  --lyrics "..." \
  --duration 90 --bpm 96 --keyscale "C major" --timesignature "4/4" \
  --seed 42 --metadata --output song.flac
```

### Options

| Option | Default | Meaning |
|---|---|---|
| `--model`, `-m` | `ace-step` | Model alias. Only the ACE-Step family is supported here. |
| `--model-path` | `$MFLUX_CACHE_DIR/ace-step` | Prepared package directory. |
| `--prompt`, `--caption` | — | Style description of the song (upstream's `caption`). Required. |
| `--lyrics` | — | Lyrics to sing. Omit them to let the planner LM write them. |
| `--lyrics-file` | — | UTF-8 lyrics file; `-` reads stdin. |
| `--duration` | `60` | Song length in seconds; the maximum is `600`. |
| `--bpm` | — | Tempo in beats per minute. |
| `--keyscale` | — | Key and scale, e.g. `C major`. |
| `--timesignature` | — | Time signature, e.g. `4/4`. |
| `--instrumental` | off | Render an instrumental track with no vocals. |
| `--vocal-language` | `zh` | Language of the sung lyrics. |
| `--use-lm` / `--no-use-lm` | on | Use the planner LM that writes the lyrics and the blueprint. |
| `--keep-prompt` | off | Keep `--prompt` verbatim instead of letting the planner LM rewrite it. |
| `--seed`, `-s` | `42` | Random seed. |
| `--output`, `-o` | `song.flac` | Output audio path; the audio container follows the suffix (default `flac`). |
| `--metadata` | off | Write an `<output>.json` sidecar next to the audio. |
| `--json-events` | off | Emit machine-readable runtime events on stdout. |
| `--debug` | off | Stream worker logs (stderr) through to the terminal. |
| `--replace` / `--no-replace` | replace | Replace the output, or pick the next free `name_1.ext`. |

### The metadata sidecar

`--metadata` writes `<output>.json` next to the audio with the facts of the run (illustrative values):

```json
{
  "task": "text-to-music",
  "model": "ace-step",
  "weights_repo": "ACE-Step/Ace-Step1.5",
  "prompt": "一首中文流行歌曲，钢琴伴奏，副歌激昂",
  "lyrics": "[Verse]\n夜色落在旧城的窗\n...",
  "duration": 60.0,
  "bpm": null,
  "keyscale": null,
  "timesignature": null,
  "instrumental": false,
  "vocal_language": "zh",
  "seed": 42,
  "output_path": "song.flac",
  "audio_seconds": 60.3,
  "sample_rate": 44100,
  "gen_seconds": 41.2,
  "package": "/Users/you/Library/Caches/mflux/ace-step"
}
```

`lyrics` holds the lyrics the model actually sang: the ones you passed in, or — when the planner LM wrote them — the ones the LM produced from your prompt. The same applies to `bpm`, `keyscale` and `timesignature`, which the LM fills in when you leave them unset.

## Runtime characteristics

The isolated worker loads the DiT decoder, the audio VAE, the text encoder and (unless `--no-lm`) the planner LM, then renders the song with the MLX backend on Apple Silicon. The model load is a fixed per-process cost; the diffusion itself scales with `--duration`. Generation is local — the weight download and the venv install are the only network steps.

The worker venv is deliberately pinned away from mlx-gen's own dependency set:

- **`transformers>=4.51,<4.58`** — on 5.x the Qwen3 text encoder and the planner LM change attention/mask semantics and the conditioning breaks.
- **`torch>=2.9.1`** (CPU/MPS build on macOS) — the upstream runtime loads the DiT checkpoint with torch and converts it to the MLX decoder.
- **`mlx>=0.25.2`** and **`mlx-lm>=0.20.0`** — the native Apple Silicon path for the DiT, the VAE and the planner LM.
- The training extras, the Gradio UI and the API server are deliberately absent: the worker only calls the inference API.

## Limitations

- The music backend is a packaged runtime, not a plain weight download: `mlxgen download --model ace-step` refuses and points at `mlxgen prepare --model ace-step`, exactly like the CosyVoice 3 TTS backend.
- `--duration` is capped at 600 seconds. Longer songs need to be rendered in parts and stitched.
- With `--no-use-lm`, lyrics are mandatory unless the track is instrumental.
- The output container is chosen by the `--output` suffix and passed through to the upstream writer; FLAC is the default.
- The upstream runtime tree is pinned to a specific commit. A new upstream release is not picked up until the pin in `mflux/models/ace_step/constants.py` is moved.
