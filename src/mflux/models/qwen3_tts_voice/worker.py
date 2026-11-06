"""Qwen3-TTS VoiceDesign synthesis worker.

This module runs as a subprocess of the MLX-Gen environment
(`python worker.py ...`), not inside MLX-Gen's own import path. Qwen3-TTS is
MLX-native, so the worker only needs `mlx-audio` from the current environment —
never import mflux here.

Protocol: one JSON runtime event per line on stdout (the MLX-Gen JSONL contract);
all library logging stays on stderr. The parent process (`runner.py`) relays or
renders these events.
"""

import argparse
import json
import os
import sys
import time
import traceback


def _emit(command, model, seed, phase, **fields):
    payload = {
        "type": "runtime",
        "command": command,
        "model": model,
        "seed": seed,
        "phase": phase,
        "step": fields.pop("step", 0),
        "total_steps": fields.pop("total_steps", 0),
        "progress": fields.pop("progress", 0.0),
    }
    payload.update({key: value for key, value in fields.items() if value is not None})
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _write_wav(path, samples, sample_rate):
    """Write mono 16-bit PCM. Uses the stdlib so the worker needs no audio extras."""
    import wave

    import numpy as np

    pcm = np.asarray(samples, dtype=np.float32)
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return pcm.shape[0] / float(sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser(prog="qwen3-tts-voice-design-worker")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--voice", required=True, help="Natural-language description of the wanted voice.")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command", default="mlxgen voice-design")
    parser.add_argument("--model-label", default="qwen3-tts-voice-design")
    args = parser.parse_args()

    command, label, seed = args.command, args.model_label, args.seed
    try:
        _emit(command, label, seed, "loading", task="voice-design")
        with open(args.text_file, encoding="utf-8") as text_file:
            text = text_file.read().strip()
        if not text:
            raise ValueError("Synthesis text is empty.")
        voice = args.voice.strip()
        if not voice:
            raise ValueError("Voice description is empty.")

        import mlx.core as mx

        mx.random.seed(seed)

        from mlx_audio.tts.utils import load_model

        load_start = time.time()
        model = load_model(str(args.model_dir))
        _emit(
            command,
            label,
            seed,
            "loaded",
            task="voice-design",
            progress=0.0,
            load_seconds=round(time.time() - load_start, 2),
        )

        synth_start = time.time()
        options = {}
        if args.temperature is not None:
            options["temperature"] = args.temperature
        chunks = []
        sample_rate = 24000
        for index, result in enumerate(
            model.generate_voice_design(
                text=text,
                instruct=voice,
                language=args.language,
                stream=True,
                **options,
            ),
            start=1,
        ):
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            sample_rate = int(getattr(result, "sample_rate", sample_rate) or sample_rate)
            chunks.append(audio)
            _emit(
                command,
                label,
                seed,
                "synthesizing",
                task="voice-design",
                step=index,
                audio_seconds=round(sum(chunk.size for chunk in chunks) / sample_rate, 2),
            )

        if not chunks:
            raise RuntimeError("Qwen3-TTS produced no audio for the given text.")

        import numpy as np

        audio = np.concatenate([np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in chunks])
        synth_seconds = time.time() - synth_start
        audio_seconds = _write_wav(args.output, audio, sample_rate)
        _emit(
            command,
            label,
            seed,
            "complete",
            task="voice-design",
            step=1,
            total_steps=1,
            progress=1.0,
            output_path=args.output,
            sample_rate=sample_rate,
            audio_seconds=round(audio_seconds, 2),
            synth_seconds=round(synth_seconds, 2),
            rtf=round(synth_seconds / audio_seconds, 3) if audio_seconds > 0 else None,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - the worker reports every failure as an event
        traceback.print_exc(file=sys.stderr)
        _emit(
            command,
            label,
            seed,
            "failed",
            task="voice-design",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
