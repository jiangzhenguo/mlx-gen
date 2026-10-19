"""CosyVoice3 synthesis worker.

This module runs INSIDE the prepared package's isolated venv
(`<package>/venv/bin/python worker.py ...`), not inside MLX-Gen's own environment.
It must stay self-contained: standard library plus the venv's packages only — never
import mflux here.

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


def main() -> int:
    parser = argparse.ArgumentParser(prog="cosyvoice3-worker")
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", default=None)
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--mode", choices=["zero_shot", "instruct2", "cross_lingual"], required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command", default="mlxgen tts")
    parser.add_argument("--model-label", default="cosyvoice3")
    args = parser.parse_args()

    command, label, seed = args.command, args.model_label, args.seed
    try:
        _emit(command, label, seed, "loading", task="text-to-speech")
        runtime_dir = os.path.abspath(args.runtime_dir)
        sys.path.insert(0, os.path.join(runtime_dir, "third_party", "Matcha-TTS"))
        sys.path.insert(0, runtime_dir)

        import random

        import numpy as np
        import torch

        from cosyvoice.cli.cosyvoice import AutoModel

        with open(args.text_file, encoding="utf-8") as text_file:
            text = text_file.read().strip()
        if not text:
            raise ValueError("Synthesis text is empty.")

        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)

        load_start = time.time()
        model = AutoModel(model_dir=args.model_dir)
        _emit(
            command,
            label,
            seed,
            "loaded",
            task="text-to-speech",
            progress=0.0,
            load_seconds=round(time.time() - load_start, 2),
            sample_rate=model.sample_rate,
        )

        synth_start = time.time()
        chunks = []
        if args.mode == "instruct2":
            iterator = model.inference_instruct2(text, args.instruct, args.ref_audio, stream=False)
        elif args.mode == "zero_shot":
            iterator = model.inference_zero_shot(text, args.ref_text, args.ref_audio, stream=False)
        else:
            iterator = model.inference_cross_lingual(text, args.ref_audio, stream=False)

        for index, output in enumerate(iterator, start=1):
            chunks.append(output["tts_speech"])
            audio_seconds = sum(chunk.shape[1] for chunk in chunks) / model.sample_rate
            _emit(
                command,
                label,
                seed,
                "synthesizing",
                task="text-to-speech",
                step=index,
                audio_seconds=round(audio_seconds, 2),
            )

        if not chunks:
            raise RuntimeError("CosyVoice3 produced no audio for the given text.")
        speech = torch.cat(chunks, dim=1)
        audio_seconds = speech.shape[1] / model.sample_rate
        synth_seconds = time.time() - synth_start

        import soundfile as sf

        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        sf.write(args.output, speech.squeeze(0).numpy(), model.sample_rate)
        _emit(
            command,
            label,
            seed,
            "complete",
            task="text-to-speech",
            step=1,
            total_steps=1,
            progress=1.0,
            output_path=args.output,
            sample_rate=model.sample_rate,
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
            task="text-to-speech",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
