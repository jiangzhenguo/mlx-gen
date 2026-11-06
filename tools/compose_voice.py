#!/usr/bin/env python3
"""Compose one WAV out of several designed-voice segments.

Long audio with more than one "beat" (a drawl, a pause, a shout, a different
emotion) is best built from separately synthesized pieces: a single generation
with a long run of "嗯……啊……" drifts into a babble loop and swallows the
words at the end. Each segment gets its own description and its own gain, and
this tool stitches them together.

Usage:

    mlxgen voice-design ...            # one segment
    python tools/compose_voice.py manifest.json --output out/final.wav

Manifest:

    {
      "language": "chinese",
      "seed": 42,
      "peak": 0.95,
      "segments": [
        {"text": "嗯……啊……", "voice": "一位三十五岁左右的女性，声音低沉磁性", "gain_db": 0},
        {"text": "舒服死了！", "voice": "同一位女性，情绪爆发，用尽全力大喊", "gain_db": 7}
      ]
    }

Per-segment keys: text, voice (required), gain_db (default 0), gap_after_s
(default 0.12), temperature, seed, language.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mflux.models.qwen3_tts_voice import constants  # noqa: E402
from mflux.models.qwen3_tts_voice.runner import resolve_package, synthesize  # noqa: E402


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"{path} must be mono 16-bit PCM.")
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32), sample_rate


def _trim(x: np.ndarray, sample_rate: int, threshold: float = 120.0) -> np.ndarray:
    """Drop the leading/trailing silence, keeping a small margin."""
    loud = np.where(np.abs(x) > threshold)[0]
    if len(loud) == 0:
        return x
    margin = int(sample_rate * 0.05)
    return x[max(0, loud[0] - margin) : min(len(x), loud[-1] + margin)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="JSON manifest describing the segments.")
    parser.add_argument("--output", "-o", default="composed.wav", help="Output wav path (default composed.wav).")
    parser.add_argument("--model-path", default=None, help=f"Prepared package directory (default: {constants.default_package_dir()}).")
    parser.add_argument("--temp-dir", default=None, help="Directory for the intermediate segment wavs (default: a temp dir).")
    parser.add_argument("--peak", type=float, default=None, help="Peak level of the result, 0..1 (default: from the manifest, or 0.95).")
    parser.add_argument("--metadata", action="store_true", help="Write <output>.json sidecar.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments = manifest.get("segments") or []
    if not segments:
        raise SystemExit("The manifest has no segments.")

    package = resolve_package(args.model_path)
    language = manifest.get("language", "auto")
    seed = manifest.get("seed", 42)
    peak_target = args.peak if args.peak is not None else manifest.get("peak", 0.95)

    import tempfile

    temp_dir = Path(args.temp_dir) if args.temp_dir else Path(tempfile.mkdtemp(prefix="mlxgen-compose-"))
    temp_dir.mkdir(parents=True, exist_ok=True)

    parts: list[np.ndarray] = []
    sample_rate = 24000
    report: list[dict[str, object]] = []

    for index, segment in enumerate(segments):
        text = str(segment.get("text", "")).strip()
        voice = str(segment.get("voice", "")).strip()
        if not text or not voice:
            raise SystemExit(f"Segment {index} needs both 'text' and 'voice'.")
        segment_path = temp_dir / f"segment_{index:03d}.wav"
        result = synthesize(
            package,
            text=text,
            output_path=segment_path,
            voice_description=voice,
            language=constants.normalize_language(segment.get("language", language)),
            seed=int(segment.get("seed", seed)),
            json_events=False,
            command="compose-voice",
            temperature=segment.get("temperature"),
        )
        data, sample_rate = _read_wav(segment_path)
        data = _trim(data, sample_rate)
        gain_db = float(segment.get("gain_db", 0.0))
        if gain_db:
            data = data * (10 ** (gain_db / 20.0))
        parts.append(data)
        report.append(
            {
                "index": index,
                "text": text,
                "gain_db": gain_db,
                "duration": round(len(data) / sample_rate, 3),
                "rtf": result.rtf,
            }
        )
        gap = float(segment.get("gap_after_s", 0.12))
        if gap > 0:
            parts.append(np.zeros(int(sample_rate * gap), dtype=np.float32))

    if parts and np.all(np.abs(parts[-1]) == 0):  # a trailing gap is not wanted
        parts.pop()

    composed = np.concatenate(parts)
    peak = float(np.abs(composed).max())
    if peak > 0:
        composed = composed / peak * peak_target * 32767
    data = np.clip(composed, -32768, 32767).astype(np.int16)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(data.tobytes())

    print(f"Composed {len(report)} segments -> {output_path} ({len(data) / sample_rate:.2f}s)")
    for entry in report:
        print(f"  [{entry['index']}] {entry['duration']:.2f}s  gain {entry['gain_db']:+.1f} dB  rtf {entry['rtf']}  {entry['text'][:24]}")

    if args.metadata:
        sidecar = output_path.with_suffix(output_path.suffix + ".json")
        sidecar.write_text(
            json.dumps(
                {
                    "task": "compose-voice",
                    "model": constants.MODEL_ALIAS,
                    "output_path": str(output_path),
                    "duration": round(len(data) / sample_rate, 3),
                    "sample_rate": sample_rate,
                    "language": language,
                    "seed": seed,
                    "segments": report,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Metadata: {sidecar}")


if __name__ == "__main__":
    main()
