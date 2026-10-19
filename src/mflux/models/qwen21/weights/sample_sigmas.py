import json
from pathlib import Path


def load_sample_sigmas(model_path: str) -> list[float] | None:
    """Sampling schedule embedded by the checkpoint author (Qwen-Image-2.1-Turbo), if any.

    Diffusers stores it as `sample_sigmas` in model_index.json (pipeline config, terminal
    zero excluded); plain Qwen-Image-2.1 has no such key and keeps the computed schedule.
    Only readable for local checkpoints — a hub repo id simply yields None here.
    """
    index_path = Path(model_path) / "model_index.json"
    if not index_path.exists():
        return None
    sigmas = json.loads(index_path.read_text()).get("sample_sigmas")
    if not sigmas:
        return None
    return [float(s) for s in sigmas]
