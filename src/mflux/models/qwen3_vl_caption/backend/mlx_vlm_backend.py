from pathlib import Path
from typing import Any

from PIL import Image

from mflux.models.qwen3_vl_caption.model.util import Qwen3VLCaptionUtil

MLX_VLM_MIN_VERSION = "0.5.0"


def _import_mlx_vlm():
    # Function-local on purpose: this module is imported by the caption route, and `import mflux`
    # plus the default native path must stay free of the optional mlx-vlm dependency.
    try:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
    except ImportError as exc:  # pragma: no cover - exercised only without the optional package
        raise ImportError(
            "The 'mlx-vlm' backend needs the optional mlx-vlm package. "
            f"Install it with: pip install 'mlx-vlm>={MLX_VLM_MIN_VERSION}'"
        ) from exc
    return load, generate, apply_chat_template, load_config


class MlxVlmCaptionBackend:
    """Optional parity backend: run the same checkpoint through the mlx-vlm runtime.

    Useful to compare caption quality and speed against mlx-gen's own Qwen3-VL route, and as a
    fallback for checkpoints mlx-gen's route does not accept.
    """

    def __init__(self, model_path: str, max_pixels: int | None = None):
        load, _generate, _apply_chat_template, load_config = _import_mlx_vlm()
        self.model_path = model_path
        self.model, self.processor = load(model_path)
        self.config = load_config(model_path)
        if max_pixels is not None:
            image_processor = getattr(self.processor, "image_processor", None)
            if image_processor is not None and hasattr(image_processor, "max_pixels"):
                image_processor.max_pixels = int(max_pixels)

    def caption(
        self,
        image: Image.Image | str | Path,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.9,
        seed: int | None = None,
        verbose: bool = False,
    ) -> str:
        _load, generate, apply_chat_template, _load_config = _import_mlx_vlm()
        resolved_image = Qwen3VLCaptionUtil.load_image(image)
        formatted_prompt = apply_chat_template(self.processor, self.config, prompt, num_images=1)

        kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
        if seed is not None:
            kwargs["seed"] = seed
        if verbose:
            kwargs["verbose"] = True

        try:
            return generate(self.model, self.processor, formatted_prompt, [resolved_image], **kwargs)
        except TypeError:
            # Older mlx-vlm releases accept a smaller keyword set; degrade to the core contract.
            return generate(
                self.model,
                self.processor,
                formatted_prompt,
                [resolved_image],
                max_tokens=max_tokens,
                temperature=temperature,
            )
