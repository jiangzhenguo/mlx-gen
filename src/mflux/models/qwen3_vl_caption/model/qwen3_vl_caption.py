from pathlib import Path
from typing import Any, List

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.common_models.qwen3_vl.qwen3_vl_util import Qwen3VLUtil
from mflux.models.qwen3_vl_caption.backend.mlx_vlm_backend import MlxVlmCaptionBackend
from mflux.models.qwen3_vl_caption.model.util import (
    DEFAULT_CAPTION_PROMPT,
    STOP_SEQUENCES,
    Qwen3VLCaptionUtil,
)
from mflux.models.qwen3_vl_caption.qwen3_vl_caption_initializer import Qwen3VLCaptionInitializer
from mflux.models.qwen3_vl_caption.tokenizer.qwen3vl_processor import Qwen3VLProcessor

BACKEND_MLXGEN = "mlxgen"
BACKEND_MLX_VLM = "mlx-vlm"
BACKENDS = (BACKEND_MLXGEN, BACKEND_MLX_VLM)

# ~800x800: the size the V4.5 caption fine-tune reports training on. Full checkpoints allow up to
# 4096x4096, but the image-token count grows with the pixel area and prefill cost grows with its
# square, so the smaller default keeps a single caption interactive.
DEFAULT_MAX_PIXELS = 640_000
DEFAULT_MODEL_ID = "mlx-community/Qwen3-VL-4B-Instruct-4bit"


class Qwen3VLCaption:
    """Describe (caption) an image with a locally available MLX Qwen3-VL checkpoint.

    Two backends share this API: `mlxgen` (default) runs the model through mlx-gen's own Qwen3-VL
    decoder and vision tower, `mlx-vlm` delegates to the optional `mlx-vlm` package for parity
    checks against that runtime.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        model_path: str | None = None,
        quantize: int | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = DEFAULT_MAX_PIXELS,
        backend: str = BACKEND_MLXGEN,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose one of: {', '.join(BACKENDS)}")

        self.backend = backend
        self.model_path = model_path if model_path else model_id
        self.bits: int | None = None
        self.decoder: Any = None
        self.tokenizers: dict = {}
        self._delegate: MlxVlmCaptionBackend | None = None

        if backend == BACKEND_MLX_VLM:
            self._delegate = MlxVlmCaptionBackend(model_path=self.model_path, max_pixels=max_pixels)
            return

        Qwen3VLCaptionInitializer.init(
            model=self,
            model_path=self.model_path,
            quantize=quantize,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    @property
    def processor(self) -> Qwen3VLProcessor:
        return self.tokenizers["qwen3_vl"].processor

    def caption(
        self,
        image: Image.Image | str | Path,
        prompt: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.9,
        seed: int | None = None,
        stop: List[str] | None = None,
        verbose: bool = False,
    ) -> str:
        """Return a free-form description of `image`. temperature <= 0 decodes greedily."""
        if self._delegate is not None:
            return self._delegate.caption(
                image=image,
                prompt=prompt if prompt is not None else DEFAULT_CAPTION_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                verbose=verbose,
            )

        resolved_image = Qwen3VLCaptionUtil.load_image(image)
        resolved_prompt = (prompt if prompt is not None else DEFAULT_CAPTION_PROMPT).strip()
        if not resolved_prompt:
            raise ValueError("Caption prompt must not be empty.")

        processor = self.processor
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resolved_image},
                    {"type": "text", "text": resolved_prompt},
                ],
            }
        ]
        chat_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted = processor(text=[chat_text], images=[resolved_image], padding=True, return_tensors="np")

        input_ids = mx.array(formatted["input_ids"])
        attention_mask = mx.array(formatted["attention_mask"]) if formatted.get("attention_mask") is not None else None
        image_grid_thw = mx.array(formatted["image_grid_thw"])
        # The vision tower follows its own weight dtype (bf16 for published Qwen3-VL repos).
        vision_dtype = self.decoder.visual.patch_embed.proj.weight.dtype
        pixel_values = mx.array(formatted["pixel_values"]).astype(vision_dtype)

        stop_sequences = stop if stop is not None else STOP_SEQUENCES
        stop_token_sequences = [
            processor.tokenizer.encode(sequence, add_special_tokens=False) for sequence in stop_sequences
        ]

        if verbose:
            grid = np.array(image_grid_thw).reshape(-1).tolist()
            print(
                f"caption: {input_ids.shape[1]} prompt tokens "
                f"({int(np.prod(grid)) // (processor.image_processor.merge_size**2)} image tokens, grid {grid}), "
                f"max_pixels={processor.max_pixels}"
            )

        generated_ids = Qwen3VLUtil.generate_text(
            decoder=self.decoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            stop_token_sequences=stop_token_sequences,
            eos_token_id=processor.tokenizer.eos_token_id,
            seed=seed,
        )
        generated_tokens = generated_ids[:, input_ids.shape[1] :]
        return processor.tokenizer.decode(np.array(generated_tokens[0]), skip_special_tokens=True).strip()

    def caption_file(self, image_path: str | Path, **kwargs: Any) -> str:
        return self.caption(image=Qwen3VLCaptionUtil.load_image(image_path), **kwargs)
