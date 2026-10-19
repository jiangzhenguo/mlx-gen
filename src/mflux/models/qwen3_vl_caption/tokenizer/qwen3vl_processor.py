from mflux.models.fibo_vlm.tokenizer.qwen2vl_processor import Qwen2VLProcessor

# Qwen3-VL image processor defaults, used only when a checkpoint does not publish its own budget.
DEFAULT_MIN_PIXELS = 256 * 256
DEFAULT_MAX_PIXELS = 4096 * 4096


class Qwen3VLProcessor(Qwen2VLProcessor):
    """Qwen3-VL processor: Qwen2-VL's image-pad expansion with a checkpoint-defined pixel budget.

    Qwen3-VL keeps the same `<|vision_start|><|image_pad|><|vision_end|>` contract (image token id
    151655), so only the resize budget differs from the FIBO-VLM route: Qwen2-VL's hardcoded
    256*28*28 / 1024*28*28 would over-resize relative to what Qwen3-VL checkpoints declare
    (`min_pixels` / `max_pixels`, or `size.shortest_edge` / `size.longest_edge`).
    """

    def __init__(
        self,
        tokenizer,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ):
        super().__init__(tokenizer)
        if min_pixels is not None:
            self.image_processor.min_pixels = int(min_pixels)
        if max_pixels is not None:
            self.image_processor.max_pixels = int(max_pixels)

    def set_pixel_budget(self, min_pixels: int | None, max_pixels: int | None) -> None:
        if min_pixels is not None:
            self.image_processor.min_pixels = int(min_pixels)
        if max_pixels is not None:
            self.image_processor.max_pixels = int(max_pixels)

    @property
    def min_pixels(self) -> int:
        return int(self.image_processor.min_pixels)

    @property
    def max_pixels(self) -> int:
        return int(self.image_processor.max_pixels)
