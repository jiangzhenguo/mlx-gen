from pathlib import Path

from PIL import Image

from mflux.utils.exceptions import PromptFileReadError

# The upstream V4.5 NSFW-caption model card ships this Chinese long-form instruction; it also works
# as a generic "describe everything" prompt for the stock Qwen3-VL instruct checkpoints.
DEFAULT_CAPTION_PROMPT = (
    "请用自然流畅的中文对以下图片进行全面而详细的描述。包括所有可见元素及其属性（如颜色、大小、形状、质地），"
    "它们的空间关系，以及任何显著特征或上下文。确保用自然流畅的中文描述清晰、生动，能够捕捉图片的每一个方面，"
    "不遗漏任何重要细节。"
)

ENGLISH_CAPTION_PROMPT = "Describe this image in detail, including every visible element, its attributes, spatial arrangement and any notable context."

STOP_SEQUENCES = ["<|im_end|>", "<|endoftext|>"]


class Qwen3VLCaptionUtil:
    @staticmethod
    def load_image(image: Image.Image | str | Path) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        image_path = Path(image)
        if not image_path.exists():
            raise PromptFileReadError(f"Image file does not exist: {image_path}")
        try:
            return Image.open(image_path).convert("RGB")
        except (OSError, ValueError) as exc:
            raise PromptFileReadError(f"Failed to load image: {exc}") from exc

    @staticmethod
    def save_caption(output_path: Path, caption: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(caption + "\n", encoding="utf-8")
