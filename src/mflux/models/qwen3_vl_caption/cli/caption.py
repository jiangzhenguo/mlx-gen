import sys
from pathlib import Path

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.qwen3_vl_caption.model.qwen3_vl_caption import (
    BACKEND_MLX_VLM,
    BACKEND_MLXGEN,
    DEFAULT_MAX_PIXELS,
    Qwen3VLCaption,
)
from mflux.models.qwen3_vl_caption.model.util import DEFAULT_CAPTION_PROMPT, ENGLISH_CAPTION_PROMPT
from mflux.utils.exceptions import PromptFileReadError

# The stock Qwen3-VL instruct checkpoints answer in the prompt's language, so the default prompt is
# the Chinese long-form instruction from the NSFW-caption fine-tune this route was built for.
PROMPT_CHOICES = {
    "zh": DEFAULT_CAPTION_PROMPT,
    "en": ENGLISH_CAPTION_PROMPT,
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Describe (caption) an image with a local MLX Qwen3-VL model.")
    # fmt: off
    parser.add_argument("--image-path", type=Path, required=True, help="Path to the image to describe")
    # `--model` (not `--model-path`): CommandLineParser.parse_args recomputes `namespace.model_path`
    # from `--model` for every route, so a hand-rolled `--model-path` flag would be overwritten.
    parser.add_argument("--model", "-m", type=str, required=True, help="Local path or Hugging Face repo of an MLX-format Qwen3-VL checkpoint")
    parser.add_argument("--backend", type=str, choices=[BACKEND_MLXGEN, BACKEND_MLX_VLM], default=BACKEND_MLXGEN, help=f"'{BACKEND_MLXGEN}' runs mlx-gen's own Qwen3-VL route; '{BACKEND_MLX_VLM}' delegates to the optional mlx-vlm package (default: {BACKEND_MLXGEN})")
    parser.add_argument("--prompt", type=str, default=None, help="Caption instruction (default: the model card's Chinese long-form prompt)")
    parser.add_argument("--prompt-preset", type=str, choices=sorted(PROMPT_CHOICES), default=None, help="Use a built-in prompt preset instead of --prompt")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of generated tokens (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature; 0 decodes greedily (default: 0.0)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling cutoff, ignored at temperature 0 (default: 0.9)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--min-pixels", type=int, default=None, help="Lower bound of the image resize budget (default: the checkpoint's own value)")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help=f"Upper bound of the image resize budget (default: {DEFAULT_MAX_PIXELS}, ~800x800)")
    parser.add_argument("-q", "--quantize", type=int, choices=[3, 4, 5, 6, 8], default=None, help="Quantize a bf16 checkpoint; ignored for already-quantized checkpoints")
    parser.add_argument("--output", type=Path, default=None, help="Write the caption to this file instead of printing it")
    # fmt: on
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    prompt = args.prompt
    if prompt is None and args.prompt_preset is not None:
        prompt = PROMPT_CHOICES[args.prompt_preset]

    try:
        caption_model = Qwen3VLCaption(
            # parse_args() derives model_path from --model and leaves it None for predefined names.
            model_path=args.model_path or args.model,
            quantize=args.quantize,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            backend=args.backend,
        )
        caption = caption_model.caption(
            image=args.image_path,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            verbose=True,
        )
    except (PromptFileReadError, ValueError, FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(caption + "\n", encoding="utf-8")
        print(f"Caption written to {args.output}")
    else:
        print(caption)


if __name__ == "__main__":
    main()
