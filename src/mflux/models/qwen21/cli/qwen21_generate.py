from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser
from mflux.cli.runtime_events import CliRuntimeEventStream, cli_print
from mflux.models.common.config.inference_defaults import default_inference_steps
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.qwen21.latent_creator.qwen21_latent_creator import Qwen21LatentCreator
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.utils.dimension_resolver import DimensionResolver
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.prompt_util import PromptUtil

DEFAULT_MODEL = "qwen-image-2.1"


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Qwen Image 2.1 model.")
    parser.add_general_arguments()
    # mlx-gen's parser has no `default_model` hook (upstream resolves the --steps default from
    # it); the same default is applied below via default_inference_steps, so an omitted --model
    # still runs Qwen-Image-2.1 while namespace.model keeps its "user named nothing" signal.
    parser.add_model_arguments(require_model_arg=False)
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    model_name = args.model or DEFAULT_MODEL
    model_config = ModelConfig.from_name(model_name=model_name, base_model=args.base_model)

    qwen = QwenImage21(
        quantize=args.quantize,
        model_path=args.model_path,
        model_config=model_config,
    )

    memory_saver = CallbackManager.register_callbacks(
        args=args,
        model=qwen,
        latent_creator=Qwen21LatentCreator,
    )

    try:
        width, height = DimensionResolver.resolve(
            width=args.width,
            height=args.height,
            reference_image_path=args.image_path,
        )

        steps = args.steps if args.steps is not None else default_inference_steps(model_config, fallback=40)

        for seed in args.seed:
            # 进度事件流(--json-events):与 z-image/flux2 等路由同构,消费者(App / 脚本)
            # 依赖 stdout 的 JSONL 渲染实时进度条。缺这段则 --json-events 被静默忽略,
            # 长任务(2.1 默认 40 步)全程无任何进度输出。
            events = CliRuntimeEventStream(
                enabled=bool(args.json_events),
                command="mlxgen generate",
                model=model_config.model_name,
                seed=seed,
            )
            output_path = args.output.format(seed=seed)
            events.set_output_path(output_path)
            unsubscribe = events.subscribe_model(qwen, map_complete_to_generated=True)
            try:
                image = qwen.generate_image(
                    seed=seed,
                    prompt=PromptUtil.read_prompt(args),
                    negative_prompt=PromptUtil.read_negative_prompt(args),
                    width=width,
                    height=height,
                    guidance=args.guidance if args.guidance is not None else 1.0,
                    scheduler=args.scheduler,
                    image_path=args.image_path,
                    num_inference_steps=steps,
                    image_strength=args.image_strength,
                )
                events.emit_save()
                image.save(path=output_path, export_json_metadata=args.metadata)
                events.emit_complete()
            except Exception as exc:
                events.emit_failed(error=exc)
                raise
            finally:
                if unsubscribe is not None:
                    unsubscribe()
    except (StopImageGenerationException, PromptFileReadError) as exc:
        cli_print(str(exc), json_events=bool(args.json_events))
    finally:
        if memory_saver:
            cli_print(memory_saver.memory_stats(), json_events=bool(args.json_events))


if __name__ == "__main__":
    main()
