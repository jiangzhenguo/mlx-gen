"""Verify that every weight of a local Qwen-Image-2.1 package lands on this fork's modules.

The text-to-image loader runs `Module.update(..., strict=False)`, so a package whose key layout
does not match this fork is loaded *silently*: unrecognised tensors are dropped and the affected
modules stay at their random initialisation (generation then returns noise, with no error).
The reference-editing loader is strict instead, but it fails only after loading the components.

Either way it is cheaper to resolve the keys first. This script performs the same key resolution
as loading, but only compares key sets and shapes, so it never runs a denoise step.

Usage:
    .venv/bin/python tools/verify_qwen21_package.py ~/ai/models/<package>              # text-to-image
    .venv/bin/python tools/verify_qwen21_package.py ~/ai/models/<package> --route edit # reference editing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mlx.utils import tree_flatten

from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_vae.qwen21_vae import Qwen21VAE
from mflux.models.qwen21.weights.qwen21_weight_definition import Qwen21WeightDefinition

# Rebuilt at module init from the config, so checkpoints are free to omit them.
COMPUTED_KEYS = {
    "txt2img": ("pos_embed.cos_tables", "pos_embed.sin_tables", "time_text_embed.time_proj.freqs", ".inv_freq"),
    "edit": (".inv_freq",),
}


def txt2img_modules() -> dict:
    return {"transformer": Qwen21Transformer(), "text_encoder": Qwen21TextEncoder(), "vae": Qwen21VAE()}


def edit_modules(root: Path) -> dict:
    from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.text_encoder import QwenImage21TextEncoder
    from mflux.models.qwen21.reference.model.qwen_image21_transformer.transformer import QwenImage21Transformer
    from mflux.models.qwen21.reference.model.qwen_image21_vae.vae import QwenImage21VAE as ReferenceVAE

    configs = {}
    for name in ("vae", "transformer", "text_encoder"):
        path = root / name / "config.json"
        if not path.exists():
            raise SystemExit(
                f"{path} is missing: the reference-editing route reads component geometry from it. "
                "Weight-only exports need this file added before editing can run."
            )
        configs[name] = json.loads(path.read_text())
    return {
        "transformer": QwenImage21Transformer(configs["transformer"]),
        "text_encoder": QwenImage21TextEncoder(configs["text_encoder"]),
        "vae": ReferenceVAE(configs["vae"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="local model directory")
    parser.add_argument("--route", choices=("txt2img", "edit"), default="txt2img")
    args = parser.parse_args()
    root = args.model_path.expanduser()
    if not root.exists():
        raise SystemExit(f"no such directory: {root}")

    if args.route == "edit":
        from mflux.models.qwen21.reference.weights.qwen_image21_weight_definition import (
            QwenImage21WeightDefinition as Definition,
        )

        modules = edit_modules(root)
    else:
        Definition = Qwen21WeightDefinition
        modules = txt2img_modules()

    ok = True
    for component in Definition.get_components():
        weights, q_level, version = WeightLoader._load_component(root, component)
        supplied = dict(tree_flatten(weights))
        expected = dict(tree_flatten(modules[component.name].parameters()))

        computed = COMPUTED_KEYS[args.route]
        missing = {
            key
            for key in expected
            if key not in supplied and not any(part in key for part in computed)
        }
        unexpected = set(supplied) - set(expected)
        mismatched = [
            (key, tuple(expected[key].shape), tuple(supplied[key].shape))
            for key in expected.keys() & supplied.keys()
            if tuple(expected[key].shape) != tuple(supplied[key].shape)
        ]
        bad = bool(missing or unexpected or mismatched)
        ok = ok and not bad
        print(
            f"[{component.name:12s}] {'OK' if not bad else 'BAD':3s} "
            f"supplied={len(supplied):4d} expected={len(expected):4d} "
            f"missing={len(missing):3d} unexpected={len(unexpected):3d} shape_mismatch={len(mismatched):3d} "
            f"(stored q={q_level}, mflux_version={version})"
        )
        if missing:
            print("   missing sample:", sorted(missing)[:4])
        if unexpected:
            print("   unexpected sample:", sorted(unexpected)[:4])
        if mismatched:
            print("   shape sample:", mismatched[:2])

    print("RESULT:", f"{args.route}: all components map cleanly" if ok else f"{args.route}: PACKAGE MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
