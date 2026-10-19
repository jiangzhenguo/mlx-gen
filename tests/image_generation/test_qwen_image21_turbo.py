"""Qwen-Image-2.1-Turbo: the 8-step distilled checkpoint of Qwen-Image-2.1.

Same architecture, same components — the only contract difference is the sampling
schedule embedded in model_index.json (`sample_sigmas`), which must be used verbatim:
no mu shifting, no terminal stretching, and its length fixes the step count.
"""

import json

import mlx.core as mx
import pytest

from mflux.models.common.config.config import Config
from mflux.models.common.config.inference_defaults import default_inference_steps
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.common.schedulers.linear_scheduler import LinearScheduler
from mflux.models.qwen21.weights.sample_sigmas import load_sample_sigmas

TURBO_SAMPLE_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]


class TestQwenImage21TurboRegistry:
    @pytest.mark.parametrize(
        "name", ["qwen-image-2.1-turbo", "qwen-2.1-turbo", "qwen-image-21-turbo", "Qwen/Qwen-Image-2.1-Turbo"]
    )
    def test_aliases_resolve_to_turbo(self, name):
        model_config = ModelConfig.from_name(name)
        assert model_config.model_name == "Qwen/Qwen-Image-2.1-Turbo"
        assert default_inference_steps(model_config, model_name=name) == 8

    def test_turbo_disables_sigma_shifting(self):
        model_config = ModelConfig.qwen_image_21_turbo()
        assert model_config.requires_sigma_shift is False
        assert model_config.sigma_shift_terminal is None

    def test_plain_21_keeps_computed_schedule(self):
        model_config = ModelConfig.qwen_image_21()
        assert model_config.requires_sigma_shift is True
        assert model_config.sigma_shift_terminal == 0.02

    def test_edit_cli_family_accepts_turbo(self):
        resolved = ConfigResolution.resolve_restricted(
            "qwen-image-2.1-turbo", "qwen-image-2.1", extra_keys=("qwen-image-2.1-turbo",)
        )
        assert resolved is ModelConfig.from_name("Qwen/Qwen-Image-2.1-Turbo")

    def test_edit_cli_infers_turbo_from_checkpoint_path(self):
        # The basename holds the longest family alias, so a Turbo export must not be
        # claimed by the plain "qwen-image-2.1" entry it contains as a substring.
        resolved = ConfigResolution.resolve_restricted(
            None,
            "qwen-image-2.1",
            model_path="/models/Qwen-Image-2.1-Turbo",
            extra_keys=("qwen-image-2.1-turbo",),
        )
        assert resolved.model_name == "Qwen/Qwen-Image-2.1-Turbo"

    def test_turbo_is_qwen21_for_task_inference(self):
        from mflux.task_inference import get_model_capabilities

        capabilities = get_model_capabilities(model="qwen-image-2.1-turbo").capabilities
        defaults = {c.public_task: c.id for c in capabilities if c.default_for_task}
        assert defaults["image-to-image"] == "qwen21.edit"


class TestSampleSigmasLoading:
    def test_reads_sample_sigmas_from_model_index(self, tmp_path):
        (tmp_path / "model_index.json").write_text(
            json.dumps({"_class_name": "QwenImage21Pipeline", "sample_sigmas": TURBO_SAMPLE_SIGMAS})
        )
        assert load_sample_sigmas(str(tmp_path)) == TURBO_SAMPLE_SIGMAS

    def test_plain_checkpoint_yields_none(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
        assert load_sample_sigmas(str(tmp_path)) is None

    def test_missing_or_empty_schedule_yields_none(self, tmp_path):
        assert load_sample_sigmas(str(tmp_path)) is None
        (tmp_path / "model_index.json").write_text(json.dumps({"sample_sigmas": []}))
        assert load_sample_sigmas(str(tmp_path)) is None


class TestFixedSigmaSchedule:
    @staticmethod
    def config(sigmas=TURBO_SAMPLE_SIGMAS, steps=40):
        return Config(
            model_config=ModelConfig.qwen_image_21_turbo(),
            num_inference_steps=steps,
            width=1024,
            height=1024,
            guidance=1.0,
            sigmas=sigmas,
        )

    def test_schedule_length_fixes_step_count(self):
        # Upstream ignores num_inference_steps once sample_sigmas are configured.
        assert self.config(steps=40).num_inference_steps == len(TURBO_SAMPLE_SIGMAS)

    def test_scheduler_uses_schedule_verbatim(self):
        sigmas = self.config().scheduler.sigmas
        expected = mx.array(TURBO_SAMPLE_SIGMAS + [0.0], dtype=mx.float32)
        assert mx.allclose(sigmas, expected)

    def test_no_mu_shift_applied(self):
        # A computed schedule would move with the canvas; the fixed one must not, at any size.
        small = Config(
            model_config=ModelConfig.qwen_image_21_turbo(),
            width=2752,
            height=1536,
            guidance=1.0,
            sigmas=TURBO_SAMPLE_SIGMAS,
        )
        assert mx.allclose(small.scheduler.sigmas, self.config().scheduler.sigmas)

    def test_degenerate_schedule_rejected(self):
        with pytest.raises(ValueError, match="at least two"):
            self.config(sigmas=[1.0])

    def test_step_uses_schedule_deltas(self):
        scheduler = self.config().scheduler
        latents = mx.zeros((1, 4, 4))
        noise = mx.ones((1, 4, 4))
        stepped = scheduler.step(noise=noise, timestep=3, latents=latents)
        expected_delta = TURBO_SAMPLE_SIGMAS[4] - TURBO_SAMPLE_SIGMAS[3]
        assert mx.allclose(stepped, mx.full((1, 4, 4), expected_delta))

    def test_plain_config_without_sigmas_unchanged(self):
        config = Config(model_config=ModelConfig.qwen_image_21(), num_inference_steps=8, width=1024, height=1024)
        assert config.sigmas is None
        assert isinstance(config.scheduler, LinearScheduler)
        assert config.num_inference_steps == 8
