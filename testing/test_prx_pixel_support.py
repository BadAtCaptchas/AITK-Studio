import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from toolkit.prompt_utils import PromptEmbeds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRX_ROOT = PROJECT_ROOT / "extensions_built_in" / "diffusion_models" / "prx_pixel_t2i"
REGISTRY_PATH = PROJECT_ROOT / "extensions_built_in" / "diffusion_models" / "__init__.py"
CONFIG_MODULES_PATH = PROJECT_ROOT / "toolkit" / "config_modules.py"
OFFLOAD_PATH = PROJECT_ROOT / "toolkit" / "memory_management" / "offload.py"
README_PATH = PROJECT_ROOT / "README.md"
UI_OPTIONS_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "options.ts"
MEMORY_PROFILE_PATH = PROJECT_ROOT / "ui" / "src" / "utils" / "memoryProfiles.ts"
AUTO_PROFILES_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "autoTrainingProfiles.ts"


try:
    from extensions_built_in.diffusion_models.prx_pixel_t2i.prx_pixel_t2i import (
        NOISE_SCALE,
        PRXPixelT2IModel,
        PROMPT_MAX_TOKENS,
    )
    from extensions_built_in.diffusion_models.prx_pixel_t2i.src.pipeline import (
        X_PRED_T_MIN,
        PRXPixelPipeline,
    )
    from extensions_built_in.diffusion_models.prx_pixel_t2i.src.transformer_prx import (
        PRXTransformer2DModel,
    )

    PRX_IMPORT_ERROR = None
except ImportError as exc:
    PRXPixelT2IModel = None
    PRXPixelPipeline = None
    PRXTransformer2DModel = None
    PROMPT_MAX_TOKENS = None
    NOISE_SCALE = None
    X_PRED_T_MIN = None
    PRX_IMPORT_ERROR = exc


def require_prx_imports():
    if PRX_IMPORT_ERROR is not None:
        raise unittest.SkipTest(str(PRX_IMPORT_ERROR))


class PRXPixelStaticSupportTest(unittest.TestCase):
    def test_backend_registry_config_and_docs_include_prx_pixel(self):
        registry = REGISTRY_PATH.read_text(encoding="utf-8")
        config = CONFIG_MODULES_PATH.read_text(encoding="utf-8")
        offload = OFFLOAD_PATH.read_text(encoding="utf-8")
        readme = README_PATH.read_text(encoding="utf-8")

        self.assertIn('PRXPixelT2IModel, = _optional_models(', registry)
        self.assertIn('"PRXPixelT2IModel", "prx_pixel"', registry)
        self.assertIn("PRXPixelT2IModel,", registry)
        self.assertIn("'prx_pixel'", config)
        self.assertIn('"prx_pixel"', offload)
        self.assertIn("Photoroom/prxpixel-t2i", readme)
        self.assertIn("direct x0 loss targets", readme)

    def test_model_class_resolves_when_optional_dependencies_are_available(self):
        require_prx_imports()
        from extensions_built_in.diffusion_models import AI_TOOLKIT_MODELS

        resolved = next(
            model_class
            for model_class in AI_TOOLKIT_MODELS
            if model_class.arch == "prx_pixel"
        )

        self.assertIs(resolved, PRXPixelT2IModel)

    def test_optional_import_fallback_raises_on_instantiation(self):
        import extensions_built_in.diffusion_models as registry

        with mock.patch.object(
            registry,
            "import_module",
            side_effect=ImportError("missing PRX dependency"),
        ):
            MissingPRXModel, = registry._optional_models(
                ".prx_pixel_t2i", (("MissingPRXModel", "prx_pixel"),)
            )

        self.assertEqual(MissingPRXModel.__name__, "MissingPRXModel")
        self.assertEqual(MissingPRXModel.arch, "prx_pixel")
        with self.assertRaisesRegex(ImportError, "missing PRX dependency"):
            MissingPRXModel()

    def test_ui_defaults_memory_profile_and_auto_profile(self):
        options = UI_OPTIONS_PATH.read_text(encoding="utf-8")
        memory = MEMORY_PROFILE_PATH.read_text(encoding="utf-8")
        profiles = AUTO_PROFILES_PATH.read_text(encoding="utf-8")

        start = options.index("name: 'prx_pixel'")
        end = options.index("disableSections", start)
        block = options[start:end]

        self.assertIn("label: 'PRXPixel (pixel space)'", block)
        self.assertIn("'config.process[0].model.name_or_path': ['Photoroom/prxpixel-t2i', defaultNameOrPath]", block)
        self.assertIn("'config.process[0].model.quantize': [true, false]", block)
        self.assertIn("'config.process[0].model.quantize_te': [true, false]", block)
        self.assertIn("'config.process[0].model.low_vram': [true, false]", block)
        self.assertIn("'config.process[0].sample.sampler': ['flowmatch', 'flowmatch']", block)
        self.assertIn("'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch']", block)
        self.assertIn("'config.process[0].train.timestep_type': ['linear', 'sigmoid']", block)
        self.assertIn("'config.process[0].sample.width': [1024, 1024]", block)
        self.assertIn("'config.process[0].sample.height': [1024, 1024]", block)
        self.assertIn("'config.process[0].sample.sample_steps': [28, 20]", block)
        self.assertIn("'config.process[0].sample.guidance_scale': [5, 7]", block)
        self.assertIn("'prx_pixel'", memory)
        self.assertIn("'prx_pixel'", profiles)

    def test_vendored_source_has_attribution_and_no_external_pipeline_requirement(self):
        transformer = (PRX_ROOT / "src" / "transformer_prx.py").read_text(
            encoding="utf-8"
        )
        model = (PRX_ROOT / "prx_pixel_t2i.py").read_text(encoding="utf-8")
        package_readme = (PRX_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("Copyright 2025 The Photoroom and The HuggingFace Teams", transformer)
        self.assertIn("Apache License, Version 2.0", transformer)
        self.assertIn("PRXTransformer2DModel", model)
        self.assertIn("attach_layer_offloading(", model)
        self.assertIn("component=\"transformer\"", model)
        self.assertIn("component=\"text_encoder\"", model)
        self.assertIn("unreleased Diffusers `PRXPixelPipeline`", package_readme)


class PRXPixelBehaviorTest(unittest.TestCase):
    def test_tiny_transformer_forward_shape_with_attention_mask(self):
        require_prx_imports()
        transformer = PRXTransformer2DModel(
            in_channels=3,
            patch_size=2,
            context_in_dim=4,
            hidden_size=8,
            mlp_ratio=2.0,
            num_heads=2,
            depth=1,
            axes_dim=[2, 2],
            bottleneck_size=6,
            resolution_embeds=True,
        )

        hidden_states = torch.randn(1, 3, 4, 4)
        text = torch.randn(1, 3, 4)
        mask = torch.tensor([[True, True, False]])
        output = transformer(
            hidden_states=hidden_states,
            timestep=torch.tensor([0.5]),
            encoder_hidden_states=text,
            attention_mask=mask,
            return_dict=False,
        )[0]

        self.assertEqual(output.shape, hidden_states.shape)

    def test_model_hooks_match_pixel_space_contract(self):
        require_prx_imports()
        model = object.__new__(PRXPixelT2IModel)
        model.vae_scale_factor = 1
        model.patch_size = 16
        model.noise_scale = NOISE_SCALE

        latents = torch.arange(12, dtype=torch.float32).view(1, 3, 2, 2)
        batch = types.SimpleNamespace(latents=latents)

        self.assertEqual(PROMPT_MAX_TOKENS, 256)
        self.assertEqual(NOISE_SCALE, 2.0)
        self.assertEqual(model.get_bucket_divisibility(), 16)
        loss_target = model.get_loss_target(batch=batch)
        self.assertFalse(loss_target.requires_grad)
        self.assertEqual(loss_target.data_ptr(), latents.data_ptr())

        torch.manual_seed(123)
        expected = torch.randn_like(latents) * NOISE_SCALE
        torch.manual_seed(123)
        actual = model.get_latent_noise_from_latents(latents, noise_offset=0.0)
        self.assertTrue(torch.equal(actual, expected))

        state = {
            "transformer.blocks.0.attn.weight": torch.ones(1),
            "other.weight": torch.ones(1),
        }
        saved = model.convert_lora_weights_before_save(state)
        self.assertIn("diffusion_model.blocks.0.attn.weight", saved)
        loaded = model.convert_lora_weights_before_load(saved)
        self.assertIn("transformer.blocks.0.attn.weight", loaded)

    def test_get_noise_prediction_scales_timestep_and_preserves_attention_mask(self):
        require_prx_imports()

        class FakeTransformer:
            def __init__(self):
                self.device = torch.device("cpu")
                self.dtype = torch.float32
                self.received_timestep = None
                self.received_mask = None

            def to(self, *args, **kwargs):
                if args:
                    self.device = torch.device(args[0])
                return self

            def __call__(
                self,
                *,
                hidden_states,
                timestep,
                encoder_hidden_states,
                attention_mask,
                return_dict,
            ):
                self.received_timestep = timestep.detach().clone()
                self.received_mask = attention_mask.detach().clone()
                return (hidden_states + encoder_hidden_states.mean() * 0.0,)

        model = object.__new__(PRXPixelT2IModel)
        model.device_torch = torch.device("cpu")
        model.torch_dtype = torch.float32
        fake_transformer = FakeTransformer()
        model.model = fake_transformer

        embeds = PromptEmbeds(torch.ones(1, 2, 4))
        embeds.attention_mask = torch.tensor([[True, False]])
        latents = torch.zeros(1, 3, 2, 2)

        prediction = model.get_noise_prediction(
            latent_model_input=latents,
            timestep=torch.tensor([250.0]),
            text_embeddings=embeds,
        )

        self.assertEqual(prediction.shape, latents.shape)
        self.assertTrue(
            torch.equal(fake_transformer.received_timestep, torch.tensor([0.25]))
        )
        self.assertTrue(torch.equal(fake_transformer.received_mask, embeds.attention_mask))

    def test_preview_sampler_converts_x0_prediction_to_velocity_with_clamp(self):
        require_prx_imports()

        class FakeTransformer:
            in_channels = 3

            def __call__(
                self,
                *,
                hidden_states,
                timestep,
                encoder_hidden_states,
                attention_mask,
                return_dict,
            ):
                value = float(encoder_hidden_states[0, 0, 0].item()) * 0.2
                return (torch.full_like(hidden_states, value),)

        class FakeScheduler:
            def __init__(self):
                self.timesteps = None
                self.seen_velocity = None

            def set_timesteps(self, num_inference_steps, device):
                self.timesteps = torch.tensor([10.0], device=device)

            def step(self, model_output, timestep, sample, return_dict):
                self.seen_velocity = model_output.detach().clone()
                return (sample - model_output * 0.0,)

        class FakeModel:
            def __init__(self):
                self.device_torch = torch.device("cpu")
                self.torch_dtype = torch.float32
                self.transformer = FakeTransformer()
                self.noise_scale = 2.0
                self.scheduler = FakeScheduler()

            def get_train_scheduler(self):
                return self.scheduler

            def decode_latents(self, latents, device=None, dtype=None):
                return latents

        model = FakeModel()
        pipeline = PRXPixelPipeline(model)
        cond = PromptEmbeds(torch.ones(1, 1, 4))
        cond.attention_mask = torch.tensor([[True]])
        uncond = PromptEmbeds(torch.zeros(1, 1, 4))
        uncond.attention_mask = torch.tensor([[True]])
        latents = torch.ones(1, 3, 2, 2)

        image = pipeline(
            conditional_embeds=cond,
            unconditional_embeds=uncond,
            height=2,
            width=2,
            num_inference_steps=1,
            guidance_scale=2.0,
            latents=latents,
        )[0]

        expected_x0 = torch.full_like(latents, 0.4)
        expected_velocity = (latents - expected_x0) / X_PRED_T_MIN
        self.assertTrue(torch.allclose(model.scheduler.seen_velocity, expected_velocity))
        self.assertEqual(image.size, (2, 2))


if __name__ == "__main__":
    unittest.main()
