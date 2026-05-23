import ast
import hashlib
import types
import unittest
from pathlib import Path
from typing import Sequence, Tuple

import torch
import yaml
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLM_MODEL_PATH = (
    PROJECT_ROOT
    / "extensions_built_in"
    / "diffusion_models"
    / "glm_image"
    / "glm_image.py"
)
REGISTRY_PATH = PROJECT_ROOT / "extensions_built_in" / "diffusion_models" / "__init__.py"
HIDREAM_INIT_PATH = PROJECT_ROOT / "extensions_built_in" / "diffusion_models" / "hidream" / "__init__.py"
UI_OPTIONS_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "options.ts"
AUTO_PROFILES_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "autoTrainingProfiles.ts"
TRAINING_PHASES_EDITOR_PATH = (
    PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "TrainingPhasesEditor.tsx"
)
SIMPLE_JOB_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "SimpleJob.tsx"
EXAMPLE_PATH = PROJECT_ROOT / "config" / "examples" / "train_lora_glm_image_auto_24gb.yaml"


def load_glm_module():
    source = GLM_MODEL_PATH.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(GLM_MODEL_PATH))
    class_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == "GlmImageModel"
    )
    method_names = {
        "_text_embeds_to_tensor",
        "_prior_cache_key",
        "_prior_seed_for_key",
        "_ensure_prior_encoder_on_device",
        "_get_prior_tokens_for_prompts",
        "_build_prior_token_inputs",
        "_prepare_sampling_prompt_embeds",
        "_left_pad_prompt_embed_tensor",
    }
    methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    test_class = ast.ClassDef(
        name="GlmImageModel",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    test_module = ast.Module(body=[test_class], type_ignores=[])
    ast.fix_missing_locations(test_module)

    namespace = {
        "torch": torch,
        "hashlib": hashlib,
        "Sequence": Sequence,
        "Tuple": Tuple,
        "AdvancedPromptEmbeds": AdvancedPromptEmbeds,
    }
    exec(compile(test_module, str(GLM_MODEL_PATH), "exec"), namespace)
    return types.SimpleNamespace(
        GlmImageModel=namespace["GlmImageModel"],
        AdvancedPromptEmbeds=AdvancedPromptEmbeds,
    )


class GlmImageStaticSupportTest(unittest.TestCase):
    def test_backend_registry_includes_glm_image(self):
        source = REGISTRY_PATH.read_text(encoding="utf-8")

        self.assertIn('GlmImageModel, = _optional_models(".glm_image"', source)
        self.assertIn("GlmImageModel,", source)

    def test_optional_hidream_o1_import_is_guarded(self):
        registry_source = REGISTRY_PATH.read_text(encoding="utf-8")
        hidream_source = HIDREAM_INIT_PATH.read_text(encoding="utf-8")

        self.assertIn("def _unavailable_model_class", registry_source)
        self.assertIn("except ImportError as e:", registry_source)
        self.assertIn('"HidreamO1Model", "hidream_o1"', registry_source)
        self.assertNotIn("from .hidream_o1_model import HidreamO1Model", hidream_source.split("def __getattr__")[0])
        self.assertIn('if name == "HidreamO1Model":', hidream_source)

    def test_optional_qwen_image_imports_are_guarded(self):
        registry_source = REGISTRY_PATH.read_text(encoding="utf-8")

        self.assertIn('".qwen_image"', registry_source)
        self.assertIn('"QwenImageModel", "qwen_image"', registry_source)
        self.assertIn('"QwenImageEditModel", "qwen_image_edit"', registry_source)
        self.assertIn(
            '"QwenImageEditPlusModel", "qwen_image_edit_plus"', registry_source
        )

    def test_ui_model_defaults_and_default_auto_profile(self):
        source = UI_OPTIONS_PATH.read_text(encoding="utf-8")
        start = source.index("name: 'glm_image'")
        end = source.index("disableSections", start)
        block = source[start:end]

        self.assertIn("defaultAutoTrainingProfileId: 'glm-image-balanced-lora'", block)
        self.assertIn("'config.process[0].model.name_or_path': ['zai-org/GLM-Image', defaultNameOrPath]", block)
        self.assertIn("'config.process[0].model.quantize': [true, false]", block)
        self.assertIn("'config.process[0].model.quantize_te': [true, false]", block)
        self.assertIn("'config.process[0].model.low_vram': [true, false]", block)
        self.assertIn("'config.process[0].sample.sampler': ['flowmatch', 'flowmatch']", block)
        self.assertIn("'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch']", block)
        self.assertIn("'config.process[0].train.timestep_type': ['weighted', 'sigmoid']", block)
        self.assertIn("'config.process[0].sample.guidance_scale': [1.5, 4]", block)
        self.assertIn("'config.process[0].sample.sample_steps': [50, 30]", block)

    def test_glm_sampling_uses_prior_tokens_when_prompt_embeds_are_forwarded(self):
        source = GLM_MODEL_PATH.read_text(encoding="utf-8")
        start = source.index("    def generate_single_image(")
        end = source.index("    def _prepare_sampling_prompt_embeds", start)
        block = source[start:end]

        self.assertIn("prior_token_ids = self._get_prior_tokens_for_prompts", block)
        self.assertIn("prompt=None", block)
        self.assertIn("prior_token_ids=prior_token_ids", block)
        self.assertNotIn("prompt=gen_config.prompt", block)

    def test_auto_training_profiles_are_model_scoped(self):
        source = AUTO_PROFILES_PATH.read_text(encoding="utf-8")

        self.assertIn("modelArchs?: string[]", source)
        self.assertIn("id: 'glm-image-balanced-lora'", source)
        self.assertIn("id: 'glm-image-low-vram-lora'", source)
        self.assertEqual(source.count("modelArchs: ['glm_image']"), 2)
        self.assertIn("loraNetwork(32, { transformer_only: true })", source)
        self.assertIn("loraNetwork(16, { dropout: 0.05, transformer_only: true })", source)
        self.assertIn("dropout: 0.05", source)
        self.assertIn("gradient_accumulation: 2", source)
        self.assertIn("phase('Polish style', 0.000015", source)
        self.assertIn("phase('Polish style', 0.00001", source)

    def test_lokr_auto_training_profiles_do_not_set_normal_dropout(self):
        source = AUTO_PROFILES_PATH.read_text(encoding="utf-8")

        for profile_id in ("anatomy-lokr", "anatomy-realism-lokr"):
            start = source.index(f"id: '{profile_id}'")
            end = source.find("\n  {", start + 1)
            block = source[start:] if end == -1 else source[start:end]

            self.assertIn("type: 'lokr'", block)
            self.assertNotIn("dropout:", block)

    def test_training_phase_editor_filters_profiles_by_model(self):
        editor_source = TRAINING_PHASES_EDITOR_PATH.read_text(encoding="utf-8")
        simple_job_source = SIMPLE_JOB_PATH.read_text(encoding="utf-8")

        self.assertIn("modelArchName?: string", editor_source)
        self.assertIn("defaultAutoTrainingProfileId?: string", editor_source)
        self.assertIn("isAutoTrainingProfileCompatible(profile, activeArch)", editor_source)
        self.assertIn("modelArchs: activeArch ? [activeArch] : undefined", editor_source)
        self.assertIn("modelArchName={modelArch?.name}", simple_job_source)
        self.assertIn("defaultAutoTrainingProfileId={modelArch?.defaultAutoTrainingProfileId}", simple_job_source)

    def test_example_config_uses_auto_train_phases_without_fixed_phase_steps(self):
        config = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))
        process = config["config"]["process"][0]

        self.assertEqual(process["model"]["arch"], "glm_image")
        self.assertEqual(process["model"]["name_or_path"], "zai-org/GLM-Image")
        self.assertTrue(process["train"]["auto_train"])
        self.assertTrue(process["train"]["save_on_phase_change"])
        self.assertEqual(process["sample"]["guidance_scale"], 1.5)
        self.assertEqual(process["sample"]["sample_steps"], 50)
        for phase in process["train"]["phases"]:
            self.assertNotIn("steps", phase)
            self.assertIn("auto_advance", phase)


class FakePriorEncoder:
    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, device, *args, **kwargs):
        self.device = torch.device(device)
        return self


class FakeGlmPipeline:
    vae_scale_factor = 8

    def __init__(self):
        self.vision_language_encoder = FakePriorEncoder()
        self.calls = 0

    def generate_prior_tokens(self, prompt, height, width, image=None, device=None, generator=None):
        self.calls += 1
        base = len(prompt) + height + width
        return torch.full((1, 4), base, dtype=torch.long, device=device), None, None


class GlmImageModelBehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.glm_module = load_glm_module()
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc

    def make_model(self):
        model = object.__new__(self.glm_module.GlmImageModel)
        model.device_torch = torch.device("cpu")
        model.torch_dtype = torch.float32
        model.pipeline = FakeGlmPipeline()
        model._prior_token_cache = {}
        return model

    def test_prior_tokens_are_cached_and_cfg_drops_unconditional_rows(self):
        model = self.make_model()

        prior_token_ids, prior_token_drop = model._build_prior_token_inputs(
            ["a cat"],
            latent_batch_size=2,
            pixel_height=128,
            pixel_width=128,
        )
        second_ids, second_drop = model._build_prior_token_inputs(
            ["a cat"],
            latent_batch_size=2,
            pixel_height=128,
            pixel_width=128,
        )

        self.assertEqual(model.pipeline.calls, 1)
        self.assertTrue(torch.equal(prior_token_ids, second_ids))
        self.assertTrue(torch.equal(prior_token_drop, second_drop))
        self.assertEqual(prior_token_ids.shape, (2, 4))
        self.assertTrue(prior_token_drop[0].all())
        self.assertFalse(prior_token_drop[1].any())

    def test_text_embedding_lists_are_left_padded_for_glm_transformer(self):
        model = self.make_model()
        embeds = self.glm_module.AdvancedPromptEmbeds(
            text_embeds=[
                torch.ones((2, 3), dtype=torch.float32),
                torch.full((4, 3), 2.0, dtype=torch.float32),
            ]
        )

        padded = model._text_embeds_to_tensor(embeds)

        self.assertEqual(padded.shape, (2, 4, 3))
        self.assertTrue(torch.equal(padded[0, :2], torch.zeros((2, 3))))
        self.assertTrue(torch.equal(padded[0, 2:], torch.ones((2, 3))))
        self.assertTrue(torch.equal(padded[1], torch.full((4, 3), 2.0)))

    def test_sampling_prompt_embeds_are_padded_to_matching_sequence_lengths(self):
        model = self.make_model()
        conditional = self.glm_module.AdvancedPromptEmbeds(
            text_embeds=[torch.ones((3, 2), dtype=torch.float32)]
        )
        unconditional = self.glm_module.AdvancedPromptEmbeds(
            text_embeds=[torch.full((1, 2), 2.0, dtype=torch.float32)]
        )

        prompt_embeds, negative_prompt_embeds = model._prepare_sampling_prompt_embeds(
            conditional,
            unconditional,
        )

        self.assertEqual(prompt_embeds.shape, (1, 3, 2))
        self.assertEqual(negative_prompt_embeds.shape, (1, 3, 2))
        self.assertTrue(torch.equal(negative_prompt_embeds[0, :2], torch.zeros((2, 2))))
        self.assertTrue(
            torch.equal(negative_prompt_embeds[0, 2:], torch.full((1, 2), 2.0))
        )


if __name__ == "__main__":
    unittest.main()
