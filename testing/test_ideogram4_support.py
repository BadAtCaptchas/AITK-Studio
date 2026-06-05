import json
import tempfile
import types
import unittest
from pathlib import Path
from typing import get_args
from unittest import mock

import torch
import yaml
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IDEOGRAM_ROOT = PROJECT_ROOT / "extensions_built_in" / "diffusion_models" / "ideogram4"
REGISTRY_PATH = PROJECT_ROOT / "extensions_built_in" / "diffusion_models" / "__init__.py"
CONFIG_MODULES_PATH = PROJECT_ROOT / "toolkit" / "config_modules.py"
LORA_SPECIAL_PATH = PROJECT_ROOT / "toolkit" / "lora_special.py"
NETWORK_MIXINS_PATH = PROJECT_ROOT / "toolkit" / "network_mixins.py"
UI_OPTIONS_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "options.ts"
MEMORY_PROFILE_PATH = PROJECT_ROOT / "ui" / "src" / "utils" / "memoryProfiles.ts"
AUTO_PROFILES_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "autoTrainingProfiles.ts"
ADVISOR_PATH = PROJECT_ROOT / "ui" / "src" / "server" / "trainingAdvisor.ts"
SDTRAINER_PATH = PROJECT_ROOT / "extensions_built_in" / "sd_trainer" / "SDTrainer.py"


try:
    from extensions_built_in.diffusion_models.ideogram4.ideogram4_model import (
        Ideogram4Model,
        Ideogram4PipelineConfig,
        _load_subfolder_state_dict_local_or_hf,
        dequantize_fp8_linears,
        infer_ideogram4_quantization,
        patchify_latents,
        unpatchify_latents,
    )
    from extensions_built_in.diffusion_models.ideogram4.src.caption_verifier import (
        CaptionVerifier,
    )
    from extensions_built_in.diffusion_models.ideogram4.src.autoencoder import (
        AutoEncoder,
    )
    from extensions_built_in.diffusion_models.ideogram4.src.quantized_loading import (
        Fp8Linear,
    )

    IDEOGRAM_IMPORT_ERROR = None
except ImportError as exc:
    Ideogram4Model = None
    Ideogram4PipelineConfig = None
    CaptionVerifier = None
    AutoEncoder = None
    Fp8Linear = None
    _load_subfolder_state_dict_local_or_hf = None
    IDEOGRAM_IMPORT_ERROR = exc


def require_ideogram_imports():
    if IDEOGRAM_IMPORT_ERROR is not None:
        raise unittest.SkipTest(str(IDEOGRAM_IMPORT_ERROR))


class Ideogram4StaticSupportTest(unittest.TestCase):
    def test_registry_and_model_arch_include_ideogram4(self):
        registry_source = REGISTRY_PATH.read_text(encoding="utf-8")
        config_source = CONFIG_MODULES_PATH.read_text(encoding="utf-8")

        self.assertIn('Ideogram4Model, = _optional_models(', registry_source)
        self.assertIn('"Ideogram4Model", "ideogram4"', registry_source)
        self.assertIn("Ideogram4Model,", registry_source)
        self.assertIn("'ideogram4'", config_source)

    def test_model_class_resolves_when_optional_dependencies_are_available(self):
        require_ideogram_imports()
        torchaudio_module = types.ModuleType("torchaudio")
        torchaudio_module.save = mock.Mock()
        album_artwork_module = types.ModuleType("toolkit.audio.album_artwork")
        album_artwork_module.add_album_artwork = mock.Mock()

        try:
            with mock.patch.dict(
                "sys.modules",
                {
                    "torchaudio": torchaudio_module,
                    "toolkit.audio.album_artwork": album_artwork_module,
                },
            ):
                from toolkit.config_modules import ModelConfig
                from toolkit.util.get_model import get_model_class
        except ImportError as exc:
            raise unittest.SkipTest(str(exc)) from exc

        config = ModelConfig(
            arch="ideogram4",
            name_or_path="ideogram-ai/ideogram-4-nf4",
            model_kwargs={"quantization": "nf4"},
        )

        self.assertIs(get_model_class(config), Ideogram4Model)

    def test_vendored_source_has_attribution_and_no_external_api_modules(self):
        readme = (IDEOGRAM_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertTrue((IDEOGRAM_ROOT / "LICENSE.md").exists())
        self.assertIn("https://github.com/ideogram-oss/ideogram4", readme)
        self.assertIn("Apache-2.0", readme)
        self.assertFalse((IDEOGRAM_ROOT / "src" / "magic_prompt.py").exists())
        self.assertFalse((IDEOGRAM_ROOT / "src" / "safety.py").exists())

        combined_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (IDEOGRAM_ROOT / "src").glob("*.py")
        )
        self.assertNotIn("developer.ideogram", combined_source)
        self.assertNotIn("IDEOGRAM_API_KEY", combined_source)
        self.assertNotIn("MAGIC_PROMPT_API_KEY", combined_source)
        self.assertNotIn("import requests", combined_source)

    def test_lora_allowlists_include_fp8_linear(self):
        self.assertIn("Fp8Linear", LORA_SPECIAL_PATH.read_text(encoding="utf-8"))
        self.assertIn("Fp8Linear", NETWORK_MIXINS_PATH.read_text(encoding="utf-8"))

    def test_sample_prompt_cache_uses_model_sample_prepare_hook(self):
        trainer_source = SDTRAINER_PATH.read_text(encoding="utf-8")

        self.assertIn("prepare_sample_image_config_for_encoding", trainer_source)

    def test_ui_defaults_memory_profile_and_auto_profile(self):
        options = UI_OPTIONS_PATH.read_text(encoding="utf-8")
        memory = MEMORY_PROFILE_PATH.read_text(encoding="utf-8")
        profiles = AUTO_PROFILES_PATH.read_text(encoding="utf-8")
        advisor = ADVISOR_PATH.read_text(encoding="utf-8")

        self.assertIn("name: 'ideogram4'", options)
        self.assertIn("label: 'Ideogram 4 NF4'", options)
        self.assertIn("'config.process[0].model.name_or_path': ['ideogram-ai/ideogram-4-nf4', defaultNameOrPath]", options)
        self.assertIn("name: 'ideogram4:fp8'", options)
        self.assertIn("label: 'Ideogram 4 FP8'", options)
        self.assertIn("'config.process[0].model.name_or_path': ['ideogram-ai/ideogram-4-fp8', defaultNameOrPath]", options)
        self.assertIn("require_json_captions: true", options)
        self.assertIn("caption_strict: false", options)
        self.assertIn("'ideogram4'", memory)
        self.assertIn("id: 'ideogram4-balanced-lora'", profiles)
        self.assertIn("const ideogram4Archs = ['ideogram4', 'ideogram4:fp8']", profiles)
        self.assertIn("ideogram", advisor)

    def test_example_configs_use_json_caption_defaults(self):
        examples = [
            ("train_lora_ideogram4_48gb.yaml", "ideogram-ai/ideogram-4-nf4", "nf4"),
            ("train_lora_ideogram4_fp8_48gb.yaml", "ideogram-ai/ideogram-4-fp8", "fp8"),
            ("train_full_fine_tune_ideogram4.yaml", "ideogram-ai/ideogram-4-fp8", "fp8"),
        ]

        for filename, repo, quantization in examples:
            with self.subTest(filename=filename):
                config = yaml.safe_load(
                    (PROJECT_ROOT / "config" / "examples" / filename).read_text(
                        encoding="utf-8"
                    )
                )
                process = config["config"]["process"][0]
                model = process["model"]

                self.assertEqual(model["arch"], "ideogram4")
                self.assertEqual(model["name_or_path"], repo)
                self.assertEqual(model["model_kwargs"]["quantization"], quantization)
                self.assertTrue(model["model_kwargs"]["require_json_captions"])
                self.assertFalse(model["model_kwargs"]["caption_strict"])
                self.assertFalse(process["train"]["train_text_encoder"])
                self.assertEqual(process["sample"]["sample_steps"], 20)
                self.assertIn('"compositional_deconstruction"', process["sample"]["prompts"][0])

        full_config = yaml.safe_load(
            (PROJECT_ROOT / "config" / "examples" / "train_full_fine_tune_ideogram4.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(
            full_config["config"]["process"][0]["model"]["model_kwargs"][
                "dequantize_fp8_transformer"
            ]
        )


class Ideogram4HelperBehaviorTest(unittest.TestCase):
    def setUp(self):
        require_ideogram_imports()

    def test_quantization_inference(self):
        self.assertEqual(
            infer_ideogram4_quantization("ideogram-ai/ideogram-4-nf4", {}),
            "nf4",
        )
        self.assertEqual(
            infer_ideogram4_quantization("ideogram-ai/ideogram-4-fp8", {}),
            "fp8",
        )
        self.assertEqual(
            infer_ideogram4_quantization(
                "local/path", {"quantization": "fp8"}
            ),
            "fp8",
        )
        with self.assertRaisesRegex(ValueError, "quantization"):
            infer_ideogram4_quantization("local/path", {"quantization": "int8"})

    def test_text_encoder_shard_loader_reports_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            text_encoder_dir = root / "text_encoder"
            text_encoder_dir.mkdir()
            save_file({"a": torch.ones(1)}, text_encoder_dir / "model-00001.safetensors")
            save_file({"b": torch.full((1,), 2.0)}, text_encoder_dir / "model-00002.safetensors")
            (text_encoder_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {
                            "a": "model-00001.safetensors",
                            "b": "model-00002.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )
            messages = []

            state_dict = _load_subfolder_state_dict_local_or_hf(
                str(root), "text_encoder", "model", status_callback=messages.append
            )

        self.assertEqual(set(state_dict.keys()), {"a", "b"})
        self.assertTrue(torch.equal(state_dict["a"], torch.ones(1)))
        self.assertTrue(torch.equal(state_dict["b"], torch.full((1,), 2.0)))
        self.assertEqual(len(messages), 2)
        self.assertIn("shard 1/2", messages[0])
        self.assertIn("shard 2/2", messages[1])

    def test_patchify_unpatchify_round_trip(self):
        latents = torch.arange(1 * 32 * 4 * 6, dtype=torch.float32).view(1, 32, 4, 6)

        patched = patchify_latents(latents)
        round_trip = unpatchify_latents(patched)

        self.assertEqual(patched.shape, (1, 128, 2, 3))
        self.assertTrue(torch.equal(round_trip, latents))

    def test_text_embedding_space_versions_default_and_ideogram_override(self):
        from toolkit.models.base_model import BaseModel
        from toolkit.stable_diffusion_model import StableDiffusion

        base_model = object.__new__(BaseModel)
        base_model.arch = "custom_arch"
        self.assertEqual(base_model.text_embedding_space_version, "custom_arch")

        sd_model = object.__new__(StableDiffusion)
        sd_model.arch = "sdxl"
        self.assertEqual(sd_model.text_embedding_space_version, "sdxl")

        ideogram_model = object.__new__(Ideogram4Model)
        self.assertEqual(ideogram_model.text_embedding_space_version, "ideogram4_te_v2")

    def test_ideogram4_text_token_cap_defaults_and_overrides(self):
        self.assertEqual(Ideogram4PipelineConfig().max_text_tokens, 3072)

        model = object.__new__(Ideogram4Model)
        model.model_config = types.SimpleNamespace(model_kwargs={})
        self.assertEqual(model._resolve_max_text_tokens(), 3072)

        model.model_config = types.SimpleNamespace(model_kwargs={"max_text_tokens": 4096})
        self.assertEqual(model._resolve_max_text_tokens(), 4096)

        model.model_config = types.SimpleNamespace(model_kwargs={"max_text_length": 1536})
        self.assertEqual(model._resolve_max_text_tokens(), 1536)

        model.model_config = types.SimpleNamespace(
            model_kwargs={"max_text_tokens": 2048, "max_text_length": 1536}
        )
        self.assertEqual(model._resolve_max_text_tokens(), 2048)

        model.model_config = types.SimpleNamespace(model_kwargs={"max_text_tokens": 0})
        with self.assertRaisesRegex(ValueError, "max_text_tokens"):
            model._resolve_max_text_tokens()

    def test_transformer_inputs_pad_llm_features_to_full_sequence(self):
        from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
        from toolkit.prompt_utils import PromptEmbeds
        from extensions_built_in.diffusion_models.ideogram4.src.constants import (
            OUTPUT_IMAGE_INDICATOR,
        )

        model = object.__new__(Ideogram4Model)
        model.device_torch = torch.device("cpu")
        first = torch.arange(2 * 5, dtype=torch.float32).view(2, 5)
        second = torch.arange(10, 25, dtype=torch.float32).view(3, 5)

        llm_features, position_ids, segment_ids, indicator, max_text_tokens = (
            model._build_transformer_inputs_from_embeds(
                AdvancedPromptEmbeds(text_embeds=[first, second]),
                latent_h=2,
                latent_w=4,
                include_text=True,
            )
        )

        self.assertEqual(max_text_tokens, 3)
        self.assertEqual(llm_features.shape, (2, 11, 5))
        self.assertEqual(position_ids.shape, (2, 11, 3))
        self.assertEqual(segment_ids.shape, (2, 11))
        self.assertEqual(indicator.shape, (2, 11))
        expected_text = torch.zeros(2, 3, 5)
        expected_text[0, 1:] = first
        expected_text[1] = second
        self.assertTrue(torch.equal(llm_features[:, :3], expected_text))
        self.assertTrue(torch.equal(llm_features[:, 3:], torch.zeros(2, 8, 5)))
        self.assertTrue(torch.all(indicator[:, 3:] == OUTPUT_IMAGE_INDICATOR))

        neg_llm, neg_position_ids, neg_segment_ids, neg_indicator, neg_text_tokens = (
            model._build_transformer_inputs_from_embeds(
                AdvancedPromptEmbeds(text_embeds=[first, second]),
                latent_h=2,
                latent_w=4,
                include_text=False,
            )
        )
        self.assertEqual(neg_text_tokens, 0)
        self.assertEqual(neg_llm.shape, (2, 8, 5))
        self.assertEqual(neg_position_ids.shape, (2, 8, 3))
        self.assertEqual(neg_segment_ids.shape, (2, 8))
        self.assertEqual(neg_indicator.shape, (2, 8))
        self.assertTrue(torch.equal(neg_llm, torch.zeros(2, 8, 5)))

        legacy_text = torch.arange(2 * 3 * 5, dtype=torch.float32).view(2, 3, 5)
        attention_mask = torch.tensor(
            [
                [False, True, True],
                [True, True, True],
            ]
        )
        legacy_features, _, _, _, legacy_text_tokens = (
            model._build_transformer_inputs_from_embeds(
                PromptEmbeds(legacy_text, attention_mask=attention_mask),
                latent_h=2,
                latent_w=4,
                include_text=True,
            )
        )
        expected_legacy = torch.zeros(2, 3, 5)
        expected_legacy[0, 1:] = legacy_text[0, 1:]
        expected_legacy[1] = legacy_text[1]
        self.assertEqual(legacy_text_tokens, 3)
        self.assertTrue(torch.equal(legacy_features[:, :3], expected_legacy))

    def test_autoencoder_exposes_diffusers_style_dtype_and_device(self):
        autoencoder = object.__new__(AutoEncoder)
        torch.nn.Module.__init__(autoencoder)
        autoencoder.register_parameter(
            "probe",
            torch.nn.Parameter(torch.zeros(1, dtype=torch.float32)),
        )

        self.assertEqual(autoencoder.device, torch.device("cpu"))
        self.assertEqual(autoencoder.dtype, torch.float32)

        autoencoder.to(dtype=torch.bfloat16)
        self.assertEqual(autoencoder.dtype, torch.bfloat16)

    def test_caption_validation_requires_non_empty_json_by_default(self):
        model = object.__new__(Ideogram4Model)
        model.model_config = types.SimpleNamespace(model_kwargs={})
        model.caption_verifier = CaptionVerifier()

        valid_caption = (
            '{"high_level_description":"A simple object.","compositional_deconstruction":'
            '{"background":"A plain background.","elements":[{"type":"obj","desc":"A cube."}]}}'
        )
        model._validate_caption(valid_caption)
        model._validate_caption("")

        with self.assertRaisesRegex(ValueError, "must be JSON"):
            model._validate_caption("plain text prompt")

    def test_plain_sample_prompts_are_wrapped_without_relaxing_training_captions(self):
        model = object.__new__(Ideogram4Model)
        model.model_config = types.SimpleNamespace(model_kwargs={})
        model.caption_verifier = CaptionVerifier()

        with self.assertWarnsRegex(UserWarning, "sample prompt"):
            wrapped = model._sample_prompt_to_json_caption("plain sample prompt")

        parsed = json.loads(wrapped)
        self.assertEqual(parsed["high_level_description"], "plain sample prompt")
        self.assertEqual(
            parsed["compositional_deconstruction"]["elements"][0]["desc"],
            "plain sample prompt",
        )
        model._validate_caption(wrapped)

        with self.assertRaisesRegex(ValueError, "must be JSON"):
            model._validate_caption("plain training caption")

    def test_json_sample_prompts_are_left_unchanged(self):
        model = object.__new__(Ideogram4Model)
        model.model_config = types.SimpleNamespace(model_kwargs={})
        model.caption_verifier = CaptionVerifier()
        prompt = (
            '{"high_level_description":"A simple object.","compositional_deconstruction":'
            '{"background":"A plain background.","elements":[{"type":"obj","desc":"A cube."}]}}'
        )

        self.assertEqual(model._sample_prompt_to_json_caption(prompt), prompt)

    def test_sample_prompt_prepare_hook_wraps_before_cache_encoding(self):
        from toolkit.config_modules import GenerateImageConfig

        model = object.__new__(Ideogram4Model)
        model.model_config = types.SimpleNamespace(model_kwargs={})
        model.caption_verifier = CaptionVerifier()
        sample_config = GenerateImageConfig(
            prompt="plain sample prompt",
            negative_prompt="plain negative sample prompt",
            output_path=str(PROJECT_ROOT / "tmp_sample.png"),
        )

        with self.assertWarnsRegex(UserWarning, "sample prompt"):
            model.prepare_sample_image_config_for_encoding(sample_config)

        self.assertEqual(
            json.loads(sample_config.prompt)["high_level_description"],
            "plain sample prompt",
        )
        self.assertEqual(
            json.loads(sample_config.negative_prompt)["high_level_description"],
            "plain negative sample prompt",
        )
        model._validate_caption(sample_config.prompt)
        model._validate_caption(sample_config.negative_prompt)

    def test_caption_schema_issues_warn_unless_strict(self):
        model = object.__new__(Ideogram4Model)
        model.caption_verifier = CaptionVerifier()
        model.model_config = types.SimpleNamespace(
            model_kwargs={"require_json_captions": True, "caption_strict": False}
        )

        with self.assertWarnsRegex(UserWarning, "caption verifier"):
            model._validate_caption("{}")

        model.model_config = types.SimpleNamespace(
            model_kwargs={"require_json_captions": True, "caption_strict": True}
        )
        with self.assertRaisesRegex(ValueError, "caption verifier"):
            model._validate_caption("{}")

    def test_dequantize_fp8_linears_replaces_buffers_with_trainable_linears(self):
        fp8 = Fp8Linear(3, 2, bias=True, compute_dtype=torch.float32)
        fp8.weight.copy_(
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to(fp8.weight.dtype)
        )
        fp8.weight_scale.copy_(torch.tensor([0.5, 0.25], dtype=torch.float32))
        fp8.bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float32))
        module = torch.nn.Sequential(fp8)

        replaced = dequantize_fp8_linears(
            module, dtype=torch.float32, device=torch.device("cpu")
        )

        self.assertEqual(replaced, 1)
        self.assertIsInstance(module[0], torch.nn.Linear)
        self.assertTrue(module[0].weight.requires_grad)
        self.assertTrue(module[0].bias.requires_grad)
        self.assertTrue(
            torch.allclose(
                module[0].weight,
                torch.tensor([[0.5, 1.0, 1.5], [1.0, 1.25, 1.5]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
