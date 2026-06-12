import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from toolkit.prompt_utils import PromptEmbeds


PROJECT_ROOT = Path(__file__).resolve().parents[1]
I1_DIR = PROJECT_ROOT / "extensions_built_in" / "diffusion_models" / "i1"
I1_MODEL_PATH = I1_DIR / "i1_model.py"
I1_INIT_PATH = I1_DIR / "__init__.py"
I1_PIPELINE_PATH = I1_DIR / "src" / "pipeline.py"
REGISTRY_PATH = PROJECT_ROOT / "extensions_built_in" / "diffusion_models" / "__init__.py"
TRAIN_PROCESS_PATH = PROJECT_ROOT / "jobs" / "process" / "BaseSDTrainProcess.py"
UI_OPTIONS_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "options.ts"
AUTO_PROFILES_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "autoTrainingProfiles.ts"
EXAMPLE_PATH = PROJECT_ROOT / "config" / "examples" / "train_lora_i1_24gb.yaml"


def load_i1_pipeline_module():
    spec = importlib.util.spec_from_file_location("i1_pipeline_for_test", I1_PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_i1_conversion_class():
    source = I1_MODEL_PATH.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(I1_MODEL_PATH))
    class_node = next(
        node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "I1Model"
    )
    method_names = {
        "convert_lora_weights_before_save",
        "convert_lora_weights_before_load",
        "get_transformer_block_names",
    }
    methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    test_class = ast.ClassDef(
        name="I1Model",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    test_module = ast.Module(body=[test_class], type_ignores=[])
    ast.fix_missing_locations(test_module)
    namespace = {"Optional": Optional, "List": List}
    exec(compile(test_module, str(I1_MODEL_PATH), "exec"), namespace)
    return namespace["I1Model"]


class I1StaticSupportTest(unittest.TestCase):
    def test_backend_registry_includes_i1_and_lazy_import_errors_are_guarded(self):
        source = REGISTRY_PATH.read_text(encoding="utf-8")

        self.assertIn('I1Model, = _optional_models(".i1"', source)
        self.assertIn('"I1Model", "i1"', source)
        self.assertIn("except ImportError as e:", source)
        self.assertIn("getattr(module, class_name)", source)

    def test_i1_package_exports_are_lazy_for_light_helper_imports(self):
        source = I1_INIT_PATH.read_text(encoding="utf-8")

        self.assertIn("def __getattr__(name):", source)
        self.assertIn('if name == "I1Model":', source)
        self.assertNotIn("from .i1_model import I1Model\n\n__all__", source)

    def test_ui_model_defaults(self):
        source = UI_OPTIONS_PATH.read_text(encoding="utf-8")
        start = source.index("name: 'i1'")
        end = source.index("disableSections", start)
        block = source[start:end]

        self.assertIn("'config.process[0].model.name_or_path': ['zlab-princeton/i1-3B', defaultNameOrPath]", block)
        self.assertIn("'config.process[0].model.dtype': ['bf16', 'bf16']", block)
        self.assertIn("'config.process[0].sample.sampler': ['flowmatch', 'flowmatch']", block)
        self.assertIn("'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch']", block)
        self.assertIn("'config.process[0].train.timestep_type': ['i1_lognorm', 'sigmoid']", block)
        self.assertIn("'config.process[0].sample.width': [1024, 1024]", block)
        self.assertIn("'config.process[0].sample.guidance_scale': [12, 4]", block)
        self.assertIn("'config.process[0].sample.guidance_rescale': [1.0, 0.0]", block)
        self.assertIn("'config.process[0].datasets[x].resolution': [[1024], [512, 768, 1024]]", block)
        self.assertIn("'config.process[0].datasets[x].square_crop': [true, false]", block)
        self.assertIn("'config.process[0].sample.sample_steps': [50, 30]", block)
        self.assertIn("checkpoint_filename: '1024_resolution_checkpoint_torch.pt'", block)
        self.assertIn("rewriter_model: 'Qwen/Qwen3-4B-Instruct-2507'", block)
        self.assertIn("rewrite_prompt: false", block)

    def test_auto_training_profiles_include_i1_as_flow_image_arch(self):
        source = AUTO_PROFILES_PATH.read_text(encoding="utf-8")
        start = source.index("const flowImageArchs")
        end = source.index("];", start)
        block = source[start:end]

        self.assertIn("'i1'", block)

    def test_example_config_defaults(self):
        config = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))
        process = config["config"]["process"][0]

        self.assertEqual(process["model"]["arch"], "i1")
        self.assertEqual(process["model"]["name_or_path"], "zlab-princeton/i1-3B")
        self.assertTrue(process["model"]["quantize"])
        self.assertTrue(process["model"]["quantize_te"])
        self.assertEqual(process["network"]["linear"], 32)
        self.assertTrue(process["network"]["transformer_only"])
        self.assertEqual(process["train"]["noise_scheduler"], "flowmatch")
        self.assertEqual(process["train"]["timestep_type"], "i1_lognorm")
        self.assertTrue(process["train"]["cache_text_embeddings"])
        self.assertEqual(process["datasets"][0]["resolution"], [1024])
        self.assertTrue(process["datasets"][0]["square_crop"])
        self.assertEqual(process["sample"]["guidance_scale"], 12)
        self.assertEqual(process["sample"]["guidance_rescale"], 1.0)
        self.assertEqual(process["sample"]["sample_steps"], 50)
        self.assertEqual(
            process["model"]["model_kwargs"]["text_encoder_name_or_path"],
            "google/t5gemma-2b-2b-ul2-it",
        )

    def test_i1_save_checkpoint_filename_is_restricted_to_basename(self):
        source = I1_MODEL_PATH.read_text(encoding="utf-8")

        self.assertIn("def _safe_i1_checkpoint_save_filename", source)
        self.assertIn("os.path.basename(save_filename) != save_filename", source)
        self.assertIn("ntpath.basename(save_filename) != save_filename", source)
        self.assertIn(
            "os.path.join(\n"
            "                output_path, _safe_i1_checkpoint_save_filename(self.checkpoint_filename)\n"
            "            )",
            source,
        )
        self.assertNotIn(
            'os.path.join(output_path, self.checkpoint_filename.replace(".pt", ".safetensors"))',
            source,
        )

    def test_training_loop_has_i1_lognorm_defaults_and_branch(self):
        source = TRAIN_PROCESS_PATH.read_text(encoding="utf-8")

        self.assertIn("model_config.get('arch') == 'i1'", source)
        self.assertIn("raw_train_config.setdefault('timestep_type', 'i1_lognorm')", source)
        self.assertIn("raw_dataset['resolution'] = [1024]", source)
        self.assertIn("raw_dataset['square_crop'] = True", source)
        self.assertIn("self.train_config.timestep_type == 'i1_lognorm'", source)
        self.assertIn("shift = 0.3", source)
        self.assertIn("((1.0 - t) * (num_train_timesteps - 1)).long()", source)

    def test_gated_flux2_vae_has_clear_error_and_token_support(self):
        source = I1_MODEL_PATH.read_text(encoding="utf-8")

        self.assertIn("from huggingface_hub import get_token, hf_hub_download", source)
        self.assertIn("self.model_config.model_kwargs.get(\"use_auth_token\", None)", source)
        self.assertIn("def _hf_token_kwargs", source)
        self.assertIn("from toolkit.exceptions import UserFacingError", source)
        self.assertIn("except (GatedRepoError, OSError) as exc:", source)
        self.assertIn("raise UserFacingError", source)
        self.assertIn("Access required for black-forest-labs/FLUX.2-dev", source)
        self.assertIn("Request and accept", source)
        self.assertIn("https://huggingface.co/black-forest-labs/FLUX.2-dev", source)
        self.assertIn("i1 uses the gated FLUX.2 VAE", source)
        self.assertIn("hf auth login", source)
        self.assertIn("def _vae_load_kwargs", source)


class I1BehaviorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_i1_pipeline_module()
        cls.I1ConversionModel = load_i1_conversion_class()

    def test_rectified_flow_timestep_sampling_and_velocity_target(self):
        torch.manual_seed(0)
        sampled = self.pipeline.sample_i1_lognorm_timesteps(128, torch.device("cpu"), shift=0.3)
        self.assertTrue(torch.all(sampled > 0.0))
        self.assertTrue(torch.all(sampled < 1.0))

        latents = torch.full((3, 1, 1, 1), 10.0)
        noise = torch.full((3, 1, 1, 1), 2.0)
        timesteps = torch.tensor([0.0, 500.0, 1000.0])
        noisy = self.pipeline.i1_rectified_flow_noisy_latents(latents, noise, timesteps)

        self.assertTrue(torch.equal(noisy[:, 0, 0, 0], torch.tensor([2.0, 6.0, 10.0])))
        self.assertTrue(torch.equal(self.pipeline.i1_velocity_target(latents, noise), latents - noise))

    def test_flux2_latent_scale_round_trip_preserves_shape_and_values(self):
        latents = torch.randn((2, 32, 8, 8), dtype=torch.float32)

        scaled = self.pipeline.scale_flux2_latents(latents)
        restored = self.pipeline.reverse_scale_flux2_latents(scaled)

        self.assertEqual(scaled.shape, latents.shape)
        self.assertEqual(restored.shape, latents.shape)
        self.assertTrue(torch.allclose(restored, latents, atol=1e-5, rtol=1e-5))

    def test_prompt_embed_object_carries_hidden_states_and_attention_mask(self):
        hidden = torch.randn((2, 256, 2304))
        mask = torch.ones((2, 256), dtype=torch.bool)

        embeds = PromptEmbeds(hidden, attention_mask=mask)

        self.assertIs(embeds.text_embeds, hidden)
        self.assertIs(embeds.attention_mask, mask)

    def test_latent_initialization_accepts_cpu_generator(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(123)

        latents = self.pipeline.randn_i1_latents(
            (1, 32, 8, 8),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            generator=generator,
        )

        self.assertEqual(latents.shape, (1, 32, 8, 8))
        self.assertEqual(latents.dtype, torch.bfloat16)

    def test_i1_tensor_prep_normalizes_aspect_bucket_shapes(self):
        image = torch.zeros((3, 576, 1024), dtype=torch.float32)
        prepared_image = self.pipeline.prepare_i1_image_tensor(image, resolution=1024)

        self.assertEqual(prepared_image.shape, (3, 1024, 1024))

        latents = torch.zeros((2, 32, 72, 128), dtype=torch.float32)
        prepared_latents = self.pipeline.prepare_i1_latent_tensor(latents, latent_size=128)

        self.assertEqual(prepared_latents.shape, (2, 32, 128, 128))

    def test_cfg_and_rescale_match_upstream_formula(self):
        cond = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        uncond = torch.zeros_like(cond)
        guidance = 12.0

        guided = self.pipeline.apply_i1_cfg_rescale(cond, uncond, guidance, None)
        expected_guided = cond + (guidance - 1.0) * (cond - uncond)
        self.assertTrue(torch.equal(guided, expected_guided))

        rescaled = self.pipeline.apply_i1_cfg_rescale(cond, uncond, guidance, 1.0)
        axes = tuple(range(1, expected_guided.ndim))
        factor = cond.float().std(dim=axes, keepdim=True) / (
            expected_guided.float().std(dim=axes, keepdim=True) + 1e-8
        )
        expected_rescaled = expected_guided * factor
        self.assertTrue(torch.allclose(rescaled, expected_rescaled))

    def test_lora_target_paths_and_prefix_conversion(self):
        model = self.I1ConversionModel()

        self.assertEqual(
            model.get_transformer_block_names(),
            ["in_blocks", "mid_block", "out_blocks", "text_encoder_adapter"],
        )
        saved = model.convert_lora_weights_before_save(
            {"transformer.in_blocks.0.attn.qkv.lora_down.weight": torch.ones(1)}
        )
        self.assertIn("diffusion_model.in_blocks.0.attn.qkv.lora_down.weight", saved)

        loaded = model.convert_lora_weights_before_load(
            {"diffusion_model.out_blocks.0.attn.proj.lora_up.weight": torch.ones(1)}
        )
        self.assertIn("transformer.out_blocks.0.attn.proj.lora_up.weight", loaded)


if __name__ == "__main__":
    unittest.main()
