import importlib.util
import unittest
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOGU_MODEL_PATH = (
    PROJECT_ROOT
    / "extensions_built_in"
    / "diffusion_models"
    / "boogu_image"
    / "boogu_image.py"
)
BOOGU_VENDOR_ROOT = (
    PROJECT_ROOT
    / "extensions_built_in"
    / "diffusion_models"
    / "boogu_image"
    / "src"
    / "boogu"
)
BOOGU_TRANSFORMER_PATH = (
    BOOGU_VENDOR_ROOT / "models" / "transformers" / "transformer_boogu.py"
)
BOOGU_ROPE_PATH = BOOGU_VENDOR_ROOT / "models" / "transformers" / "rope.py"
BOOGU_PIPELINE_PATH = BOOGU_VENDOR_ROOT / "pipelines" / "boogu" / "pipeline_boogu.py"
BOOGU_TURBO_PIPELINE_PATH = (
    BOOGU_VENDOR_ROOT / "pipelines" / "boogu" / "pipeline_boogu_turbo.py"
)
REGISTRY_PATH = PROJECT_ROOT / "extensions_built_in" / "diffusion_models" / "__init__.py"
CONFIG_MODULES_PATH = PROJECT_ROOT / "toolkit" / "config_modules.py"
MEMORY_OFFLOAD_PATH = PROJECT_ROOT / "toolkit" / "memory_management" / "offload.py"
UI_MEMORY_PATH = PROJECT_ROOT / "ui" / "src" / "utils" / "memoryProfiles.ts"
UI_OPTIONS_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "options.ts"
AUTO_PROFILES_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "autoTrainingProfiles.ts"
README_PATH = PROJECT_ROOT / "README.md"
EXAMPLES = {
    "boogu_image": PROJECT_ROOT / "config" / "examples" / "train_lora_boogu_image_24gb.yaml",
    "boogu_image_edit": PROJECT_ROOT / "config" / "examples" / "train_lora_boogu_image_edit_24gb.yaml",
    "boogu_image_turbo": PROJECT_ROOT
    / "config"
    / "examples"
    / "train_lora_boogu_image_turbo_experimental_24gb.yaml",
}


def ts_object_block(source: str, arch_name: str) -> str:
    start = source.index(f"name: '{arch_name}'")
    end = source.find("\n  {", start + 1)
    return source[start:] if end == -1 else source[start:end]


class BooguImageSupportTest(unittest.TestCase):
    def test_backend_registry_includes_boogu_family(self):
        source = REGISTRY_PATH.read_text(encoding="utf-8")

        self.assertIn('".boogu_image"', source)
        self.assertIn('"BooguImageModel", "boogu_image"', source)
        self.assertIn('"BooguImageEditModel", "boogu_image_edit"', source)
        self.assertIn('"BooguImageTurboModel", "boogu_image_turbo"', source)
        self.assertIn("BooguImageModel,", source)
        self.assertIn("BooguImageEditModel,", source)
        self.assertIn("BooguImageTurboModel,", source)

    def test_model_adapter_has_training_and_edit_conditioning_hooks(self):
        source = BOOGU_MODEL_PATH.read_text(encoding="utf-8")

        self.assertIn('self.target_lora_modules = ["BooguImageTransformer2DModel"]', source)
        self.assertIn("return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)", source)
        self.assertIn("timestep = 1 - (timestep / 1000)", source)
        self.assertIn("return (batch.latents - noise).detach()", source)
        self.assertIn("ref_image_hidden_states=self._control_latent", source)
        self.assertIn('encode_control_in_text_embeddings = True', source)
        self.assertIn('has_multiple_control_images = True', source)
        self.assertIn('use_raw_control_images = True', source)
        self.assertIn('multi_controls = getattr(batch, "control_tensor_list", None)', source)
        self.assertIn("input_pil_images=input_images", source)
        self.assertIn('"ref_image_refiner"', source)
        self.assertIn("def _get_mllm_for_generation(self):", source)
        self.assertIn("if self.sample_prompts_cache is not None:", source)
        self.assertIn("return None", source)
        self.assertIn("def _is_fake_text_encoder(module) -> bool:", source)
        self.assertIn("mllm=self._get_mllm_for_generation()", source)
        self.assertIn('"device": str(self.device_torch)', source)
        self.assertIn("instruction = gen_config.prompt or \"\"", source)
        self.assertIn("negative_instruction = gen_config.negative_prompt or \"\"", source)
        self.assertIn('"instruction": instruction', source)
        self.assertIn('"negative_instruction": negative_instruction', source)
        self.assertIn("unexpected_device_kwarg = (", source)
        self.assertIn("if not unexpected_device_kwarg:", source)
        self.assertNotIn("mllm=unwrap_model(self.text_encoder[0])", source)

    def test_model_loader_uses_expected_boogu_components_and_rejects_fp8(self):
        source = BOOGU_MODEL_PATH.read_text(encoding="utf-8")

        self.assertIn('BOOGU_BASE_MODEL_PATH = "Boogu/Boogu-Image-0.1-Base"', source)
        self.assertIn('BOOGU_EDIT_MODEL_PATH = "Boogu/Boogu-Image-0.1-Edit"', source)
        self.assertIn('BOOGU_TURBO_MODEL_PATH = "Boogu/Boogu-Image-0.1-Turbo"', source)
        self.assertIn("BooguImageTransformer2DModel.from_pretrained", source)
        self.assertIn("Qwen3VLForConditionalGeneration.from_pretrained", source)
        self.assertIn("Qwen3VLProcessor.from_pretrained", source)
        self.assertIn("AutoencoderKL.from_pretrained", source)
        self.assertIn("BooguFlowMatchEuler.from_pretrained", source)
        self.assertIn("BooguImagePipeline", source)
        self.assertIn("BooguImageTurboPipeline", source)
        self.assertIn('transformer_subfolder = "transformer"', source)
        self.assertIn("subfolder=\"mllm\"", source)
        self.assertIn("subfolder=\"processor\"", source)
        self.assertIn("subfolder=\"scheduler\"", source)
        self.assertIn("subfolder=\"vae\"", source)
        self.assertIn("pipe.scheduler = self.get_train_scheduler()", source)
        self.assertIn("not supported for", source)
        self.assertIn("AITK training", source)
        self.assertIn("model.quantize: true", source)

    def test_vendored_boogu_stack_is_present(self):
        expected_files = [
            "models/attention_processor.py",
            "models/transformers/transformer_boogu.py",
            "models/transformers/rope.py",
            "pipelines/image_processor.py",
            "pipelines/boogu/pipeline_boogu.py",
            "pipelines/boogu/pipeline_boogu_turbo.py",
            "schedulers/scheduling_flow_match_euler_discrete_time_shifting.py",
        ]

        for relative_path in expected_files:
            self.assertTrue((BOOGU_VENDOR_ROOT / relative_path).exists(), relative_path)

    def test_vendored_boogu_sampling_guards(self):
        transformer_source = BOOGU_TRANSFORMER_PATH.read_text(encoding="utf-8")
        rope_source = BOOGU_ROPE_PATH.read_text(encoding="utf-8")
        pipeline_source = BOOGU_PIPELINE_PATH.read_text(encoding="utf-8")
        turbo_pipeline_source = BOOGU_TURBO_PIPELINE_PATH.read_text(encoding="utf-8")

        self.assertIn("patch_size = int(patch_size)", transformer_source)
        self.assertIn("p = int(self.config.patch_size)", transformer_source)
        self.assertIn("self.patch_size = int(patch_size)", rope_source)
        self.assertIn("p = int(self.patch_size)", rope_source)
        self.assertIn("from tqdm import tqdm", pipeline_source)
        self.assertIn("tqdm.write(f\"[Pipeline Processing]", pipeline_source)
        self.assertNotIn("✅", pipeline_source)
        self.assertNotIn("⚠", pipeline_source)
        self.assertIn("from tqdm import tqdm", turbo_pipeline_source)
        self.assertIn(
            "tqdm.write(\"[Turbo Pipeline Processing]", turbo_pipeline_source
        )

    def test_rope_accepts_float_attention_masks_and_lengths(self):
        spec = importlib.util.spec_from_file_location(
            "boogu_rope_for_test", BOOGU_ROPE_PATH
        )
        rope_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rope_module)

        embedder = rope_module.BooguImageDoubleStreamRotaryPosEmbed(
            theta=10000,
            axes_dim=(2, 2, 2),
            axes_lens=(8, 8, 8),
            patch_size=2.0,
        )
        freqs_cis = embedder.get_freqs_cis((2, 2, 2), (8, 8, 8), theta=10000)

        outputs = embedder(
            freqs_cis,
            torch.ones(1, 3, dtype=torch.float32),
            [[0.0]],
            [4.0],
            [None],
            [(4, 4)],
            torch.device("cpu"),
        )

        self.assertEqual(outputs[4], [3])
        self.assertEqual(outputs[5], [7])

    def test_model_arch_and_validation_guards(self):
        source = CONFIG_MODULES_PATH.read_text(encoding="utf-8")

        self.assertIn("'boogu_image'", source)
        self.assertIn("'boogu_image_edit'", source)
        self.assertIn("'boogu_image_turbo'", source)
        self.assertIn("boogu_arches = {'boogu_image', 'boogu_image_edit', 'boogu_image_turbo'}", source)
        self.assertIn("full base-model fine-tuning is not supported", source)
        self.assertIn("freezes Qwen3-VL", source)
        self.assertIn("network.type lora only", source)
        self.assertIn("Cannot unload the text encoder with boogu_image_edit", source)
        self.assertIn("Boogu fp8 repositories are inference-oriented", source)

    def test_memory_profiles_include_boogu_arches(self):
        backend_source = MEMORY_OFFLOAD_PATH.read_text(encoding="utf-8")
        ui_source = UI_MEMORY_PATH.read_text(encoding="utf-8")

        for arch in ("boogu_image", "boogu_image_edit", "boogu_image_turbo"):
            self.assertIn(f'"{arch}"', backend_source)
            self.assertIn(f"'{arch}'", ui_source)

    def test_ui_model_defaults(self):
        source = UI_OPTIONS_PATH.read_text(encoding="utf-8")

        base = ts_object_block(source, "boogu_image")
        edit = ts_object_block(source, "boogu_image_edit")
        turbo = ts_object_block(source, "boogu_image_turbo")

        self.assertIn("defaultAutoTrainingProfileId: 'boogu-image-balanced-lora'", base)
        self.assertIn("'config.process[0].model.name_or_path': ['Boogu/Boogu-Image-0.1-Base'", base)
        self.assertIn("'config.process[0].sample.guidance_scale': [4, 4]", base)
        self.assertIn("'config.process[0].sample.sample_steps': [50, 30]", base)
        self.assertIn("'config.process[0].network.linear': [32, defaultLinearRank]", base)
        self.assertNotIn("-fp8", base.lower())

        self.assertIn("defaultAutoTrainingProfileId: 'boogu-image-edit-lora'", edit)
        self.assertIn("'config.process[0].model.name_or_path': ['Boogu/Boogu-Image-0.1-Edit'", edit)
        self.assertIn("'config.process[0].sample.guidance_scale': [5, 4]", edit)
        self.assertIn("'datasets.control_path'", edit)
        self.assertIn("'datasets.multi_control_paths'", edit)
        self.assertIn("'sample.ctrl_img'", edit)
        self.assertIn("'sample.multi_ctrl_imgs'", edit)
        self.assertIn("'train.unload_text_encoder'", edit)
        self.assertNotIn("-fp8", edit.lower())

        self.assertIn("defaultAutoTrainingProfileId: 'boogu-image-turbo-experimental-lora'", turbo)
        self.assertIn("'config.process[0].model.name_or_path': ['Boogu/Boogu-Image-0.1-Turbo'", turbo)
        self.assertIn("'config.process[0].sample.guidance_scale': [0, 4]", turbo)
        self.assertIn("'config.process[0].sample.sample_steps': [4, 30]", turbo)
        self.assertIn("'config.process[0].network.linear': [16, defaultLinearRank]", turbo)
        self.assertNotIn("-fp8", turbo.lower())

    def test_auto_training_profiles_are_scoped(self):
        source = AUTO_PROFILES_PATH.read_text(encoding="utf-8")

        self.assertIn("const booguImageArchs = ['boogu_image'];", source)
        self.assertIn("const booguEditArchs = ['boogu_image_edit'];", source)
        self.assertIn("const booguTurboArchs = ['boogu_image_turbo'];", source)
        self.assertIn("id: 'boogu-image-balanced-lora'", source)
        self.assertIn("id: 'boogu-image-edit-lora'", source)
        self.assertIn("id: 'boogu-image-turbo-experimental-lora'", source)
        self.assertIn("modelArchs: booguImageArchs", source)
        self.assertIn("modelArchs: booguEditArchs", source)
        self.assertIn("modelArchs: booguTurboArchs", source)
        self.assertIn("loraNetwork(32, { transformer_only: true })", source)
        self.assertIn("loraNetwork(16, { transformer_only: true })", source)
        self.assertIn("unload_text_encoder: false", source)
        self.assertIn("phase('Tiny polish', 0.000005", source)

    def test_example_configs(self):
        for arch, path in EXAMPLES.items():
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            process = config["config"]["process"][0]

            self.assertEqual(process["model"]["arch"], arch)
            self.assertTrue(process["model"]["quantize"])
            self.assertTrue(process["model"]["quantize_te"])
            self.assertEqual(process["model"]["qtype"], "qfloat8")
            self.assertEqual(process["model"]["qtype_te"], "qfloat8")
            self.assertEqual(process["network"]["type"], "lora")
            self.assertTrue(process["network"]["transformer_only"])
            self.assertTrue(process["train"]["cache_text_embeddings"])
            self.assertFalse(process["train"]["train_text_encoder"])
            self.assertEqual(process["train"]["noise_scheduler"], "flowmatch")
            self.assertEqual(process["train"]["timestep_type"], "weighted")
            self.assertEqual(process["sample"]["sampler"], "flowmatch")

        edit = yaml.safe_load(EXAMPLES["boogu_image_edit"].read_text(encoding="utf-8"))["config"]["process"][0]
        self.assertIn("control_path", edit["datasets"][0])
        self.assertFalse(edit["train"]["unload_text_encoder"])
        self.assertIn("ctrl_img", edit["sample"]["samples"][0])
        self.assertIn("multi_ctrl_imgs", edit["sample"]["samples"][1])

        turbo = yaml.safe_load(EXAMPLES["boogu_image_turbo"].read_text(encoding="utf-8"))["config"]["process"][0]
        self.assertEqual(turbo["network"]["linear"], 16)
        self.assertEqual(turbo["sample"]["guidance_scale"], 0)
        self.assertEqual(turbo["sample"]["sample_steps"], 4)

    def test_readme_documents_supported_models_and_fp8_guidance(self):
        source = README_PATH.read_text(encoding="utf-8")

        self.assertIn("Boogu/Boogu-Image-0.1-Base", source)
        self.assertIn("Boogu/Boogu-Image-0.1-Edit", source)
        self.assertIn("Boogu/Boogu-Image-0.1-Turbo", source)
        self.assertIn("The official Boogu fp8 repos are not first-class training presets", source)
        self.assertIn("train_lora_boogu_image_24gb.yaml", source)
        self.assertIn("train_lora_boogu_image_edit_24gb.yaml", source)
        self.assertIn("train_lora_boogu_image_turbo_experimental_24gb.yaml", source)


if __name__ == "__main__":
    unittest.main()
