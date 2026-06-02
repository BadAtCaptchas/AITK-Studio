import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SD_TRAIN_PROCESS_PATH = PROJECT_ROOT / "jobs" / "process" / "BaseSDTrainProcess.py"
GENERATE_PAGE_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "generate" / "page.tsx"
JOB_UTILS_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "utils.ts"
JOB_CONFIG_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "jobConfig.ts"
TYPES_PATH = PROJECT_ROOT / "ui" / "src" / "types.ts"


def load_base_process_methods(method_names: set[str]):
    source = BASE_SD_TRAIN_PROCESS_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(BASE_SD_TRAIN_PROCESS_PATH))
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseSDTrainProcess"
    )
    methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    test_class = ast.ClassDef(
        name="BaseSDTrainProcessSampleOffload",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    test_module = ast.Module(body=[test_class], type_ignores=[])
    ast.fix_missing_locations(test_module)

    namespace = {
        "GenerateImageConfig": object,
        "List": List,
        "print_acc": lambda *_args, **_kwargs: None,
    }
    exec(compile(test_module, str(BASE_SD_TRAIN_PROCESS_PATH), "exec"), namespace)
    return namespace["BaseSDTrainProcessSampleOffload"]


class SampleGenerationOffloadTest(unittest.TestCase):
    def test_sample_generation_temporarily_disables_cpu_offload_flags(self):
        process_cls = load_base_process_methods({"_generate_sample_images"})
        process = process_cls()
        process.model_config = SimpleNamespace(
            low_vram=True,
            layer_offloading=True,
            layer_offloading_backend="legacy",
        )
        process.sd = SimpleNamespace(
            low_vram=True,
            model_config=SimpleNamespace(
                low_vram=True,
                layer_offloading=True,
                layer_offloading_backend="legacy",
            ),
        )

        def generate_images(image_configs, sampler=None):
            self.assertEqual(image_configs, ["sample"])
            self.assertEqual(sampler, "flowmatch")
            self.assertFalse(process.sd.low_vram)
            self.assertFalse(process.model_config.low_vram)
            self.assertFalse(process.model_config.layer_offloading)
            self.assertFalse(process.sd.model_config.low_vram)
            self.assertFalse(process.sd.model_config.layer_offloading)

        process.sd.generate_images = generate_images

        process._generate_sample_images(["sample"], sampler="flowmatch")

        self.assertTrue(process.sd.low_vram)
        self.assertTrue(process.model_config.low_vram)
        self.assertTrue(process.model_config.layer_offloading)
        self.assertTrue(process.sd.model_config.low_vram)
        self.assertTrue(process.sd.model_config.layer_offloading)

    def test_sample_generation_keeps_block_offloading_enabled(self):
        process_cls = load_base_process_methods({"_generate_sample_images"})
        process = process_cls()
        process.model_config = SimpleNamespace(
            low_vram=False,
            layer_offloading=True,
            layer_offloading_backend="block",
        )
        process.sd = SimpleNamespace(
            low_vram=False,
            model_config=SimpleNamespace(
                low_vram=False,
                layer_offloading=True,
                layer_offloading_backend="block",
            ),
        )

        def generate_images(image_configs, sampler=None):
            self.assertEqual(image_configs, ["sample"])
            self.assertTrue(process.model_config.layer_offloading)
            self.assertTrue(process.sd.model_config.layer_offloading)

        process.sd.generate_images = generate_images

        process._generate_sample_images(["sample"], sampler="flowmatch")

        self.assertTrue(process.model_config.layer_offloading)
        self.assertTrue(process.sd.model_config.layer_offloading)

    def test_sample_generation_restores_cpu_offload_flags_after_failure(self):
        process_cls = load_base_process_methods({"_generate_sample_images"})
        process = process_cls()
        process.model_config = SimpleNamespace(low_vram=True, layer_offloading=False)
        process.sd = SimpleNamespace(
            low_vram=True,
            model_config=SimpleNamespace(low_vram=True, layer_offloading=False),
        )

        def generate_images(_image_configs, sampler=None):
            raise RuntimeError("sample failed")

        process.sd.generate_images = generate_images

        with self.assertRaisesRegex(RuntimeError, "sample failed"):
            process._generate_sample_images(["sample"], sampler="flowmatch")

        self.assertTrue(process.sd.low_vram)
        self.assertTrue(process.model_config.low_vram)
        self.assertFalse(process.model_config.layer_offloading)
        self.assertTrue(process.sd.model_config.low_vram)
        self.assertFalse(process.sd.model_config.layer_offloading)

    def test_trainer_sample_uses_offload_guard(self):
        source = BASE_SD_TRAIN_PROCESS_PATH.read_text(encoding="utf-8")
        self.assertIn("self._generate_sample_images(gen_img_config_list, sampler=sample_config.sampler)", source)
        self.assertNotIn("self.sd.generate_images(gen_img_config_list, sampler=sample_config.sampler)", source)


class GeneratePageMemoryDefaultsTest(unittest.TestCase):
    def test_generate_page_does_not_inherit_training_cpu_offload_defaults(self):
        source = GENERATE_PAGE_PATH.read_text(encoding="utf-8")
        start = source.index("function getDefaultModelConfig")
        end = source.index("function getDefaultSampler", start)
        block = source[start:end]

        self.assertIn("low_vram: false", block)
        self.assertIn("layer_offloading: false", block)
        self.assertIn("layer_offloading_backend:", block)
        self.assertNotIn("low_vram: Boolean(getArchDefault", block)
        self.assertNotIn("layer_offloading: Boolean(getArchDefault", block)

    def test_generate_page_keeps_manual_memory_toggles_when_lora_selected(self):
        source = GENERATE_PAGE_PATH.read_text(encoding="utf-8")
        start = source.index("const applyLoraModelDefaults")
        end = source.index("const handleLoraPathChange", start)
        block = source[start:end]

        self.assertIn("low_vram: current.low_vram", block)
        self.assertIn("layer_offloading: current.layer_offloading", block)
        self.assertIn("layer_offloading_backend: current.layer_offloading_backend", block)
        self.assertIn("layer_offloading_transformer_percent: current.layer_offloading_transformer_percent", block)
        self.assertIn("layer_offloading_text_encoder_percent: current.layer_offloading_text_encoder_percent", block)

    def test_generate_page_exposes_layer_offloading_backend_selector(self):
        source = GENERATE_PAGE_PATH.read_text(encoding="utf-8")

        self.assertIn("layerOffloadingBackendOptions", source)
        self.assertIn('label="Offload Backend"', source)
        self.assertIn("getLayerOffloadingMemoryProfile", source)


class TrainingUiMemoryProfileTest(unittest.TestCase):
    def test_training_arch_changes_use_memory_profile_defaults_without_overwriting_manual_values(self):
        source = JOB_UTILS_PATH.read_text(encoding="utf-8")

        self.assertIn("getLayerOffloadingMemoryProfile", source)
        self.assertIn("memoryProfile.backend", source)
        self.assertIn("if (!('layer_offloading_backend' in jobConfig.config.process[0].model))", source)
        self.assertIn("delete newModel.layer_offloading_backend", source)

    def test_config_migration_and_types_include_backend(self):
        job_config_source = JOB_CONFIG_PATH.read_text(encoding="utf-8")
        types_source = TYPES_PATH.read_text(encoding="utf-8")

        self.assertIn("layer_offloading_backend ??= memoryProfile.backend", job_config_source)
        self.assertIn("layer_offloading_backend?: 'block' | 'legacy'", types_source)


if __name__ == "__main__":
    unittest.main()
