import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_SD_TRAIN_PROCESS_PATH = PROJECT_ROOT / "jobs" / "process" / "BaseSDTrainProcess.py"
CONFIG_MODULES_PATH = PROJECT_ROOT / "toolkit" / "config_modules.py"
GENERATE_PAGE_PATH = (
    PROJECT_ROOT / "ui" / "src" / "app" / "generate" / "GeneratePageContent.tsx"
)
SIMPLE_JOB_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "SimpleJob.tsx"
JOB_UTILS_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "utils.ts"
JOB_CONFIG_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "jobConfig.ts"
TYPES_PATH = PROJECT_ROOT / "ui" / "src" / "types.ts"


def load_base_process_methods(method_names: set[str], print_acc=None):
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
        "print_acc": print_acc or (lambda *_args, **_kwargs: None),
    }
    exec(compile(test_module, str(BASE_SD_TRAIN_PROCESS_PATH), "exec"), namespace)
    return namespace["BaseSDTrainProcessSampleOffload"]


class SampleGenerationOffloadTest(unittest.TestCase):
    def test_disabled_memory_options_stay_disabled_without_offload_message(self):
        messages = []
        process = load_base_process_methods({"_generate_sample_images"}, messages.append)()
        process.model_config = SimpleNamespace(
            low_vram=False, layer_offloading=False, layer_offloading_backend="legacy",
        )
        calls = []

        def generate_images(image_configs, sampler=None):
            calls.append((image_configs, sampler))
            self.assertFalse(process.sd.low_vram)
            self.assertFalse(process.model_config.low_vram)
            self.assertFalse(process.model_config.layer_offloading)

        process.sd = SimpleNamespace(
            low_vram=False, model_config=process.model_config, generate_images=generate_images,
        )
        process._generate_sample_images(["sample"], sampler="flowmatch")
        self.assertEqual(calls, [(["sample"], "flowmatch")])
        self.assertEqual(messages, [])

    def test_legacy_offload_is_named_correctly_and_restored_even_on_failure(self):
        for shared_config in (False, True):
            for fail in (False, True):
                with self.subTest(shared_config=shared_config, fail=fail):
                    messages = []
                    process = load_base_process_methods({"_generate_sample_images"}, messages.append)()
                    process.model_config = SimpleNamespace(
                        low_vram=False, layer_offloading=True, layer_offloading_backend="legacy",
                    )
                    sd_config = process.model_config if shared_config else SimpleNamespace(
                        low_vram=False, layer_offloading=True, layer_offloading_backend="legacy",
                    )

                    def generate_images(_image_configs, sampler=None):
                        self.assertFalse(process.sd.low_vram)
                        for config in (process.model_config, sd_config):
                            self.assertFalse(config.low_vram)
                            self.assertFalse(config.layer_offloading)
                        if fail:
                            raise RuntimeError("sample failed")

                    process.sd = SimpleNamespace(
                        low_vram=False, model_config=sd_config, generate_images=generate_images,
                    )
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "sample failed"):
                            process._generate_sample_images(["sample"])
                    else:
                        process._generate_sample_images(["sample"])
                    self.assertEqual(messages, [
                        "Temporarily disabling legacy layer offloading for sample generation (Low VRAM is off)",
                    ])
                    self.assertFalse(process.sd.low_vram)
                    for config in (process.model_config, sd_config):
                        self.assertFalse(config.low_vram)
                        self.assertTrue(config.layer_offloading)

    def test_sample_generation_keeps_low_vram_flags_enabled(self):
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
            self.assertTrue(process.sd.low_vram)
            self.assertTrue(process.model_config.low_vram)
            self.assertTrue(process.model_config.layer_offloading)
            self.assertTrue(process.sd.model_config.low_vram)
            self.assertTrue(process.sd.model_config.layer_offloading)

        process.sd.generate_images = generate_images

        process._generate_sample_images(["sample"], sampler="flowmatch")

        self.assertTrue(process.sd.low_vram)
        self.assertTrue(process.model_config.low_vram)
        self.assertTrue(process.model_config.layer_offloading)
        self.assertTrue(process.sd.model_config.low_vram)
        self.assertTrue(process.sd.model_config.layer_offloading)

    def test_sample_generation_can_keep_cpu_offload_flags_enabled(self):
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
            self.assertTrue(process.sd.low_vram)
            self.assertTrue(process.model_config.low_vram)
            self.assertTrue(process.model_config.layer_offloading)
            self.assertTrue(process.sd.model_config.low_vram)
            self.assertTrue(process.sd.model_config.layer_offloading)

        process.sd.generate_images = generate_images

        process._generate_sample_images(
            ["sample"],
            sampler="flowmatch",
            keep_low_vram_for_samples=True,
        )

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
        self.assertIn("keep_low_vram_for_samples=sample_config.keep_low_vram_for_samples", source)
        self.assertIn("sample_uses_low_vram", source)
        self.assertNotIn("self.sd.generate_images(gen_img_config_list, sampler=sample_config.sampler)", source)

    def test_sample_config_option_defaults_disabled(self):
        source = CONFIG_MODULES_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "self.keep_low_vram_for_samples = kwargs.get('keep_low_vram_for_samples', False)",
            source,
        )


class GeneratePageMemoryDefaultsTest(unittest.TestCase):
    # Defaults and LoRA selection behavior are exercised in
    # ui/scripts/generationConfig.test.mjs against the shared TypeScript helpers.

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

    def test_training_low_vram_defaults_keep_low_vram_samples(self):
        simple_job_source = SIMPLE_JOB_PATH.read_text(encoding="utf-8")
        utils_source = JOB_UTILS_PATH.read_text(encoding="utf-8")
        job_config_source = JOB_CONFIG_PATH.read_text(encoding="utf-8")

        self.assertIn("setJobConfig(true, 'config.process[0].sample.keep_low_vram_for_samples')", simple_job_source)
        self.assertIn("const modelLowVramKey = 'config.process[0].model.low_vram'", utils_source)
        self.assertIn(
            "const keepLowVramForSamplesKey = 'config.process[0].sample.keep_low_vram_for_samples'",
            utils_source,
        )
        self.assertIn("newDefaults[keepLowVramForSamplesKey] = [true, false]", utils_source)
        self.assertIn("setJobConfig(false, keepLowVramForSamplesKey)", utils_source)
        self.assertIn("sample.keep_low_vram_for_samples = true", job_config_source)

    def test_config_migration_and_types_include_backend(self):
        job_config_source = JOB_CONFIG_PATH.read_text(encoding="utf-8")
        types_source = TYPES_PATH.read_text(encoding="utf-8")

        self.assertIn("layer_offloading_backend ??= memoryProfile.backend", job_config_source)
        self.assertIn("layer_offloading_backend?: 'block' | 'legacy'", types_source)


if __name__ == "__main__":
    unittest.main()
