import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAGEFLOW_ROOT = (
    PROJECT_ROOT / "extensions_built_in" / "diffusion_models" / "mageflow"
)
REGISTRY_PATH = (
    PROJECT_ROOT / "extensions_built_in" / "diffusion_models" / "__init__.py"
)
UI_OPTIONS_PATH = (
    PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "options.ts"
)
README_PATH = PROJECT_ROOT / "README.md"


class MageFlowStaticSupportTest(unittest.TestCase):
    def test_all_vendored_modules_compile(self):
        paths = sorted(MAGEFLOW_ROOT.rglob("*.py"))
        self.assertEqual(len(paths), 8)
        for path in paths:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_optional_registry_contains_both_architectures(self):
        registry = REGISTRY_PATH.read_text(encoding="utf-8")

        self.assertIn("MageFlowModel, MageFlowEditModel = _optional_models(", registry)
        self.assertIn('("MageFlowModel", "mageflow")', registry)
        self.assertIn('("MageFlowEditModel", "mageflow_edit")', registry)
        self.assertIn("MageFlowModel,", registry)
        self.assertIn("MageFlowEditModel,", registry)

    def test_model_contract_and_scheduler_match_upstream(self):
        model = (MAGEFLOW_ROOT / "mageflow.py").read_text(encoding="utf-8")
        pipeline = (MAGEFLOW_ROOT / "src" / "pipeline.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('arch = "mageflow"', model)
        self.assertIn('arch = "mageflow_edit"', model)
        self.assertIn('"shift": 6.0', model)
        self.assertIn("return (noise - batch.latents).detach()", model)
        self.assertIn("def pack_text_features(", pipeline)
        self.assertIn("def predict_velocity(", pipeline)
        self.assertIn("class MageFlowPipeline:", pipeline)

    def test_ui_defaults_and_docs_include_both_models(self):
        options = UI_OPTIONS_PATH.read_text(encoding="utf-8")
        readme = README_PATH.read_text(encoding="utf-8")

        base_start = options.index("name: 'mageflow'")
        edit_start = options.index("name: 'mageflow_edit'")
        base_block = options[base_start:edit_start]
        edit_block = options[edit_start:options.index("name: 'flux2_klein_9b'")]

        self.assertIn("microsoft/Mage-Flow-Base", base_block)
        self.assertIn("'config.process[0].model.quantize': [true, false]", base_block)
        self.assertIn("'config.process[0].sample.guidance_scale': [4, 4]", base_block)
        self.assertIn("'config.process[0].sample.sample_steps': [25, 25]", base_block)
        self.assertIn("microsoft/Mage-Flow-Edit-Base", edit_block)
        self.assertIn("'datasets.multi_control_paths'", edit_block)
        self.assertIn("'sample.multi_ctrl_imgs'", edit_block)
        self.assertIn("microsoft/Mage-Flow-Base", readme)
        self.assertIn("microsoft/Mage-Flow-Edit-Base", readme)


if __name__ == "__main__":
    unittest.main()
