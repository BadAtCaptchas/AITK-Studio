import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def class_method_return(path: Path, class_name: str, method_name: str):
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    klass = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in klass.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    returned = next(node for node in method.body if isinstance(node, ast.Return))
    return ast.literal_eval(returned.value)


class UpstreamJulyBehaviorTest(unittest.TestCase):
    def test_sensitive_layer_exclusions_are_adapted_to_existing_models(self):
        krea = class_method_return(
            PROJECT_ROOT
            / "extensions_built_in"
            / "diffusion_models"
            / "krea2"
            / "krea2.py",
            "Krea2Model",
            "get_quantization_exclude_modules",
        )
        z_image = class_method_return(
            PROJECT_ROOT
            / "extensions_built_in"
            / "diffusion_models"
            / "z_image"
            / "z_image.py",
            "ZImageModel",
            "get_quantization_exclude_modules",
        )
        self.assertEqual(
            krea,
            ["first", "tmlp*", "tproj*", "txtmlp*", "txtfusion.projector", "last*"],
        )
        self.assertEqual(
            z_image,
            [
                "t_embedder*",
                "cap_embedder*",
                "all_x_embedder*",
                "all_final_layer*",
                "siglip_embedder*",
            ],
        )
        self.assertFalse((PROJECT_ROOT / "toolkit" / "models" / "v2").exists())

    def test_captioner_traceback_and_hugging_face_defaults_preserve_overrides(self):
        captioner = (
            PROJECT_ROOT
            / "extensions_built_in"
            / "captioner"
            / "Qwen3VLCaptioner.py"
        ).read_text(encoding="utf-8")
        run_source = (PROJECT_ROOT / "run.py").read_text(encoding="utf-8")

        self.assertIn("import traceback", captioner)
        self.assertIn("traceback.print_exc()", captioner)
        self.assertIn('default_disable_xet = "1" if os.name == "nt" else "0"', run_source)
        self.assertIn(
            'os.environ["HF_HUB_DISABLE_XET"] = os.getenv(',
            run_source,
        )
        self.assertIn('default_hf_transfer = "0" if os.name == "nt" else "1"', run_source)

    def test_compile_workaround_is_feature_detected(self):
        source = (
            PROJECT_ROOT / "jobs" / "process" / "BaseSDTrainProcess.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"coalesce_tiling_analysis"', source)
        self.assertIn("torch._inductor.config.triton.coalesce_tiling_analysis = False", source)


if __name__ == "__main__":
    unittest.main()
