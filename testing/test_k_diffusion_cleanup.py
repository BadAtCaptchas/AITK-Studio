import ast
import copy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLER_PATH = ROOT / "toolkit" / "sampler.py"


def load_get_sampler():
    source = SAMPLER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_sampler"
    )

    class FakeScheduler:
        @classmethod
        def from_config(cls, config):
            return cls, config

    namespace = {
        "copy": copy,
        "sd_config": {},
        "pixart_config": {},
        "EulerDiscreteScheduler": FakeScheduler,
        "LMSDiscreteScheduler": FakeScheduler,
    }
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(SAMPLER_PATH), "exec"), namespace)
    return namespace["get_sampler"], FakeScheduler


class KDiffusionCleanupTests(unittest.TestCase):
    def test_legacy_sampler_names_are_rejected_clearly(self):
        get_sampler, _ = load_get_sampler()
        with self.assertRaisesRegex(
            ValueError,
            "Legacy k-diffusion sampler.*Diffusers-backed",
        ):
            get_sampler("sample_dpmpp_2m")

    def test_diffusers_karras_aliases_remain_available(self):
        get_sampler, scheduler = load_get_sampler()
        for name in ("k_euler", "k_lms"):
            with self.subTest(name=name):
                scheduler_class, config = get_sampler(name)
                self.assertIs(scheduler_class, scheduler)
                self.assertTrue(config["use_karras_sigmas"])

    def test_runtime_sources_have_no_k_diffusion_imports_or_pipeline(self):
        for relative_path in (
            "toolkit/pipelines.py",
            "toolkit/sampler.py",
            "toolkit/stable_diffusion_model.py",
        ):
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            ast.parse(source, filename=relative_path)
            self.assertNotIn("from k_diffusion", source)
            self.assertNotIn("import k_diffusion", source)
            self.assertNotIn("StableDiffusionKDiffusion", source)

        self.assertNotIn(
            "k-diffusion",
            (ROOT / "requirements_base.txt").read_text(encoding="utf-8"),
        )
        self.assertNotIn(
            '"k-diffusion"',
            (ROOT / "run_modal.py").read_text(encoding="utf-8"),
        )
        self.assertFalse((ROOT / "toolkit" / "import_compat.py").exists())


if __name__ == "__main__":
    unittest.main()
