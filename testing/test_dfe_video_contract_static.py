import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DFE_PATH = ROOT / "toolkit" / "models" / "diffusion_feature_extraction.py"


class DFEVideoContractStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = DFE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def class_forward_source(self, class_name):
        class_node = next(
            node for node in self.tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        forward_node = next(
            node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )
        return ast.get_source_segment(self.source, forward_node)

    def test_all_video_dfe_paths_fold_frames_and_repeat_timesteps(self):
        for class_name in (
            "DiffusionFeatureExtractor4",
            "DiffusionFeatureExtractor6",
            "DiffusionFeatureExtractor7",
            "DiffusionFeatureExtractor9",
            "DiffusionFeatureExtractor10",
        ):
            with self.subTest(class_name=class_name):
                forward = self.class_forward_source(class_name)
                self.assertIn("_fold_frames_to_batch(noise_pred)", forward)
                self.assertIn("timesteps.repeat_interleave(num_frames)", forward)
                self.assertIn("tensors.reshape(-1, *tensors.shape[2:])", forward)

    def test_x0_and_partial_video_contracts_are_present(self):
        for class_name in (
            "DiffusionFeatureExtractor7",
            "DiffusionFeatureExtractor9",
            "DiffusionFeatureExtractor10",
        ):
            with self.subTest(class_name=class_name):
                forward = self.class_forward_source(class_name)
                self.assertIn('"x0_pred"', forward)
                self.assertIn("_fold_frames_to_batch(target_latents)", forward)

        for class_name in (
            "DiffusionFeatureExtractor3",
            "DiffusionFeatureExtractor4",
            "DiffusionFeatureExtractor6",
        ):
            with self.subTest(class_name=class_name):
                self.assertIn("'x0_pred'", self.class_forward_source(class_name))

        self.assertIn(
            "self.x0_pred = False",
            (ROOT / "toolkit" / "models" / "base_model.py").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "self.x0_pred = True",
            (
                ROOT
                / "extensions_built_in"
                / "diffusion_models"
                / "prx_pixel_t2i"
                / "prx_pixel_t2i.py"
            ).read_text(encoding="utf-8"),
        )
        self.assertIn(
            "if self.sd.x0_pred:",
            (ROOT / "extensions_built_in" / "sd_trainer" / "SDTrainer.py").read_text(encoding="utf-8"),
        )
        self.assertIn(
            "model=self.sd",
            (ROOT / "extensions_built_in" / "sd_trainer" / "SDTrainer.py").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
