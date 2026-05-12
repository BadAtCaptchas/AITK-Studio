import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIDREAM_O1_MODEL_PATH = (
    PROJECT_ROOT
    / "extensions_built_in"
    / "diffusion_models"
    / "hidream"
    / "hidream_o1_model.py"
)
SD_TRAINER_PATH = (
    PROJECT_ROOT
    / "extensions_built_in"
    / "sd_trainer"
    / "SDTrainer.py"
)


def load_class_with_methods(source_path: Path, source_class_name: str, method_names: set[str], output_class_name: str):
    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(source_path))
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == source_class_name
    )
    methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    test_class = ast.ClassDef(
        name=output_class_name,
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    test_module = ast.Module(body=[test_class], type_ignores=[])
    ast.fix_missing_locations(test_module)

    namespace = {"torch": torch, "T_EPS": 0.001}
    exec(compile(test_module, str(source_path), "exec"), namespace)
    return namespace[output_class_name]


class HidreamO1LossWeightTest(unittest.TestCase):
    def setUp(self):
        model_cls = load_class_with_methods(
            HIDREAM_O1_MODEL_PATH,
            "HidreamO1Model",
            {"get_loss_weight"},
            "HidreamO1LossWeight",
        )
        self.model = model_cls()

    def test_sigma_squared_weights_shrink_low_timestep_spikes(self):
        timesteps = torch.tensor([1.0, 1000.0])
        weights = self.model.get_loss_weight(
            timesteps=timesteps,
            loss=torch.ones((2, 3, 4, 4)),
        )

        self.assertEqual(weights.shape, (2,))
        self.assertAlmostEqual(weights[0].item(), 1e-6, places=12)
        self.assertAlmostEqual(weights[1].item(), 1.0, places=7)

        raw_loss = torch.tensor([1_000_000.0, 1.0]).view(2, 1, 1, 1)
        weighted_loss = raw_loss * weights.view(2, 1, 1, 1)

        self.assertAlmostEqual(weighted_loss[0].item(), 1.0, places=5)
        self.assertAlmostEqual(weighted_loss[1].item(), 1.0, places=7)

    def test_skips_non_velocity_loss_contexts(self):
        timesteps = torch.tensor([1.0])
        loss = torch.ones((1, 3, 4, 4))

        self.assertIsNone(self.model.get_loss_weight(timesteps, loss, t0_loss_target=True))
        self.assertIsNone(self.model.get_loss_weight(timesteps, loss, loss_target="source"))


class TrainerLossWeightBroadcastTest(unittest.TestCase):
    def setUp(self):
        trainer_cls = load_class_with_methods(
            SD_TRAINER_PATH,
            "SDTrainer",
            {"apply_model_loss_weight"},
            "TrainerLossWeight",
        )
        self.trainer = trainer_cls()
        self.trainer.train_config = SimpleNamespace(t0_loss_target=False)

    def test_model_loss_weight_broadcasts_over_4d_image_loss(self):
        class FakeModel:
            def get_loss_weight(self, **kwargs):
                return torch.tensor([0.25, 1.0])

        self.trainer.sd = FakeModel()
        loss = torch.ones((2, 3, 4, 5))

        weighted = self.trainer.apply_model_loss_weight(
            loss=loss,
            timesteps=torch.tensor([250.0, 1000.0]),
            noise_pred=torch.zeros_like(loss),
            target=torch.zeros_like(loss),
            noisy_latents=torch.zeros_like(loss),
            noise=torch.zeros_like(loss),
            batch=object(),
            loss_target="default",
        )

        self.assertEqual(weighted.shape, loss.shape)
        self.assertTrue(torch.allclose(weighted[0], torch.full_like(weighted[0], 0.25)))
        self.assertTrue(torch.allclose(weighted[1], torch.ones_like(weighted[1])))


if __name__ == "__main__":
    unittest.main()
