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
BASE_SD_TRAIN_PROCESS_PATH = PROJECT_ROOT / "jobs" / "process" / "BaseSDTrainProcess.py"
UI_OPTIONS_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "options.ts"
HIDREAM_O1_EXAMPLE_PATH = PROJECT_ROOT / "config" / "examples" / "train_lora_hidream_o1_48gb.yaml"


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

    def test_velocity_fallback_sigma_squared_weights_shrink_low_timestep_spikes(self):
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

    def test_t0_loss_formula_recovers_o1_x0_prediction(self):
        latents = torch.arange(24, dtype=torch.float32).view(2, 3, 2, 2) / 24.0
        noise = torch.linspace(-1.0, 1.0, 24, dtype=torch.float32).view(2, 3, 2, 2)
        x0_pred = latents + torch.tensor([0.25, -0.125]).view(2, 1, 1, 1)
        sigma = (torch.tensor([1.0, 500.0]) / 1000.0).clamp_min(0.001).view(2, 1, 1, 1)
        noisy_latents = (1.0 - sigma) * latents + sigma * (noise * 8.0)

        velocity_pred = (noisy_latents - x0_pred) / sigma
        reconstructed_t0 = noisy_latents - sigma * velocity_pred

        self.assertTrue(torch.allclose(reconstructed_t0, x0_pred))


class HidreamO1SchedulerBroadcastTest(unittest.TestCase):
    def setUp(self):
        scheduler_cls = load_class_with_methods(
            HIDREAM_O1_MODEL_PATH,
            "HidreamO1FlowmatchScheduler",
            {"add_noise"},
            "HidreamO1FlowmatchSchedulerForTest",
        )
        self.scheduler = scheduler_cls()
        self.scheduler.noise_scale = 8.0

    def test_add_noise_broadcasts_timesteps_over_batch_not_width(self):
        original = torch.zeros((2, 3, 4, 5), dtype=torch.float32)
        noise = torch.ones_like(original)
        timesteps = torch.tensor([250.0, 750.0])

        noisy = self.scheduler.add_noise(original, noise, timesteps)

        self.assertEqual(noisy.shape, original.shape)
        self.assertTrue(torch.allclose(noisy[0], torch.full_like(noisy[0], 2.0)))
        self.assertTrue(torch.allclose(noisy[1], torch.full_like(noisy[1], 6.0)))


class HidreamO1DefaultConfigTest(unittest.TestCase):
    def test_backend_defaults_o1_to_t0_loss_when_omitted(self):
        source = BASE_SD_TRAIN_PROCESS_PATH.read_text(encoding="utf-8")

        self.assertIn("model_config.get('arch') == 'hidream_o1'", source)
        self.assertIn("raw_train_config.setdefault('noise_scheduler', 'flowmatch')", source)
        self.assertIn("raw_train_config.setdefault('dtype', 'bf16')", source)
        self.assertIn("raw_train_config.setdefault('batch_size', 2)", source)
        self.assertIn("raw_train_config.setdefault('steps', 8000)", source)
        self.assertIn("raw_train_config.setdefault('optimizer', 'adamw8bit')", source)
        self.assertIn("raw_train_config.setdefault('lr', 0.00003)", source)
        self.assertIn("raw_train_config.setdefault('timestep_type', 'sigmoid')", source)
        self.assertIn("optimizer_params.setdefault('weight_decay', 0.0001)", source)
        self.assertIn("raw_train_config['t0_loss_target'] = True", source)

    def test_ui_preset_defaults_o1_to_t0_loss(self):
        source = UI_OPTIONS_PATH.read_text(encoding="utf-8")
        start = source.index("name: 'hidream_o1'")
        end = source.index("disableSections", start)
        hidream_o1_block = source[start:end]

        self.assertIn("'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch']", hidream_o1_block)
        self.assertIn("'config.process[0].train.dtype': ['bf16', 'bf16']", hidream_o1_block)
        self.assertIn("'config.process[0].train.batch_size': [2, 1]", hidream_o1_block)
        self.assertIn("'config.process[0].train.steps': [8000, 3000]", hidream_o1_block)
        self.assertIn("'config.process[0].train.optimizer': ['adamw8bit', 'adamw8bit']", hidream_o1_block)
        self.assertIn("'config.process[0].train.lr': [0.00003, 0.0001]", hidream_o1_block)
        self.assertIn("'config.process[0].train.optimizer_params.weight_decay': [0.0001, 0.0001]", hidream_o1_block)
        self.assertIn("'config.process[0].train.timestep_type': ['sigmoid', 'sigmoid']", hidream_o1_block)
        self.assertIn("'config.process[0].train.content_or_style': ['balanced', 'balanced']", hidream_o1_block)
        self.assertIn("'config.process[0].train.loss_type': ['mse', 'mse']", hidream_o1_block)
        self.assertIn("'config.process[0].train.t0_loss_target': [true, undefined]", hidream_o1_block)
        self.assertIn("'config.process[0].logging.monitor_every': [1, 10]", hidream_o1_block)

    def test_example_config_targets_hidream_o1_image(self):
        source = HIDREAM_O1_EXAMPLE_PATH.read_text(encoding="utf-8")

        self.assertIn('arch: "hidream_o1"', source)
        self.assertIn('name_or_path: "HiDream-ai/HiDream-O1-Image"', source)
        self.assertIn('noise_scheduler: "flowmatch"', source)
        self.assertIn('t0_loss_target: true', source)
        self.assertIn('monitor_every: 1', source)


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
