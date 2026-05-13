import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SD_TRAINER_PATH = PROJECT_ROOT / "extensions_built_in" / "sd_trainer" / "SDTrainer.py"
BASE_SD_TRAIN_PROCESS_PATH = PROJECT_ROOT / "jobs" / "process" / "BaseSDTrainProcess.py"
CONFIG_MODULES_PATH = PROJECT_ROOT / "toolkit" / "config_modules.py"
JOB_LOSS_GRAPH_PATH = PROJECT_ROOT / "ui" / "src" / "components" / "JobLossGraph.tsx"


class TrainingMonitorMetricsTest(unittest.TestCase):
    def test_backend_records_new_training_diagnostics(self):
        trainer_source = SD_TRAINER_PATH.read_text(encoding="utf-8")
        base_source = BASE_SD_TRAIN_PROCESS_PATH.read_text(encoding="utf-8")
        config_source = CONFIG_MODULES_PATH.read_text(encoding="utf-8")

        self.assertIn("self.monitor_every", config_source)
        self.assertIn("self.monitor_tensor_stats", config_source)
        self.assertIn("consume_monitor_metrics", trainer_source)
        self.assertIn("train/grad_norm", trainer_source)
        self.assertIn("self._record_tensor_stats('train/noise_pred'", trainer_source)
        self.assertIn("train/timestep_mean", base_source)
        self.assertIn("train/gpu_mem_allocated_gb", base_source)

    def test_ui_exposes_diagnostic_tabs_and_cards(self):
        source = JOB_LOSS_GRAPH_PATH.read_text(encoding="utf-8")

        self.assertIn("'timesteps'", source)
        self.assertIn("'gradients'", source)
        self.assertIn("'memory'", source)
        self.assertIn("Latest training stats", source)
        self.assertIn("train/gpu_mem_reserved_gb", source)
        self.assertIn("train/loss_unclipped", source)


if __name__ == "__main__":
    unittest.main()
