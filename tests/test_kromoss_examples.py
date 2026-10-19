"""Validate the shipped Kromoss examples without weights, datasets, or a GPU."""

from copy import deepcopy
from pathlib import Path
import unittest

import yaml

from toolkit.config_modules import (
    DatasetConfig, ModelConfig, NetworkConfig, SaveConfig, TrainConfig,
    preprocess_dataset_raw_config, validate_configs,
)


class KromossExampleTests(unittest.TestCase):
    def test_examples_support_lora_and_switching_only_the_network_type_to_lokr(self):
        examples = Path(__file__).resolve().parents[1] / "config" / "examples"
        for suffix in ("", "_turbo"):
            with (examples / f"train_lora_kromoss{suffix}_16gb.yaml").open(encoding="utf-8") as stream:
                process = yaml.safe_load(stream)["config"]["process"][0]
            for kind in ("lora", "lokr"):
                with self.subTest(variant=suffix or "full", network=kind):
                    raw = deepcopy(process)
                    raw["network"]["type"] = kind
                    network = NetworkConfig(**raw["network"])
                    train = TrainConfig(**raw["train"])
                    model = ModelConfig(**raw["model"])
                    datasets = [DatasetConfig(**dataset) for dataset in preprocess_dataset_raw_config(raw["datasets"])]
                    validate_configs(train, model, SaveConfig(**raw["save"]), datasets, network)
                    self.assertFalse(network.lokr_full_rank)
                    self.assertEqual(network.lokr_factor, -1)
                    self.assertEqual(network.linear, 16)
                    self.assertEqual(network.linear_alpha, 16)
                    self.assertFalse(network.lokr_weight_decompose)
                    self.assertFalse(train.train_text_encoder)
                    self.assertEqual(model.vae_dtype, "fp32")


if __name__ == "__main__":
    unittest.main()
