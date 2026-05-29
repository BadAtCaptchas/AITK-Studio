import unittest

import torch

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork


class ZImageTransformer2DModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "proj": torch.nn.Linear(4, 4),
                    }
                )
            ]
        )


class _BaseModel:
    arch = "zimage"
    use_old_lokr_format = False

    def get_transformer_block_names(self):
        return ["layers"]


class LoRASpecialFilterTest(unittest.TestCase):
    def _build_network(self, only_if_contains):
        network_config = NetworkConfig(
            type="lora",
            linear=2,
            linear_alpha=2,
            transformer_only=True,
        )
        return LoRASpecialNetwork(
            text_encoder=None,
            unet=ZImageTransformer2DModel(),
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=True,
            target_lin_modules=["ZImageTransformer2DModel"],
            base_model=_BaseModel(),
            only_if_contains=only_if_contains,
        )

    def test_empty_only_if_contains_does_not_filter_everything(self):
        network = self._build_network([])

        self.assertEqual(len(network.get_all_modules()), 1)

    def test_populated_only_if_contains_still_filters(self):
        network = self._build_network(["does_not_exist"])

        self.assertEqual(len(network.get_all_modules()), 0)


if __name__ == "__main__":
    unittest.main()
