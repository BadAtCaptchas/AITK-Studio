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


class WrappedZImageTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wrapped = ZImageTransformer2DModel()


class Linear4bit(torch.nn.Linear):
    pass


class Ideogram4Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "qkv": Linear4bit(4, 4),
                    }
                )
            ]
        )


class _BaseModel:
    arch = "zimage"
    use_old_lokr_format = False

    def get_transformer_block_names(self):
        return ["layers"]


class _IdeogramBaseModel(_BaseModel):
    arch = "ideogram4"


class LoRASpecialFilterTest(unittest.TestCase):
    def _build_network(
        self,
        only_if_contains=None,
        unet=None,
        target_lin_modules=None,
        base_model=None,
    ):
        network_config = NetworkConfig(
            type="lora",
            linear=2,
            linear_alpha=2,
            transformer_only=True,
        )
        return LoRASpecialNetwork(
            text_encoder=None,
            unet=unet or ZImageTransformer2DModel(),
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=True,
            target_lin_modules=target_lin_modules or ["ZImageTransformer2DModel"],
            base_model=base_model or _BaseModel(),
            only_if_contains=only_if_contains,
        )

    def test_empty_only_if_contains_does_not_filter_everything(self):
        network = self._build_network([])

        self.assertEqual(len(network.get_all_modules()), 1)

    def test_populated_only_if_contains_still_filters(self):
        network = self._build_network(["does_not_exist"])

        self.assertEqual(len(network.get_all_modules()), 0)

    def test_zimage_turbo_style_config_creates_modules(self):
        network = self._build_network([])

        self.assertEqual(len(network.get_all_modules()), 1)
        self.assertTrue(network.can_merge_in)

    def test_mergeable_dense_transformer_modules_can_still_merge(self):
        network = self._build_network([])

        network.merge_in()
        self.assertTrue(network.is_merged_in)

        network.merge_out()
        self.assertFalse(network.is_merged_in)

    def test_assistant_lora_wrapped_model_still_creates_training_modules(self):
        transformer = ZImageTransformer2DModel()
        assistant_network = self._build_network(unet=transformer)
        assistant_network.apply_to(
            None,
            transformer,
            apply_text_encoder=False,
            apply_unet=True,
        )

        training_network = self._build_network(unet=transformer)

        self.assertEqual(len(training_network.get_all_modules()), 1)

    def test_transformer_block_scan_handles_root_class_mismatch(self):
        network = self._build_network(
            unet=WrappedZImageTransformer(),
            target_lin_modules=["DoesNotMatchWrapper"],
        )

        self.assertEqual(len(network.get_all_modules()), 1)

    def test_nf4_linear4bit_transformer_modules_are_targeted(self):
        network = self._build_network(
            unet=Ideogram4Transformer(),
            target_lin_modules=["Ideogram4Transformer"],
            base_model=_IdeogramBaseModel(),
        )

        self.assertEqual(len(network.get_all_modules()), 1)
        self.assertIn("layers", network.get_all_modules()[0].lora_name)
        self.assertFalse(network.can_merge_in)
        self.assertFalse(network.get_all_modules()[0].can_merge_in)

        network.merge_in()
        self.assertFalse(network.is_merged_in)

    def test_empty_network_error_includes_targeting_details(self):
        network = self._build_network(["does_not_exist"])

        with self.assertRaisesRegex(ValueError, "only_if_contains=\\['does_not_exist'\\]"):
            network._update_torch_multiplier()


if __name__ == "__main__":
    unittest.main()
