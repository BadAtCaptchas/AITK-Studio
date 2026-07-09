import unittest
import sys
import types

import torch

diffusers_module = types.ModuleType("diffusers")
diffusers_module.UNet2DConditionModel = type("UNet2DConditionModel", (), {})
diffusers_module.PixArtTransformer2DModel = type("PixArtTransformer2DModel", (), {})
diffusers_module.AuraFlowTransformer2DModel = type("AuraFlowTransformer2DModel", (), {})
diffusers_module.WanTransformer3DModel = type("WanTransformer3DModel", (), {})
sys.modules.setdefault("diffusers", diffusers_module)
transformers_module = types.ModuleType("transformers")
transformers_module.CLIPTextModel = type("CLIPTextModel", (), {})
sys.modules.setdefault("transformers", transformers_module)
optimum_module = types.ModuleType("optimum")
quanto_module = types.ModuleType("optimum.quanto")
quanto_module.QTensor = type("QTensor", (), {})
quanto_module.QBytesTensor = type("QBytesTensor", (), {})
sys.modules.setdefault("optimum", optimum_module)
sys.modules.setdefault("optimum.quanto", quanto_module)
torchaudio_module = types.ModuleType("torchaudio")
torchaudio_module.save = lambda *args, **kwargs: None
sys.modules.setdefault("torchaudio", torchaudio_module)
album_artwork_module = types.ModuleType("toolkit.audio.album_artwork")
album_artwork_module.add_album_artwork = lambda *args, **kwargs: None
sys.modules.setdefault("toolkit.audio.album_artwork", album_artwork_module)
config_modules_module = types.ModuleType("toolkit.config_modules")


class NetworkConfig(types.SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.old_lokr_format = kwargs.get("old_lokr_format", False)


config_modules_module.NetworkConfig = NetworkConfig
sys.modules.setdefault("toolkit.config_modules", config_modules_module)
lokr_module = types.ModuleType("toolkit.models.lokr")
lokr_module.LokrModule = type("LokrModule", (torch.nn.Module,), {})
sys.modules.setdefault("toolkit.models.lokr", lokr_module)
dora_module = types.ModuleType("toolkit.models.DoRA")
dora_module.DoRAModule = type("DoRAModule", (torch.nn.Module,), {})
sys.modules.setdefault("toolkit.models.DoRA", dora_module)
lorm_module = types.ModuleType("toolkit.lorm")


def _count_parameters(module):
    if not hasattr(module, "parameters"):
        return 0
    count = sum(p.numel() for p in module.parameters())
    if (
        getattr(module, "logical_weight_numel", None) is not None
        and "weight" not in module._parameters
    ):
        count += int(module.logical_weight_numel)
    return count


lorm_module.count_parameters = _count_parameters
lorm_module.extract_conv = lambda *args, **kwargs: {}
lorm_module.extract_linear = lambda *args, **kwargs: {}
sys.modules.setdefault("toolkit.lorm", lorm_module)
metadata_module = types.ModuleType("toolkit.metadata")
metadata_module.add_model_hash_to_meta = lambda state_dict, meta: meta
sys.modules.setdefault("toolkit.metadata", metadata_module)
saving_module = types.ModuleType("toolkit.saving")
saving_module.get_lora_keymap_from_model_keymap = lambda keymap: keymap
sys.modules.setdefault("toolkit.saving", saving_module)
kohya_lora_module = types.ModuleType("toolkit.kohya_lora")


class LoRANetwork(torch.nn.Module):
    UNET_TARGET_REPLACE_MODULE = ["UNet2DConditionModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["UNet2DConditionModel"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]

    def prepare_optimizer_params(self, *args, **kwargs):
        return []

    def apply_max_norm_regularization(self, *args, **kwargs):
        return None

    def apply_to(
        self,
        text_encoder,
        unet,
        apply_text_encoder=True,
        apply_unet=True,
    ):
        for lora_module in self.get_all_modules():
            lora_module.apply_to()


kohya_lora_module.LoRANetwork = LoRANetwork
sys.modules.setdefault("toolkit.kohya_lora", kohya_lora_module)

quantize_module = types.ModuleType("toolkit.util.quantize")
quantize_module.is_quantized_tensor = lambda tensor: bool(
    getattr(tensor, "_is_ostris_weight", False)
)
quantize_module.get_torchao_config = lambda _qtype: None


def _requantize_module_weight(module, fp_weight, orig_dtype, _config):
    if getattr(module, "is_ostris_quantized", False):
        module.requantize_(fp_weight)
    else:
        module.weight = torch.nn.Parameter(
            fp_weight.to(orig_dtype),
            requires_grad=False,
        )


quantize_module.requantize_module_weight = _requantize_module_weight
sys.modules.setdefault("toolkit.util.quantize", quantize_module)

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.util.ostris_quant import OstrisQuantizer, convert_linear_to_ostris


class _ExactQuantizer(OstrisQuantizer):
    def quantize_(self, module, weight_fp32):
        module.register_buffer("exact_weight", weight_fp32.clone(), persistent=False)

    def dequantize(self, module):
        return module.exact_weight.float()

    def requantize_(self, module, fp_weight):
        module.exact_weight = fp_weight.to(module.exact_weight)


class ZImageTransformer2DModel(torch.nn.Module):
    def __init__(self, quantized=False):
        super().__init__()
        projection = torch.nn.Linear(4, 4)
        if quantized:
            convert_linear_to_ostris(projection, _ExactQuantizer())
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "proj": projection,
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


class NestedLanguageTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "q_proj": torch.nn.Linear(4, 4),
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


class _NestedLanguageBaseModel(_BaseModel):
    arch = "hidream_o1"

    def get_transformer_block_names(self):
        return ["model.language_model.layers"]


class LoRASpecialFilterTest(unittest.TestCase):
    def _build_network(
        self,
        only_if_contains=None,
        unet=None,
        target_lin_modules=None,
        base_model=None,
        full_if_contains=None,
        parameter_threshold=0,
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
            full_if_contains=full_if_contains,
            parameter_threshold=parameter_threshold,
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

    def test_ostris_linear_uses_logical_size_for_parameter_threshold(self):
        transformer = ZImageTransformer2DModel(quantized=True)

        included = self._build_network(
            unet=transformer,
            parameter_threshold=20,
        )
        excluded = self._build_network(
            unet=ZImageTransformer2DModel(quantized=True),
            parameter_threshold=21,
        )

        self.assertEqual(len(included.get_all_modules()), 1)
        self.assertEqual(len(excluded.get_all_modules()), 0)

    def test_ostris_lora_merge_requantizes_without_replacing_module(self):
        transformer = ZImageTransformer2DModel(quantized=True)
        layer = transformer.layers[0]["proj"]
        network = self._build_network(unet=transformer)
        lora_module = network.get_all_modules()[0]

        with torch.no_grad():
            lora_module.lora_down.weight.fill_(0.2)
            lora_module.lora_up.weight.fill_(0.3)
        original = layer.dequantize_weight().clone()
        expected_delta = (
            lora_module.lora_up.weight.float()
            @ lora_module.lora_down.weight.float()
        ) * lora_module.scale

        lora_module.merge_in()

        self.assertNotIn("weight", layer._parameters)
        self.assertTrue(
            torch.allclose(
                layer.dequantize_weight(),
                original + expected_delta,
            )
        )

    def test_ostris_full_if_contains_uses_functional_delta_and_requantizes(self):
        transformer = ZImageTransformer2DModel(quantized=True)
        layer = transformer.layers[0]["proj"]
        network = self._build_network(
            unet=transformer,
            full_if_contains=["proj"],
        )
        full_module = network.get_all_modules()[0]
        self.assertEqual(full_module.__class__.__name__, "FullModule")

        x = torch.randn(2, 4)
        base_output = layer(x).detach()
        with torch.no_grad():
            full_module.diff.fill_(0.25)
            if full_module.diff_b is not None:
                full_module.diff_b.zero_()
        expected_delta = torch.nn.functional.linear(x, full_module.diff)

        full_module.apply_to()
        network.is_active = True
        network._multiplier = 1.0
        network._update_torch_multiplier()
        actual = layer(x)

        self.assertNotIn("weight", layer._parameters)
        self.assertTrue(torch.allclose(actual, base_output + expected_delta))

        before_merge = layer.dequantize_weight().clone()
        full_module.merge_in()
        self.assertNotIn("weight", layer._parameters)
        self.assertTrue(
            torch.allclose(
                layer.dequantize_weight(),
                before_merge + full_module.diff,
            )
        )

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

    def test_nested_transformer_block_names_match_clean_lora_paths(self):
        network = self._build_network(
            unet=NestedLanguageTransformer(),
            target_lin_modules=["NestedLanguageTransformer"],
            base_model=_NestedLanguageBaseModel(),
        )

        self.assertEqual(len(network.get_all_modules()), 1)
        self.assertIn(
            "model$$language_model$$layers",
            network.get_all_modules()[0].lora_name,
        )

    def test_empty_network_error_includes_targeting_details(self):
        network = self._build_network(["does_not_exist"])

        with self.assertRaisesRegex(ValueError, "only_if_contains=\\['does_not_exist'\\]"):
            network._update_torch_multiplier()


if __name__ == "__main__":
    unittest.main()
