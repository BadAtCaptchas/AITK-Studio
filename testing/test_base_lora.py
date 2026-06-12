import os
import tempfile
import types
import unittest
import importlib.util
from importlib.machinery import ModuleSpec
from unittest import mock
from types import SimpleNamespace

import torch
import safetensors.torch

album_artwork_module = types.ModuleType("toolkit.audio.album_artwork")
album_artwork_module.add_album_artwork = mock.Mock()

prompt_utils_module = types.ModuleType("toolkit.prompt_utils")
prompt_utils_module.PromptEmbeds = type("PromptEmbeds", (), {})

def stub_module(name):
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name, loader=None)
    return module


DummyModel = type("DummyModel", (torch.nn.Module,), {})

diffusers_module = stub_module("diffusers")
diffusers_module.__path__ = []
diffusers_module.AutoencoderKL = DummyModel
diffusers_module.UNet2DConditionModel = DummyModel
diffusers_module.PixArtTransformer2DModel = DummyModel
diffusers_module.AuraFlowTransformer2DModel = DummyModel
diffusers_module.WanTransformer3DModel = DummyModel
for scheduler_name in [
    "DDPMScheduler",
    "EulerAncestralDiscreteScheduler",
    "DPMSolverMultistepScheduler",
    "DPMSolverSinglestepScheduler",
    "LMSDiscreteScheduler",
    "PNDMScheduler",
    "DDIMScheduler",
    "EulerDiscreteScheduler",
    "HeunDiscreteScheduler",
    "KDPM2DiscreteScheduler",
    "KDPM2AncestralDiscreteScheduler",
]:
    setattr(diffusers_module, scheduler_name, DummyModel)
diffusers_utils_module = stub_module("diffusers.utils")
diffusers_utils_module.__path__ = []
diffusers_torch_utils_module = stub_module("diffusers.utils.torch_utils")
diffusers_torch_utils_module.is_compiled_module = lambda module: False

transformers_module = stub_module("transformers")
transformers_module.CLIPTextModel = DummyModel
transformers_module.T5Tokenizer = DummyModel
transformers_module.T5EncoderModel = DummyModel
transformers_module.UMT5EncoderModel = DummyModel

optimum_module = stub_module("optimum")
optimum_quanto_module = stub_module("optimum.quanto")
optimum_quanto_module.QBytesTensor = type("QBytesTensor", (), {})
optimum_quanto_module.QTensor = type("QTensor", (), {})

torchao_module = stub_module("torchao")
torchao_dtypes_module = stub_module("torchao.dtypes")
torchao_quantization_module = stub_module("torchao.quantization")
torchao_quant_primitives_module = stub_module("torchao.quantization.quant_primitives")
torchao_dtypes_module.AffineQuantizedTensor = type("AffineQuantizedTensor", (), {})
torchao_quant_primitives_module._DTYPE_TO_BIT_WIDTH = {}

mocked_modules = {
    "diffusers": diffusers_module,
    "diffusers.utils": diffusers_utils_module,
    "diffusers.utils.torch_utils": diffusers_torch_utils_module,
    "transformers": transformers_module,
    "optimum": optimum_module,
    "optimum.quanto": optimum_quanto_module,
    "torchao": torchao_module,
    "torchao.dtypes": torchao_dtypes_module,
    "torchao.quantization": torchao_quantization_module,
    "torchao.quantization.quant_primitives": torchao_quant_primitives_module,
    "toolkit.audio.album_artwork": album_artwork_module,
    "toolkit.prompt_utils": prompt_utils_module,
}

if importlib.util.find_spec("torchaudio") is None:
    torchaudio_module = stub_module("torchaudio")
    mocked_modules["torchaudio"] = torchaudio_module

with mock.patch.dict(
    "sys.modules",
    mocked_modules,
):
    from toolkit.base_lora import fuse_base_lora_into_model
    from toolkit.config_modules import NetworkConfig
    from toolkit.lora_special import LoRASpecialNetwork


class TinyRoot(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
        torch.nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


class FakeBaseModel:
    def __init__(self, model, base_lora_path=None, base_lora_strength=1.0):
        self.model = model
        self.unet = model
        self.model_config = SimpleNamespace(
            base_lora_path=base_lora_path,
            base_lora_strength=base_lora_strength,
        )
        self.device_torch = torch.device("cpu")
        self.torch_dtype = torch.float32
        self.is_transformer = True
        self.is_xl = False
        self.is_ssd = False
        self.is_v2 = False
        self.is_v3 = False
        self.is_pixart = False
        self.is_auraflow = False
        self.is_flux = False
        self.is_lumina2 = False
        self.is_vega = False
        self.use_old_lokr_format = True
        self.target_lora_modules = ["TinyRoot"]

    def get_model_to_train(self):
        return self.model

    def get_transformer_block_names(self):
        return None

    def convert_lora_weights_before_load(self, state_dict):
        return state_dict

    def convert_lora_weights_before_save(self, state_dict):
        return state_dict


def save_tiny_lora(path):
    model = TinyRoot()
    base_model = FakeBaseModel(model)
    network_config = NetworkConfig(
        type="lora",
        linear=1,
        linear_alpha=1,
        transformer_only=True,
    )
    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=model,
        lora_dim=1,
        multiplier=1.0,
        alpha=1,
        train_unet=True,
        train_text_encoder=False,
        network_config=network_config,
        network_type=network_config.type,
        transformer_only=network_config.transformer_only,
        is_transformer=True,
        target_lin_modules=base_model.target_lora_modules,
        base_model=base_model,
    )
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    module = network.get_all_modules()[0]
    module.lora_down.weight.data = torch.tensor([[1.0, 2.0]])
    module.lora_up.weight.data = torch.tensor([[3.0], [4.0]])
    network.save_weights(path, dtype=torch.float32, metadata={})


class BaseLoraMergeHelperTest(unittest.TestCase):
    def test_fuses_weights_once_and_leaves_adapter_inactive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "tiny.safetensors")
            save_tiny_lora(lora_path)

            model = TinyRoot()
            base_model = FakeBaseModel(
                model,
                base_lora_path=lora_path,
                base_lora_strength=0.5,
            )

            result = fuse_base_lora_into_model(base_model)

            expected_weight = torch.tensor([[1.5, 3.0], [2.0, 4.0]])
            self.assertEqual(result.num_modules, 1)
            self.assertTrue(torch.allclose(model.proj.weight.detach(), expected_weight))

            x = torch.tensor([[1.0, 1.0]])
            expected_output = x @ expected_weight.t()
            self.assertTrue(torch.allclose(model(x), expected_output))

            [network] = base_model._fused_base_lora_networks
            self.assertFalse(network.is_active)
            self.assertTrue(network.is_merged_in)
            self.assertFalse(network.can_merge_in)

    def test_rejects_dora_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "dora.safetensors")
            safetensors.torch.save_file(
                {
                    "proj.lora_down.weight": torch.ones(1, 2),
                    "proj.lora_up.weight": torch.ones(2, 1),
                    "proj.dora_scale": torch.ones(2),
                },
                lora_path,
            )

            with self.assertRaisesRegex(ValueError, "DoRA"):
                fuse_base_lora_into_model(FakeBaseModel(TinyRoot(), base_lora_path=lora_path))

    def test_rejects_zero_matched_modules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = os.path.join(tmpdir, "unmatched.safetensors")
            safetensors.torch.save_file(
                {
                    "not_the_model.lora_down.weight": torch.ones(1, 2),
                    "not_the_model.lora_up.weight": torch.ones(2, 1),
                },
                lora_path,
            )

            with self.assertRaisesRegex(ValueError, "matched zero"):
                fuse_base_lora_into_model(FakeBaseModel(TinyRoot(), base_lora_path=lora_path))


if __name__ == "__main__":
    unittest.main()
