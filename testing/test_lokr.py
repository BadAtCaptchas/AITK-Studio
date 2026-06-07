import importlib.util
import types
import unittest
from importlib.machinery import ModuleSpec
from unittest import mock

import torch


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

album_artwork_module = stub_module("toolkit.audio.album_artwork")
album_artwork_module.add_album_artwork = mock.Mock()
prompt_utils_module = stub_module("toolkit.prompt_utils")
prompt_utils_module.PromptEmbeds = type("PromptEmbeds", (), {})
accelerate_module = stub_module("accelerate")
accelerate_module.Accelerator = type("Accelerator", (), {})

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
    "accelerate": accelerate_module,
}

if importlib.util.find_spec("torchaudio") is None:
    torchaudio_module = stub_module("torchaudio")
    mocked_modules["torchaudio"] = torchaudio_module

with mock.patch.dict("sys.modules", mocked_modules):
    from toolkit.base_lora import _infer_network_config
    from toolkit.config_modules import NetworkConfig
    from toolkit.lora_special import LoRASpecialNetwork
    from toolkit.models.lokr import LokrModule, factorization, legacy_factorization


class NetworkStub:
    network_type = "lokr"
    is_lorm = False
    is_active = True
    is_merged_in = False
    _multiplier = 1.0

    def __init__(self):
        self.torch_multiplier = torch.tensor([1.0])


class TinyRoot(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.proj(x)


class FakeBaseModel:
    arch = "tiny"
    use_old_lokr_format = False

    def get_transformer_block_names(self):
        return None


def make_lokr(module=None, **kwargs):
    network = NetworkStub()
    lokr = LokrModule(
        "tiny",
        module or torch.nn.Linear(4, 4, bias=False),
        lora_dim=kwargs.pop("lora_dim", 1),
        alpha=kwargs.pop("alpha", 1),
        network=network,
        **kwargs,
    )
    lokr._test_network = network
    return lokr


class LokrModuleTest(unittest.TestCase):
    def test_factorization_matches_current_upstream(self):
        self.assertEqual(factorization(128, 16), (8, 16))
        self.assertEqual(factorization(250, -1), (10, 25))
        self.assertEqual(factorization(360, 16), (15, 24))
        self.assertEqual(legacy_factorization(128, 16), (16, 8))

    def test_rank_dropout_one_drops_all_weight_rows(self):
        module = make_lokr(rank_dropout=1.0)
        module.train()
        with torch.no_grad():
            for param in module.parameters():
                param.fill_(1.0)

        weight = module.get_weight((4, 4))

        self.assertTrue(torch.equal(weight, torch.zeros_like(weight)))

    def test_rank_dropout_zero_keeps_weight_rows(self):
        module = make_lokr(rank_dropout=0.0)
        module.train()
        with torch.no_grad():
            for param in module.parameters():
                param.fill_(1.0)

        weight = module.get_weight((4, 4))

        self.assertGreater(weight.abs().sum().item(), 0.0)

    def test_scalar_is_folded_into_state_dict(self):
        module = make_lokr(use_scalar=True)
        with torch.no_grad():
            module.lokr_w1.fill_(2.0)
            module.lokr_w2.fill_(3.0)
            module.scalar.fill_(0.5)

        state_dict = module.state_dict()

        self.assertNotIn("scalar", state_dict)
        self.assertTrue(torch.equal(state_dict["lokr_w1"], torch.ones_like(state_dict["lokr_w1"])))
        self.assertTrue(torch.equal(state_dict["lokr_w2"], torch.full_like(state_dict["lokr_w2"], 3.0)))

    def test_bypass_forward_matches_rebuild_forward_for_linear(self):
        torch.manual_seed(1)
        base = torch.nn.Linear(4, 4, bias=True)
        bypass_base = torch.nn.Linear(4, 4, bias=True)
        bypass_base.load_state_dict(base.state_dict())

        module = make_lokr(base)
        bypass = make_lokr(bypass_base, bypass_mode=True)
        with torch.no_grad():
            for param in module.parameters():
                param.normal_()
        bypass.load_state_dict(module.state_dict(), strict=False)

        module.org_forward = base.forward
        bypass.org_forward = bypass_base.forward
        x = torch.randn(3, 4)

        self.assertTrue(torch.allclose(module._call_forward(x), bypass._call_forward(x), atol=1e-5))

    def test_network_config_passes_lokr_flags_to_module(self):
        network_config = NetworkConfig(
            type="lokr",
            linear=1,
            linear_alpha=1,
            transformer_only=False,
            lokr_use_scalar=True,
            lokr_weight_decompose=True,
            lokr_bypass_mode=True,
            lokr_rs_lora=True,
            lokr_factor=2,
        )
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=TinyRoot(),
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=True,
            target_lin_modules=["TinyRoot"],
            base_model=FakeBaseModel(),
        )

        [module] = network.get_all_modules()

        self.assertIsInstance(module.scalar, torch.nn.Parameter)
        self.assertTrue(module.wd)
        self.assertTrue(module.bypass_mode)
        self.assertTrue(module.rs_lora)

    def test_base_lora_infers_lokr_dora_scale_as_weight_decompose(self):
        state_dict = {
            "lycoris_proj.lokr_w1": torch.ones(2, 2),
            "lycoris_proj.lokr_w2": torch.ones(2, 2),
            "lycoris_proj.dora_scale": torch.ones(4, 1),
        }

        config, network_kwargs, network_type = _infer_network_config(state_dict)

        self.assertEqual(network_type, "lokr")
        self.assertTrue(config.lokr_weight_decompose)
        self.assertTrue(config.lokr_legacy_factorization)
        self.assertEqual(network_kwargs["only_if_contains"], ["proj"])


if __name__ == "__main__":
    unittest.main()
