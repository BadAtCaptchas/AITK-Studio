import importlib.util
import sys
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

if "torchaudio" not in sys.modules and importlib.util.find_spec("torchaudio") is None:
    torchaudio_module = stub_module("torchaudio")
    mocked_modules["torchaudio"] = torchaudio_module

with mock.patch.dict("sys.modules", mocked_modules):
    from toolkit.base_lora import _infer_network_config
    from toolkit.config_modules import NetworkConfig
    from toolkit.lora_special import LoRAModule, LoRASpecialNetwork
    from toolkit.models.DoRA import DoRAModule
    from toolkit.models.lokr import LokrModule, balanced_factorization, factorization, legacy_factorization
    from toolkit.util.orbit_quant import OrbitQuantizer
    from toolkit.util.ostris_quant import convert_linear_to_ostris


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


def make_orbit_linear(source=None):
    if source is None:
        source = torch.nn.Linear(32, 32, bias=True)
    module = torch.nn.Linear(source.in_features, source.out_features, bias=source.bias is not None)
    module.load_state_dict(source.state_dict())
    convert_linear_to_ostris(
        module,
        OrbitQuantizer(4, kernel="torch", max_workspace_mb=1),
    )
    return module


def make_adapter_network(network_type, multiplier=1.0):
    network = NetworkStub()
    network.network_type = network_type
    network._multiplier = multiplier
    network.torch_multiplier = torch.tensor([multiplier])
    return network


def forbid_full_dequantization(module):
    failure = mock.Mock(side_effect=AssertionError("full base dequantization"))
    module.ostris_quantizer.dequantize = failure
    if hasattr(module.ostris_quantizer, "dequantize_to"):
        module.ostris_quantizer.dequantize_to = failure
    return failure


class ByteBackedLinear(torch.nn.Module):
    def __init__(self, in_features=4, out_features=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = torch.float32
        self.register_buffer("weight", torch.ones(out_features, in_features, dtype=torch.uint8))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.linear(x.to(self.compute_dtype), self.weight.to(self.compute_dtype), self.bias)


class Fp8Linear(torch.nn.Module):
    def __init__(self, in_features=4, out_features=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = torch.float32
        logical_weight = torch.arange(out_features * in_features, dtype=torch.float32).reshape(out_features, in_features)
        self.register_buffer("weight", logical_weight.reshape(-1, 1))
        self.register_buffer("weight_scale", torch.ones(out_features, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        weight = self.weight.reshape(self.out_features, self.in_features).to(x.dtype)
        return torch.nn.functional.linear(x, weight, self.bias.to(x.dtype))


class FakeNf4Weight:
    quant_state = True

    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = (tensor.numel() // 2, 1)

    @property
    def device(self):
        return self.tensor.device

    def to(self, device):
        return FakeNf4Weight(self.tensor.to(device))

    def dequantize(self):
        return self.tensor


class Linear4bit(torch.nn.Module):
    def __init__(self, in_features=4, out_features=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = torch.float32
        self.weight = FakeNf4Weight(torch.ones(out_features, in_features, dtype=torch.float32))
        self.bias = None

    def forward(self, x):
        return torch.nn.functional.linear(x.to(self.compute_dtype), self.weight.dequantize())


class LokrModuleTest(unittest.TestCase):
    def test_factorization_matches_current_upstream(self):
        self.assertEqual(factorization(128, 16), (16, 8))
        self.assertEqual(factorization(250, -1), (10, 25))
        self.assertEqual(factorization(360, 16), (15, 24))
        self.assertEqual(balanced_factorization(128, 16), (8, 16))
        self.assertEqual(legacy_factorization(128, 16), (16, 8))

    def test_default_lokr_module_uses_upstream_factor_shapes(self):
        lokr = make_lokr(torch.nn.Linear(128, 128, bias=False), factor=16, lora_dim=16)

        self.assertEqual(tuple(lokr.lokr_w1.shape), (16, 16))
        self.assertEqual(tuple(lokr.lokr_w2.shape), (8, 8))

    def test_opt_in_balanced_factorization_preserves_balanced_shapes(self):
        lokr = make_lokr(
            torch.nn.Linear(128, 128, bias=False),
            factor=16,
            lora_dim=16,
            legacy_factorization=False,
        )

        self.assertEqual(tuple(lokr.lokr_w1.shape), (8, 8))
        self.assertEqual(tuple(lokr.lokr_w2.shape), (16, 16))

    def test_linear_shape_uses_weight_shape_when_metadata_mismatches(self):
        module = torch.nn.Linear(1, 4, bias=False)
        module.out_features = 8

        lokr = make_lokr(module, factor=2)

        self.assertEqual(tuple(lokr.get_weight(module.weight.shape).shape), tuple(module.weight.shape))

    def test_forward_promotes_byte_backed_weight_to_compute_dtype(self):
        module = ByteBackedLinear()

        lokr = make_lokr(module)
        lokr.org_forward = module.forward
        x = torch.ones(2, 4, dtype=torch.uint8)

        output = lokr._call_forward(x)

        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(tuple(output.shape), (2, 4))

    def test_bypass_forward_promotes_byte_backed_input_to_compute_dtype(self):
        module = ByteBackedLinear()
        lokr = make_lokr(module, bypass_mode=True)
        lokr.org_forward = module.forward
        x = torch.ones(2, 4, dtype=torch.uint8)

        output = lokr._call_forward(x)

        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(tuple(output.shape), (2, 4))

    def test_forward_uses_logical_shape_for_flat_fp8_weight(self):
        module = Fp8Linear()
        lokr = make_lokr(module)
        lokr.org_forward = module.forward
        x = torch.ones(2, module.in_features, dtype=torch.float32)

        output = lokr._call_forward(x)

        self.assertEqual(lokr.shape, (module.out_features, module.in_features))
        self.assertEqual(tuple(output.shape), (2, module.out_features))
        self.assertTrue(torch.allclose(output, module.forward(x)))

    def test_get_orig_weight_dequantizes_nf4_style_weight(self):
        module = Linear4bit()
        lokr = make_lokr(module)
        x = torch.ones(2, module.in_features, dtype=torch.float32)

        weight = lokr.get_orig_weight(x.device)

        self.assertEqual(lokr.shape, (module.out_features, module.in_features))
        self.assertEqual(tuple(weight.shape), (module.out_features, module.in_features))
        self.assertTrue(torch.allclose(weight, module.weight.dequantize()))

    def test_weight_decompose_returns_adapted_output_only(self):
        module = torch.nn.Linear(4, 4, bias=True)
        lokr = make_lokr(module, weight_decompose=True)
        lokr.org_forward = module.forward
        x = torch.randn(2, 4)

        output = lokr._call_forward(x)

        self.assertTrue(torch.allclose(output, module.forward(x), atol=1e-5))

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

    def test_orbit_lokr_uses_factorized_forward_and_round_trips_state(self):
        torch.manual_seed(41)
        source = torch.nn.Linear(32, 32, bias=True)
        base = make_orbit_linear(source)
        packed_before = base.orbit_packed.clone()
        network = make_adapter_network("lokr")
        adapter = LokrModule(
            "orbit_lokr",
            base,
            lora_dim=4,
            alpha=4,
            factor=4,
            network=network,
        )
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.normal_(std=0.1)
        adapter.apply_to()

        # A compatibility dequantization on the hot path is a hard failure.
        forbid_full_dequantization(base)
        x = torch.randn(3, 32, requires_grad=True)
        output = base(x)
        output.square().mean().backward()

        self.assertTrue(adapter.bypass_mode)
        self.assertFalse(adapter.can_merge_in)
        self.assertTrue(torch.equal(base.orbit_packed, packed_before))
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in adapter.parameters()
        ))

        saved = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
        self.assertTrue(all(value.numel() < base.logical_weight_numel for value in saved.values()))

        reloaded_base = make_orbit_linear(source)
        reloaded_network = make_adapter_network("lokr")
        reloaded = LokrModule(
            "orbit_lokr",
            reloaded_base,
            lora_dim=4,
            alpha=4,
            factor=4,
            network=reloaded_network,
        )
        reloaded._test_network = reloaded_network
        reloaded.load_state_dict(saved)
        reloaded.apply_to()
        self.assertTrue(torch.allclose(reloaded_base(x.detach()), output.detach(), atol=1e-5))

    def test_orbit_lokr_rejects_weight_decomposition(self):
        base = make_orbit_linear()
        with self.assertRaisesRegex(ValueError, "full dense base weight"):
            LokrModule(
                "orbit_lokr_wd",
                base,
                lora_dim=4,
                alpha=4,
                factor=4,
                weight_decompose=True,
                network=make_adapter_network("lokr"),
            )

    def test_orbit_lokr_factorized_rank_dropout_drops_delta(self):
        torch.manual_seed(44)
        base = make_orbit_linear()
        inputs = torch.randn(2, 32)
        expected = base(inputs)
        network = make_adapter_network("lokr")
        adapter = LokrModule(
            "orbit_lokr_dropout",
            base,
            lora_dim=4,
            alpha=4,
            factor=4,
            rank_dropout=1.0,
            network=network,
        )
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.normal_(std=0.1)
        adapter.train()
        adapter.apply_to()
        forbid_full_dequantization(base)

        self.assertTrue(torch.allclose(base(inputs), expected, atol=1e-6))

    def test_orbit_dora_is_factorized_matches_reference_and_round_trips(self):
        torch.manual_seed(42)
        source = torch.nn.Linear(32, 32, bias=True)
        base = make_orbit_linear(source)
        packed_before = base.orbit_packed.clone()
        network = make_adapter_network("dora", multiplier=0.7)
        adapter = DoRAModule(
            "orbit_dora",
            base,
            lora_dim=4,
            alpha=2,
            network=network,
        )
        with torch.no_grad():
            adapter.lora_up.weight.normal_(std=0.1)
            adapter.lora_down.weight.normal_(std=0.1)

        x = torch.randn(3, 32, requires_grad=True)
        dense_base = base.dequantize_weight()
        delta = (
            adapter.lora_up.weight @ adapter.lora_down.weight
        ) * adapter.scale * network.torch_multiplier.mean()
        norm = (dense_base.float() + delta.float()).norm(dim=1)
        ratio = adapter.magnitude.detach().float() / norm
        expected = base.bias + (
            torch.nn.functional.linear(x, dense_base, None)
            + torch.nn.functional.linear(x, delta, None)
        ) * ratio

        adapter.apply_to()
        forbid_full_dequantization(base)
        output = base(x)
        self.assertTrue(torch.allclose(output, expected, atol=1e-5))
        output.square().mean().backward()

        self.assertTrue(torch.equal(base.orbit_packed, packed_before))
        self.assertGreater(adapter.lora_up.weight.grad.abs().sum().item(), 0)
        self.assertGreater(adapter.lora_down.weight.grad.abs().sum().item(), 0)
        self.assertGreater(adapter.magnitude.grad.abs().sum().item(), 0)

        saved = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
        self.assertTrue(all(value.numel() < base.logical_weight_numel for value in saved.values()))

        reloaded_base = make_orbit_linear(source)
        reloaded_network = make_adapter_network("dora", multiplier=0.7)
        reloaded = DoRAModule(
            "orbit_dora",
            reloaded_base,
            lora_dim=4,
            alpha=2,
            network=reloaded_network,
        )
        reloaded._test_network = reloaded_network
        reloaded.load_state_dict(saved)
        reloaded.apply_to()
        self.assertTrue(torch.allclose(reloaded_base(x.detach()), output.detach(), atol=1e-5))

    def test_orbit_lora_stays_parallel_has_gradients_and_round_trips(self):
        torch.manual_seed(43)
        source = torch.nn.Linear(32, 32, bias=True)
        base = make_orbit_linear(source)
        packed_before = base.orbit_packed.clone()
        network = make_adapter_network("lora")
        adapter = LoRAModule(
            "orbit_lora",
            base,
            lora_dim=4,
            alpha=4,
            network=network,
        )
        with torch.no_grad():
            adapter.lora_up.weight.normal_(std=0.1)
        adapter.apply_to()
        forbid_full_dequantization(base)

        x = torch.randn(3, 32, requires_grad=True)
        output = base(x)
        output.square().mean().backward()
        self.assertFalse(adapter.can_merge_in)
        self.assertTrue(torch.equal(base.orbit_packed, packed_before))
        self.assertGreater(adapter.lora_up.weight.grad.abs().sum().item(), 0)
        self.assertGreater(adapter.lora_down.weight.grad.abs().sum().item(), 0)

        saved = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
        reloaded_base = make_orbit_linear(source)
        reloaded_network = make_adapter_network("lora")
        reloaded = LoRAModule(
            "orbit_lora",
            reloaded_base,
            lora_dim=4,
            alpha=4,
            network=reloaded_network,
        )
        reloaded._test_network = reloaded_network
        reloaded.load_state_dict(saved)
        reloaded.apply_to()
        self.assertTrue(torch.allclose(reloaded_base(x.detach()), output.detach(), atol=1e-5))

    def test_orbit_rejects_logical_weight_sized_full_rank_adapter(self):
        with self.assertRaisesRegex(ValueError, "Full-rank adapters"):
            LoRAModule(
                "orbit_fullrank",
                make_orbit_linear(),
                lora_dim=32,
                alpha=32,
                network=make_adapter_network("fullrank"),
            )


if __name__ == "__main__":
    unittest.main()
