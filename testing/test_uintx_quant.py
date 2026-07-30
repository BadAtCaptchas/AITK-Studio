import copy
import os
import tempfile
import unittest
import weakref
from importlib import metadata

import torch

from toolkit.quantized_cache import QuantizedModelCache, quantized_cache_key
from toolkit.util.ostris_quant import (
    OstrisLinear,
    get_ostris_backend_registry,
    load_quantized_layers,
    save_quantized_layers,
)
from toolkit.util.quantize import get_qtype, quantize, requantize_module_weight
from toolkit.util.uintx_quant import pack_uintx, unpack_uintx, unpack_uintx_range

try:
    TORCHAO_010 = metadata.version("torchao").startswith("0.10.")
except metadata.PackageNotFoundError:
    TORCHAO_010 = False


class UIntXModel(torch.nn.Module):
    def __init__(self, in_features=128, out_features=16):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)


def deterministic_linear(dtype=torch.float32):
    layer = torch.nn.Linear(128, 16, bias=True, dtype=dtype)
    with torch.no_grad():
        values = torch.linspace(-1.75, 2.25, layer.weight.numel(), dtype=torch.float32)
        layer.weight.copy_(values.view_as(layer.weight).to(dtype))
        layer.bias.copy_(torch.linspace(-0.2, 0.2, layer.out_features).to(dtype))
    return layer


class UIntXPackingTests(unittest.TestCase):
    def test_pack_unpack_and_bounded_ranges_round_trip(self):
        generator = torch.Generator().manual_seed(17)
        for bits in range(2, 8):
            with self.subTest(bits=bits):
                codes = torch.randint(
                    0,
                    1 << bits,
                    (1031,),
                    generator=generator,
                    dtype=torch.uint8,
                )
                packed = pack_uintx(codes, bits)
                self.assertTrue(torch.equal(unpack_uintx(packed, bits, codes.numel()), codes))
                self.assertTrue(
                    torch.equal(
                        unpack_uintx_range(packed, bits, codes.numel(), 37, 997),
                        codes[37:997],
                    )
                )


class UIntXBackendTests(unittest.TestCase):
    def test_registry_and_uint8_rejection(self):
        registry = get_ostris_backend_registry()
        for bits in range(2, 9):
            qtype = f"uint{bits}"
            with self.subTest(qtype=qtype):
                self.assertEqual(registry[qtype].bits, bits)
                self.assertIn("packed_cache", registry[qtype].capabilities)
                self.assertEqual(get_qtype(qtype).__class__.__name__, "ostristype")

        layer = deterministic_linear()
        report = quantize(layer, weights="uint8")
        self.assertNotIsInstance(layer, OstrisLinear)
        self.assertEqual(report.quantized_modules, 0)
        self.assertEqual(report.skip_reasons["unsupported_shape"].modules, 1)

    def test_unsupported_shape_stays_dense(self):
        layer = torch.nn.Linear(96, 8)
        report = quantize(layer, weights="uint4")
        self.assertNotIsInstance(layer, OstrisLinear)
        self.assertEqual(report.skip_reasons["unsupported_shape"].modules, 1)

    def test_forward_backward_requantize_and_telemetry(self):
        layer = deterministic_linear()
        report = quantize(
            layer,
            weights="uint4",
            max_workspace_mb=1,
            component_label="uintx-test",
        )
        self.assertIsInstance(layer, OstrisLinear)
        self.assertEqual(report.backend, "ostris")
        self.assertEqual(report.quantized_modules, 1)
        self.assertEqual(report.compressed_bytes, layer.uintx_packed.numel())
        self.assertGreater(report.metadata_bytes, 0)

        activation = torch.randn(2, 3, 128, requires_grad=True)
        reference_activation = activation.detach().clone().requires_grad_(True)
        materialized_weight = layer.dequantize_weight().detach()
        quantized_output = layer(activation)
        reference_output = torch.nn.functional.linear(
            reference_activation,
            materialized_weight,
            layer.bias.detach(),
        )
        quantized_output.square().mean().backward()
        reference_output.square().mean().backward()
        self.assertTrue(torch.equal(quantized_output, reference_output))
        self.assertTrue(
            torch.allclose(
                activation.grad,
                reference_activation.grad,
                atol=1e-6,
                rtol=1e-6,
            )
        )

        original = materialized_weight.clone()
        requantize_module_weight(
            layer,
            original + 0.25,
            layer.ostris_orig_dtype,
            get_qtype("uint4"),
        )
        self.assertIsInstance(layer, OstrisLinear)
        self.assertGreater(float((layer.dequantize_weight() - original).abs().mean()), 0.1)

    def test_conversion_disables_accuracy_adapter_merging(self):
        class Adapter:
            can_merge_in = True

        layer = deterministic_linear()
        adapter = Adapter()
        layer.ara_lora_ref = weakref.ref(adapter)
        quantize(layer, weights="uint3")
        self.assertFalse(adapter.can_merge_in)

    def test_packed_layer_and_model_cache_round_trips(self):
        source = torch.nn.Sequential(deterministic_linear())
        quantize(source, weights="uint5", max_workspace_mb=1)
        activation = torch.randn(2, 128)
        expected = source(activation)

        with tempfile.TemporaryDirectory() as root:
            packed_path = os.path.join(root, "uintx.safetensors")
            save_quantized_layers({"0": source[0]}, packed_path)
            packed_target = torch.nn.Sequential(torch.nn.Linear(128, 16))
            self.assertEqual(load_quantized_layers(packed_target, packed_path), 1)
            self.assertIsInstance(packed_target[0], OstrisLinear)
            self.assertTrue(torch.equal(source[0].uintx_packed, packed_target[0].uintx_packed))
            self.assertTrue(torch.equal(expected, packed_target(activation)))

            model = UIntXModel()
            model.linear.load_state_dict(copy.deepcopy(deterministic_linear().state_dict()))
            quantize(model, weights="uint6", max_workspace_mb=1)
            cache_key, payload = quantized_cache_key(
                "uintx-test",
                {
                    "dtype": "float32",
                    "qtype": "uint6",
                    "quantize_kwargs": {"max_workspace_mb": 1},
                },
                sources=[],
            )
            cache = QuantizedModelCache(root)
            cache.save(model, "uintx-test", cache_key, payload)
            restored = UIntXModel()
            cache.load(restored, "uintx-test", cache_key, device=torch.device("cpu"))
            self.assertIsInstance(restored.linear, OstrisLinear)
            self.assertTrue(torch.equal(model.linear.uintx_packed, restored.linear.uintx_packed))
            self.assertTrue(torch.equal(model(activation), restored(activation)))

    @unittest.skipUnless(TORCHAO_010, "TorchAO 0.10 is required for the compatibility reference")
    def test_uint2_through_uint7_match_torchao_010_byte_for_byte(self):
        from torchao.quantization.quant_api import UIntXWeightOnlyConfig
        from torchao.quantization.quant_api import quantize_ as torchao_quantize_

        activation = torch.linspace(-0.75, 0.75, 256, dtype=torch.bfloat16).view(2, 128)
        for bits in range(2, 8):
            with self.subTest(bits=bits):
                source = deterministic_linear(dtype=torch.bfloat16)
                reference = copy.deepcopy(source)
                ours = copy.deepcopy(source)
                torchao_quantize_(
                    reference,
                    UIntXWeightOnlyConfig(getattr(torch, f"uint{bits}")),
                )
                quantize(ours, weights=f"uint{bits}")

                reference_weight = reference.weight.dequantize().to(torch.bfloat16).contiguous()
                ours_weight = ours.dequantize_weight(dtype=torch.bfloat16).contiguous()
                self.assertTrue(
                    torch.equal(
                        reference_weight.view(torch.uint8),
                        ours_weight.view(torch.uint8),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        reference(activation).contiguous().view(torch.uint8),
                        ours(activation).contiguous().view(torch.uint8),
                    )
                )


if __name__ == "__main__":
    unittest.main()
