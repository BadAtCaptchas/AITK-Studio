import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from toolkit.util import convrot_quant
from toolkit.util.ostris_quant import (
    OstrisLazyWeight,
    OstrisLinear,
    convert_linear_to_ostris,
    get_ostris_quantizer,
    load_quantized_layers,
    save_quantized_layers,
)
from toolkit.util.quantize import get_qtype, quantize

CONVROT_QTYPES = (
    "convrot4",
    "convrot8",
    "convrotint8",
    "convrotint7",
    "convrotint6",
    "convrotint5",
    "convrotint4",
    "convrotint3",
    "convrotint2",
    "convrotbitnet",
)


def make_layer(qtype: str, *, in_features=64, out_features=64, dtype=torch.float32):
    torch.manual_seed(101)
    layer = torch.nn.Linear(in_features, out_features, dtype=dtype)
    quantize(
        layer,
        weights=get_qtype(qtype, kernel="torch", max_workspace_mb=1),
    )
    if not isinstance(layer, OstrisLinear):
        raise AssertionError("ConvRot test layer was not converted")
    return layer


class ConvRotPrimitiveTest(unittest.TestCase):
    def test_regular_hadamard_rotation_is_self_inverse(self):
        value = torch.randn(3, 64)
        rotated = convrot_quant.rotate(value, 64)
        restored = convrot_quant.rotate(rotated, 64)
        self.assertTrue(torch.allclose(value, restored, atol=1e-5, rtol=1e-5))

    def test_zero_weights_remain_finite(self):
        for qtype in CONVROT_QTYPES:
            with self.subTest(qtype=qtype):
                layer = torch.nn.Linear(64, 64, bias=False)
                with torch.no_grad():
                    layer.weight.zero_()
                quantize(layer, weights=get_qtype(qtype, kernel="torch", max_workspace_mb=1))
                output = layer(torch.randn(2, 64))
                self.assertTrue(torch.equal(output, torch.zeros_like(output)))


class ConvRotLinearTest(unittest.TestCase):
    def test_forward_and_input_gradient_contract(self):
        for qtype in CONVROT_QTYPES:
            with self.subTest(qtype=qtype):
                layer = make_layer(qtype)
                x = torch.randn(3, 64, requires_grad=True)
                output = layer(x)
                grad_output = torch.randn_like(output)
                output.backward(grad_output)
                logical_weight = layer.dequantize_weight().detach()
                expected_grad = F.linear(grad_output, logical_weight.transpose(0, 1))
                self.assertEqual(output.shape, (3, 64))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.allclose(x.grad, expected_grad, atol=1e-5, rtol=1e-5))

    def test_dtype_moves_preserve_byte_encoded_metadata(self):
        for qtype, byte_buffers in (
            ("convrot4", ("cr_scales", "cr_scales_blocked", "cr_pts")),
            ("convrot8", ("cr8_scales",)),
            ("convrotint4", ("crn_scales", "crn_gratio")),
            ("convrotbitnet", ("crn_scales", "crn_gratio")),
        ):
            with self.subTest(qtype=qtype):
                layer = make_layer(qtype).double()
                output = layer(torch.randn(2, 64, dtype=torch.float64))
                self.assertEqual(output.dtype, torch.float64)
                for name in byte_buffers:
                    self.assertEqual(getattr(layer, name).dtype, torch.uint8)

    def test_state_dict_contains_plain_portable_weight(self):
        for qtype in CONVROT_QTYPES:
            with self.subTest(qtype=qtype):
                layer = make_layer(qtype)
                state = layer.state_dict()
                self.assertIsInstance(state["weight"], OstrisLazyWeight)
                self.assertTrue(
                    torch.allclose(state["weight"].dequantize(), layer.weight)
                )
                restored = torch.nn.Linear(64, 64)
                restored.load_state_dict(state)
                self.assertEqual(set(state), {"weight", "bias"})
                self.assertTrue(torch.allclose(layer.weight, restored.weight))

    def test_arbitrary_bit_packing_round_trip(self):
        for bits in range(2, 9):
            with self.subTest(bits=bits):
                qmax = (1 << (bits - 1)) - 1
                values = torch.randint(-qmax, qmax + 1, (3, 64), dtype=torch.int8)
                packed = convrot_quant.pack_intn_rows(values, bits)
                restored = convrot_quant.unpack_intn_rows(packed, bits, 3, 64)
                self.assertTrue(torch.equal(values, restored))
                self.assertEqual(packed.numel(), values.numel() * bits // 8)

    def test_comfy_w4a4_export_and_qat_helper(self):
        layer = make_layer(
            "convrotcomfyw4a4", in_features=256, out_features=16
        )
        exported = convrot_quant.export_comfy_convrot_w4a4(layer, "layer.")
        self.assertEqual(exported["layer.weight"].shape, (16, 128))
        self.assertEqual(exported["layer.weight_scale"].shape, (16,))
        self.assertIn("layer.comfy_quant", exported)

        layer.register_parameter(
            "qat_master",
            torch.nn.Parameter(layer.dequantize_weight().detach().clone()),
        )
        x = torch.randn(2, 256, requires_grad=True)
        output = convrot_quant.convrot_qat_forward(layer, x)
        output.square().mean().backward()
        self.assertIsNotNone(layer.qat_master.grad)
        self.assertIsNotNone(x.grad)

    def test_packed_layer_save_load_round_trip(self):
        source = torch.nn.Sequential(make_layer("convrotint3"))
        activation = torch.randn(2, 64)
        expected = source(activation)
        target = torch.nn.Sequential(torch.nn.Linear(64, 64))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "layers.safetensors")
            save_quantized_layers({"0": source[0]}, path, metadata={"test": "1"})
            restored = load_quantized_layers(target, path)
        self.assertEqual(restored, 1)
        self.assertIsInstance(target[0], OstrisLinear)
        self.assertEqual(target[0].ostris_backend_name, "convrotint3")
        self.assertTrue(torch.equal(target[0].crn_qdata, source[0].crn_qdata))
        self.assertTrue(torch.allclose(target(activation), expected))

    def test_quantization_honors_bounded_row_workspace(self):
        for qtype in (*CONVROT_QTYPES, "convrotcomfyw4a4"):
            with self.subTest(qtype=qtype):
                layer = torch.nn.Linear(1024, 128)
                quantizer = get_ostris_quantizer(
                    qtype, kernel="torch", max_workspace_mb=1
                )
                seen_rows = []
                original_rotate = convrot_quant.rotate

                def tracking_rotate(value, rot_size):
                    if value.ndim == 2 and value.shape[1] == 1024:
                        seen_rows.append(value.shape[0])
                    return original_rotate(value, rot_size)

                with mock.patch.object(
                    convrot_quant, "rotate", side_effect=tracking_rotate
                ):
                    self.assertTrue(convert_linear_to_ostris(layer, quantizer))

                self.assertGreater(len(seen_rows), 1)
                self.assertLess(max(seen_rows), 128)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_hardware_paths_are_finite_and_close_to_fallbacks(self):
        for qtype, support_name, tolerance in (
            ("convrot4", "_fp4_gemm_supported", 0.35),
            ("convrot8", "_int8_gemm_supported", 0.10),
        ):
            with self.subTest(qtype=qtype):
                layer = torch.nn.Linear(64, 64, device="cuda", dtype=torch.bfloat16)
                quantize(layer, weights=get_qtype(qtype, kernel="torch", max_workspace_mb=1))
                x = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
                fast = layer(x)
                with mock.patch.object(convrot_quant, support_name, return_value=False):
                    fallback = layer(x)
                self.assertTrue(torch.isfinite(fast).all())
                self.assertTrue(torch.isfinite(fallback).all())
                mean_error = float((fast.float() - fallback.float()).abs().mean())
                self.assertLess(mean_error, tolerance)


if __name__ == "__main__":
    unittest.main()
