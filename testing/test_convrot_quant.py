import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from toolkit.util import convrot_quant
from toolkit.util.ostris_quant import OstrisLinear
from toolkit.util.quantize import get_qtype, quantize


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
        for qtype in ("convrot4", "convrot8"):
            with self.subTest(qtype=qtype):
                layer = torch.nn.Linear(64, 64, bias=False)
                with torch.no_grad():
                    layer.weight.zero_()
                quantize(layer, weights=get_qtype(qtype, kernel="torch", max_workspace_mb=1))
                output = layer(torch.randn(2, 64))
                self.assertTrue(torch.equal(output, torch.zeros_like(output)))


class ConvRotLinearTest(unittest.TestCase):
    def test_forward_and_input_gradient_contract(self):
        for qtype in ("convrot4", "convrot8"):
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
        ):
            with self.subTest(qtype=qtype):
                layer = make_layer(qtype).double()
                output = layer(torch.randn(2, 64, dtype=torch.float64))
                self.assertEqual(output.dtype, torch.float64)
                for name in byte_buffers:
                    self.assertEqual(getattr(layer, name).dtype, torch.uint8)

    def test_state_dict_contains_plain_portable_weight(self):
        for qtype in ("convrot4", "convrot8"):
            with self.subTest(qtype=qtype):
                layer = make_layer(qtype)
                state = layer.state_dict()
                restored = torch.nn.Linear(64, 64)
                restored.load_state_dict(state)
                self.assertEqual(set(state), {"weight", "bias"})
                self.assertTrue(torch.allclose(layer.weight, restored.weight))

    def test_quantization_honors_bounded_row_workspace(self):
        layer = torch.nn.Linear(1024, 128)
        quantizer = convrot_quant.ConvRotQuantizer(
            kernel="torch",
            max_workspace_mb=1,
        )
        seen_rows = []
        original_rotate = convrot_quant.rotate

        def tracking_rotate(value, rot_size):
            if value.ndim == 2 and value.shape[1] == 1024:
                seen_rows.append(value.shape[0])
            return original_rotate(value, rot_size)

        with mock.patch.object(convrot_quant, "rotate", side_effect=tracking_rotate):
            from toolkit.util.ostris_quant import convert_linear_to_ostris

            self.assertTrue(convert_linear_to_ostris(layer, quantizer))

        self.assertGreater(len(seen_rows), 2)
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
