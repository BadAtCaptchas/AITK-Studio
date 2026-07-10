import unittest
from unittest import mock
import warnings

import torch

from toolkit.memory_management.block_offload import BlockOffloadManager
from toolkit.util import orbit_quant
from toolkit.util.orbit_quant import OrbitQuantizer, pack_codes, unpack_codes
from toolkit.util.orbit_vq_quant import ORBIT_VQ_QTYPES, OrbitVQQuantizer
from toolkit.util.ostris_quant import convert_linear_to_ostris


def make_layer(
    *,
    in_features=4096,
    out_features=48,
    max_workspace_mb=1,
    dtype=torch.float32,
):
    torch.manual_seed(29)
    layer = torch.nn.Linear(in_features, out_features, dtype=dtype)
    quantizer = OrbitQuantizer(
        4,
        kernel="torch",
        max_workspace_mb=max_workspace_mb,
    )
    converted = convert_linear_to_ostris(layer, quantizer)
    if not converted:
        raise AssertionError("test linear was unexpectedly ineligible for Orbit")
    return layer, quantizer


class TwoOrbitBlockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(2):
            layer = torch.nn.Linear(64, 64)
            if not convert_linear_to_ostris(
                layer,
                OrbitQuantizer(4, kernel="torch"),
            ):
                raise AssertionError("test linear was unexpectedly ineligible for Orbit")
            layers.append(torch.nn.Sequential(layer, torch.nn.SiLU()))
        self.blocks = torch.nn.ModuleList(layers)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class OrbitWorkspaceTest(unittest.TestCase):
    def test_orbit4_uses_legacy_compatible_direct_nibbles(self):
        codes = torch.tensor([0, 1, 7, 15, 12], dtype=torch.uint8)
        packed = pack_codes(codes, 4)

        self.assertTrue(
            torch.equal(
                packed,
                torch.tensor([0x01, 0x7F, 0xC0], dtype=torch.uint8),
            )
        )
        self.assertTrue(torch.equal(unpack_codes(packed, 4, 5), codes))

    def test_quantization_casts_and_rotates_only_bounded_row_tiles(self):
        torch.manual_seed(31)
        source = torch.nn.Linear(4096, 48, dtype=torch.bfloat16)
        reference = torch.nn.Linear(4096, 48, dtype=torch.bfloat16)
        reference.load_state_dict(source.state_dict())
        bounded = OrbitQuantizer(4, kernel="torch", max_workspace_mb=1)
        unchunked = OrbitQuantizer(4, kernel="torch", max_workspace_mb=64)

        with mock.patch.object(
            orbit_quant,
            "_quantize_rows",
            wraps=orbit_quant._quantize_rows,
        ) as quantize_rows:
            self.assertTrue(convert_linear_to_ostris(source, bounded))
        self.assertTrue(convert_linear_to_ostris(reference, unchunked))

        tile_rows = [call.args[0].shape[0] for call in quantize_rows.call_args_list]
        self.assertGreater(len(tile_rows), 1)
        self.assertLess(max(tile_rows) * source.in_features, source.logical_weight_numel)
        self.assertEqual(source.orbit_row_norms.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(source.orbit_packed, reference.orbit_packed))
        self.assertTrue(
            torch.equal(source.orbit_row_norms, reference.orbit_row_norms)
        )

    def test_torch_forward_and_rotated_backward_never_decode_full_weight(self):
        layer, quantizer = make_layer()
        dense_weight = layer.dequantize_weight().detach()
        bias = layer.bias.detach().clone()
        decoded_shapes = []
        original_decode = quantizer._decode_rotated_rows

        def tracking_decode(module, start, end, dtype):
            decoded_shapes.append((end - start, module.in_features))
            return original_decode(module, start, end, dtype)

        x = torch.randn(2, 3, 4096, requires_grad=True)
        x_reference = x.detach().clone().requires_grad_(True)
        with mock.patch.object(
            quantizer,
            "_decode_rotated_rows",
            side_effect=tracking_decode,
        ), mock.patch.object(
            quantizer,
            "dequantize",
            side_effect=AssertionError("hot path materialized the full weight"),
        ):
            output = layer(x)
            output.square().mean().backward()

        reference = torch.nn.functional.linear(x_reference, dense_weight, bias)
        reference.square().mean().backward()
        self.assertEqual(output.shape, (2, 3, 48))
        self.assertGreater(len(decoded_shapes), 2)
        self.assertTrue(
            all(rows * columns < layer.logical_weight_numel for rows, columns in decoded_shapes)
        )
        self.assertTrue(torch.allclose(output, reference, atol=2e-6, rtol=2e-6))
        self.assertTrue(
            torch.allclose(x.grad, x_reference.grad, atol=2e-6, rtol=2e-6)
        )

    def test_packed_weight_and_row_scales_stay_below_point_52_bytes_per_weight(self):
        layer, _ = make_layer(out_features=256, max_workspace_mb=8)
        weight_storage = (
            layer.orbit_packed.numel() * layer.orbit_packed.element_size()
            + layer.orbit_row_norms.numel() * layer.orbit_row_norms.element_size()
        )
        self.assertLessEqual(weight_storage / layer.logical_weight_numel, 0.52)

    def test_explicit_triton_request_falls_back_cleanly_on_cpu(self):
        layer = torch.nn.Linear(64, 8)
        quantizer = OrbitQuantizer(4, kernel="triton")
        self.assertTrue(convert_linear_to_ostris(layer, quantizer))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = layer(torch.randn(2, 64))
        self.assertEqual(output.shape, (2, 8))
        self.assertTrue(any("bounded Torch kernel" in str(item.message) for item in caught))

    def test_invalid_runtime_options_are_rejected(self):
        with self.assertRaises(ValueError):
            OrbitQuantizer(4, kernel="cuda")
        with self.assertRaises(ValueError):
            OrbitQuantizer(4, max_workspace_mb=0)
        with self.assertRaises(TypeError):
            OrbitQuantizer(4, max_workspace_mb=1.5)

    def test_cached_rotation_templates_do_not_alias_module_buffers(self):
        model = TwoOrbitBlockModel()
        first = model.blocks[0][0]
        second = model.blocks[1][0]

        for name in (
            "orbit_codebook",
            "orbit_perm",
            "orbit_inv_perm",
            "orbit_signs",
        ):
            with self.subTest(buffer=name):
                first_buffer = getattr(first, name)
                second_buffer = getattr(second, name)
                self.assertIsNot(first_buffer, second_buffer)
                self.assertNotEqual(first_buffer.data_ptr(), second_buffer.data_ptr())

        vq_config = ORBIT_VQ_QTYPES["orbitvq3"]
        first_vq = torch.nn.Linear(64, 8)
        second_vq = torch.nn.Linear(64, 8)
        self.assertTrue(
            convert_linear_to_ostris(first_vq, OrbitVQQuantizer(**vq_config))
        )
        self.assertTrue(
            convert_linear_to_ostris(second_vq, OrbitVQQuantizer(**vq_config))
        )
        self.assertIsNot(first_vq.ovq_signs, second_vq.ovq_signs)
        self.assertNotEqual(first_vq.ovq_signs.data_ptr(), second_vq.ovq_signs.data_ptr())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_block_offload_preserves_independent_mixed_residency(self):
        model = TwoOrbitBlockModel()
        first = model.blocks[0][0]
        second = model.blocks[1][0]
        manager = BlockOffloadManager.attach(
            model,
            torch.device("cuda"),
            offload_fraction=0.5,
            block_paths=["blocks"],
        )
        try:
            self.assertEqual(manager.strategy.resident_indices, (0,))
            self.assertEqual(manager.strategy.offloaded_indices, (1,))
            for name in (
                "orbit_packed",
                "orbit_row_norms",
                "orbit_codebook",
                "orbit_perm",
                "orbit_inv_perm",
                "orbit_signs",
            ):
                with self.subTest(buffer=name):
                    self.assertEqual(getattr(first, name).device.type, "cuda")
                    self.assertEqual(getattr(second, name).device.type, "cpu")

            output = model(torch.randn(2, 64, device="cuda"))
            torch.cuda.synchronize()
            self.assertEqual(output.device.type, "cuda")
            self.assertTrue(torch.isfinite(output).all())
        finally:
            manager.deactivate_to_cpu()
            manager.detach()

    def test_cpu_autocast_preserves_forward_and_input_gradient_contract(self):
        layer = torch.nn.Linear(64, 16)
        self.assertTrue(
            convert_linear_to_ostris(
                layer,
                OrbitQuantizer(4, kernel="torch", max_workspace_mb=1),
            )
        )
        x = torch.randn(3, 64, dtype=torch.float32, requires_grad=True)

        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = layer(x)
            output.float().square().mean().backward()

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(x.grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(x.grad).all())

    @unittest.skipUnless(hasattr(torch, "compile"), "torch.compile is unavailable")
    def test_torch_compile_executes_forward_and_backward(self):
        layer = torch.nn.Linear(64, 16)
        self.assertTrue(
            convert_linear_to_ostris(
                layer,
                OrbitQuantizer(4, kernel="torch", max_workspace_mb=1),
            )
        )
        compiled = torch.compile(layer, backend="eager")
        x = torch.randn(2, 64, requires_grad=True)

        output = compiled(x)
        output.sum().backward()

        self.assertEqual(output.shape, (2, 16))
        self.assertEqual(x.grad.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
