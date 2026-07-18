import concurrent.futures
import unittest
from unittest import mock
import weakref

import torch
import torch.utils.checkpoint

from toolkit.dequantize import patch_dequantization_on_save
from toolkit.lorm import count_parameters
from toolkit.memory_management.block_offload import BlockOffloadManager
from toolkit.memory_management.manager import MemoryManager
from toolkit.util.orbit_quant import (
    OrbitQuantizer,
    gaussian_lloyd_max,
    pack_codes,
    rpbh_forward,
    rpbh_inverse,
    rpbh_params,
    unpack_codes,
)
from toolkit.util.orbit_vq_quant import (
    _cache_lock as vq_cache_lock,
    _device_tables,
    _master_tables,
    get_vq_tables,
    pack_indices,
    unpack_indices,
)
from toolkit.util.ostris_quant import (
    OstrisLazyWeight,
    OstrisLinear,
    convert_linear_to_ostris,
)
from toolkit.util.quantize import (
    get_qtype,
    get_torchao_config,
    is_quantized_tensor,
    quantize,
    requantize_module_weight,
)


class TinyBlockModel(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Sequential(layer, torch.nn.SiLU())])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def make_quantized_linear(qtype="orbit4", in_features=64, out_features=16, bias=True):
    torch.manual_seed(11)
    layer = torch.nn.Linear(in_features, out_features, bias=bias)
    quantize(layer, weights=get_qtype(qtype))
    return layer


class OrbitPrimitiveTest(unittest.TestCase):
    def test_scalar_code_packing_round_trips(self):
        generator = torch.Generator().manual_seed(3)
        for bits in (2, 3, 4):
            codes = torch.randint(
                0,
                2 ** bits,
                (257,),
                generator=generator,
                dtype=torch.uint8,
            )
            unpacked = unpack_codes(pack_codes(codes, bits), bits, codes.numel())
            self.assertTrue(torch.equal(unpacked, codes))

    def test_vector_index_packing_round_trips(self):
        generator = torch.Generator().manual_seed(5)
        for bits in (12, 16):
            indices = torch.randint(
                0,
                2 ** bits,
                (131,),
                generator=generator,
                dtype=torch.int32,
            )
            unpacked = unpack_indices(
                pack_indices(indices, bits),
                bits,
                indices.numel(),
            )
            self.assertTrue(torch.equal(unpacked, indices))

    def test_rpbh_round_trip_is_deterministic(self):
        x = torch.randn(2, 3, 64)
        permutation, signs = rpbh_params(64)
        repeated_permutation, repeated_signs = rpbh_params(64)
        self.assertIs(permutation, repeated_permutation)
        self.assertIs(signs, repeated_signs)

        rotated = rpbh_forward(x, permutation.to(torch.int32), signs, 64)
        restored = rpbh_inverse(
            rotated,
            torch.argsort(permutation).to(torch.int32),
            signs,
            64,
        )
        self.assertTrue(torch.allclose(restored, x, atol=5e-7, rtol=5e-7))

    def test_shared_caches_initialize_once_under_concurrency(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            centroids = list(executor.map(lambda _: gaussian_lloyd_max(4), range(16)))
        self.assertTrue(all(item is centroids[0] for item in centroids))

        with vq_cache_lock:
            _master_tables.pop(("D4", 2 ** 12), None)
            _device_tables.pop(("D4", 2 ** 12, "cpu"), None)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            tables = list(
                executor.map(
                    lambda _: get_vq_tables("D4", 2 ** 12, torch.device("cpu")),
                    range(16),
                )
            )
        self.assertTrue(all(item is tables[0] for item in tables))


class OstrisLinearTest(unittest.TestCase):
    def test_all_custom_qtypes_resolve_without_ui_registration(self):
        for qtype in (
            "orbit2",
            "orbit3",
            "orbit4",
            "orbitvq2",
            "orbitvq3",
            "orbitvq4",
        ):
            with self.subTest(qtype=qtype):
                self.assertEqual(get_qtype(qtype).__class__.__name__, "ostristype")

    def test_conversion_preserves_identity_and_is_idempotent(self):
        layer = torch.nn.Linear(64, 16)
        identity = id(layer)
        model = torch.nn.Sequential(layer)
        weights = get_qtype("orbit4")

        quantize(model, weights=weights)
        packed = layer.orbit_packed
        quantize(model, weights=weights)

        self.assertEqual(id(layer), identity)
        self.assertIsInstance(layer, OstrisLinear)
        self.assertIs(layer.orbit_packed, packed)

    def test_conversion_disables_weakly_referenced_accuracy_adapter_merging(self):
        class Adapter:
            can_merge_in = True

        layer = torch.nn.Linear(64, 16)
        adapter = Adapter()
        layer.ara_lora_ref = weakref.ref(adapter)

        self.assertTrue(convert_linear_to_ostris(layer, OrbitQuantizer(4)))

        self.assertFalse(adapter.can_merge_in)

    def test_include_exclude_and_unsupported_shapes_remain_dense(self):
        model = torch.nn.ModuleDict(
            {
                "included": torch.nn.Linear(64, 8),
                "excluded": torch.nn.Linear(64, 8),
                "unsupported": torch.nn.Linear(48, 8),
            }
        )
        quantize(
            model,
            weights=get_qtype("orbit4"),
            include=["included", "unsupported"],
            exclude=["excluded"],
        )

        self.assertIsInstance(model["included"], OstrisLinear)
        self.assertIsInstance(model["excluded"], torch.nn.Linear)
        self.assertNotIsInstance(model["excluded"], OstrisLinear)
        self.assertIsInstance(model["unsupported"], torch.nn.Linear)
        self.assertNotIsInstance(model["unsupported"], OstrisLinear)

    def test_forward_and_backward_match_materialized_weight(self):
        for qtype in ("orbit4", "orbitvq3"):
            with self.subTest(qtype=qtype):
                layer = make_quantized_linear(qtype)
                x_quantized = torch.randn(2, 3, 64, requires_grad=True)
                x_reference = x_quantized.detach().clone().requires_grad_(True)
                weight = layer.dequantize_weight().detach()
                bias = layer.bias.detach().clone()

                quantized_output = layer(x_quantized)
                reference_output = torch.nn.functional.linear(
                    x_reference,
                    weight,
                    bias,
                )
                quantized_output.square().mean().backward()
                reference_output.square().mean().backward()

                self.assertTrue(
                    torch.allclose(
                        quantized_output,
                        reference_output,
                        atol=1e-6,
                        rtol=1e-6,
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        x_quantized.grad,
                        x_reference.grad,
                        atol=1e-6,
                        rtol=1e-6,
                    )
                )

    def test_autograd_does_not_retain_full_dequantized_weight(self):
        layer = make_quantized_linear("orbit4")
        x = torch.randn(2, 64, requires_grad=True)
        saved_numels = []

        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: (saved_numels.append(tensor.numel()), tensor)[1],
            lambda tensor: tensor,
        ):
            output = layer(x)
        output.sum().backward()

        self.assertNotIn(layer.logical_weight_numel, saved_numels)

    def test_state_dict_is_plain_and_loadable_by_linear(self):
        layer = make_quantized_linear("orbit4")
        restored = torch.nn.Linear(64, 16)
        with mock.patch.object(
            layer.ostris_quantizer,
            "dequantize_to",
            wraps=layer.ostris_quantizer.dequantize_to,
        ) as dequantize_to:
            state_dict = layer.state_dict()
            self.assertIsNone(dequantize_to.call_args)
            restored.load_state_dict(state_dict)
            self.assertEqual(
                torch.device(dequantize_to.call_args.args[1]).type,
                "cpu",
            )
        x = torch.randn(2, 64)

        self.assertEqual(set(state_dict), {"weight", "bias"})
        self.assertIsInstance(state_dict["weight"], OstrisLazyWeight)
        self.assertTrue(torch.allclose(layer(x), restored(x), atol=1e-6, rtol=1e-6))

    def test_model_save_patch_keeps_only_plain_linear_weights(self):
        layer = make_quantized_linear("orbit4")
        model = torch.nn.Sequential(layer)
        patch_dequantization_on_save(model)

        state_dict = model.state_dict()

        self.assertEqual(set(state_dict), {"0.weight", "0.bias"})
        self.assertFalse(any("orbit_" in key for key in state_dict))

    def test_dtype_moves_and_logical_parameter_count(self):
        layer = make_quantized_linear("orbit4").double()
        output = layer(torch.randn(2, 64, dtype=torch.float64))

        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(layer.orbit_codebook.dtype, torch.uint8)
        self.assertEqual(layer.ostris_quantizer._codebook(layer).dtype, torch.float32)
        self.assertEqual(count_parameters(layer), 64 * 16 + 16)

    def test_requantize_preserves_custom_module(self):
        layer = make_quantized_linear("orbit4")
        original = layer.dequantize_weight().clone()
        target = original + 0.125
        tagged_weight = layer.weight

        self.assertTrue(is_quantized_tensor(tagged_weight))
        requantize_module_weight(
            layer,
            target,
            tagged_weight.dtype,
            get_torchao_config("orbit4"),
        )

        self.assertIsInstance(layer, OstrisLinear)
        self.assertGreater(
            float((layer.dequantize_weight() - original).abs().mean()),
            0.01,
        )

    def test_zero_weights_are_finite_for_scalar_and_vector_backends(self):
        for qtype in ("orbit4", "orbitvq3"):
            with self.subTest(qtype=qtype):
                layer = torch.nn.Linear(64, 8, bias=False)
                with torch.no_grad():
                    layer.weight.zero_()
                quantize(layer, weights=get_qtype(qtype))
                output = layer(torch.randn(2, 64))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_direct_conversion_rejects_small_hadamard_blocks(self):
        layer = torch.nn.Linear(48, 8)
        converted = convert_linear_to_ostris(layer, OrbitQuantizer(4))
        self.assertFalse(converted)
        self.assertNotIsInstance(layer, OstrisLinear)


class OrbitOffloadTest(unittest.TestCase):
    def test_legacy_offload_keeps_compressed_layer_as_resident_module(self):
        layer = make_quantized_linear("orbit4")
        model = torch.nn.Sequential(layer)
        MemoryManager.attach(model, torch.device("cpu"))
        try:
            self.assertIn(layer, model._memory_manager.unmanaged_modules)
            self.assertEqual(model._memory_manager.unmanaged_modules.count(layer), 1)
            model.to("cpu")
            output = model(torch.randn(2, 64))
            self.assertEqual(output.shape, (2, 16))
            self.assertEqual(layer.orbit_packed.device.type, "cpu")
        finally:
            MemoryManager.detach(model)

    def test_block_offload_tracks_quantized_buffers(self):
        layer = make_quantized_linear("orbit4")
        model = TinyBlockModel(layer)
        manager = BlockOffloadManager.attach(
            model,
            torch.device("cpu"),
            offload_fraction=1.0,
            block_paths=["blocks"],
        )
        try:
            tracked_buffer_ids = {
                id(buffer)
                for entry in manager.layers
                for buffer in entry.buffers
            }
            self.assertIn(id(layer.orbit_packed), tracked_buffer_ids)
            self.assertIn(id(layer.orbit_row_norms), tracked_buffer_ids)
            self.assertEqual(model(torch.randn(2, 64)).shape, (2, 16))
        finally:
            manager.detach()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_round_trip_moves_registered_buffers(self):
        layer = make_quantized_linear("orbit4").to("cuda")
        output = layer(torch.randn(2, 64, device="cuda", requires_grad=True))
        output.sum().backward()
        self.assertEqual(output.device.type, "cuda")
        self.assertEqual(layer.orbit_packed.device.type, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_checkpoint_recomputation_with_compressed_block_offload(self):
        layer = make_quantized_linear("orbit4").to("cuda")
        model = TinyBlockModel(layer)
        manager = BlockOffloadManager.attach(
            model,
            torch.device("cuda"),
            offload_fraction=1.0,
            block_paths=["blocks"],
        )
        try:
            x = torch.randn(2, 64, device="cuda", requires_grad=True)
            output = torch.utils.checkpoint.checkpoint(
                model.blocks[0],
                x,
                use_reentrant=False,
            )
            output.square().mean().backward()
            torch.cuda.synchronize()
            manager._wait_for_entry_transfer(manager.layers[0])

            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertEqual(layer.orbit_packed.device.type, "cpu")
        finally:
            manager.detach()


if __name__ == "__main__":
    unittest.main()
