import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models import AI_TOOLKIT_MODELS, MinimaxH3Model
from extensions_built_in.diffusion_models.minimax_h3.minimax_h3 import (
    MiniMaxH3VaeBundle,
)
from extensions_built_in.diffusion_models.minimax_h3.src.audio_vae import (
    MiniMaxH3AudioVAE,
)
from extensions_built_in.diffusion_models.minimax_h3.src import packing
from extensions_built_in.diffusion_models.minimax_h3.src.transformer import (
    MiniMaxH3Transformer,
    MiniMaxH3TransformerParams,
)
from toolkit.quantized_cache import QuantizedModelCache, quantized_cache_key
from toolkit.util.convrot_quant import dequantize_nvfp4
from toolkit.util.comfy_quant_import import (
    Int8Embedding,
    import_comfy_quantized_layers,
    parse_comfy_quant_blob,
)
from toolkit.util.nvfp4_quant import swap_nvfp4_nibbles, unswizzle_nvfp4_scales
from toolkit.util.ostris_quant import (
    OstrisLinear,
    get_ostris_backend_registry,
    load_quantized_layers,
    save_quantized_layers,
)
from toolkit.util.quantize import quantize, report_prequantized_ostris_model


def _comfy_blob(payload: dict) -> torch.Tensor:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(encoded), dtype=torch.uint8)


class TinyNvfp4Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128, bias=False)


class TinyMappedInt8Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.target = torch.nn.Module()
        self.target.linear = torch.nn.Linear(16, 8, bias=True)


class TwoLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(16, 8, bias=False)
        self.second = torch.nn.Linear(64, 128, bias=False)


class TinyMixedPackedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 16)
        self.linear = torch.nn.Linear(64, 128, bias=False)


def _nvfp4_state() -> dict[str, torch.Tensor]:
    return {
        "linear.comfy_quant": _comfy_blob({"format": "nvfp4"}),
        # Comfy stores the first value in the high nibble. Import swaps 0x12 -> 0x21.
        "linear.weight": torch.full((128, 32), 0x12, dtype=torch.uint8),
        # 128 rows x (64 / 16) block scales is one complete cuBLAS scale tile.
        "linear.weight_scale": torch.ones(
            (128, 4), dtype=torch.float8_e4m3fn
        ),
        "linear.weight_scale_2": torch.tensor([0.25], dtype=torch.float32),
        "linear.pre_quant_scale": torch.linspace(0.5, 1.5, 64),
        "unrelated": torch.tensor([7.0]),
    }


def _mixed_packed_state() -> dict[str, torch.Tensor]:
    state = _nvfp4_state()
    state.update(
        {
            "embedding.comfy_quant": _comfy_blob({"format": "int8_tensorwise"}),
            "embedding.weight": (
                torch.arange(32 * 16, dtype=torch.int16).reshape(32, 16) % 127
            ).to(torch.int8),
            "embedding.weight_scale": torch.linspace(0.005, 0.02, 32),
        }
    )
    return state


class MiniMaxH3RegistryTests(unittest.TestCase):
    def test_optional_registry_and_frame_contract(self):
        self.assertEqual(MinimaxH3Model.arch, "minimax_h3")
        self.assertIn(MinimaxH3Model, AI_TOOLKIT_MODELS)
        self.assertEqual(packing.align_num_frames_down(39), 39)
        self.assertEqual(packing.align_num_frames_down(38), 22)
        self.assertEqual(packing.video_latent_num_frames(39), 12)
        self.assertEqual(packing.build_sigma_schedule(28).numel() - 1, 28)
        latents = torch.arange(2 * 24 * 2 * 4 * 6).reshape(2, 24, 2, 4, 6)
        rows = packing.patchify_video_latents(latents)
        restored = packing.unpatchify_video_tokens(rows, 2, 4, 6)
        self.assertTrue(torch.equal(restored, latents))
        self.assertEqual(float(packing.remap_sigma(torch.tensor(0.0))), 0.0)
        self.assertEqual(float(packing.remap_sigma(torch.tensor(1.0))), 1.0)

    def test_h3_rejects_mutating_packed_weight_modes_before_loading(self):
        model = object.__new__(MinimaxH3Model)
        model.model_config = SimpleNamespace(
            layer_offloading=False,
            quantize=True,
            qtype="convrot8",
            quantize_te=True,
            qtype_te="nvfp4",
            te_name_or_path=None,
        )
        model._validate_prequantized_configuration()

        model.model_config.layer_offloading = True
        with self.assertRaisesRegex(ValueError, "does not support layer_offloading"):
            model._validate_prequantized_configuration()

        model.model_config.layer_offloading = False
        model.model_config.qtype = "convrot4"
        with self.assertRaisesRegex(ValueError, "requires.*ConvRot8"):
            model._validate_prequantized_configuration()

    def test_partition_routing_and_pruned_topology_are_explicit(self):
        model = object.__new__(MinimaxH3Model)
        model.model_config = SimpleNamespace(model_kwargs={})
        self.assertEqual(model._partition(), "fl2va_pruned")
        expected = {
            "fl2va": True,
            "fl2va_pruned": True,
            "ref2va": False,
            "ref2va_pruned": False,
        }
        for partition, uses_spatial_keyframe in expected.items():
            with self.subTest(partition=partition):
                model.model_config.model_kwargs["partition"] = partition
                self.assertEqual(model._dit_component(), f"dit_{partition}")
                self.assertEqual(
                    model._uses_latent_keyframe_conditioning(),
                    uses_spatial_keyframe,
                )

        model.model_config.model_kwargs["partition"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "partition must be"):
            model._dit_component()

        dense = model._transformer_params_for_state({})
        self.assertIsNone(dense.adaln_t_table_size)
        self.assertTrue(dense.adaln_apply_silu)
        self.assertFalse(dense.adaln_bias)
        pruned = model._transformer_params_for_state(
            {"adaln_t_table": torch.zeros(7, 11)}
        )
        self.assertEqual(pruned.adaln_t_table_size, 7)
        self.assertEqual(pruned.time_embed_dim, 11)
        self.assertFalse(pruned.adaln_apply_silu)
        self.assertTrue(pruned.adaln_bias)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            model._transformer_params_for_state(
                {"adaln_t_table": torch.zeros(7)}
            )


class MiniMaxH3MixedPrecisionTests(unittest.TestCase):
    @staticmethod
    def _tiny_transformer() -> MiniMaxH3Transformer:
        return MiniMaxH3Transformer(
            MiniMaxH3TransformerParams(
                hidden_size=8,
                num_layers=1,
                token_refiner_num_layers=1,
                num_attention_heads=1,
                attention_head_dim=8,
                ffn_hidden_size=16,
                latents_dim=2,
                audio_latents_dim=2,
                patch_size=(1, 2, 2),
                text_dim=6,
                timestep_input_dim=4,
                time_embed_hidden_size=8,
                time_embed_dim=4,
                rope_inv_freq_len=1,
            )
        )

    def test_transformer_to_preserves_checkpoint_dtypes_and_memory_format(self):
        transformer = self._tiny_transformer()
        # Build the same mixed-dtype shape as a loaded checkpoint without
        # invoking the H3 override under test.
        torch.nn.Module.to(transformer, dtype=torch.bfloat16)
        for prefix in transformer.FP32_KEY_PREFIXES:
            transformer.get_submodule(prefix).to(dtype=torch.float32)
        transformer.register_buffer(
            "memory_format_probe",
            torch.randn(2, 3, 4, 5, dtype=torch.float32),
        )

        before = {
            name: value.dtype
            for name, value in list(transformer.named_parameters())
            + list(transformer.named_buffers())
        }
        self.assertIn(torch.bfloat16, before.values())
        self.assertIn(torch.float32, before.values())
        for name, dtype in before.items():
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in transformer.FP32_KEY_PREFIXES
            ):
                self.assertEqual(dtype, torch.float32, name)

        returned = transformer.to(
            device=torch.device("cpu"),
            dtype=torch.float16,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        self.assertIs(returned, transformer)
        after = {
            name: value.dtype
            for name, value in list(transformer.named_parameters())
            + list(transformer.named_buffers())
        }
        self.assertEqual(after, before)
        self.assertTrue(
            transformer.memory_format_probe.is_contiguous(
                memory_format=torch.channels_last
            )
        )

        # The tensor overload carries both a target device and dtype; only its
        # device is relevant for H3.
        before_tensor_overload = {
            name: value.dtype
            for name, value in list(transformer.named_parameters())
            + list(transformer.named_buffers())
        }
        transformer.to(torch.empty((), dtype=torch.float64))
        self.assertEqual(
            {
                name: value.dtype
                for name, value in list(transformer.named_parameters())
                + list(transformer.named_buffers())
            },
            before_tensor_overload,
        )

    def test_vae_bundle_keeps_audio_fp32_during_generic_dtype_cast(self):
        class TinyVideoVae(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.probe = torch.nn.Parameter(torch.ones(1, 1, 2, 2))

            @property
            def device(self):
                return self.probe.device

            @property
            def dtype(self):
                return self.probe.dtype

        class TinyPinnedAudioVae(MiniMaxH3AudioVAE):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.probe = torch.nn.Parameter(torch.ones(1, 1, 2))

        bundle = MiniMaxH3VaeBundle(TinyVideoVae(), TinyPinnedAudioVae())
        bundle.to(device=torch.device("cpu"), dtype=torch.bfloat16)
        self.assertEqual(bundle.video_vae.probe.dtype, torch.bfloat16)
        self.assertEqual(bundle.audio_vae.probe.dtype, torch.float32)


class ComfyQuantImportTests(unittest.TestCase):
    def test_nvfp4_layout_conversion_matches_cublas_tile_order(self):
        swizzled = torch.arange(128 * 4).reshape(128, 4)
        row_major = unswizzle_nvfp4_scales(swizzled, 128, 4)
        self.assertTrue(torch.equal(row_major[0], torch.tensor([0, 1, 2, 3])))
        self.assertTrue(torch.equal(row_major[1], torch.tensor([16, 17, 18, 19])))
        self.assertTrue(torch.equal(row_major[32], torch.tensor([4, 5, 6, 7])))
        self.assertTrue(
            torch.equal(row_major[127], torch.tensor([508, 509, 510, 511]))
        )
        packed = torch.tensor([0x12, 0xF0, 0xA5], dtype=torch.uint8)
        self.assertTrue(
            torch.equal(
                swap_nvfp4_nibbles(packed),
                torch.tensor([0x21, 0x0F, 0x5A], dtype=torch.uint8),
            )
        )

    def test_metadata_boundary_rejects_non_object_and_invalid_utf8(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            parse_comfy_quant_blob(_comfy_blob(["nvfp4"]))
        with self.assertRaisesRegex(ValueError, "UTF-8 JSON"):
            parse_comfy_quant_blob(torch.tensor([0xFF], dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "one-dimensional uint8"):
            parse_comfy_quant_blob(torch.ones(2, dtype=torch.int16))

    def test_nvfp4_import_preserves_packed_storage_and_telemetry(self):
        model = TinyNvfp4Model()
        remaining, converted = import_comfy_quantized_layers(
            model,
            _nvfp4_state(),
            orig_dtype=torch.bfloat16,
        )
        self.assertEqual(converted, 1)
        self.assertEqual(set(remaining), {"unrelated"})
        self.assertIsInstance(model.linear, OstrisLinear)
        self.assertEqual(model.linear.ostris_backend_name, "nvfp4")
        self.assertTrue(
            torch.equal(
                model.linear.nv4_qdata,
                torch.full((128, 32), 0x21, dtype=torch.uint8),
            )
        )
        self.assertEqual(model.linear.nv4_pre_scale.numel(), 64 * 4)

        report = report_prequantized_ostris_model(
            model,
            component_label="text_encoder",
        )
        self.assertEqual(report.qtype, "nvfp4")
        self.assertEqual(report.quantized_modules, 1)
        self.assertEqual(report.quantized_weight_count, 128 * 64)
        self.assertEqual(report.compressed_bytes, 128 * 32)
        self.assertGreater(report.metadata_bytes, 0)
        self.assertIs(model._aitk_quantization_report, report)

        value = torch.randn(2, 64, requires_grad=True)
        reference_value = value.detach().clone().requires_grad_(True)
        output = model.linear(value)
        raw_weight = dequantize_nvfp4(
            model.linear.nv4_qdata,
            model.linear.nv4_scales.view(torch.float8_e4m3fn),
            model.linear.nv4_pts.view(torch.float32).reshape(()),
            128,
            64,
            torch.float32,
        )
        pre_scale = model.linear.nv4_pre_scale.view(torch.float32)
        reference = torch.nn.functional.linear(
            reference_value * pre_scale,
            raw_weight,
        )
        output.float().mean().backward()
        reference.float().mean().backward()
        self.assertTrue(torch.allclose(output, reference, atol=1e-4, rtol=1e-4))
        self.assertTrue(
            torch.allclose(value.grad, reference_value.grad, atol=1e-5, rtol=1e-5)
        )

        with self.assertRaisesRegex(RuntimeError, "cannot be merged or requantized"):
            model.linear.requantize_(torch.zeros(128, 64))

    def test_int8_embedding_is_included_in_packed_telemetry(self):
        model = TinyMixedPackedModel()
        remaining, converted = import_comfy_quantized_layers(
            model,
            _mixed_packed_state(),
            orig_dtype=torch.bfloat16,
        )
        self.assertEqual(converted, 2)
        self.assertEqual(set(remaining), {"unrelated"})
        self.assertIsInstance(model.embedding, Int8Embedding)

        report = report_prequantized_ostris_model(model)
        self.assertEqual(report.qtype, "int8_embedding+nvfp4")
        self.assertEqual(report.quantized_modules, 2)
        self.assertEqual(report.quantized_weight_count, 128 * 64 + 32 * 16)
        self.assertEqual(
            report.compressed_bytes,
            model.linear.nv4_qdata.numel() + model.embedding.qweight.numel(),
        )
        self.assertGreaterEqual(report.metadata_bytes, model.embedding.scales.numel())

    def test_convrot8_import_honors_key_mapping_without_requantizing(self):
        model = TinyMappedInt8Model()
        qweight = torch.arange(8 * 16, dtype=torch.int8).reshape(8, 16)
        state = {
            "source.linear.comfy_quant": _comfy_blob(
                {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": 16,
                }
            ),
            "source.linear.weight": qweight,
            "source.linear.weight_scale": torch.linspace(0.01, 0.08, 8),
            "source.linear.bias": torch.linspace(-0.2, 0.2, 8),
        }
        remaining, converted = import_comfy_quantized_layers(
            model,
            state,
            key_map=lambda prefix: prefix.replace("source", "target", 1),
        )
        self.assertEqual(converted, 1)
        self.assertFalse(remaining)
        self.assertIsInstance(model.target.linear, OstrisLinear)
        self.assertEqual(model.target.linear.ostris_backend_name, "convrot8")
        self.assertEqual(model.target.linear.cr8_rot_size, 16)
        self.assertTrue(torch.equal(model.target.linear.cr8_qdata, qweight))
        self.assertTrue(
            torch.equal(
                model.target.linear.bias,
                torch.linspace(-0.2, 0.2, 8),
            )
        )

    def test_missing_nvfp4_scale_fails_with_context(self):
        state = _nvfp4_state()
        del state["linear.weight_scale_2"]
        with self.assertRaisesRegex(ValueError, "weight_scale_2"):
            import_comfy_quantized_layers(TinyNvfp4Model(), state)

    def test_preflight_does_not_partially_convert_on_a_later_bad_marker(self):
        model = TwoLinearModel()
        state = {
            "first.comfy_quant": _comfy_blob({"format": "int8_tensorwise"}),
            "first.weight": torch.zeros(8, 16, dtype=torch.int8),
            "first.weight_scale": torch.ones(8),
            "second.comfy_quant": _comfy_blob({"format": "nvfp4"}),
            "second.weight": torch.zeros(128, 32, dtype=torch.uint8),
            "second.weight_scale": torch.ones(128, 4, dtype=torch.float8_e4m3fn),
        }
        with self.assertRaisesRegex(ValueError, "second.weight_scale_2"):
            import_comfy_quantized_layers(model, state)
        self.assertIs(type(model.first), torch.nn.Linear)
        self.assertIs(type(model.second), torch.nn.Linear)


class Nvfp4PersistenceTests(unittest.TestCase):
    def _import_model(self) -> TinyNvfp4Model:
        model = TinyNvfp4Model()
        import_comfy_quantized_layers(model, _nvfp4_state())
        return model

    def test_registry_and_layers_only_round_trip(self):
        metadata = get_ostris_backend_registry()["nvfp4"]
        self.assertTrue(metadata.experimental)
        self.assertEqual(metadata.bits, 4)
        self.assertIn("packed_cache", metadata.capabilities)
        self.assertIn("awq_pre_scale", metadata.capabilities)

        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "layers.safetensors")
            source = self._import_model()
            save_quantized_layers({"linear": source.linear}, path)

            restored = TinyNvfp4Model()
            self.assertEqual(load_quantized_layers(restored, path), 1)
            self.assertIsInstance(restored.linear, OstrisLinear)
            self.assertEqual(restored.linear.ostris_backend_name, "nvfp4")
            for name in ("nv4_qdata", "nv4_scales", "nv4_pts", "nv4_pre_scale"):
                self.assertTrue(
                    torch.equal(
                        getattr(restored.linear, name),
                        getattr(source.linear, name),
                    )
                )

    def test_local_nvfp4_forward_backward_and_storage_accounting(self):
        layer = torch.nn.Linear(16, 16)
        report = quantize(layer, weights="nvfp4", component_label="nvfp4-local")
        self.assertIsInstance(layer, OstrisLinear)
        self.assertEqual(report.quantized_modules, 1)
        self.assertEqual(report.compressed_bytes, layer.nv4_qdata.numel())
        self.assertGreater(report.metadata_bytes, 0)
        self.assertEqual(layer.nv4_pre_scale.numel(), 0)

        value = torch.randn(3, 16, requires_grad=True)
        reference_value = value.detach().clone().requires_grad_(True)
        output = layer(value)
        reference = torch.nn.functional.linear(
            reference_value,
            layer.dequantize_weight().detach(),
            layer.bias.detach(),
        )
        output.square().mean().backward()
        reference.square().mean().backward()
        self.assertTrue(torch.equal(output, reference))
        self.assertTrue(
            torch.allclose(value.grad, reference_value.grad, atol=1e-6, rtol=1e-6)
        )

    def test_model_cache_round_trip_keeps_awq_buffer(self):
        with tempfile.TemporaryDirectory() as root:
            source = self._import_model()
            key, payload = quantized_cache_key(
                "minimax-h3-te",
                {"qtype": "nvfp4", "prequantized": True},
            )
            cache = QuantizedModelCache(root)
            cache.save(source, "text_encoder", key, payload)

            with torch.device("meta"):
                restored = TinyNvfp4Model()
            cache.load(restored, "text_encoder", key, device=torch.device("cpu"))
            self.assertIsInstance(restored.linear, OstrisLinear)
            self.assertEqual(restored.linear.ostris_backend_name, "nvfp4")
            self.assertTrue(
                torch.equal(restored.linear.nv4_pre_scale, source.linear.nv4_pre_scale)
            )

    def test_model_cache_round_trip_keeps_int8_embedding_buffers(self):
        with tempfile.TemporaryDirectory() as root:
            source = TinyMixedPackedModel()
            import_comfy_quantized_layers(source, _mixed_packed_state())
            key, payload = quantized_cache_key(
                "minimax-h3-te-mixed",
                {"qtype": "nvfp4", "prequantized": True},
            )
            cache = QuantizedModelCache(root)
            cache.save(source, "text_encoder", key, payload)

            with torch.device("meta"):
                restored = TinyMixedPackedModel()
            cache.load(restored, "text_encoder", key, device=torch.device("cpu"))
            self.assertIsInstance(restored.embedding, Int8Embedding)
            self.assertTrue(
                torch.equal(restored.embedding.qweight, source.embedding.qweight)
            )
            self.assertTrue(
                torch.equal(restored.embedding.scales, source.embedding.scales)
            )
            ids = torch.tensor([[0, 7, 31], [4, 11, 19]])
            self.assertTrue(
                torch.equal(restored.embedding(ids), source.embedding(ids))
            )


if __name__ == "__main__":
    unittest.main()
