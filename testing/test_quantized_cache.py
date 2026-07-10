import os
import shutil
import sys
import unittest

import torch
from optimum.quanto import freeze, qfloat8
from optimum.quanto.quantize import quantize
from safetensors.torch import load_file, save_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.quantized_cache import (
    CACHE_MANIFEST_NAME,
    QuantizedModelCache,
    get_raw_state_dict,
    is_quantized_cache_qtype,
    is_quanto_qtype,
    quantized_cache_key,
)
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.util.ostris_quant import (
    OstrisLinear,
    get_ostris_backend_metadata,
    get_ostris_backend_registry,
)
from toolkit.util.quantize import (
    enforce_orbit4_low_vram_coverage,
    get_qtype,
    quantize,
    quantize_component_in_stages,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)


class OrbitTinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 16)
        self.unsupported = torch.nn.Linear(48, 8)


class StagedOrbitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(torch.nn.Linear(64, 32)),
                torch.nn.Sequential(torch.nn.Linear(64, 32)),
            ]
        )
        self.extra = torch.nn.Linear(64, 16)
        self.excluded = torch.nn.Linear(64, 8)


def _tmp_root():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp")
    os.makedirs(root, exist_ok=True)
    return root


def _make_test_dir(name):
    path = os.path.join(_tmp_root(), name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


class QuantizedCacheTest(unittest.TestCase):
    def test_quanto_qtype_detection(self):
        self.assertTrue(is_quanto_qtype("qfloat8"))
        self.assertFalse(is_quanto_qtype("float8"))
        self.assertFalse(is_quanto_qtype("uint4"))
        self.assertTrue(is_quantized_cache_qtype("orbit4"))
        self.assertFalse(is_quantized_cache_qtype("uint4"))

    def test_orbit_registry_marks_only_q4_stable_and_passes_options(self):
        registry = get_ostris_backend_registry()
        self.assertEqual(registry["orbit4"].status, "stable")
        self.assertEqual(registry["orbit4"].bits, 4)
        self.assertIn("packed_cache", registry["orbit4"].capabilities)
        self.assertTrue(get_ostris_backend_metadata("orbit3").experimental)
        resolved = get_qtype("orbit4", kernel="torch", max_workspace_mb=7)
        self.assertEqual(resolved.quantizer.kernel, "torch")
        self.assertEqual(resolved.quantizer.max_workspace_mb, 7)

    def test_quantized_cache_round_trip_from_meta_model(self):
        temp_dir = _make_test_dir("test_quantized_cache_round_trip")
        try:
            model = TinyModel()
            quantize(model, weights=qfloat8)
            freeze(model)

            cache_key, payload = quantized_cache_key(
                "tiny",
                {"dtype": "float32", "qtype": "qfloat8"},
                sources=[],
            )
            cache = QuantizedModelCache(temp_dir)
            cache.save(model, "tiny", cache_key, payload)

            with torch.device("meta"):
                restored = TinyModel()
            cache.load(restored, "tiny", cache_key, device=torch.device("cpu"))

            self.assertEqual(restored.linear.__class__.__name__, "QLinear")
            self.assertTrue(all(param.device.type == "cpu" for param in restored.parameters()))
            self.assertIn("linear.weight._data", restored.state_dict())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_quantized_cache_key_changes_with_qtype_and_source(self):
        temp_dir = _make_test_dir("test_quantized_cache_key")
        try:
            source_path = os.path.join(temp_dir, "source.safetensors")
            save_file({"x": torch.ones(1)}, source_path)

            key_1, _ = quantized_cache_key(
                "tiny",
                {"dtype": "float32", "qtype": "qfloat8"},
                sources=[source_path],
            )
            key_2, _ = quantized_cache_key(
                "tiny",
                {"dtype": "float32", "qtype": "qint8"},
                sources=[source_path],
            )

            self.assertNotEqual(key_1, key_2)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_orbit_cache_round_trip_from_meta_without_dense_weight(self):
        temp_dir = _make_test_dir("test_orbit_cache_round_trip")
        try:
            torch.manual_seed(23)
            model = OrbitTinyModel()
            report = quantize(model, weights="orbit4", component_label="tiny")
            expected = model.linear(torch.randn(2, 64))
            self.assertIsInstance(model.linear, OstrisLinear)
            self.assertEqual(report.component, "tiny")
            self.assertEqual(report.quantized_modules, 1)
            self.assertGreater(report.eligible_bytes, 0)
            self.assertGreater(report.quantized_original_bytes, 0)
            self.assertEqual(report.quantized_weight_count, 64 * 16)
            self.assertGreater(report.compressed_bytes, 0)
            self.assertEqual(report.compressed_bytes_per_weight, 0.5)
            self.assertIn("unsupported_shape", report.skip_reasons)
            self.assertLess(report.coverage, 0.95)
            with self.assertRaisesRegex(ValueError, "unsupported_shape"):
                enforce_orbit4_low_vram_coverage(report)

            cache_key, payload = quantized_cache_key(
                "orbit_tiny",
                {
                    "dtype": "float32",
                    "qtype": "orbit4",
                    "quantize_kwargs": {
                        "kernel": "torch",
                        "max_workspace_mb": 7,
                        "exclude": ["output"],
                    },
                },
                sources=[],
            )
            cache = QuantizedModelCache(temp_dir)
            cache_dir = cache.save(model, "orbit_tiny", cache_key, payload)
            packed_state = load_file(os.path.join(cache_dir, "model.safetensors"))
            self.assertIn("linear.orbit_packed", packed_state)
            self.assertNotIn("linear.weight", packed_state)
            self.assertTrue(os.path.exists(os.path.join(cache_dir, CACHE_MANIFEST_NAME)))

            with torch.device("meta"):
                restored = OrbitTinyModel()
            metadata = cache.load(
                restored,
                "orbit_tiny",
                cache_key,
                device=torch.device("cpu"),
            )
            self.assertEqual(metadata["cache_backend"], "ostris")
            self.assertIsInstance(restored.linear, OstrisLinear)
            self.assertEqual(restored.linear.orbit_packed.device.type, "cpu")
            self.assertTrue(torch.equal(restored.linear.orbit_packed, model.linear.orbit_packed))
            # Reuse the same activation used for the cached model.
            torch.manual_seed(99)
            activation = torch.randn(2, 64)
            self.assertTrue(
                torch.allclose(model.linear(activation), restored.linear(activation))
            )
            self.assertEqual(expected.shape, (2, 16))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_corrupt_orbit_manifest_is_invalidated_on_load(self):
        temp_dir = _make_test_dir("test_corrupt_orbit_cache")
        try:
            model = OrbitTinyModel()
            quantize(model, weights="orbit4")
            cache_key, payload = quantized_cache_key(
                "orbit_tiny",
                {"qtype": "orbit4"},
            )
            cache = QuantizedModelCache(temp_dir)
            cache_dir = cache.save(model, "orbit_tiny", cache_key, payload)
            manifest_path = os.path.join(cache_dir, CACHE_MANIFEST_NAME)
            with open(manifest_path, "w", encoding="utf-8") as manifest_file:
                manifest_file.write("{not valid json")
            with self.assertRaises(Exception):
                cache.load(OrbitTinyModel(), "orbit_tiny", cache_key)
            self.assertFalse(os.path.exists(cache_dir))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_same_shape_packed_tensor_bit_flip_is_rejected_before_conversion(self):
        temp_dir = _make_test_dir("test_orbit_cache_tensor_digest")
        try:
            model = OrbitTinyModel()
            quantize(model, weights="orbit4")
            cache_key, payload = quantized_cache_key(
                "orbit_tiny",
                {"qtype": "orbit4"},
            )
            cache = QuantizedModelCache(temp_dir)
            cache_dir = cache.save(model, "orbit_tiny", cache_key, payload)
            weights_path = os.path.join(cache_dir, "model.safetensors")
            state = load_file(weights_path)
            corrupted = state["linear.orbit_packed"].clone()
            corrupted[0] = corrupted[0] ^ 1
            state["linear.orbit_packed"] = corrupted
            replacement_path = f"{weights_path}.corrupt"
            save_file(state, replacement_path)
            os.replace(replacement_path, weights_path)

            restored = OrbitTinyModel()
            with self.assertRaisesRegex(ValueError, "checksum failed"):
                cache.load(restored, "orbit_tiny", cache_key)
            self.assertNotIsInstance(restored.linear, OstrisLinear)
            self.assertFalse(os.path.exists(cache_dir))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_staged_quantization_preserves_absolute_filters(self):
        model = StagedOrbitModel()
        report = quantize_component_in_stages(
            model,
            weights="orbit4",
            device=torch.device("cpu"),
            dtype=torch.float32,
            block_paths=["blocks"],
            exclude=["blocks.1.*", "excluded"],
            options={"kernel": "torch", "max_workspace_mb": 2},
            component_label="staged",
        )
        self.assertIsInstance(model.blocks[0][0], OstrisLinear)
        self.assertNotIsInstance(model.blocks[1][0], OstrisLinear)
        self.assertIsInstance(model.extra, OstrisLinear)
        self.assertNotIsInstance(model.excluded, OstrisLinear)
        self.assertEqual(report.quantized_modules, 2)
        self.assertEqual(report.skip_reasons["excluded"].modules, 2)
        self.assertEqual(model.blocks[0][0].orbit_kernel, "torch")
        self.assertEqual(model.blocks[0][0].orbit_max_workspace_mb, 2)

    def test_dequantized_save_patch_is_idempotent_and_preserves_raw_state_dict(self):
        model = TinyModel()
        quantize(model, weights=qfloat8)
        freeze(model)

        patch_dequantization_on_save(model)
        patch_dequantization_on_save(model)

        save_state_dict = model.state_dict()
        raw_state_dict = get_raw_state_dict(model)

        self.assertIn("linear.weight", save_state_dict)
        self.assertNotIn("linear.weight._data", save_state_dict)
        self.assertIn("linear.weight._data", raw_state_dict)


if __name__ == "__main__":
    unittest.main()
