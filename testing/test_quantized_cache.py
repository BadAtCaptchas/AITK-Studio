import os
import shutil
import sys
import unittest

import torch
from optimum.quanto import freeze, qfloat8
from optimum.quanto.quantize import quantize
from safetensors.torch import save_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.quantized_cache import (
    QuantizedModelCache,
    get_raw_state_dict,
    is_quanto_qtype,
    quantized_cache_key,
)
from toolkit.dequantize import patch_dequantization_on_save


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)


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
