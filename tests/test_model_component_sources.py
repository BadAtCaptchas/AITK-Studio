"""Component source-policy pooling checks without loading model weights."""

import unittest
from unittest.mock import patch

import torch

from toolkit.models.v2._mixin import OstrisModelMixin
from toolkit.models.v2.pool import ComponentPool


class TinySourceModel(torch.nn.Module, OstrisModelMixin):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.quantization_requests = []

    def quantize_(self, qtype, **kwargs):
        self.quantization_requests.append(qtype)
        self.aitk_is_quantized = True
        self.aitk_qtype = qtype
        return self


def fake_resolve_source(name_or_path, *, subfolder, qtype):
    precision = "int8_convrot" if qtype == "convrot8" else "bf16"
    return f"fake/component_{precision}.safetensors"


def fake_load_source(file_path, **kwargs):
    return TinySourceModel(file_path)


class ComponentSourcePoolTests(unittest.TestCase):
    def test_deferred_quantization_reuses_equal_sources_but_reloads_full_precision(self):
        pool = ComponentPool()
        with patch.object(ComponentPool, "current", pool), \
             patch.object(TinySourceModel, "resolve_comfy_weights", side_effect=fake_resolve_source) as resolve, \
             patch.object(TinySourceModel, "_resolve_single_file", side_effect=lambda path: path), \
             patch.object(TinySourceModel, "_load_single_file", side_effect=fake_load_source) as load, \
             patch("toolkit.util.quantize.quantize_module") as quantize:
            pool.begin_request()
            quantized = TinySourceModel.load("test/component", qtype="convrot8")
            self.assertTrue(quantized.aitk_is_quantized)
            self.assertEqual(quantized.source, "fake/component_int8_convrot.safetensors")

            pool.begin_request()
            self.assertIs(TinySourceModel.load("test/component", qtype="convrot8"), quantized)
            full_precision = TinySourceModel.load("test/component", qtype=None)
            self.assertIsNot(full_precision, quantized)
            self.assertEqual(full_precision.source, "fake/component_bf16.safetensors")
            self.assertFalse(full_precision.aitk_is_quantized)
            # A source-policy change must not dequantize the resident module
            # and silently reuse its already-rounded weights.
            self.assertTrue(quantized.aitk_is_quantized)
            self.assertIs(TinySourceModel.load("test/component", qtype=None), full_precision)

        self.assertEqual(resolve.call_count, 2)
        self.assertEqual(load.call_count, 2)
        quantize.assert_called_once()
        self.assertEqual(pool.hits, 2)
        self.assertEqual(pool.misses, 2)

    def test_eager_and_deferred_quantization_do_not_share_resident_modules(self):
        pool = ComponentPool()
        with patch.object(ComponentPool, "current", pool), \
             patch.object(TinySourceModel, "resolve_comfy_weights", side_effect=fake_resolve_source), \
             patch.object(TinySourceModel, "_resolve_single_file", side_effect=lambda path: path), \
             patch.object(TinySourceModel, "_load_single_file", side_effect=fake_load_source) as load:
            deferred = TinySourceModel.load_model(
                "test/component", qtype="convrot8", quantize_on_load=False,
            )
            eager = TinySourceModel.load_model(
                "test/component", qtype="convrot8", quantize_on_load=True,
            )
            self.assertIsNot(deferred, eager)
            self.assertFalse(deferred.aitk_is_quantized)
            self.assertTrue(eager.aitk_is_quantized)
            self.assertIs(TinySourceModel.load_model(
                "test/component", qtype="convrot8", quantize_on_load=True,
            ), eager)

        self.assertEqual(eager.quantization_requests, ["convrot8"])
        self.assertEqual(deferred.quantization_requests, [])
        self.assertEqual(load.call_count, 2)
        self.assertEqual(pool.hits, 1)
        self.assertEqual(pool.misses, 2)


if __name__ == "__main__":
    unittest.main()
