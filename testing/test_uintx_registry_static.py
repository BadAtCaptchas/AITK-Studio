import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UIntXRegistryStaticTests(unittest.TestCase):
    def test_sources_compile_and_torchao_pin_is_unchanged(self):
        for relative_path in (
            "toolkit/util/uintx_quant.py",
            "toolkit/util/ostris_quant.py",
            "toolkit/util/quantize.py",
            "toolkit/quantized_cache.py",
            "testing/test_uintx_quant.py",
        ):
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            ast.parse(source, filename=relative_path)
        self.assertIn(
            "torchao==0.10.0",
            (ROOT / "requirements_base.txt").read_text(encoding="utf-8"),
        )

    def test_registry_cache_and_telemetry_hooks_cover_uintx(self):
        registry = (ROOT / "toolkit" / "util" / "ostris_quant.py").read_text(encoding="utf-8")
        quantize = (ROOT / "toolkit" / "util" / "quantize.py").read_text(encoding="utf-8")
        cache = (ROOT / "toolkit" / "quantized_cache.py").read_text(encoding="utf-8")
        self.assertIn("for name, bits in UINTX_QTYPES.items()", registry)
        self.assertIn('"uintx_packed"', quantize)
        for token in (
            '"uintx_packed"',
            '"uintx_scale"',
            '"uintx_zero_point"',
            '"power2_shards_v1"',
        ):
            self.assertIn(token, cache)
        quantize_tree = ast.parse(quantize)
        imported_names = {
            alias.name
            for node in ast.walk(quantize_tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertNotIn("UIntXWeightOnlyConfig", imported_names)


if __name__ == "__main__":
    unittest.main()
