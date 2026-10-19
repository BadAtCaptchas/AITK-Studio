"""Weight-free inference catalog coverage for registered architecture variants."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from toolkit.models.registry import ARCH_REGISTRY, describe_archs, list_archs


class ModelPresetRegistryTests(unittest.TestCase):
    def test_lists_registered_variants_only_when_the_base_is_available(self):
        model_module = SimpleNamespace(
            LEGACY_ARCHS={"flux"},
            get_all_models=lambda: [SimpleNamespace(arch="moss_image")],
        )
        with patch.dict(sys.modules, {"toolkit.util.get_model": model_module}):
            names = list_archs()
            self.assertIn("moss_image:kromoss", names)
            self.assertIn("moss_image:kromoss_turbo", names)
            self.assertIn("flux", names)
            self.assertEqual(names, sorted(set(names)))
            for name in ARCH_REGISTRY:
                if ":" in name:
                    self.assertEqual(name in names, name.split(":", 1)[0] in {"moss_image", "flux"})

    def test_describes_kromoss_variants_with_their_distinct_defaults(self):
        model_module = SimpleNamespace(
            LEGACY_ARCHS=set(),
            get_all_models=lambda: [SimpleNamespace(arch="moss_image")],
        )
        with patch.dict(sys.modules, {"toolkit.util.get_model": model_module}):
            entries = {entry["arch"]: entry for entry in describe_archs()}
        full = entries["moss_image:kromoss"]
        turbo = entries["moss_image:kromoss_turbo"]
        self.assertEqual(full["sample"]["num_inference_steps"], 50)
        self.assertEqual(full["sample"]["guidance_scale"], 3.0)
        self.assertNotIn("distillation_lora_path", full["model"]["model_kwargs"])
        self.assertEqual(turbo["sample"]["num_inference_steps"], 8)
        self.assertEqual(turbo["sample"]["guidance_scale"], 1.0)
        self.assertEqual(turbo["model"]["model_kwargs"]["distillation_lora_path"],
                         "lodestones/Kromoss/8-steps-dmd-lora.safetensors")
        for entry in (full, turbo):
            self.assertEqual(entry["model"]["vae_dtype"], "fp32")


if __name__ == "__main__":
    unittest.main()
