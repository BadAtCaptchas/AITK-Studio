import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIDREAM_O1_PIPELINE_PATH = (
    PROJECT_ROOT
    / "extensions_built_in"
    / "diffusion_models"
    / "hidream"
    / "src"
    / "hidream_o1"
    / "pipeline.py"
)


IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
VISION_START_TOKEN_ID = 151652


def load_hidream_o1_conditioning_namespace():
    # Avoid importing the full model stack just to exercise pure tensor layout helpers.
    source = HIDREAM_O1_PIPELINE_PATH.read_text()
    module = ast.parse(source, filename=str(HIDREAM_O1_PIPELINE_PATH))
    nodes = []
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {"TIMESTEP_TOKEN_NUM", "PATCH_SIZE"}
            for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef):
            nodes.append(node)

    test_module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(test_module)
    namespace = {"torch": torch, "List": List, "Optional": Optional}
    exec(compile(test_module, str(HIDREAM_O1_PIPELINE_PATH), "exec"), namespace)
    return namespace


def model_config():
    return SimpleNamespace(
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_TOKEN_ID,
    )


class HidreamO1ConditioningTest(unittest.TestCase):
    def setUp(self):
        self.namespace = load_hidream_o1_conditioning_namespace()
        self.build_sample = self.namespace["_build_t2i_sample_from_input_ids"]
        self.config = model_config()

    def test_valid_prompt_ids_build_expected_layout(self):
        input_ids = torch.tensor([101, 102, 103], dtype=torch.long)

        sample = self.build_sample(input_ids, 64, 64, self.config)

        self.assertEqual(sample["input_ids"].shape, (1, 3))
        self.assertEqual(sample["position_ids"].shape, (3, 1, 7))
        self.assertEqual(sample["token_types"].shape, (1, 7))
        self.assertEqual(sample["vinput_mask"].shape, (1, 7))
        self.assertEqual(sample["vinput_mask"].sum().item(), 4)
        self.assertEqual(sample["token_types"].sum().item(), 5)
        self.assertEqual(sample["input_ids"].device.type, "cpu")
        self.assertEqual(sample["position_ids"].device.type, "cpu")

    def test_rejects_unaligned_dimensions(self):
        input_ids = torch.tensor([101, 102, 103], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "patch-aligned"):
            self.build_sample(input_ids, 64, 65, self.config)

    def test_rejects_tiny_dimensions_without_image_token_after_vision_start(self):
        input_ids = torch.tensor([101, 102, 103], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "at least two image patch tokens"):
            self.build_sample(input_ids, 32, 32, self.config)

    def test_rejects_prompt_vision_tokens_before_rope_indexing(self):
        def fail_if_called(*args, **kwargs):
            raise AssertionError("rope indexing should not run for malformed input")

        self.namespace["_get_rope_index_t2i"] = fail_if_called
        input_ids = torch.tensor([101, VISION_START_TOKEN_ID, 103], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "vision layout tokens"):
            self.build_sample(input_ids, 64, 64, self.config)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_input_builds_rope_index_on_cpu_and_returns_cuda_tensors(self):
        real_rope_index = self.namespace["_get_rope_index_t2i"]
        seen_devices = []

        def assert_cpu_rope_index(*args, **kwargs):
            seen_devices.append(kwargs["input_ids"].device.type)
            return real_rope_index(*args, **kwargs)

        self.namespace["_get_rope_index_t2i"] = assert_cpu_rope_index
        input_ids = torch.tensor([101, 102, 103], dtype=torch.long, device="cuda")

        sample = self.build_sample(input_ids, 64, 64, self.config)

        self.assertEqual(seen_devices, ["cpu"])
        self.assertEqual(sample["input_ids"].device.type, "cuda")
        self.assertEqual(sample["position_ids"].device.type, "cuda")
        self.assertEqual(sample["token_types"].device.type, "cuda")
        self.assertEqual(sample["vinput_mask"].device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
