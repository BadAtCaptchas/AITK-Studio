import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, TypedDict, Unpack

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DFE_PATH = PROJECT_ROOT / "toolkit" / "models" / "diffusion_feature_extraction.py"
HIDREAM_QWEN_PATH = (
    PROJECT_ROOT
    / "extensions_built_in"
    / "diffusion_models"
    / "hidream"
    / "src"
    / "hidream_o1"
    / "qwen3_vl_transformers.py"
)


class TransformersKwargs(TypedDict, total=False):
    pass


def load_hidream_attention_functions():
    source = HIDREAM_QWEN_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(HIDREAM_QWEN_PATH))
    nodes = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"repeat_kv", "eager_attention_forward"}
    ]
    test_module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(test_module)

    namespace = {
        "torch": torch,
        "nn": nn,
        "Optional": Optional,
        "TransformersKwargs": TransformersKwargs,
        "Unpack": Unpack,
    }
    exec(compile(test_module, str(HIDREAM_QWEN_PATH), "exec"), namespace)
    return namespace["eager_attention_forward"]


def load_dfe7_forward_class():
    source = DFE_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(DFE_PATH))
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DiffusionFeatureExtractor7"
    )
    forward_node = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    test_class = ast.ClassDef(
        name="DiffusionFeatureExtractor7ForwardForTest",
        bases=[],
        keywords=[],
        body=[forward_node],
        decorator_list=[],
    )
    test_module = ast.Module(body=[test_class], type_ignores=[])
    ast.fix_missing_locations(test_module)

    namespace = {
        "torch": torch,
        "DataLoaderBatchDTO": object,
        "CustomFlowMatchEulerDiscreteScheduler": object,
    }
    exec(compile(test_module, str(DFE_PATH), "exec"), namespace)
    return namespace["DiffusionFeatureExtractor7ForwardForTest"]


def fake_tips_prediction(tensor):
    return SimpleNamespace(
        head=tensor[:, :1] * 0.5,
        depth=tensor[:, :1] * 2.0,
        normals=tensor[:, :3],
        segmentation=tensor.repeat(1, 50, 1, 1),
    )


class FakeSD:
    def __init__(self):
        self.vae = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)

    def decode_latents(self, latents):
        return latents


def build_dfe_for_test(do_partial_step=False):
    dfe_cls = load_dfe7_forward_class()
    dfe = dfe_cls()
    dfe.do_partial_step = do_partial_step
    dfe.losses = {}
    dfe.log_every = 100
    dfe.step = 0
    dfe.model = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32)
    dfe.sd_ref = lambda: FakeSD()
    dfe.get_pred = fake_tips_prediction
    return dfe


def dfe_inputs():
    return {
        "noise": torch.linspace(-0.25, 0.25, 96, dtype=torch.float32).view(2, 3, 4, 4),
        "noise_pred": torch.linspace(0.1, 1.0, 96, dtype=torch.float32).view(2, 3, 4, 4),
        "noisy_latents": torch.linspace(-1.0, 1.0, 96, dtype=torch.float32).view(2, 3, 4, 4),
        "batch": SimpleNamespace(
            tensor=torch.linspace(-0.5, 0.5, 96, dtype=torch.float32).view(2, 3, 4, 4),
            latents=torch.linspace(-0.75, 0.75, 96, dtype=torch.float32).view(2, 3, 4, 4),
        ),
    }


def dfe_loss_from_predictions(pred, target, tv, weighted):
    head_loss = torch.nn.functional.mse_loss(
        pred.head.float(), target.head.float(), reduction="none"
    )
    depth_loss = torch.nn.functional.l1_loss(
        pred.depth.float(), target.depth.float(), reduction="none"
    )
    normals_loss = torch.nn.functional.l1_loss(
        pred.normals.float(), target.normals.float(), reduction="none"
    )
    segmentation_loss = torch.nn.functional.l1_loss(
        pred.segmentation.float(), target.segmentation.float(), reduction="none"
    )

    if weighted:
        velocity_equiv_weight = 1.0 / torch.clamp(tv, min=0.1)
        head_loss = head_loss * velocity_equiv_weight**2
        depth_loss = depth_loss * velocity_equiv_weight
        normals_loss = normals_loss * velocity_equiv_weight
        segmentation_loss = segmentation_loss * velocity_equiv_weight

    return (
        head_loss.mean()
        + depth_loss.mean()
        + normals_loss.mean()
        + segmentation_loss.mean()
    ) / 4.0


class HidreamEagerAttentionTest(unittest.TestCase):
    def setUp(self):
        self.eager_attention_forward = load_hidream_attention_functions()
        self.module = SimpleNamespace(num_key_value_groups=1, training=False)

    def test_noncausal_attention_matches_manual_softmax(self):
        torch.manual_seed(1)
        query = torch.randn(1, 2, 4, 3)
        key = torch.randn(1, 2, 4, 3)
        value = torch.randn(1, 2, 4, 3)
        scaling = 3**-0.5

        _, weights = self.eager_attention_forward(
            self.module, query, key, value, None, scaling, is_causal=False
        )
        expected = torch.softmax(
            torch.matmul(query, key.transpose(2, 3)) * scaling,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)

        self.assertTrue(torch.allclose(weights, expected))

    def test_causal_attention_masks_future_keys(self):
        torch.manual_seed(2)
        query = torch.randn(1, 2, 4, 3)
        key = torch.randn(1, 2, 4, 3)
        value = torch.randn(1, 2, 4, 3)

        _, noncausal_weights = self.eager_attention_forward(
            self.module, query, key, value, None, 1.0, is_causal=False
        )
        _, causal_weights = self.eager_attention_forward(
            self.module, query, key, value, None, 1.0, is_causal=True
        )

        future_mask = torch.ones(4, 4, dtype=torch.bool).triu(diagonal=1)
        future_mask = future_mask.view(1, 1, 4, 4)

        self.assertFalse(torch.allclose(causal_weights, noncausal_weights))
        self.assertLess(causal_weights.masked_select(future_mask).abs().max().item(), 1e-6)
        self.assertTrue(torch.allclose(causal_weights.sum(dim=-1), torch.ones(1, 2, 4)))


class DFE7VelocityWeightingTest(unittest.TestCase):
    def test_full_step_dfe7_applies_loss_specific_velocity_weights(self):
        dfe = build_dfe_for_test(do_partial_step=False)
        inputs = dfe_inputs()
        timesteps = torch.tensor([50.0, 500.0], dtype=torch.float32)

        actual = dfe.forward(
            noise=inputs["noise"],
            noise_pred=inputs["noise_pred"],
            noisy_latents=inputs["noisy_latents"],
            timesteps=timesteps,
            batch=inputs["batch"],
            scheduler=None,
        )

        tv = timesteps.view(2, 1, 1, 1) / 1000.0
        target = fake_tips_prediction((inputs["batch"].tensor.to(torch.bfloat16) + 1) / 2)
        stepped_latents = inputs["noisy_latents"] - tv * inputs["noise_pred"]
        pred = fake_tips_prediction((stepped_latents + 1) / 2)
        expected = dfe_loss_from_predictions(pred, target, tv.clamp(min=0.001), weighted=True)

        self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))

    def test_partial_step_dfe8_path_keeps_unweighted_loss_then_existing_multiplier(self):
        dfe = build_dfe_for_test(do_partial_step=True)
        inputs = dfe_inputs()
        timesteps = torch.tensor([500.0, 700.0], dtype=torch.float32)
        original_rand_like = torch.rand_like
        torch.rand_like = lambda tensor: torch.full_like(tensor, 0.5)
        try:
            actual = dfe.forward(
                noise=inputs["noise"],
                noise_pred=inputs["noise_pred"],
                noisy_latents=inputs["noisy_latents"],
                timesteps=timesteps,
                batch=inputs["batch"],
                scheduler=None,
            )
        finally:
            torch.rand_like = original_rand_like

        tv = timesteps.view(2, 1, 1, 1) / 1000.0
        step = torch.full_like(tv, 0.5) * 0.03 + 0.02
        next_step = torch.clamp(tv - step, min=0.0)
        stepped_latents = inputs["noisy_latents"] + (next_step - tv) * inputs["noise_pred"]
        target_latents = (1.0 - next_step) * inputs["batch"].latents + next_step * inputs["noise"]

        target = fake_tips_prediction((target_latents + 1) / 2)
        pred = fake_tips_prediction((stepped_latents + 1) / 2)
        expected = dfe_loss_from_predictions(pred, target, tv, weighted=False) * 10.0

        self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
