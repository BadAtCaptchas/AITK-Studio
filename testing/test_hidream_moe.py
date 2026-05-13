import ast
import unittest
from pathlib import Path
from typing import Optional

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIDREAM_MOE_PATH = (
    PROJECT_ROOT
    / "extensions_built_in"
    / "diffusion_models"
    / "hidream"
    / "src"
    / "models"
    / "moe.py"
)
HIDREAM_ATTENTION_PATH = HIDREAM_MOE_PATH.with_name("attention.py")
HIDREAM_EXAMPLE_PATH = PROJECT_ROOT / "config" / "examples" / "train_lora_hidream_48.yaml"
UI_OPTIONS_PATH = PROJECT_ROOT / "ui" / "src" / "app" / "jobs" / "new" / "options.ts"


def load_moe_namespace():
    namespace = {"torch": torch, "nn": torch.nn, "Optional": Optional}

    attention_source = HIDREAM_ATTENTION_PATH.read_text(encoding="utf-8")
    attention_module = ast.parse(attention_source, filename=str(HIDREAM_ATTENTION_PATH))
    feed_forward_node = next(
        node
        for node in attention_module.body
        if isinstance(node, ast.ClassDef) and node.name == "FeedForwardSwiGLU"
    )
    ast.fix_missing_locations(feed_forward_node)
    exec(compile(ast.Module(body=[feed_forward_node], type_ignores=[]), str(HIDREAM_ATTENTION_PATH), "exec"), namespace)

    moe_source = HIDREAM_MOE_PATH.read_text(encoding="utf-8")
    moe_module = ast.parse(moe_source, filename=str(HIDREAM_MOE_PATH))
    moe_nodes = [
        node
        for node in moe_module.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "attention")
    ]
    ast.fix_missing_locations(ast.Module(body=moe_nodes, type_ignores=[]))
    exec(compile(ast.Module(body=moe_nodes, type_ignores=[]), str(HIDREAM_MOE_PATH), "exec"), namespace)
    return namespace


class HidreamMoETest(unittest.TestCase):
    def setUp(self):
        namespace = load_moe_namespace()
        self.MOEFeedForwardSwiGLU = namespace["MOEFeedForwardSwiGLU"]

    def test_moe_routes_and_records_detached_stats(self):
        moe = self.MOEFeedForwardSwiGLU(
            dim=8,
            hidden_dim=32,
            num_routed_experts=4,
            num_activated_experts=2,
        )
        moe.train()
        x = torch.randn(2, 3, 8)

        output = moe(x)
        stats = moe.gate.last_routing_stats

        self.assertEqual(output.shape, x.shape)
        self.assertIsNotNone(stats)
        self.assertIn("aux_loss", stats)
        self.assertTrue(all(not value.requires_grad for value in stats.values()))

    def test_aux_loss_backprops_to_gate_without_output_gradient(self):
        moe = self.MOEFeedForwardSwiGLU(
            dim=8,
            hidden_dim=32,
            num_routed_experts=4,
            num_activated_experts=2,
        )
        moe.eval()
        moe.gate.alpha = 0.5
        with torch.no_grad():
            moe.gate.weight.zero_()
        x = torch.ones(2, 3, 8)

        output = moe(x)
        loss = (output * 0.0).sum()
        loss.backward()

        self.assertIsNotNone(moe.gate.last_routing_stats)
        self.assertIsNotNone(moe.gate.weight.grad)
        self.assertGreater(moe.gate.weight.grad.abs().sum().item(), 0.0)

    def test_expert_paths_receive_gradients(self):
        moe = self.MOEFeedForwardSwiGLU(
            dim=8,
            hidden_dim=32,
            num_routed_experts=4,
            num_activated_experts=2,
        )
        moe.train()
        moe.gate.alpha = 0.0
        x = torch.randn(2, 3, 8)

        loss = moe(x).pow(2).mean()
        loss.backward()

        expert_grad = 0.0
        for expert in moe.experts:
            for param in expert.parameters():
                if param.grad is not None:
                    expert_grad += param.grad.abs().sum().item()
        self.assertGreater(expert_grad, 0.0)

    def test_routing_stats_do_not_keep_graphs_across_forwards(self):
        moe = self.MOEFeedForwardSwiGLU(
            dim=8,
            hidden_dim=32,
            num_routed_experts=4,
            num_activated_experts=2,
        )
        moe.train()

        for _ in range(3):
            moe(torch.randn(2, 3, 8))
            stats = moe.gate.last_routing_stats
            self.assertTrue(all(not value.requires_grad for value in stats.values()))


class HidreamMoEDefaultsTest(unittest.TestCase):
    def test_example_trains_experts_by_default(self):
        config = yaml.safe_load(HIDREAM_EXAMPLE_PATH.read_text(encoding="utf-8"))
        process = config["config"]["process"][0]
        ignored = process["network"]["network_kwargs"]["ignore_if_contains"]

        self.assertNotIn("ff_i.experts", ignored)
        self.assertIn("ff_i.gate", ignored)
        self.assertEqual(process["train"]["moe_aux_loss_alpha"], 0.01)

    def test_ui_presets_train_experts_by_default(self):
        source = UI_OPTIONS_PATH.read_text(encoding="utf-8")
        for name in ("hidream", "hidream_e1"):
            start = source.index(f"name: '{name}'")
            end = source.index("disableSections", start)
            block = source[start:end]

            self.assertIn("'config.process[0].train.moe_aux_loss_alpha': [0.01, undefined]", block)
            self.assertIn("'config.process[0].network.network_kwargs.ignore_if_contains': [['ff_i.gate'], []]", block)
            self.assertNotIn("ff_i.experts", block)


if __name__ == "__main__":
    unittest.main()
