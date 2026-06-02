import unittest
from types import SimpleNamespace

import torch

from toolkit.memory_management.block_offload import BlockOffloadManager, LayerOffloadStrategy
from toolkit.memory_management.offload import resolve_layer_offloading_backend


class TinyBlockModel(torch.nn.Module):
    def __init__(self, block_count=4):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.SiLU())
            for _ in range(block_count)
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class LayerOffloadStrategyTest(unittest.TestCase):
    def test_selects_deterministic_whole_block_suffix(self):
        first = LayerOffloadStrategy((10, 10, 10, 10), 0.7)
        second = LayerOffloadStrategy((10, 10, 10, 10), 0.7)

        self.assertEqual(first.offloaded_indices, (1, 2, 3))
        self.assertEqual(first.offloaded_indices, second.offloaded_indices)
        self.assertEqual(first.resident_indices, (0,))

    def test_uneven_layer_sizes_use_closest_suffix(self):
        strategy = LayerOffloadStrategy((60, 1, 1, 1), 0.5)

        self.assertEqual(strategy.offloaded_indices, (1, 2, 3))
        self.assertEqual(strategy.resident_indices, (0,))

    def test_windows_follow_forward_and_backward_order(self):
        strategy = LayerOffloadStrategy((10, 10, 10, 10, 10), 0.6)

        self.assertEqual(strategy.forward_forward_window(2), (2, 3))
        self.assertEqual(strategy.forward_backward_window(3), (2, 3))
        self.assertEqual(strategy.backward_forward_window(3), (3, 4))
        self.assertEqual(strategy.forward_forward_window(0), ())


class BlockOffloadManagerTest(unittest.TestCase):
    def test_attach_is_idempotent_and_forward_still_runs_on_cpu(self):
        model = TinyBlockModel()
        x = torch.randn(2, 4)
        expected = model(x)

        manager = BlockOffloadManager.attach(
            model,
            torch.device("cpu"),
            offload_fraction=0.5,
            block_paths=["blocks"],
        )
        second = BlockOffloadManager.attach(
            model,
            torch.device("cpu"),
            offload_fraction=0.5,
            block_paths=["blocks"],
        )

        self.assertIs(manager, second)
        self.assertTrue(hasattr(model, "_block_offload_manager"))
        self.assertTrue(torch.allclose(model(x), expected))

    def test_detach_restores_original_forwards(self):
        model = TinyBlockModel()
        manager = BlockOffloadManager.attach(
            model,
            torch.device("cpu"),
            offload_fraction=1.0,
            block_paths=["blocks"],
        )

        manager.detach()

        self.assertFalse(hasattr(model, "_block_offload_manager"))
        self.assertFalse(hasattr(model, "_block_offload_original_to"))
        self.assertTrue(callable(model.blocks[0].forward))

    def test_unsupported_block_paths_fail_clearly(self):
        model = TinyBlockModel()

        with self.assertRaisesRegex(ValueError, "could not find ordered block layers"):
            BlockOffloadManager.attach(
                model,
                torch.device("cpu"),
                offload_fraction=1.0,
                block_paths=["missing.blocks"],
            )

    def test_default_block_backend_falls_back_to_legacy_without_cuda(self):
        model = TinyBlockModel()
        config = SimpleNamespace(
            arch="flux",
            layer_offloading=True,
            layer_offloading_backend="block",
            _layer_offloading_backend_explicit=True,
        )

        if torch.cuda.is_available():
            self.skipTest("CUDA is available; this fallback is only CPU-safe.")

        self.assertEqual(
            resolve_layer_offloading_backend(config, model, torch.device("cpu"), ["blocks"]),
            "legacy",
        )


if __name__ == "__main__":
    unittest.main()
