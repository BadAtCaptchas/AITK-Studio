import importlib
import sys
import types
import unittest
from unittest import mock

for module_name in ("toolkit.memory_management", "toolkit.memory_management.offload"):
    sys.modules.pop(module_name, None)

train_process_module = importlib.import_module("jobs.process.BaseSDTrainProcess")
BaseSDTrainProcess = train_process_module.BaseSDTrainProcess
torch = train_process_module.torch


class OptimalNoisePairingTest(unittest.TestCase):
    def test_scores_each_candidate_before_generating_next_candidate(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.train_config = types.SimpleNamespace(optimal_noise_pairing_samples=4)
        latents = torch.zeros((2, 1, 1, 1), dtype=torch.float32)
        state = {
            "allocations": 0,
            "pending_score": 0,
            "scores": 0,
        }

        def fake_randn_like(chunk, device=None, dtype=None):
            self.assertEqual(
                state["pending_score"],
                0,
                "generated a new candidate before scoring the previous one",
            )
            state["allocations"] += 1
            state["pending_score"] = 1
            return torch.full_like(
                chunk,
                fill_value=state["allocations"],
                device=device,
                dtype=dtype,
            )

        def fake_mse_loss(chunk, noise):
            self.assertEqual(state["pending_score"], 1)
            state["pending_score"] = 0
            state["scores"] += 1
            return float(state["scores"])

        with mock.patch.object(train_process_module.torch, "randn_like", side_effect=fake_randn_like), \
                mock.patch.object(train_process_module.torch.nn.functional, "mse_loss", side_effect=fake_mse_loss):
            noise = process.get_optimal_noise(latents, dtype=torch.float32)

        self.assertEqual(state["allocations"], 8)
        self.assertEqual(state["scores"], 8)
        self.assertEqual(state["pending_score"], 0)
        self.assertEqual(noise.shape, latents.shape)


if __name__ == "__main__":
    unittest.main()
