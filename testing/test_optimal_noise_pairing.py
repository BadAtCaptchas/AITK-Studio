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


    def test_selects_one_whole_candidate_by_mse_not_channel_projection(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.train_config = types.SimpleNamespace(optimal_noise_pairing_samples=2)
        latents = torch.ones(1, 2, 1, 1)
        candidates = [torch.tensor([2., 0.]).reshape_as(latents),
                      torch.tensor([0., 3.]).reshape_as(latents)]
        # Candidate 2 has greater projection, but its larger norm makes MSE worse.
        self.assertGreater((latents * candidates[1]).sum(), (latents * candidates[0]).sum())
        with mock.patch.object(torch, "randn_like", side_effect=candidates) as draw:
            result = process.get_optimal_noise(latents)
        torch.testing.assert_close(result, candidates[0])
        self.assertEqual(draw.call_count, 2)
        self.assertFalse(torch.equal(result, torch.tensor([2., 3.]).reshape_as(latents)))

    def test_zero_latents_still_choose_smallest_norm_candidate(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.train_config = types.SimpleNamespace(optimal_noise_pairing_samples=2)
        latents = torch.zeros(1, 2, 2, 2)
        candidates = [torch.full_like(latents, 3), torch.full_like(latents, .25)]
        with mock.patch.object(torch, "randn_like", side_effect=candidates):
            result = process.get_optimal_noise(latents)
        torch.testing.assert_close(result, candidates[1])

    def test_video_batches_keep_first_tie_and_requested_dtype(self):
        process = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
        process.train_config = types.SimpleNamespace(optimal_noise_pairing_samples=2)
        latents = torch.zeros(2, 2, 2, 2, 2)
        values = iter((1., -1., 3., .5))

        def draw(chunk, device, dtype):
            self.assertEqual(device, latents.device)
            self.assertEqual(dtype, torch.float64)
            return torch.full_like(chunk, next(values), device=device, dtype=dtype)

        with mock.patch.object(torch, "randn_like", side_effect=draw):
            result = process.get_optimal_noise(latents, dtype=torch.float64)
        expected = torch.stack((torch.ones_like(latents[0]), torch.full_like(latents[0], .5))).double()
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
