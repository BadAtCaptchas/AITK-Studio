import os
import tempfile
import unittest
from unittest import mock

import torch

from toolkit.optimizer_checkpoint import (
    OptimizerCheckpointValidationError,
    atomic_save_optimizer_checkpoint,
    load_optimizer_checkpoint,
    parse_optimizer_checkpoint,
)


def optimizer_state(value=1.0):
    return {
        "state": {0: {"momentum": torch.tensor([value])}},
        "param_groups": [{"params": [0], "lr": 0.001}],
    }


class OptimizerCheckpointTest(unittest.TestCase):
    def test_atomic_checkpoint_round_trip_records_model_step(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "optimizer.pt")
            atomic_save_optimizer_checkpoint(
                path,
                optimizer_state(3.0),
                step=42,
                checkpoint_path=os.path.join(folder, "model_000000042.safetensors"),
            )

            loaded = load_optimizer_checkpoint(path, expected_step=42)

        self.assertEqual(loaded["param_groups"][0]["lr"], 0.001)
        self.assertTrue(torch.equal(loaded["state"][0]["momentum"], torch.tensor([3.0])))

    def test_stale_optimizer_step_is_rejected(self):
        payload = {
            "format_version": 1,
            "step": 10,
            "model_checkpoint": "model_000000010.safetensors",
            "optimizer_state_dict": optimizer_state(),
        }

        with self.assertRaisesRegex(OptimizerCheckpointValidationError, "does not match"):
            parse_optimizer_checkpoint(payload, expected_step=11)

    def test_legacy_unversioned_optimizer_is_rejected(self):
        with self.assertRaisesRegex(OptimizerCheckpointValidationError, "legacy"):
            parse_optimizer_checkpoint(optimizer_state(), expected_step=1)

    def test_failed_save_invalidates_previous_optimizer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "optimizer.pt")
            torch.save({"old": "state"}, path)

            with mock.patch(
                "toolkit.optimizer_checkpoint.torch.save",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    atomic_save_optimizer_checkpoint(path, optimizer_state(), step=2)

            self.assertFalse(os.path.exists(path))
            self.assertEqual(
                [name for name in os.listdir(folder) if name.endswith(".tmp")],
                [],
            )


if __name__ == "__main__":
    unittest.main()
