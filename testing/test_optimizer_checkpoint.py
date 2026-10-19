import os
import tempfile
import unittest
from unittest import mock

import torch

from toolkit.optimizer_checkpoint import (
    OptimizerCheckpointValidationError,
    atomic_save_optimizer_checkpoint,
    build_optimizer_checkpoint,
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

    def test_version_two_requires_raw_state_and_digest_and_preserves_version_one(self):
        ordinary = build_optimizer_checkpoint(optimizer_state(), step=3)
        self.assertEqual(ordinary["format_version"], 1)
        self.assertIs(parse_optimizer_checkpoint(ordinary, 3), ordinary["optimizer_state_dict"])
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = os.path.join(folder, "adapter.safetensors")
            with open(checkpoint, "wb") as output:
                output.write(b"adapter identity")
            payload = build_optimizer_checkpoint(
                optimizer_state(), step=3, checkpoint_path=checkpoint,
                adapter_resume_state={"format_version": 1},
            )
            optimizer_path = os.path.join(folder, "optimizer.pt")
            torch.save(payload, optimizer_path)
            with self.assertRaisesRegex(OptimizerCheckpointValidationError, "load_optimizer_checkpoint_for_resume"):
                load_optimizer_checkpoint(optimizer_path, expected_step=3)
        self.assertEqual(payload["format_version"], 2)
        self.assertIs(parse_optimizer_checkpoint(payload, 3), payload["optimizer_state_dict"])
        for key in ("adapter_resume_state", "model_checkpoint_sha256"):
            bad = dict(payload)
            del bad[key]
            with self.assertRaises(OptimizerCheckpointValidationError):
                parse_optimizer_checkpoint(bad, 3)
        for version in (4, True, "2"):
            bad = dict(payload, format_version=version)
            with self.assertRaisesRegex(OptimizerCheckpointValidationError, "unsupported format"):
                parse_optimizer_checkpoint(bad, 3)

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
