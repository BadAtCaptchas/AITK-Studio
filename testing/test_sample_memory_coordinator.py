import unittest
from unittest import mock
from types import SimpleNamespace

import torch

from toolkit.memory_management import SampleMemoryCoordinator


class FakeModule:
    def __init__(self, name, device="cpu"):
        self.name = name
        self.device = torch.device(device)
        self.training = True
        self.moves = []

    def to(self, device, *args, **kwargs):
        self.device = torch.device(device)
        self.moves.append(self.device)
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def parameters(self):
        return iter(())

    def buffers(self):
        return iter(())


class SampleMemoryCoordinatorTest(unittest.TestCase):
    def make_owner(self):
        return SimpleNamespace(
            device_torch=torch.device("cuda"),
            model=FakeModule("model"),
            vae=FakeModule("vae"),
            text_encoder=[FakeModule("text")],
            unconditional_transformer=FakeModule("unconditional"),
            adapter=None,
            refiner_unet=None,
        )

    def test_phase_activation_moves_only_requested_components(self):
        owner = self.make_owner()
        coordinator = SampleMemoryCoordinator(owner)

        coordinator.activate(("text_encoder",), phase_name="text")
        self.assertEqual(owner.text_encoder[0].device, torch.device("cuda"))
        self.assertEqual(owner.model.device, torch.device("cpu"))
        self.assertEqual(owner.vae.device, torch.device("cpu"))
        self.assertEqual(owner.unconditional_transformer.device, torch.device("cpu"))

        coordinator.activate(("unet", "vae"), phase_name="generate")
        self.assertEqual(owner.model.device, torch.device("cuda"))
        self.assertEqual(owner.vae.device, torch.device("cuda"))
        self.assertEqual(owner.text_encoder[0].device, torch.device("cpu"))
        self.assertEqual(owner.unconditional_transformer.device, torch.device("cpu"))

    def test_context_restores_state_after_exception(self):
        owner = self.make_owner()
        owner.model.training = False
        owner.vae.device = torch.device("meta")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            with SampleMemoryCoordinator(owner) as coordinator:
                coordinator.activate(("text_encoder",), phase_name="text")
                raise RuntimeError("boom")

        self.assertEqual(owner.model.device, torch.device("cpu"))
        self.assertFalse(owner.model.training)
        self.assertEqual(owner.vae.device, torch.device("meta"))

    def test_null_components_and_requires_grad_are_safe(self):
        linear = torch.nn.Linear(1, 1)
        linear.bias.requires_grad_(False)
        owner = SimpleNamespace(
            device_torch=torch.device("cpu"),
            model=linear,
            vae=None,
            text_encoder=[],
            unconditional_transformer=None,
            adapter=None,
            refiner_unet=None,
        )

        with SampleMemoryCoordinator(owner) as coordinator:
            coordinator.activate(
                ("unet", "vae", "text_encoder", "unconditional_transformer"),
                phase_name="safe",
            )

        self.assertTrue(linear.weight.requires_grad)
        self.assertFalse(linear.bias.requires_grad)

    def test_block_offload_activation_uses_partial_forward_activation(self):
        module = FakeModule("model")
        module._block_offload_manager = SimpleNamespace(
            activate_for_forward=mock.Mock(),
            deactivate_to_cpu=mock.Mock(),
        )
        owner = SimpleNamespace(
            device_torch=torch.device("cuda"),
            model=module,
            vae=None,
            text_encoder=[],
            unconditional_transformer=None,
            adapter=None,
            refiner_unet=None,
        )
        coordinator = SampleMemoryCoordinator(owner)

        coordinator.activate(("unet",), phase_name="generate")

        module._block_offload_manager.activate_for_forward.assert_called_once_with(
            torch.device("cuda")
        )

    def test_status_messages_use_tqdm_write_and_owner_status_hook(self):
        owner = self.make_owner()
        owner._status_update = mock.Mock()
        callback = mock.Mock()
        coordinator = SampleMemoryCoordinator(owner, status_callback=callback)

        with mock.patch("toolkit.memory_management.sample_coordinator.tqdm.write") as write:
            coordinator._status("Low-VRAM sample 1/3: encoding prompts")

        write.assert_called_once_with("Low-VRAM sample 1/3: encoding prompts")
        owner._status_update.assert_called_once_with(
            "Low-VRAM sample 1/3: encoding prompts"
        )
        callback.assert_not_called()

    def test_status_messages_fall_back_to_callback_without_owner_status_hook(self):
        owner = self.make_owner()
        callback = mock.Mock()
        coordinator = SampleMemoryCoordinator(owner, status_callback=callback)

        with mock.patch("toolkit.memory_management.sample_coordinator.tqdm.write") as write:
            coordinator._status("Low-VRAM sample 1/3: generating image")

        write.assert_called_once_with("Low-VRAM sample 1/3: generating image")
        callback.assert_called_once_with("Low-VRAM sample 1/3: generating image")

    def test_status_messages_are_quiet_without_status_sink(self):
        owner = self.make_owner()
        coordinator = SampleMemoryCoordinator(owner)

        with mock.patch("toolkit.memory_management.sample_coordinator.tqdm.write") as write:
            coordinator._status("Low-VRAM sample 1/3: encoding prompts")

        write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
