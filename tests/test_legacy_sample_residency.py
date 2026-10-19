"""Real legacy phase residency, including quantized MOSS persistent adapters."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from toolkit.memory_management import MemoryManager, SampleMemoryCoordinator


class ResidentModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.root = torch.nn.Parameter(torch.randn(4))
        self.root_alias = self.root
        self.register_buffer("root_buffer", torch.randn(4))
        self.norm = torch.nn.LayerNorm(4)
        self.norm.bias.requires_grad_(False)
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(self.norm(x)) + self.root + self.root_buffer


class LegacySampleResidencyCpuTests(unittest.TestCase):
    def test_explicit_ignore_tensor_outside_module_tree_keeps_device_moves(self):
        module = torch.nn.Module()
        module.external_table = torch.ones(4)
        manager = MemoryManager(module, torch.device("cpu"))
        manager.unmanaged_modules = [module.external_table]
        moves = []
        original_to = torch.Tensor.to

        def tracked_to(tensor, *args, **kwargs):
            moves.append((tensor.shape, args))
            return original_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", tracked_to):
            manager.memory_managed_to(torch.device("cpu"))
            manager.deactivate_to_cpu()
        self.assertEqual(moves, [(torch.Size([4]), (torch.device("cpu"),))] * 2)
        self.assertFalse(list(module.parameters()))
        self.assertFalse(list(module.buffers()))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for real phase residency")
class LegacySampleResidencyTests(unittest.TestCase):
    def test_explicit_parking_preserves_managed_storage_ties_grads_and_restore(self):
        device = torch.device("cuda", torch.cuda.current_device())
        module = ResidentModule().to(device)
        module.root.grad = torch.ones_like(module.root)
        MemoryManager.attach(module, device)
        self.addCleanup(MemoryManager.free, module)
        manager = module._memory_manager
        x = torch.randn(2, 4, device=device)
        with torch.no_grad():
            expected = module(x)
        module.to("cpu")  # Ordinary legacy parking remains a no-op.
        self.assertEqual(module.root.device, device)
        manager.deactivate_to_cpu()
        self.assertEqual(module.device.type, "cpu")
        self.assertEqual(manager.process_device, device)
        self.assertEqual(module.root.device.type, "cpu")
        self.assertEqual(module.root.grad.device.type, "cpu")
        self.assertEqual(module.root_buffer.device.type, "cpu")
        self.assertIs(module.root, module.root_alias)
        module.to(device)  # Existing holder/training restore path.
        self.assertEqual(module.root.device, device)
        self.assertEqual(module.root.grad.device, device)
        self.assertEqual(module.root_buffer.device, device)
        self.assertEqual(module.linear.weight.device.type, "cpu")
        self.assertIs(module.root, module.root_alias)
        with torch.no_grad():
            torch.testing.assert_close(module(x), expected, rtol=0, atol=0)

        owner = SimpleNamespace(model=module, text_encoder=[], vae=None, device_torch=device)
        coordinator = SampleMemoryCoordinator(owner)
        coordinator.capture()
        coordinator.offload_all()
        module.requires_grad_(False)
        module.eval()
        coordinator.restore()
        self.assertEqual(module.root.device, device)
        self.assertTrue(module.root.requires_grad)
        self.assertFalse(module.norm.bias.requires_grad)
        self.assertTrue(module.training)

    def test_quantized_moss_and_adapters_survive_low_vram_phase_and_holder_restore(self):
        from tests.test_kromoss import holder, tiny_model, tiny_distillation, TURBO
        from toolkit.inference_lora import InferenceLoRA
        from toolkit.util.convrot_quant import get_convrot_quantizer
        from toolkit.util.ostris_quant import convert_linear_to_ostris
        from extensions_built_in.diffusion_models.moss_image.src.distillation import FrozenMossDistillationAdapter

        device = torch.device("cuda", torch.cuda.current_device())
        model = tiny_model().bfloat16().to(device).requires_grad_(False)
        fixed = tiny_distillation(model)
        linear = model.layers[0].proj["qkv"]
        self.assertTrue(convert_linear_to_ostris(linear, get_convrot_quantizer("convrot8", kernel="torch")))
        MemoryManager.attach(model, device)
        self.addCleanup(MemoryManager.free, model)
        controller = FrozenMossDistillationAdapter.attach(model, fixed, identity="fixed")
        owner = holder(low_vram=True, model_kwargs={"distillation_lora_path": TURBO})
        owner.model = model
        owner.device_torch = device
        owner.text_encoder = ResidentModule().to(device)
        MemoryManager.attach(owner.text_encoder, device)
        self.addCleanup(MemoryManager.free, owner.text_encoder)
        owner.vae = None

        with TemporaryDirectory() as root:
            path = str(Path(root) / "user.safetensors")
            save_file(owner.convert_lora_weights_before_save({
                "transformer.layers.0.proj.o.lora_A.weight": torch.randn(2, 32) * .1,
                "transformer.layers.0.proj.o.lora_B.weight": torch.randn(32, 2) * .1,
            }), path)
            user = InferenceLoRA(path).load(owner).attach()
            model._moss_user_lora = user
            model._moss_user_lora_identity = path
            x = torch.randn(1, 4, 2, 2, device=device, dtype=torch.bfloat16)
            t = torch.tensor([500.], device=device)
            context = torch.randn(1, 3, 16, device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                expected = model(x, t, context)
            original_hooks = list(user._hooks)
            owner._park_loaded_component(model)
            owner._park_loaded_component(owner.text_encoder)
            self.assertEqual(linear.quantized_device.type, "cpu")
            self.assertEqual(model.device.type, "cpu")
            self.assertEqual(owner.text_encoder.device.type, "cpu")
            self.assertTrue(all(delta.down.device.type == "cpu" and delta.down.dtype == torch.float32
                                for parts in controller._targets.values() for delta in parts))
            self.assertNotEqual(original_hooks, user._hooks)
            owner.save_device_state()
            self.assertEqual(owner.device_state["unet"]["device"].type, "cpu")
            self.assertEqual(owner.device_state["text_encoder"]["device"].type, "cpu")
            coordinator = SampleMemoryCoordinator(owner, device=device)
            coordinator.capture()
            coordinator.activate(("text_encoder",), phase_name="text")
            self.assertEqual(model.device.type, "cpu")
            coordinator.activate(("unet",), phase_name="generate")
            self.assertEqual(owner.text_encoder.device.type, "cpu")
            self.assertEqual(linear.quantized_device, device)
            with torch.no_grad():
                torch.testing.assert_close(model(x, t, context), expected, rtol=0, atol=0)
            coordinator.offload_all()
            coordinator.restore()
            owner.restore_device_state()
            self.assertEqual(model.device.type, "cpu")
            self.assertEqual(owner.text_encoder.device.type, "cpu")
            self.assertTrue(all(delta.down.device.type == "cpu" for parts in controller._targets.values() for delta in parts))
            model.to(device)  # BaseModel.generate_images final restore.
            self.assertEqual(owner.text_encoder.device.type, "cpu")
            with torch.no_grad():
                torch.testing.assert_close(model(x, t, context), expected, rtol=0, atol=0)
            self.assertEqual(len(model.layers[0].proj["qkv"]._forward_hooks), 1)
            self.assertEqual(len(model.layers[0].proj["o"]._forward_hooks), 2)


if __name__ == "__main__":
    unittest.main()
