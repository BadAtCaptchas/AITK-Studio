"""Opt-in MiniMax-H3 hardware smoke test.

This suite downloads the released checkpoints (roughly 43 GB) and is therefore
never enabled by normal discovery. Run it explicitly with
``AITK_RUN_MINIMAX_H3_SMOKE=1 python -m unittest testing.test_minimax_h3_smoke``.
"""

import gc
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from extensions_built_in.diffusion_models.minimax_h3 import MinimaxH3Model
from extensions_built_in.diffusion_models.minimax_h3.src import packing
from toolkit.config_modules import ModelConfig
from toolkit.lora_special import LoRAModule
from toolkit.util import convrot_quant
from toolkit.util.ostris_quant import OstrisLinear


RUN_HARDWARE_SMOKE = os.getenv("AITK_RUN_MINIMAX_H3_SMOKE") == "1"


class _SmokeNetwork:
    network_type = "lora"
    is_active = True
    is_merged_in = False
    is_lorm = False
    _multiplier = 1.0

    def __init__(self, device: torch.device):
        self.torch_multiplier = torch.ones(1, device=device)


@unittest.skipUnless(
    RUN_HARDWARE_SMOKE,
    "set AITK_RUN_MINIMAX_H3_SMOKE=1 to download and run MiniMax-H3",
)
@unittest.skipUnless(torch.cuda.is_available(), "MiniMax-H3 smoke requires CUDA")
class MiniMaxH3HardwareSmokeTest(unittest.TestCase):
    def test_load_adapter_step_reload_and_representative_samples(self):
        device = torch.device("cuda")
        size = int(os.getenv("AITK_MINIMAX_H3_SMOKE_SIZE", "256"))
        if size < 64 or size % 32:
            self.fail("AITK_MINIMAX_H3_SMOKE_SIZE must be >= 64 and divisible by 32")

        config = ModelConfig(
            name_or_path="Comfy-Org/MiniMax-H3",
            arch="minimax_h3",
            dtype="bf16",
            vae_dtype="bf16",
            te_dtype="bf16",
            quantize=True,
            qtype="convrot8",
            quantize_te=True,
            qtype_te="nvfp4",
            low_vram=True,
            layer_offloading=False,
            assistant_lora_path=(
                "ostris/minimax_h3_training_adapter/"
                "minimax_h3_training_adapter_v1.safetensors"
            ),
            model_kwargs={
                "partition": "fl2va_pruned",
                "max_text_length": 512,
                "sample_audio": True,
            },
        )
        model = MinimaxH3Model(device, config, dtype="bf16")
        model.load_model()
        self.assertGreater(model.quantization_report.quantized_modules, 0)
        self.assertGreater(model.text_encoder_quantization_report.quantized_modules, 0)
        assistant = model.assistant_lora
        self.assertIsNotNone(assistant)
        self.assertTrue(assistant.is_active)
        self.assertFalse(assistant.is_merged_in)
        self.assertFalse(assistant.can_merge_in)
        self.assertFalse(any(parameter.requires_grad for parameter in assistant.parameters()))

        prompt = ["A small paper boat moving across a calm pond"]
        prompt_embeds = model.get_prompt_embeds(prompt)
        model.text_encoder.to("cpu")
        torch.cuda.empty_cache()

        keyframe = Image.new("RGB", (size, size), color=(80, 120, 180))
        keyframe_embeds = model.get_prompt_embeds(prompt, control_images=[keyframe])
        model.text_encoder.to("cpu")
        torch.cuda.empty_cache()

        model.model.to(device)
        pipeline = model.get_generation_pipeline()
        # Sampling disables the live de-distillation assistant; training turns
        # it back on. Exercise both wrapped-forward states without ever merging
        # into the packed base weights.
        assistant.is_active = False
        self.assertFalse(assistant.is_active)
        image = pipeline(
            conditional_embeds=prompt_embeds,
            height=size,
            width=size,
            num_frames=1,
            num_inference_steps=1,
            generator=torch.Generator(device="cpu").manual_seed(11),
            with_audio=False,
        )
        self.assertEqual(len(image), 1)
        self.assertIsInstance(image[0], Image.Image)

        t2v = pipeline(
            conditional_embeds=prompt_embeds,
            height=size,
            width=size,
            num_frames=5,
            num_inference_steps=1,
            generator=torch.Generator(device="cpu").manual_seed(12),
            with_audio=True,
        )
        self.assertEqual(t2v["video"].shape[0], 5)
        self.assertIsNotNone(t2v["audio"])

        i2v = pipeline(
            conditional_embeds=keyframe_embeds,
            height=size,
            width=size,
            num_frames=5,
            num_inference_steps=1,
            generator=torch.Generator(device="cpu").manual_seed(13),
            ctrl_img=keyframe,
            with_audio=False,
        )
        self.assertEqual(i2v["video"].shape[0], 5)
        assistant.is_active = True
        self.assertTrue(assistant.is_active)

        packed_layer = next(
            module for module in model.model.modules() if isinstance(module, OstrisLinear)
        )
        # Mirror the trainer: the packed base stays frozen and only the live
        # adapter participates in autograd.
        model.model.requires_grad_(False)
        packed_layer.to(device)
        network = _SmokeNetwork(device)
        adapter = LoRAModule(
            "minimax_h3_smoke",
            packed_layer,
            multiplier=1.0,
            lora_dim=2,
            alpha=2,
            network=network,
        ).to(device=device, dtype=model.torch_dtype)
        adapter.apply_to()
        self.assertFalse(adapter.can_merge_in)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
        latent_frames = packing.video_latent_num_frames(5)
        noisy_latents = torch.randn(
            1,
            24,
            latent_frames,
            size // 16,
            size // 16,
            device=device,
            dtype=model.torch_dtype,
        )
        audio_rows = packing.audio_latent_num_frames(5) * packing.AUDIO_CHANNELS
        batch = SimpleNamespace(
            dataset_config=SimpleNamespace(do_i2v=False, do_audio=True),
            num_frames=5,
            first_frame_latents=None,
            tensor=None,
            audio_latents=torch.randn(
                1, audio_rows, 32, device=device, dtype=torch.float32
            ),
            audio_data=None,
            audio_noise=None,
            audio_target=None,
            audio_pred=None,
            audio_pred_uncond=None,
            audio_pred_prior=None,
            audio_pred_preservation=None,
            audio_pred_slot=None,
            audio_noisy=None,
            audio_sigma=None,
            audio_loss_mask=None,
            keyframe_conditioning_noise=None,
        )
        batch.set_secondary_audio_pred = lambda pred: (
            setattr(batch, batch.audio_pred_slot, pred)
            if batch.audio_pred_slot is not None
            else None
        )
        prediction = model.get_noise_prediction(
            latent_model_input=noisy_latents,
            timestep=torch.tensor([500.0], device=device),
            text_embeddings=prompt_embeds,
            batch=batch,
        )
        self.assertIsNotNone(batch.audio_pred)
        self.assertIsNotNone(batch.audio_target)
        loss = prediction.float().square().mean()
        loss = loss + torch.nn.functional.mse_loss(
            batch.audio_pred.float(), batch.audio_target.float()
        )
        loss.backward()
        optimizer.step()
        self.assertTrue(
            any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in adapter.parameters()
            )
        )

        with tempfile.TemporaryDirectory() as root:
            adapter_path = os.path.join(root, "adapter.safetensors")
            saved = {
                key: tensor.detach().to("cpu").contiguous()
                for key, tensor in adapter.state_dict().items()
            }
            save_file(saved, adapter_path)
            with torch.no_grad():
                adapter.lora_up.weight.zero_()
            adapter.load_state_dict(load_file(adapter_path), strict=True)
            self.assertTrue(
                torch.equal(
                    adapter.lora_up.weight.detach().cpu(),
                    saved["lora_up.weight"],
                )
            )

        capability = torch.cuda.get_device_capability(device)
        if capability < (8, 9):
            self.assertFalse(convrot_quant._e4m3_triton_ok(device))

        packed_layer.forward = adapter.org_forward
        adapter.to("cpu")
        assistant.force_to("cpu", model.torch_dtype)
        optimizer.zero_grad(set_to_none=True)
        prompt_embeds.text_embeds = [item.to("cpu") for item in prompt_embeds.text_embeds]
        keyframe_embeds.text_embeds = [
            item.to("cpu") for item in keyframe_embeds.text_embeds
        ]
        model.model.to("cpu")
        model.vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
