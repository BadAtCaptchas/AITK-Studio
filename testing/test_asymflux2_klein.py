import os
import tempfile
import unittest

import torch

from extensions_built_in.diffusion_models.flux2.asymflux2_components import (
    AsymFlux2Transformer2DModel,
    FlowAdapterScheduler,
    OklabColorEncoder,
)
from extensions_built_in.diffusion_models.flux2.asymflux2_klein_model import AsymFlux2Klein9BModel
from toolkit.config_modules import ModelConfig
from toolkit.util.get_model import get_model_class


def make_tiny_asymflux2_transformer():
    return AsymFlux2Transformer2DModel(
        patch_size=1,
        in_channels=3,
        base_rank=3,
        num_layers=0,
        num_single_layers=0,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=8,
        timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2),
        rope_theta=1000,
    )


class OklabColorEncoderTest(unittest.TestCase):
    def test_encode_decode_roundtrip_shape_and_tolerance(self):
        encoder = OklabColorEncoder(
            use_affine_norm=True,
            mean=(0.56, 0.0, 0.01),
            std=0.16,
        )
        image = torch.rand(2, 3, 16, 16) * 2 - 1

        latents = encoder.encode(image)
        decoded = encoder.decode(latents)

        self.assertEqual(latents.shape, image.shape)
        self.assertEqual(decoded.shape, image.shape)
        self.assertLess(torch.mean(torch.abs(decoded - image)).item(), 1e-4)

    def test_encode_accepts_float_images_when_encoder_buffers_are_bfloat16(self):
        encoder = OklabColorEncoder(
            use_affine_norm=True,
            mean=(0.56, 0.0, 0.01),
            std=0.16,
        ).to(dtype=torch.bfloat16)
        image = torch.rand(1, 3, 8, 8, dtype=torch.float32) * 2 - 1

        latents = encoder.encode(image)
        decoded = encoder.decode(latents)

        self.assertEqual(latents.dtype, image.dtype)
        self.assertEqual(decoded.dtype, image.dtype)
        self.assertEqual(latents.shape, image.shape)
        self.assertEqual(decoded.shape, image.shape)


class FlowAdapterSchedulerTest(unittest.TestCase):
    def make_scheduler(self):
        return FlowAdapterScheduler(
            shift=17.0,
            use_dynamic_shifting=True,
            base_seq_len=1024 ** 2,
            max_seq_len=2048 ** 2,
            base_logshift=torch.log(torch.tensor(17.0)).item(),
            max_logshift=torch.log(torch.tensor(34.0)).item(),
            dynamic_shifting_type="sqrt",
            base_scheduler="UniPCMultistep",
        )

    def test_sqrt_dynamic_shift_defaults(self):
        scheduler = self.make_scheduler()

        self.assertAlmostEqual(float(scheduler.get_shift(seq_len=1024 ** 2)), 17.0, places=5)
        self.assertAlmostEqual(float(scheduler.get_shift(seq_len=2048 ** 2)), 34.0, places=5)

    def test_train_timesteps_and_add_noise(self):
        scheduler = self.make_scheduler()
        latents = torch.zeros(1, 3, 32, 32)
        timesteps = scheduler.set_train_timesteps(
            4,
            device=torch.device("cpu"),
            timestep_type="shift",
            latents=latents,
            patch_size=16,
        )

        self.assertEqual(timesteps.shape, (4,))
        self.assertTrue(torch.all(timesteps[:-1] >= timesteps[1:]))

        original = torch.zeros(2, 3, 4, 4)
        noise = torch.ones_like(original)
        noisy = scheduler.add_noise(original, noise, torch.tensor([0.0, 500.0]))

        self.assertTrue(torch.allclose(noisy[0], original[0]))
        self.assertTrue(torch.allclose(noisy[1], torch.full_like(original[1], 0.5)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for scheduler device regression")
    def test_unipc_inference_steps_keep_scheduler_state_on_cuda(self):
        scheduler = self.make_scheduler()
        device = torch.device("cuda")
        latents = torch.randn(1, 3, 4, 4, device=device)

        scheduler.set_timesteps(4, seq_len=latents.shape[2:].numel(), device=device)
        for timestep in scheduler.timesteps[:2]:
            model_output = torch.randn_like(latents)
            latents = scheduler.step(model_output, timestep, latents, return_dict=False)[0]

        self.assertEqual(latents.device.type, "cuda")
        self.assertEqual(scheduler.sigmas.device.type, "cuda")
        self.assertEqual(scheduler.base_scheduler.sigmas.device.type, "cuda")


class AsymFlux2KleinModelTest(unittest.TestCase):
    def test_tiny_asymflow_transformer_forward_shape(self):
        model = make_tiny_asymflux2_transformer()
        latents = torch.randn(1, 3, 2, 2)
        prompt_embeds = torch.randn(1, 2, 8)
        text_ids = torch.zeros(1, 2, 4, dtype=torch.long)

        output = model(
            x_t=latents,
            timestep=torch.ones(1) * 0.5,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
        )

        self.assertEqual(output.shape, latents.shape)

    def test_registry_resolves_arch(self):
        config = ModelConfig(
            arch="asymflux2_klein_9b",
            name_or_path="Lakonik/AsymFLUX.2-klein-9B",
        )

        self.assertIs(get_model_class(config), AsymFlux2Klein9BModel)

    def test_full_finetune_folder_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(AsymFlux2Klein9BModel._is_full_finetune_path(tmpdir))
            os.makedirs(os.path.join(tmpdir, "transformer"))

            self.assertTrue(AsymFlux2Klein9BModel._is_full_finetune_path(tmpdir))

    def test_lora_keys_remain_peft_style(self):
        model = object.__new__(AsymFlux2Klein9BModel)
        state_dict = {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(1, 1),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(1, 1),
        }

        self.assertEqual(model.convert_lora_weights_before_save(state_dict), state_dict)
        self.assertEqual(model.convert_lora_weights_before_load(state_dict), state_dict)

    def test_full_save_contract(self):
        model = object.__new__(AsymFlux2Klein9BModel)
        model.model = make_tiny_asymflux2_transformer()
        model.device_torch = torch.device("cpu")
        model.model_config = type("ModelConfigStub", (), {"low_vram": True})()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "checkpoint.safetensors")
            model.save_model(output_path, {"test": True}, torch.float32)
            output_dir = os.path.join(tmpdir, "checkpoint")

            self.assertTrue(os.path.isdir(os.path.join(output_dir, "transformer")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "transformer", "config.json")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "aitk_meta.yaml")))
            self.assertTrue(AsymFlux2Klein9BModel._is_full_finetune_path(output_dir))

            reloaded = AsymFlux2Transformer2DModel.from_pretrained(os.path.join(output_dir, "transformer"))
            self.assertEqual(reloaded.config["_class_name"], "AsymFlux2Transformer2DModel")


if __name__ == "__main__":
    unittest.main()
