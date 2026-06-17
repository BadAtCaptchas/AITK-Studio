import types
import unittest

import torch

from extensions_built_in.diffusion_models.ltx2.ltx2 import LTX2Model
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import TrainConfig
from toolkit.samplers.custom_flowmatch_sampler import sample_shifted_logit_normal


class _FakeDeviceModule:
    def __init__(self, device=torch.device("cpu"), dtype=torch.float32):
        self.device = device
        self.dtype = dtype

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self


class _FakeConnectors(_FakeDeviceModule):
    def __call__(self, text_embeds, attention_mask, padding_side="left"):
        return text_embeds, torch.zeros_like(text_embeds), attention_mask


class _FakeRope:
    @staticmethod
    def prepare_video_coords(batch_size, frames, height, width, device, fps=24):
        return torch.zeros((batch_size, frames * height * width), device=device)

    @staticmethod
    def prepare_audio_coords(batch_size, frames, device):
        return torch.zeros((batch_size, frames), device=device)


class _FakeTransformer(_FakeDeviceModule):
    def __init__(self):
        super().__init__()
        self.rope = _FakeRope()
        self.audio_rope = _FakeRope()
        self.last_call = None

    def __call__(self, **kwargs):
        self.last_call = kwargs
        return kwargs["hidden_states"], kwargs["audio_hidden_states"]


class _FakePipeline:
    transformer_spatial_patch_size = 1
    transformer_temporal_patch_size = 1
    audio_sampling_rate = 16000
    audio_hop_length = 160
    audio_vae_temporal_compression_ratio = 1

    def __init__(self, transformer):
        self.transformer = transformer
        self.connectors = _FakeConnectors()
        self.audio_vae = types.SimpleNamespace(
            config=types.SimpleNamespace(mel_bins=80, latent_channels=1)
        )

    @staticmethod
    def _pack_latents(latents, patch_size=1, patch_size_t=1):
        batch, channels, frames, height, width = latents.shape
        return latents.permute(0, 2, 3, 4, 1).reshape(batch, frames * height * width, channels)

    @staticmethod
    def _unpack_latents(latents, num_frames, height, width, patch_size=1, patch_size_t=1):
        batch, _, channels = latents.shape
        return latents.reshape(batch, num_frames, height, width, channels).permute(0, 4, 1, 2, 3)

    @staticmethod
    def prepare_audio_latents(
        batch_size,
        num_channels_latents,
        audio_latent_length,
        num_mel_bins,
        noise_scale,
        dtype,
        device,
        generator=None,
        latents=None,
    ):
        return torch.zeros(
            (batch_size, audio_latent_length, num_channels_latents),
            dtype=dtype,
            device=device,
        )


class LTXTrainingUpdateTests(unittest.TestCase):
    def test_shifted_logit_normal_bounds_and_reproducibility(self):
        torch.manual_seed(123)
        first = sample_shifted_logit_normal(
            batch_size=8,
            seq_length=2048,
            device=torch.device("cpu"),
        )
        torch.manual_seed(123)
        second = sample_shifted_logit_normal(
            batch_size=8,
            seq_length=2048,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(first.shape), (8,))
        self.assertTrue(torch.all(first >= 0))
        self.assertTrue(torch.all(first <= 1))
        self.assertTrue(torch.equal(first, second))

    def test_shifted_logit_normal_sequence_length_shift(self):
        torch.manual_seed(7)
        short = sample_shifted_logit_normal(
            batch_size=20_000,
            seq_length=1024,
            device=torch.device("cpu"),
            uniform_prob=0.0,
        )
        torch.manual_seed(7)
        long = sample_shifted_logit_normal(
            batch_size=20_000,
            seq_length=4096,
            device=torch.device("cpu"),
            uniform_prob=0.0,
        )

        self.assertGreater(long.mean().item(), short.mean().item())

    def test_ltx_legacy_strategy_maps_i2v_and_audio(self):
        model = object.__new__(LTX2Model)
        batch = types.SimpleNamespace(
            ltx_strategy=None,
            num_frames=9,
            dataset_config=types.SimpleNamespace(do_i2v=True, do_audio=True),
        )

        video = model._get_ltx_modality_config(batch, "video")
        audio = model._get_ltx_modality_config(batch, "audio")

        self.assertTrue(video["is_generated"])
        self.assertEqual(video["conditions"], [{"type": "first_frame", "probability": 1.0}])
        self.assertTrue(audio["is_generated"])

    def test_ltx_temporal_condition_masks(self):
        model = object.__new__(LTX2Model)
        device = torch.device("cpu")

        first_frame = model._build_temporal_condition_mask(
            [{"type": "first_frame", "probability": 1.0}],
            total_units=4,
            batch_size=1,
            device=device,
            modality="video",
        )
        prefix = model._build_temporal_condition_mask(
            [{"type": "prefix", "tokens": 2, "probability": 1.0}],
            total_units=5,
            batch_size=1,
            device=device,
            modality="audio",
        )
        suffix = model._build_temporal_condition_mask(
            [{"type": "suffix", "latent_frames": 2, "probability": 1.0}],
            total_units=5,
            batch_size=1,
            device=device,
            modality="video",
        )

        self.assertEqual(first_frame.tolist(), [[True, False, False, False]])
        self.assertEqual(prefix.tolist(), [[True, True, False, False, False]])
        self.assertEqual(suffix.tolist(), [[False, False, False, True, True]])

    def test_ltx_strategy_can_mark_audio_as_frozen_conditioning(self):
        model = object.__new__(LTX2Model)
        batch = types.SimpleNamespace(
            ltx_strategy={
                "audio": {
                    "is_generated": False,
                    "conditions": [{"type": "prefix", "tokens": 2}],
                }
            },
            num_frames=9,
            dataset_config=types.SimpleNamespace(do_i2v=False, do_audio=True),
        )

        audio = model._get_ltx_modality_config(batch, "audio")

        self.assertFalse(audio["is_generated"])
        self.assertEqual(audio["conditions"][0]["type"], "prefix")

    def _fake_ltx_model(self):
        model = object.__new__(LTX2Model)
        transformer = _FakeTransformer()
        model.model = _FakeDeviceModule()
        model.transformer = transformer
        model.pipeline = _FakePipeline(transformer)
        model.device_torch = torch.device("cpu")
        model.torch_dtype = torch.float32
        model.ltx_version = "2.3"
        model.add_noise = (
            lambda original_samples, noise, timesteps: original_samples
            + noise * (timesteps.to(original_samples.dtype).view(-1, 1, 1) / 1000.0)
        )
        return model, transformer

    def _run_fake_ltx_prediction(
        self,
        ltx_strategy=None,
        do_i2v=False,
        do_audio=False,
        audio_latents=None,
    ):
        model, transformer = self._fake_ltx_model()
        clean_latents = torch.tensor([[[[[10.0]], [[20.0]], [[30.0]]]]])
        noisy_latents = torch.zeros_like(clean_latents)
        batch = types.SimpleNamespace(
            latents=clean_latents,
            ltx_strategy=ltx_strategy,
            num_frames=17,
            dataset_config=types.SimpleNamespace(
                do_i2v=do_i2v,
                do_audio=do_audio,
                fps=24,
            ),
            audio_latents=audio_latents,
            audio_tensor=None,
            audio_target=None,
            audio_pred=None,
            audio_loss_mask=None,
            video_loss_mask=None,
        )
        text_embeddings = types.SimpleNamespace(
            text_embeds=torch.zeros((1, 1, 2)),
            attention_mask=torch.ones((1, 1)),
        )

        model.get_noise_prediction(
            latent_model_input=noisy_latents,
            timestep=torch.tensor([500.0]),
            text_embeddings=text_embeddings,
            batch=batch,
        )
        return batch, transformer

    def test_ltx_get_noise_prediction_applies_video_conditioning_masks(self):
        cases = [
            (
                "first_frame",
                {"video": {"is_generated": True, "conditions": [{"type": "first_frame"}]}},
                [0.0, 1.0, 1.0],
            ),
            (
                "prefix",
                {"video": {"is_generated": True, "conditions": [{"type": "prefix", "tokens": 2}]}},
                [0.0, 0.0, 1.0],
            ),
            (
                "suffix",
                {"video": {"is_generated": True, "conditions": [{"type": "suffix", "tokens": 2}]}},
                [1.0, 0.0, 0.0],
            ),
            ("legacy_i2v", None, [0.0, 1.0, 1.0]),
        ]

        for name, strategy, expected_loss_mask in cases:
            with self.subTest(name=name):
                batch, transformer = self._run_fake_ltx_prediction(
                    ltx_strategy=strategy,
                    do_i2v=name == "legacy_i2v",
                )
                video_timestep = transformer.last_call["timestep"]

                self.assertEqual(
                    batch.video_loss_mask.flatten().tolist(),
                    expected_loss_mask,
                )
                self.assertEqual(
                    video_timestep.flatten().tolist(),
                    [500.0 * value for value in expected_loss_mask],
                )

    def test_ltx_generated_audio_prefix_gets_loss_mask(self):
        audio_latents = torch.tensor([[[10.0], [20.0], [30.0]]])
        batch, transformer = self._run_fake_ltx_prediction(
            ltx_strategy={
                "video": {"is_generated": True, "conditions": []},
                "audio": {
                    "is_generated": True,
                    "conditions": [{"type": "prefix", "tokens": 2}],
                },
            },
            do_audio=True,
            audio_latents=audio_latents,
        )

        self.assertIsNotNone(batch.audio_target)
        self.assertIsNotNone(batch.audio_pred)
        self.assertEqual(batch.audio_loss_mask.tolist(), [[0.0, 0.0, 1.0]])
        self.assertEqual(transformer.last_call["audio_timestep"].flatten().tolist(), [0.0, 0.0, 500.0])
        self.assertTrue(torch.equal(transformer.last_call["audio_hidden_states"][:, :2], audio_latents[:, :2]))

    def test_ltx_frozen_audio_a2v_uses_clean_audio_without_audio_loss(self):
        audio_latents = torch.tensor([[[10.0], [20.0], [30.0]]])
        batch, transformer = self._run_fake_ltx_prediction(
            ltx_strategy={
                "video": {"is_generated": True, "conditions": []},
                "audio": {"is_generated": False, "conditions": []},
            },
            do_audio=True,
            audio_latents=audio_latents,
        )

        self.assertIsNone(batch.audio_target)
        self.assertIsNone(batch.audio_pred)
        self.assertIsNone(batch.audio_loss_mask)
        self.assertTrue(torch.equal(transformer.last_call["audio_hidden_states"], audio_latents))
        self.assertEqual(transformer.last_call["audio_timestep"].flatten().tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(transformer.last_call["audio_sigma"].tolist(), [0.0])

    def _trainer_for_loss_tests(self):
        trainer = object.__new__(SDTrainer)
        trainer.train_config = TrainConfig(noise_scheduler="flowmatch")
        trainer.device_torch = torch.device("cpu")
        trainer.sd = types.SimpleNamespace(
            is_flow_matching=True,
            prediction_type="epsilon",
            noise_scheduler=types.SimpleNamespace(),
        )
        trainer.adapter = None
        trainer.dfe = None
        trainer.snr_gos = None
        trainer.apply_model_loss_weight = lambda **kwargs: kwargs["loss"]
        trainer._record_tensor_stats = lambda *args, **kwargs: None
        trainer._record_monitor_metric = lambda *args, **kwargs: None
        return trainer

    def test_video_loss_mask_excludes_conditioned_tokens(self):
        trainer = self._trainer_for_loss_tests()
        batch = types.SimpleNamespace(
            latents=torch.zeros((1, 1, 2, 1, 1)),
            loss_multiplier_list=[1.0],
            mask_tensor=None,
            audio_pred=None,
            audio_target=None,
            video_loss_mask=torch.tensor([[[[[0.0]], [[1.0]]]]]),
            get_is_reg_list=lambda: [False],
        )

        loss = trainer.calculate_loss(
            noise_pred=torch.tensor([[[[[10.0]], [[2.0]]]]]),
            noise=torch.zeros((1, 1, 2, 1, 1)),
            noisy_latents=torch.zeros((1, 1, 2, 1, 1)),
            timesteps=torch.tensor([500.0]),
            batch=batch,
            mask_multiplier=torch.ones((1, 1, 1, 1)),
        )

        self.assertAlmostEqual(loss.item(), 4.0, places=5)

    def test_audio_loss_mask_excludes_conditioned_tokens(self):
        trainer = self._trainer_for_loss_tests()
        batch = types.SimpleNamespace(
            latents=torch.zeros((1, 1, 1, 1, 1)),
            loss_multiplier_list=[1.0],
            mask_tensor=None,
            audio_pred=torch.tensor([[[10.0], [2.0]]]),
            audio_target=torch.zeros((1, 2, 1)),
            audio_loss_mask=torch.tensor([[0.0, 1.0]]),
            video_loss_mask=None,
            get_is_reg_list=lambda: [False],
        )

        loss = trainer.calculate_loss(
            noise_pred=torch.zeros((1, 1, 1, 1, 1)),
            noise=torch.zeros((1, 1, 1, 1, 1)),
            noisy_latents=torch.zeros((1, 1, 1, 1, 1)),
            timesteps=torch.tensor([500.0]),
            batch=batch,
            mask_multiplier=torch.ones((1, 1, 1, 1)),
        )

        self.assertAlmostEqual(loss.item(), 4.0, places=5)


if __name__ == "__main__":
    unittest.main()
