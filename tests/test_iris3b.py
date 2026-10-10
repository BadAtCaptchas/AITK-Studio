"""Iris integration checks using real dependencies and tiny CPU models."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from extensions_built_in.diffusion_models import Iris3BModel
from extensions_built_in.diffusion_models.iris3b.iris3b import _resolve_checkpoint
from extensions_built_in.diffusion_models.iris3b.src.dit import IrisConfig, IrisDiT
from extensions_built_in.diffusion_models.iris3b.src.pipeline import Iris3BPipeline, time_grid
from toolkit.config_modules import ModelConfig
from toolkit.model_registry import resolve_model
from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLTextEncoder
from toolkit.prompt_utils import PromptEmbeds


def tiny_iris():
    return IrisDiT(IrisConfig.from_dict(dict(
        hidden_size=32, depth=2, dual_depth=1, num_heads=4, num_kv_heads=2,
        patch_size=2, text_dim=16, text_len=4, text_lap_num_layers=2,
        text_lap_num_heads=4,
        pixel=dict(hidden_size=4, attn_hidden_size=32, num_heads=4, depth=1),
    )))


class IrisTests(unittest.TestCase):
    def test_lazy_registration(self):
        self.assertIs(resolve_model('iris3b'), Iris3BModel)

    def test_reference_grid_and_training_conditioning(self):
        for shift in (1.0, 4.0, 8.0):
            scheduler = Iris3BModel.get_train_scheduler(shift)
            times = scheduler.set_train_timesteps(1000, 'cpu', timestep_type='shift')
            base = 1 - torch.linspace(1, .001, 1000, dtype=torch.float64)
            reference = (shift * base / (1 + (shift - 1) * base)).flip(0)
            torch.testing.assert_close(scheduler.sigmas[:-1], reference.float(), rtol=0, atol=0)
            torch.testing.assert_close(times, (reference * 1000).float(), rtol=0, atol=0)
            torch.testing.assert_close(scheduler.training_model_times(times), (reference * 1000).long().float(), rtol=0, atol=0)
            self.assertEqual(times[-1].item(), 0)
            self.assertLess(times[0].item(), 1000)
            # Sigma lookup must remain continuous, independent of integer model time.
            torch.testing.assert_close(scheduler.get_sigmas(times, 1, torch.float32, 'cpu'), reference.float())
        scheduler.set_train_timesteps(1000, 'cpu', timestep_type='linear')
        self.assertEqual(scheduler.timesteps[0].item(), 1000)

    def test_x_training_keeps_continuous_sigma_and_integer_conditioning(self):
        holder = Iris3BModel('cpu', ModelConfig(arch='iris3b', name_or_path='test'), dtype='fp32')
        holder.prediction = 'x'
        holder.noise_scheduler = holder.get_train_scheduler()
        times = holder.noise_scheduler.set_train_timesteps(1000, 'cpu')
        seen = []
        class CleanModel(torch.nn.Module):
            device = torch.device('cpu')
            def forward(self, x, t, y, y_mask):
                seen.append(t)
                return torch.full_like(x, .25)
        holder.model = CleanModel()
        embeds = PromptEmbeds(torch.zeros(1, 4, 32))
        embeds.attention_mask = torch.ones(1, 4, dtype=torch.bool)
        timestep = times[500:501]
        velocity = holder.get_noise_prediction(torch.ones(1, 3, 2, 2), timestep, embeds)
        torch.testing.assert_close(seen[0], holder.noise_scheduler.training_model_times(timestep))
        torch.testing.assert_close(velocity, torch.full_like(velocity, .75 / (timestep.item() / 1000)))

    def test_tiny_checkpoint_and_checkpointed_gradients(self):
        model = tiny_iris()
        reloaded = IrisDiT.load_from_state_dict(model.state_dict(), dtype=torch.float32, config=model.config)
        x = torch.randn(2, 3, 8, 8, requires_grad=True)
        text = torch.randn(2, 4, 32)
        mask = torch.ones(2, 4, dtype=torch.bool)
        times = torch.tensor([100., 500.])
        expected = model(x, times, text, y_mask=mask)
        reloaded.enable_gradient_checkpointing()
        actual = reloaded(x, times, text, y_mask=mask)
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(actual.shape, x.shape)
        self.assertTrue(any(p.grad is not None for p in reloaded.blocks.parameters()))

    def test_embedding_cache_changes_with_encoder_configuration(self):
        holder = Iris3BModel('cpu', ModelConfig(arch='iris3b', name_or_path='speridlabs/iris-3b'), dtype='fp32')
        initial = holder.get_text_embedding_space_version()
        for attr, value in [('text_len', 301), ('text_hidden_layers', (1, 2)), ('te_config_repo', 'another/encoder')]:
            old = getattr(holder, attr)
            setattr(holder, attr, value)
            self.assertNotEqual(initial, holder.get_text_embedding_space_version())
            setattr(holder, attr, old)
        holder.model_config.te_name_or_path = 'local-encoder.safetensors'
        self.assertNotEqual(initial, holder.get_text_embedding_space_version())

    def test_remote_subfolder_uses_posix_filename(self):
        with patch('extensions_built_in.diffusion_models.iris3b.iris3b.huggingface_hub.hf_hub_download', return_value='downloaded') as download:
            _resolve_checkpoint('vendor/repo/nested/model.safetensors')
        self.assertEqual([c.kwargs['filename'] for c in download.call_args_list], ['nested/model.safetensors', 'nested/config.yaml'])

    def test_qwen_checkpoint_aliases_and_existing_head(self):
        tensor = torch.ones(2, 2)
        old = {'model.layers.0.weight': tensor, 'model.embed_tokens.weight': tensor, 'visual.patch.weight': tensor}
        converted = Qwen3VLTextEncoder.convert_state_dict_on_load(old)
        self.assertIn('model.language_model.layers.0.weight', converted)
        self.assertIn('model.visual.patch.weight', converted)
        self.assertIs(converted['lm_head.weight'], tensor)
        self.assertIn('model.layers.0.weight', old)
        head = torch.zeros(2, 2)
        native = {'model.language_model.embed_tokens.weight': tensor, 'lm_head.weight': head}
        result = Qwen3VLTextEncoder.convert_state_dict_on_load(native)
        self.assertEqual(set(native), set(result))
        self.assertIs(result['lm_head.weight'], head)

    def test_x_prediction_sampling_uses_clean_output_at_low_sigma(self):
        # Training clamps conversion to velocity at small sigma. Sampling must
        # use x0 directly, as the reference solver does, without that clamp.
        embeds = PromptEmbeds(torch.zeros(1, 4, 32))
        embeds.attention_mask = torch.ones(1, 4, dtype=torch.bool)
        holder = SimpleNamespace(
            device_torch=torch.device('cpu'), torch_dtype=torch.float32,
            transformer=SimpleNamespace(in_channels=3), flow_shift=.01,
            num_train_timesteps=1000, prediction='x',
            model=lambda x, t, y, y_mask: torch.full_like(x, .25),
            decode_latents=lambda x, **kwargs: x,
        )
        self.assertLess(time_grid(3, .01)[-2], .05)
        image = Iris3BPipeline(holder)(embeds, None, height=2, width=2,
            num_inference_steps=3, latents=torch.zeros(1, 3, 2, 2))[0]
        self.assertEqual(image.getpixel((0, 0)), (159, 159, 159))


if __name__ == '__main__':
    unittest.main()
