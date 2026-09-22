"""Weight-free Ming adapter, AITK LoRA and output integration checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
import torch

from extensions_built_in.diffusion_models.ming_image import MingImageDesignModel, MingImageDesignLayerModel
from testing.test_ming_components import tiny_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import (DatasetConfig, GenerateImageConfig, ModelConfig, NetworkConfig,
                                    SampleConfig, SaveConfig, TrainConfig, validate_configs)
from toolkit.layered_output import LayeredImageOutput
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.model_registry import resolve_model


def adapter(layered=False):
    cls = MingImageDesignLayerModel if layered else MingImageDesignModel
    return cls('cpu', ModelConfig(name_or_path='unused', arch=cls.arch, quantize=False), dtype='float32')


def network(holder):
    result = LoRASpecialNetwork(text_encoder=[], unet=holder.model, train_text_encoder=False,
        train_unet=True, lora_dim=2, alpha=2, target_lin_modules=holder.target_lora_modules,
        network_config=NetworkConfig(type='lora', linear=2, linear_alpha=2),
        is_transformer=True, base_model=holder)
    result.apply_to([], holder.model, False, True)
    result.force_to('cpu', dtype=torch.float32)
    result._update_torch_multiplier()
    return result


class MingIntegrationTests(unittest.TestCase):
    def test_registry_and_public_imports(self):
        import extensions_built_in.diffusion_models as registry
        self.assertIs(resolve_model('ming_image_design'), MingImageDesignModel)
        self.assertIs(registry.MingImageDesignLayerModel, MingImageDesignLayerModel)

    def test_training_shift_applied_once_and_independent_of_frames(self):
        scheduler = MingImageDesignModel.get_train_scheduler()
        times = scheduler.set_train_timesteps(3, 'cpu', latents=torch.zeros(1, 16, 3, 4, 4))
        sigma = torch.linspace(1, .001, 3)
        torch.testing.assert_close(times, 1000 * 6 * sigma / (1 + 5 * sigma))
        self.assertAlmostEqual(float(times[1]), 857.3876, places=3)

    def test_precise_timestep_sign_and_reference_exclusion(self):
        holder = adapter(True)
        holder.model = tiny_model()
        embeds = AdvancedPromptEmbeds(text_embeds=[torch.randn(5, 8)],
            direct_embeds=[torch.randn(3, 16)], reference_latents=[torch.randn(4, 4, 4)])
        value, time = torch.randn(1, 4, 3, 4, 4), torch.tensor([643.271])
        recorded = []
        hook = holder.model.register_forward_pre_hook(lambda module, args, kwargs: recorded.append(kwargs['t']), with_kwargs=True)
        prediction = holder.get_noise_prediction(value, time, embeds)
        hook.remove()
        expected = holder.model(list(value), 1 - time / 1000, embeds.text_embeds,
            ref_x=[embeds.reference_latents[0][:, None]], cap_feats_2=embeds.direct_embeds,
            return_dict=False)[0][0]
        torch.testing.assert_close(prediction[0], -expected)
        torch.testing.assert_close(recorded[0], 1 - time / 1000, rtol=0, atol=0)
        noise, clean = torch.randn_like(value), torch.randn_like(value)
        torch.testing.assert_close(holder.get_loss_target(noise=noise, batch=SimpleNamespace(latents=clean)), noise-clean)

    def test_aitk_lora_step_save_reload_resume_preserves_frozen_projectors(self):
        torch.manual_seed(13)
        holder = adapter(True)
        holder.model = tiny_model().requires_grad_(False)
        original = {key: value.clone() for key, value in holder.model.state_dict().items()}
        holder.model.enable_gradient_checkpointing()
        lora = network(holder)
        holder.network = lora
        self.assertTrue(lora.unet_loras)
        self.assertFalse(any('embedder' in item.lora_name or 'final_layer' in item.lora_name for item in lora.unet_loras))
        args = ([torch.randn(4, 3, 4, 4)], torch.tensor([.375]), [torch.randn(4, 8)])
        optimizer = torch.optim.AdamW(lora.parameters(), lr=1e-3)
        with lora:
            holder.model(*args, return_dict=False)[0][0].square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora.parameters()))
            optimizer.step()
            expected = holder.model(*args, return_dict=False)[0][0].detach()
        for key, value in holder.model.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'adapter.safetensors')
            lora.save_weights(path, dtype=torch.float32)
            restored_holder = adapter(True)
            restored_holder.model = tiny_model().requires_grad_(False)
            restored_holder.model.load_state_dict(original)
            restored = network(restored_holder)
            restored.load_weights(path)
            resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            resumed_optimizer.load_state_dict(optimizer.state_dict())
            with restored:
                actual = restored_holder.model(*args, return_dict=False)[0][0]
                torch.testing.assert_close(actual, expected)
                actual.square().mean().backward()
                resumed_optimizer.step()
            self.assertTrue(all(int(state['step']) == 2 for state in resumed_optimizer.state.values()))

    def test_sample_context_and_integer_layer_count(self):
        holder = adapter(True)
        item = GenerateImageConfig(prompt='poster', width=512, height=512, num_layers=3,
            ctrl_img='reference.png', output_path='unused.png')
        holder.prepare_sample_image_config_for_encoding(item)
        holder.prepare_sample_prompt_context(item)
        self.assertEqual(item.prompt, 'Decompose the image into 3 layers.\nposter')
        for invalid in (True, 0, 33, 2.5, '2'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                SampleConfig(num_layers=invalid)
        item.negative_prompt = 'bad'
        with self.assertRaisesRegex(ValueError, 'negative'):
            holder.prepare_sample_image_config_for_encoding(item)

    def test_atomic_layer_output_is_rgba_png_not_animation(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new('RGBA', (64, 64), (255, 40, 10, 90))
            config = GenerateImageConfig(prompt='layers', num_layers=2, width=64, height=64,
                output_path=str(Path(directory) / 'preview.png'))
            config.save_image_atomic(LayeredImageOutput(image, [image, image]))
            manifest = json.loads((Path(directory) / 'preview.png.layers.json').read_text())
            self.assertEqual(manifest['order'], 'bottom-to-top')
            self.assertEqual(len(manifest['layers']), 2)
            for name in ['preview.png', *manifest['layers']]:
                with Image.open(Path(directory) / name) as actual:
                    self.assertEqual(actual.mode, 'RGBA')
                    self.assertFalse(getattr(actual, 'is_animated', False))
                    self.assertEqual(actual.getpixel((0, 0)), image.getpixel((0, 0)))

    def test_examples_pass_python_config_contract(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        for variant in ('design', 'design_layer'):
            raw = yaml.safe_load((root / f'config/examples/train_lora_ming_{variant}_16gb_experimental.yaml').read_text())['config']['process'][0]
            datasets = [DatasetConfig(**{**item, 'resolution': item['resolution'][0]}) for item in raw['datasets']]
            validate_configs(TrainConfig(**raw['train']), ModelConfig(**raw['model']),
                SaveConfig(**raw['save']), datasets, NetworkConfig(**raw['network']))

    def test_invalid_training_modes_fail_before_loading_weights(self):
        model = ModelConfig(arch='ming_image_design_layer', name_or_path='unused')
        defaults = dict(noise_scheduler='flowmatch', timestep_type='linear')
        dataset = DatasetConfig(type='layered_image', resolution=512, caption_dropout_rate=0)
        invalid = [dict(train_text_encoder=True), dict(batch_size=2), dict(standardize_images=True),
                   dict(img_multiplier=.8), dict(do_cfg=True), dict(timestep_type='sigmoid'),
                   dict(validation_config={})]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_configs(TrainConfig(**{**defaults, **values}), model, SaveConfig(),
                                 [dataset], NetworkConfig())
        for net in (None, NetworkConfig(type='dora'), NetworkConfig(all_layers=True),
                    NetworkConfig(network_kwargs={'full_if_contains': ['attention']})):
            with self.subTest(network=net), self.assertRaises(ValueError):
                validate_configs(TrainConfig(**defaults), model, SaveConfig(), [dataset], net)
        for fields in (dict(batch_size=2), dict(num_frames=3), dict(do_audio=True),
                       dict(standardize_images=True), dict(resolution=520)):
            invalid_dataset = DatasetConfig(**{'type': 'layered_image', 'resolution': 512, **fields})
            with self.subTest(dataset=fields), self.assertRaises(ValueError):
                validate_configs(TrainConfig(**defaults), model, SaveConfig(), [invalid_dataset], NetworkConfig())


if __name__ == '__main__':
    unittest.main()
