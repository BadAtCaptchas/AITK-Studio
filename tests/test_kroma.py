"""Weight-free Kroma/Krea regression tests using the real model and LoRA code."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from safetensors.torch import load_file, save_file

import extensions_built_in.diffusion_models  # exercise registry initialization
from extensions_built_in.diffusion_models.krea2 import Krea2Model
from extensions_built_in.diffusion_models.krea2.krea2 import _load_mmdit_state_dict
from extensions_built_in.diffusion_models.krea2.src.mmdit import SingleMMDiTConfig, SingleStreamDiT
from extensions_built_in.diffusion_models.krea2.src.pipeline import Krea2Pipeline, predict_velocity, timesteps
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import ModelConfig, NetworkConfig, SampleConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.util.quantize import is_quantized_tensor


MODULE = 'extensions_built_in.diffusion_models.krea2.krea2'
PIPELINE = 'extensions_built_in.diffusion_models.krea2.src.pipeline'


def tiny_config():
    return SingleMMDiTConfig(features=64, tdim=16, txtdim=32, heads=4,
                            kvheads=2, multiplier=1, layers=1, patch=2,
                            channels=2, txtheads=2, txtkvheads=2, txtlayers=2)


def holder():
    return Krea2Model('cpu', ModelConfig(arch='krea2', name_or_path='unused'), dtype='fp32')


def network(model, base, exclude=True):
    config = NetworkConfig(type='lora', linear=2, linear_alpha=2, transformer_only=False)
    result = LoRASpecialNetwork(
        text_encoder=None, unet=model, lora_dim=2, alpha=2,
        train_unet=True, train_text_encoder=False, network_config=config,
        is_transformer=True, base_model=base, target_lin_modules=['SingleStreamDiT'],
        transformer_only=False, ignore_if_contains=['txtfusion.', 'txtmlp'] if exclude else [],
    )
    result.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    result.force_to('cpu', dtype=torch.float32)
    result.is_active = True
    result._update_torch_multiplier()
    return result


class KromaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_checkpoint_sources_and_offline_errors(self):
        with TemporaryDirectory() as directory:
            local = Path(directory) / 'teacher.safetensors'
            save_file({'weight': torch.ones(2)}, str(local))
            for source, filename in [(str(local), None), (directory, None), (directory, local.name)]:
                self.assertEqual(_load_mmdit_state_dict(source, filename)['weight'].tolist(), [1, 1])
            with patch(f'{MODULE}.huggingface_hub.hf_hub_download', return_value=str(local)) as download:
                _load_mmdit_state_dict('lodestones/Kroma', local.name)
                self.assertEqual(download.call_args.kwargs['filename'], local.name)
                self.assertEqual(download.call_args.kwargs['repo_id'], 'lodestones/Kroma')
            save_file({'weight': torch.ones(2)}, str(Path(directory) / 'base.safetensors'))
            with self.assertRaisesRegex(FileNotFoundError, 'checkpoint_filename'):
                _load_mmdit_state_dict(directory, None)
        with patch(f'{MODULE}.huggingface_hub.hf_hub_download', side_effect=EntryNotFoundError('missing')):
            with self.assertRaisesRegex(FileNotFoundError, 'checkpoint_filename'):
                _load_mmdit_state_dict('lodestones/Kroma', 'missing.safetensors')
        with patch(f'{MODULE}.huggingface_hub.hf_hub_download', side_effect=LocalEntryNotFoundError('offline')):
            with patch(f'{MODULE}.is_hf_offline_mode', return_value=True):
                with self.assertRaisesRegex(RuntimeError, 'offline mode'):
                    _load_mmdit_state_dict('lodestones/Kroma', 'teacher.safetensors')

    def test_example_configs_use_explicit_empty_negative_prompts(self):
        root = Path(__file__).resolve().parents[1]
        for variant in ['base', 'teacher', 'turbo_opd']:
            example = root / f'config/examples/train_lora_kroma_{variant}_16gb.yaml'
            process = yaml.safe_load(example.read_text(encoding='utf-8'))['config']['process'][0]
            model = ModelConfig(**process['model'])
            sample = SampleConfig(**process['sample'])
            self.assertEqual(model.arch, 'krea2')
            self.assertTrue(model.model_kwargs['checkpoint_filename'].endswith('.safetensors'))
            self.assertEqual(sample.neg, '')
            self.assertEqual(sample.guidance_scale, 0 if variant == 'turbo_opd' else 4)

    def test_velocity_target_and_timestep_convention(self):
        base = holder()
        base.model = SingleStreamDiT(tiny_config())
        noise = torch.randn(1, 2, 4, 4)
        clean = torch.randn_like(noise)
        target = base.get_loss_target(noise=noise, batch=SimpleNamespace(latents=clean))
        torch.testing.assert_close(target, noise - clean)
        embeds = AdvancedPromptEmbeds(text_embeds=[torch.zeros(3, 64)])
        with patch(f'{MODULE}.predict_velocity', return_value=noise) as predict:
            base.get_noise_prediction(noise, torch.tensor([750]), embeds)
            torch.testing.assert_close(predict.call_args.args[2], torch.tensor([0.75]))

    def test_strict_loading_and_precision(self):
        original = SingleStreamDiT(tiny_config()).state_dict()
        for mixed in [False, True]:
            weights = {k: v.to(torch.bfloat16 if mixed and k.endswith('weight') else torch.float32)
                       for k, v in original.items()}
            loaded = SingleStreamDiT.load_from_state_dict(weights, torch.bfloat16, config=tiny_config())
            self.assertTrue(all(p.dtype == torch.bfloat16 and not p.is_meta for p in loaded.parameters()))
        for kind in ['shape', 'missing', 'extra']:
            weights = dict(original)
            if kind == 'shape':
                weights['first.weight'] = torch.zeros(1)
            elif kind == 'missing':
                del weights['first.weight']
            else:
                weights['unexpected.weight'] = torch.zeros(1)
            with self.assertRaises(RuntimeError):
                SingleStreamDiT.load_from_state_dict(weights, torch.float32, config=tiny_config())

    def test_quantization_hooks_reach_post_load_and_keep_sensitive_layers(self):
        model = SingleStreamDiT(tiny_config())
        with patch('toolkit.util.quantize.quantize_module') as quantize:
            model.aitk_post_load(qtype='float8', device='cpu', dtype=torch.float32)
            self.assertEqual(quantize.call_args.kwargs['block_names'], ['blocks'])
            self.assertIn('txtfusion.projector', quantize.call_args.kwargs['exclude'])
        model = SingleStreamDiT(tiny_config())
        model.aitk_post_load(qtype='float8', device='cpu', dtype=torch.float32)
        self.assertTrue(is_quantized_tensor(model.blocks[0].attn.wq.weight))
        self.assertFalse(is_quantized_tensor(model.first.weight))
        self.assertFalse(is_quantized_tensor(model.txtfusion.projector.weight))
        self.assertFalse(is_quantized_tensor(model.txtmlp[1].weight))
        self.assertEqual(holder().get_quantization_exclude_modules(), model.get_quantization_exclude_modules())

    def test_lora_backward_checkpointing_exclusions_and_export_reload(self):
        torch.manual_seed(42)
        model = SingleStreamDiT(tiny_config())
        model.requires_grad_(False)
        original = copy.deepcopy(model.state_dict())
        base = holder()
        adapter = network(model, base)
        self.assertTrue(adapter.unet_loras)
        self.assertFalse(any('txtfusion' in m.lora_name or 'txtmlp' in m.lora_name for m in adapter.unet_loras))
        model.enable_gradient_checkpointing()
        x = torch.randn(1, 2, 4, 4)
        context = torch.randn(1, 3, 64)
        mask = torch.ones(1, 3, dtype=torch.bool)
        t = torch.tensor([0.5])
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
        before = copy.deepcopy(adapter.state_dict())
        target = torch.randn_like(x)
        loss = (predict_velocity(model, x, t, context, mask) - target).square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in adapter.state_dict().items()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)
        with torch.no_grad():
            prediction = predict_velocity(model, x, t, context, mask)
        with TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'adapter.safetensors')
            adapter.save_weights(filename, dtype=torch.float32)
            saved = load_file(filename)
            self.assertTrue(all(k.startswith('diffusion_model.') for k in saved))
            self.assertTrue(all(k.endswith(('lora_A.weight', 'lora_B.weight')) for k in saved))
            restored = SingleStreamDiT(tiny_config())
            restored.load_state_dict(original)
            restored.requires_grad_(False)
            restored_adapter = network(restored, base)
            restored_adapter.load_weights(filename)
            with torch.no_grad():
                torch.testing.assert_close(predict_velocity(restored, x, t, context, mask), prediction)
        included = network(SingleStreamDiT(tiny_config()), base, exclude=False)
        self.assertTrue(any('txtfusion' in m.lora_name for m in included.unet_loras))
        self.assertTrue(any('txtmlp' in m.lora_name for m in included.unet_loras))

    def test_guidance_and_fixed_turbo_schedule(self):
        a = timesteps(1024, 10, 256, 6400, mu=1.15)
        self.assertEqual(a, timesteps(4096, 10, 256, 6400, mu=1.15))
        self.assertEqual((a[0], a[-1]), (1.0, 0.0))
        self.assertTrue(all(x > y for x, y in zip(a, a[1:])))
        self.assertNotEqual(timesteps(1024, 10, 256, 6400), timesteps(4096, 10, 256, 6400))
        for guidance, expected, calls in [(0, -2, 1), (0.5, -2.5, 2), (4, -6, 2)]:
            captured = []
            def decode(latents, **_kwargs):
                captured.append(latents.clone())
                return torch.zeros(1, 3, 16, 16)
            base = SimpleNamespace(device_torch=torch.device('cpu'), torch_dtype=torch.float32,
                                   transformer=SimpleNamespace(config=SimpleNamespace(channels=2)),
                                   patch_size=2, vae_scale_factor=8, kv_cache=False,
                                   model_config=SimpleNamespace(model_kwargs={'schedule_mu': 1.15}),
                                   decode_latents=decode)
            embeds = AdvancedPromptEmbeds(text_embeds=[torch.zeros(1, 64)])
            with patch(f'{PIPELINE}.predict_velocity', side_effect=[torch.full((1, 2, 2, 2), 2.), torch.ones(1, 2, 2, 2)]) as predict:
                Krea2Pipeline(base)(embeds, embeds, height=16, width=16, num_inference_steps=1,
                                    guidance_scale=guidance, latents=torch.zeros(1, 2, 2, 2))
                self.assertEqual(predict.call_count, calls)
            torch.testing.assert_close(captured[0], torch.full_like(captured[0], expected))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for Float8 offload backward')
    def test_cuda_float8_offload_backward(self):
        model = SingleStreamDiT(tiny_config()).to(dtype=torch.bfloat16)
        model.aitk_post_load(qtype='float8', offload=1.0, device='cpu',
                            quantize_device='cuda:0', dtype=torch.bfloat16)
        base = holder()
        adapter = network(model, base)
        adapter.force_to('cuda:0', dtype=torch.float32)
        adapter._update_torch_multiplier()
        model.enable_gradient_checkpointing()
        prediction = predict_velocity(
            model, torch.randn(1, 2, 4, 4, device='cuda', dtype=torch.bfloat16),
            torch.tensor([0.5], device='cuda'),
            torch.randn(1, 3, 64, device='cuda', dtype=torch.bfloat16),
            torch.ones(1, 3, device='cuda', dtype=torch.bool),
        )
        loss = prediction.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters()))

    def test_assistant_is_removed_for_sampling_and_not_exported(self):
        torch.manual_seed(12)
        model = SingleStreamDiT(tiny_config())
        original = copy.deepcopy(model.state_dict())
        base = holder()
        source = network(model, base, exclude=False)
        with torch.no_grad():
            for module in source.unet_loras:
                module.lora_up.weight.normal_(std=0.001)
        # The shipped training adapter uses only repeated blocks.
        weights = {k: v for k, v in source.get_state_dict(dtype=torch.float32).items() if 'blocks.' in k}
        with TemporaryDirectory() as directory:
            filename = str(Path(directory) / 'assistant.safetensors')
            save_file(weights, filename)
            fresh = SingleStreamDiT(tiny_config())
            fresh.load_state_dict(original)
            base.model_config.assistant_lora_path = filename
            base.load_training_adapter(fresh)
            assistant = base.assistant_lora
            self.assertFalse(assistant.is_active)
            self.assertTrue(base.invert_assistant_lora)
            self.assertEqual(assistant.multiplier, -1.0)
            x = torch.randn(1, 2, 4, 4)
            context = torch.randn(1, 3, 64)
            mask = torch.ones(1, 3, dtype=torch.bool)
            t = torch.tensor([0.5])
            original_model = SingleStreamDiT(tiny_config())
            original_model.load_state_dict(original)
            with torch.no_grad():
                baseline = predict_velocity(original_model, x, t, context, mask)
                merged = predict_velocity(fresh, x, t, context, mask)
                assistant.is_active = True
                assistant._update_torch_multiplier()
                removed = predict_velocity(fresh, x, t, context, mask)
            self.assertFalse(torch.allclose(merged, baseline))
            torch.testing.assert_close(removed, baseline, rtol=1e-4, atol=1e-5)
            assistant.is_active = False
            trainable = network(fresh, base)
            exported = trainable.get_state_dict(dtype=torch.float32)
            self.assertTrue(all('txtfusion' not in k for k in exported))
            self.assertTrue(all(torch.count_nonzero(v) == 0 for k, v in exported.items() if k.endswith('lora_B.weight')))


if __name__ == '__main__':
    unittest.main()
