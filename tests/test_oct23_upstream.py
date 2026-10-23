"""CPU behavioral coverage for the selective October 23 integrations."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image
from safetensors.torch import load_file

from extensions_built_in.diffusion_models import Fibo2Model, MossImageModel
from extensions_built_in.diffusion_models.fibo2.src.transformer import Fibo2Params, Fibo2Transformer
from extensions_built_in.diffusion_models.fibo2.src.pipeline import fibo2_sigmas, stack_text_embeds, Fibo2Pipeline
from extensions_built_in.diffusion_models.fibo2.src.references import prepare_references, vlm_image
from extensions_built_in.diffusion_models.fibo2.src.text_encoder import _bundles, BUNDLE_LAYERS, generation_text, NULL_CAPTION
from extensions_built_in.diffusion_models.fibo2.src import checkpoints
from toolkit.config_modules import ModelConfig, GenerateImageConfig, SampleConfig, TrainConfig, SaveConfig, NetworkConfig, validate_configs
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.models.v2._mixin import OstrisModelMixin
from toolkit.memory_management.manager import MemoryManager
from toolkit.optimizer import get_optimizer
from toolkit.optimizers.automagic3 import Automagic3
from toolkit.optimizers.automagic4 import Automagic4


def tiny_transformer():
    return Fibo2Transformer(Fibo2Params(dim=32, n_heads=4, n_layers=2,
        n_refiner_layers=1, injection_layers=(0,), cap_feat_dim=8,
        perceiver_layers=1, max_gist_tokens=4, min_gist_tokens=2, gist_step=2,
        ffn_dim=48, in_channels=4, axes_dims=(2, 2, 4)))


class October23Tests(unittest.TestCase):
    def test_fibo_forward_checkpoint_gradients_and_save_load(self):
        torch.manual_seed(2)
        model = tiny_transformer()
        x = torch.randn(2, 4, 2, 3, requires_grad=True)
        context = torch.randn(2, 5, 16)
        mask = torch.tensor([[True]*5, [True]*3+[False]*2])
        refs = [torch.randn(2, 4, 2, 2)]
        output = model(x, torch.tensor([500., 800.]), context, mask, refs)
        self.assertEqual(output.shape, x.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(model.layers[0].attention.qkv.weight.grad.abs().sum(), 0)
        model.zero_grad()
        model.enable_gradient_checkpointing()
        again = model(x, torch.tensor([500., 800.]), context, mask, refs)
        torch.testing.assert_close(output, again)
        again.sum().backward()
        with tempfile.TemporaryDirectory() as folder:
            filename = str(Path(folder) / 'model.safetensors')
            model.save_model(filename, dtype=torch.float32, quantized=False)
            restored = Fibo2Transformer.load_model(filename, dtype=torch.float32, config=model.params, quantize_on_load=False)
            torch.testing.assert_close(restored(x.detach(), torch.tensor([500., 800.]), context, mask, refs), output.detach())

    def test_fibo_checkpoint_shape_inference_without_external_config(self):
        params = Fibo2Params(dim=128, n_heads=1, n_layers=1, n_refiner_layers=1,
            injection_layers=(0,), cap_feat_dim=8, perceiver_layers=1,
            max_gist_tokens=256, ffn_dim=48, in_channels=4)
        model = Fibo2Transformer(params)
        with tempfile.TemporaryDirectory() as folder:
            filename = str(Path(folder) / 'inferred.safetensors')
            model.save_model(filename, dtype=torch.float32, quantized=False)
            restored = Fibo2Transformer.load_model(filename, dtype=torch.float32, quantize_on_load=False)
            self.assertEqual(restored.params, params)
            x, context, t = torch.randn(1, 4, 2, 2), torch.randn(1, 3, 16), torch.tensor([250.])
            torch.testing.assert_close(restored(x, t, context), model(x, t, context))

    def test_fibo_lora_hooks_have_gradients_with_frozen_base(self):
        from toolkit.lora_special import LoRASpecialNetwork
        owner = Fibo2Model('cpu', ModelConfig(arch='fibo2', name_or_path='test'), dtype='fp32')
        model = tiny_transformer().requires_grad_(False)
        owner.model = model
        network = LoRASpecialNetwork(text_encoder=None, unet=model, lora_dim=2, alpha=2,
            train_unet=True, train_text_encoder=False, network_config=NetworkConfig(type='lora'),
            is_transformer=True, base_model=owner, target_lin_modules=['Fibo2Transformer'],
            only_if_contains=['layers.0.attention.qkv'])
        network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
        network.force_to('cpu', dtype=torch.float32)
        network.is_active = True
        network._update_torch_multiplier()
        self.assertEqual(len(network.unet_loras), 1)
        model(torch.randn(1, 4, 2, 2), torch.tensor([500.]), torch.randn(1, 3, 16)).sum().backward()
        self.assertGreater(network.unet_loras[0].lora_up.weight.grad.abs().sum(), 0)
        self.assertIsNone(model.layers[0].attention.qkv.weight.grad)

    def test_fibo_sampler_cfg_and_preview_use_velocity_convention(self):
        class Transformer:
            params = SimpleNamespace(in_channels=3)
            def __call__(self, x, timestep, embeds, mask, refs):
                return torch.ones_like(x) * embeds.mean()
        decoded = []
        owner = SimpleNamespace(device_torch=torch.device('cpu'), torch_dtype=torch.float32,
            transformer=Transformer(), sample_step_hook=True, _emit_sample_step=Mock(),
            decode_latents=lambda x, **kw: decoded.append(x) or x)
        pipeline = Fibo2Pipeline(owner)
        cond = AdvancedPromptEmbeds(text_embeds=[torch.full((1, 2), 2.)])
        uncond = AdvancedPromptEmbeds(text_embeds=[torch.ones(1, 2)])
        result = pipeline(cond, uncond, height=32, width=32, num_inference_steps=3,
            guidance_scale=3., latents=torch.zeros(1, 3, 2, 2))
        torch.testing.assert_close(decoded[0], torch.full((1, 3, 2, 2), -4.))
        self.assertEqual(owner._emit_sample_step.call_count, 3)
        self.assertEqual(result[0].size, (2, 2))

    def test_fibo_text_bundles_masks_sigmas_and_reference_geometry(self):
        hidden = [torch.full((1, 3, 2), float(i+1)) for i in range(36)]
        bundles = _bundles(hidden, torch.float32)
        self.assertEqual(bundles.shape, (1, 3, len(BUNDLE_LAYERS)*3*2))
        torch.testing.assert_close(bundles, torch.ones_like(bundles))
        self.assertEqual(generation_text(''), NULL_CAPTION)
        stacked, mask = stack_text_embeds([torch.ones(2, 4), torch.ones(3, 4)], torch.device('cpu'), torch.float32)
        self.assertEqual(stacked.shape, (2, 3, 4))
        self.assertEqual(mask.tolist(), [[True, True, False], [True]*3])
        sigmas = fibo2_sigmas(8, 256)
        self.assertEqual(len(sigmas), 9)
        self.assertEqual(sigmas[0], 1)
        self.assertEqual(sigmas[-1], 0)
        self.assertTrue((sigmas[:-1] > sigmas[1:]).all())
        crops = prepare_references([torch.ones(3, 40, 80)], (64, 96), True)
        self.assertEqual(crops[0].size, (64, 96))
        self.assertTrue(all(n % 32 == 0 for n in vlm_image(crops[0]).size))

    def test_fibo_ordered_sample_references_and_cache_identity(self):
        model = Fibo2Model('cpu', ModelConfig(arch='fibo2', name_or_path='test'), dtype='fp32')
        with tempfile.TemporaryDirectory() as folder:
            filename = str(Path(folder) / 'ref.png')
            Image.new('RGB', (80, 40), 'red').save(filename)
            config = GenerateImageConfig(prompt='edit', ctrl_imgs=[filename, filename], output_path=str(Path(folder)/'out.png'))
            images = model.load_sample_control_images(config)
            self.assertEqual(len(images), 2)
            self.assertEqual(len(model._normalize_control_images(images, 1)[0]), 2)
        first = model.get_text_embedding_space_version()
        model.max_text_length += 1
        self.assertNotEqual(first, model.get_text_embedding_space_version())
        model.model_config.model_kwargs['match_target_res'] = False
        self.assertFalse(model.text_embedding_uses_target_size)
        self.assertIn('reftrained', model.get_text_embedding_space_version())
        with patch.object(checkpoints, 'download_checkpoint', return_value='file') as download:
            checkpoints.resolve_hub_file('owner/a/same.safetensors')
            one = download.call_args.args[2]
            checkpoints.resolve_hub_file('owner/b/same.safetensors')
            self.assertNotEqual(one, download.call_args.args[2])

    def test_fibo_cached_and_uncached_cfg_negatives_are_text_only(self):
        model = Fibo2Model('cpu', ModelConfig(arch='fibo2', name_or_path='test'), dtype='fp32')
        model.model = torch.nn.Linear(1, 1)
        # BaseModel.transformer is not needed by this generation hook.
        model.model.device = torch.device('cpu')
        model.load_sample_control_images = Mock(return_value=[torch.ones(3, 64, 64)])
        model._encode_crops = Mock(return_value=[torch.ones(1, 4, 4, 4)])
        negative = AdvancedPromptEmbeds(text_embeds=[torch.zeros(2, 16)])
        model.get_prompt_embeds = Mock(return_value=negative)
        pipeline = Mock(return_value=['result'])
        cfg = GenerateImageConfig(prompt='edit', negative_prompt='bad', ctrl_imgs=['ref'], width=64, height=64, output_path='out.png')
        for cache in (None, [{'conditional': 'cached'}]):
            model.sample_prompts_cache = cache
            self.assertEqual(model.generate_single_image(pipeline, cfg, None, None, torch.Generator(), {}), 'result')
            self.assertIs(pipeline.call_args.kwargs['unconditional_embeds'], negative)
            self.assertEqual(model.get_prompt_embeds.call_args.args, ('bad',))

    def test_fibo_training_guards(self):
        for setting in ('train_text_encoder', 'unload_text_encoder'):
            with self.assertRaisesRegex(ValueError, 'Fibo'):
                validate_configs(TrainConfig(**{setting: True}), ModelConfig(arch='fibo2', name_or_path='test'),
                    SaveConfig(), [], NetworkConfig(type='lora'))

    def test_moss_and_fibo_decode_preserve_input_gradients(self):
        class Vae(torch.nn.Module):
            device = torch.device('cpu')
            dtype = torch.float32
            def decode(self, x):
                return x * 3
        for cls in (MossImageModel, Fibo2Model):
            model = object.__new__(cls)
            model.vae = Vae()
            x = torch.randn(1, 4, 2, 2, requires_grad=True)
            model.decode_latents(x, device=torch.device('cpu'), dtype=torch.float32).sum().backward()
            torch.testing.assert_close(x.grad, torch.full_like(x, 3))

    def test_compiled_checkpoint_keys_round_trip_and_collision_rejection(self):
        class Model(torch.nn.Module, OstrisModelMixin):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([torch.compile(torch.nn.Linear(2, 2), backend='eager')])
        model = Model()
        with tempfile.TemporaryDirectory() as folder:
            filename = str(Path(folder) / 'compiled.safetensors')
            model.save_model(filename, quantized=False)
            saved = load_file(filename)
            self.assertEqual(set(saved), {'blocks.0.weight', 'blocks.0.bias'})
            torch.testing.assert_close(saved['blocks.0.weight'], model.blocks[0].weight)
            with patch.object(model, 'state_dict', return_value={'x': torch.ones(1), '_orig_mod.x': torch.ones(1)}):
                with self.assertRaisesRegex(ValueError, 'collision'):
                    model.save_model(filename, quantized=False)

    def test_mps_cache_release_does_not_call_unsafe_native_host_cache(self):
        for on_mps in (False, True):
            with patch('torch.backends.mps.is_available', return_value=on_mps), \
                 patch.object(torch._C, '_accelerator_emptyHostCache', create=True) as clear, \
                 patch('torch.cuda.empty_cache'), patch('gc.collect'), patch('ctypes.CDLL'):
                MemoryManager.release_cached_memory()
                self.assertEqual(clear.call_count, 0 if on_mps else 1)

    def test_mps_rotary_phases_never_request_float64(self):
        from extensions_built_in.diffusion_models.krea2.src.mmdit import rope
        class Positions:
            device = torch.device('mps')
            def __init__(self, tensor):
                self.tensor = tensor
            def to(self, dtype):
                return self.tensor.to(dtype)
            def __getitem__(self, key):
                return self.tensor[key]
        original_arange = torch.arange
        requested = []
        def arange(*args, **kwargs):
            requested.append(kwargs.get('dtype'))
            kwargs['device'] = 'cpu'
            return original_arange(*args, **kwargs)
        pos = torch.arange(3).reshape(1, 3)
        ids = torch.arange(9).reshape(3, 3)
        model = tiny_transformer()
        expected_rope = rope(pos, 8)
        expected_fibo = model._rope(ids)
        with patch('torch.arange', side_effect=arange):
            actual_rope = rope(Positions(pos), 8)
            actual_fibo = model._rope(Positions(ids))
        self.assertTrue(requested)
        self.assertNotIn(torch.float64, requested)
        torch.testing.assert_close(actual_rope, expected_rope, atol=1e-6, rtol=1e-6)
        for actual, expected in zip(actual_fibo, expected_fibo):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_fibo_reference_batch_contract(self):
        normalize = Fibo2Model._normalize_control_images
        self.assertEqual([len(row) for row in normalize(torch.ones(2, 3, 4, 4), 2)], [1, 1])
        self.assertEqual([len(row) for row in normalize(torch.ones(2, 3, 3, 4, 4), 2)], [3, 3])
        self.assertEqual(len(normalize(torch.ones(3, 4, 4), 1)[0]), 1)
        with self.assertRaises(ValueError):
            normalize(torch.ones(2, 4), 1)
        with self.assertRaisesRegex(ValueError, 'control image sets'):
            normalize(torch.ones(3, 3, 4, 4), 2)

    def test_automagic4_measurement_accumulation_resume_and_legacy_separation(self):
        p = torch.nn.Parameter(torch.ones(2, 3))
        options = {'polarity_history': 3}
        opt = get_optimizer([p], 'automagic4', 1e-3, options)
        self.assertNotIn('fused', options)
        self.assertFalse(opt.fused)
        self.assertNotIsInstance(opt, Automagic3)
        legacy = get_optimizer([torch.nn.Parameter(torch.ones(1))], 'automagic3',
            optimizer_params={'fused': False, 'min_lr': 1e-5})
        self.assertIsInstance(legacy, Automagic3)
        self.assertNotIsInstance(legacy, Automagic4)
        initial = p.detach().clone()
        (p.sum()*2).backward()
        (p.sum()*3).backward()
        torch.testing.assert_close(p.grad, torch.full_like(p, 5))
        torch.testing.assert_close(p, initial)
        opt.step()
        torch.testing.assert_close(p, initial)  # first step measures sensitivity only
        for _ in range(4):
            p.square().sum().backward()
            opt.step()
        self.assertTrue((p < initial).all())
        saved = copy.deepcopy(opt.state_dict())
        q = torch.nn.Parameter(p.detach().clone())
        other = get_optimizer([q], 'automagic4', 1e-3, options)
        other.load_state_dict(saved)
        p.square().sum().backward(); q.square().sum().backward()
        opt.step(); other.step()
        torch.testing.assert_close(p, q)
        torch.testing.assert_close(opt.state[p]['sens_ref'], other.state[q]['sens_ref'])
        with self.assertRaisesRegex(ValueError, 'fused=True'):
            get_optimizer([q], 'automagic4', optimizer_params={'fused': True})
        for kwargs in ({'step_target': 0}, {'sensitivity_quantile': 1.1}, {'min_lr': -1}, {'beta2': 1}, {'eps': 0}):
            with self.assertRaises(ValueError):
                Automagic4([q], **kwargs)

    def test_automagic4_low_precision_gradients_remain_visible_to_clipping(self):
        for dtype in (torch.float16, torch.bfloat16):
            p = torch.nn.Parameter(torch.ones(2, 3, dtype=dtype))
            opt = get_optimizer([p], 'automagic4', 1e-3)
            (p.sum()*2).backward()
            (p.sum()*3).backward()
            torch.testing.assert_close(p.grad, torch.full_like(p, 5))
            torch.nn.utils.clip_grad_norm_([p], 1.)
            opt.step()
            self.assertLess(float(opt.state[p]['sens_ref']), 1.)
            opt.zero_grad(set_to_none=True)
            self.assertIsNone(p.grad)
            self.assertFalse(hasattr(p, '_accum_grad'))

    def test_automagic4_low_precision_moments_remain_finite(self):
        for dtype in (torch.float16, torch.bfloat16):
            p = torch.nn.Parameter(torch.ones(2, 3, dtype=dtype))
            opt = get_optimizer([p], 'automagic4', 1e-3)
            for _ in range(3):
                p.grad = torch.full_like(p, 10000.)
                opt.step()
            self.assertTrue(torch.isfinite(p).all())
            self.assertTrue(torch.isfinite(opt.state[p]['sens_ref']))
            q = torch.nn.Parameter(p.detach().clone())
            restored = get_optimizer([q], 'automagic4', 1e-3)
            restored.load_state_dict(copy.deepcopy(opt.state_dict()))
            for key in ('sens_ref', 'exp_avg_sq_row', 'exp_avg_sq_col', 'lr'):
                torch.testing.assert_close(restored.state[q][key], opt.state[p][key])
            q.grad = torch.full_like(q, 10000.)
            restored.step()
            self.assertTrue(torch.isfinite(q).all())

    def test_cached_sample_geometry_includes_frames(self):
        from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
        from toolkit.prompt_utils import PromptEmbeds
        owner = SDTrainer.__new__(SDTrainer)
        owner.train_config = SimpleNamespace(disable_sampling=False)
        owner.sample_config = SampleConfig(samples=[{'prompt': 'test', 'width': 640, 'height': 384, 'num_frames': 49}])
        owner.save_root = '.'
        owner.trigger_word = None
        seen = []
        owner.sd = SimpleNamespace(sample_prompts_cache=[], encode_control_in_text_embeddings=False,
            prepare_sample_image_config_for_encoding=lambda c: seen.append((c.width, c.height, c.num_frames)),
            encode_prompt=lambda _: PromptEmbeds(torch.zeros(1, 2, 3)))
        owner.cache_sample_prompts()
        self.assertEqual(seen, [(640, 384, 49)])
