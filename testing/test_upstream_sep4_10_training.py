"""Small real-tensor regressions for the September 4-10 selective ports."""
import tempfile
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from extensions_built_in.diffusion_models.hidream.hidream_o1_model import HidreamO1Model
from toolkit.config_modules import DatasetConfig, EMAConfig, ModelConfig, TrainConfig
from toolkit.accelerator import get_accelerator
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.dto import DTO
from toolkit.ema import ExponentialMovingAverage
from toolkit.memory_management.manager import MemoryManager
from toolkit.memory_management.manager_modules import _move_own_param, _storage_device
from toolkit.optimizer import get_optimizer
from toolkit.optimizer_checkpoint import atomic_save_optimizer_checkpoint, load_optimizer_checkpoint
from toolkit.optimizers.adamconvrot import AdamConvRot
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler, force_first_timestep_indices,
)
from toolkit.timestep_weighing.default_weighing_scheme import default_weighing_scheme, x0_weighing_scheme
from toolkit.samplers.mean_flow_scheduler import MeanFlowScheduler
from testing.test_dataloader_batch_dto import make_file_item


class ConfigurationAndEMATests(unittest.TestCase):
    def test_defaults_and_invalid_probabilities(self):
        self.assertFalse(DatasetConfig().pin_memory)
        self.assertEqual(TrainConfig().first_timestep_chance, 0.0)
        self.assertEqual(EMAConfig().feedback_rate, 0.001)
        for value in (-1, 1.1, float('nan'), float('inf'), '0.2', True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    TrainConfig(first_timestep_chance=value)
                with self.assertRaises(ValueError):
                    EMAConfig(feedback_rate=value)
        with self.assertRaises(ValueError):
            DatasetConfig(pin_memory='false')

    def test_shadow_moves_toward_parameter_and_feedback_has_independent_rate(self):
        for decay in (0.9, 0.99):
            with self.subTest(decay=decay):
                p = nn.Parameter(torch.tensor([0.]))
                ema = ExponentialMovingAverage([p], decay=decay, use_feedback=True, feedback_rate=0.05)
                p.data.fill_(1.)
                ema.update()
                torch.testing.assert_close(ema.shadow_params[0], torch.tensor([1 - decay]))
                torch.testing.assert_close(p, torch.tensor([0.95]))

    def test_feedback_disabled_and_checkpoint_round_trip(self):
        p = nn.Parameter(torch.tensor([0.]))
        ema = ExponentialMovingAverage([p], decay=0.9)
        p.data.fill_(1.)
        for _ in range(20):
            ema.update()
        self.assertTrue(0 < ema.shadow_params[0].item() < 1)
        self.assertEqual(p.item(), 1.0)
        restored = ExponentialMovingAverage([p], decay=0.9)
        restored.load_state_dict(ema.state_dict())
        torch.testing.assert_close(restored.shadow_params[0], ema.shadow_params[0])

    def test_noise_probability_endpoints_and_per_sample_draws(self):
        indices = torch.full((4096,), 500, dtype=torch.long)
        state = torch.random.get_rng_state()
        self.assertIs(force_first_timestep_indices(indices, 0.0), indices)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue((force_first_timestep_indices(indices, 1.0) == 0).all())
        torch.manual_seed(5)
        sampled = force_first_timestep_indices(indices, 0.25)
        self.assertTrue(((sampled == 0) | (sampled == 500)).all())
        self.assertTrue(0.20 < (sampled == 0).float().mean().item() < 0.30)


class PinningTests(unittest.TestCase):
    def test_default_and_mixed_batches_do_not_pin(self):
        rows = [make_file_item(), make_file_item()]
        rows[0].dataset_config.pin_memory = True
        batch = DataLoaderBatchDTO(file_items=rows)
        with mock.patch.object(torch.Tensor, 'pin_memory', side_effect=AssertionError('must not pin')):
            self.assertIs(batch.pin_memory(), batch)

    @unittest.skipUnless(torch.cuda.is_available(), 'Pinned memory requires CUDA')
    def test_batch_pins_nested_payloads_without_modifying_file_cache(self):
        row = make_file_item()
        row.dataset_config.pin_memory = True
        batch = DataLoaderBatchDTO(file_items=[row])
        audio = torch.ones(1, 8)
        batch.latents = DTO(torch.ones(1, 2, 2), audio={'rows': [audio]})
        batch.audio_data = [{'waveform': audio, 'sample_rate': 16000}]
        batch.pin_memory()
        self.assertTrue(batch.tensor.is_pinned())
        self.assertTrue(batch.latents.tensor.is_pinned())
        self.assertTrue(batch.latents.get('audio')['rows'][0].is_pinned())
        self.assertTrue(batch.audio_data[0]['waveform'].is_pinned())
        self.assertFalse(row.tensor.is_pinned())
        self.assertFalse(audio.is_pinned())


def make_trainer(param, optimizer, legacy_steps=1, clip=1000.0):
    trainer = SDTrainer.__new__(SDTrainer)
    trainer.optimizer = optimizer
    trainer.params = [param]
    trainer.sd = SimpleNamespace(is_multistage=False)
    trainer.model_config = SimpleNamespace(low_vram=False)
    trainer.train_config = SimpleNamespace(gradient_accumulation_steps=legacy_steps,
                                           optimizer='adamw', max_grad_norm=clip)
    trainer.logging_config = SimpleNamespace(monitor_grad_stats=False)
    trainer.is_grad_accumulation_step = False
    trainer.steps_this_boundary = 0
    trainer.ema = trainer.adapter = trainer.embedding = None
    trainer.timer = lambda *_: nullcontext()
    trainer.cuda_memory_phase = lambda *_: nullcontext()
    trainer._record_monitor_metric = lambda *_: None
    trainer.end_of_training_loop = lambda: None
    trainer.lr_scheduler = SimpleNamespace(step=lambda: None)
    trainer.accelerator = SimpleNamespace(clip_grad_norm_=torch.nn.utils.clip_grad_norm_)

    def backward_batch(batch):
        loss = (param * batch).square().mean()
        loss.backward()
        return loss.detach()

    trainer.train_single_accumulation = backward_batch
    return trainer


class AccumulationTests(unittest.TestCase):
    def test_accelerate_environment_cannot_double_scale_toolkit_accumulation(self):
        with mock.patch.dict(os.environ, {'ACCELERATE_GRADIENT_ACCUMULATION_STEPS': '7'}), \
             mock.patch('toolkit.accelerator.global_accelerator', None):
            self.assertEqual(get_accelerator().gradient_accumulation_steps, 1)

    def run_update(self, batches, repeated=False, whole_epoch=False):
        p = nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([p], lr=0.01)
        trainer = make_trainer(p, optimizer, -1 if whole_epoch else (len(batches) if repeated else 1))
        if repeated:
            for i, batch in enumerate(batches):
                trainer.is_grad_accumulation_step = i < len(batches) - 1
                trainer.hook_train_loop(batch)
        else:
            trainer.hook_train_loop(batches)
        return p.detach(), trainer

    def test_list_and_repeated_accumulation_match_large_batch(self):
        batches = [torch.tensor([x]) for x in (1., 2., 3.)]
        expected, _ = self.run_update([torch.cat(batches)])
        for repeated in (False, True):
            actual, trainer = self.run_update(batches, repeated=repeated)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(trainer._accumulated_microbatches, 0)

    def test_whole_epoch_retains_legacy_summed_gradients(self):
        actual, _ = self.run_update([torch.tensor([1.]), torch.tensor([2.])], repeated=True, whole_epoch=True)
        torch.testing.assert_close(actual, torch.tensor([0.9]))

    def test_short_window_uses_actual_count_and_preserves_logged_loss(self):
        p = nn.Parameter(torch.tensor([1.]))
        trainer = make_trainer(p, torch.optim.SGD([p], lr=0.01), legacy_steps=8)
        trainer.is_grad_accumulation_step = True
        first = trainer.hook_train_loop(torch.tensor([1.]))
        trainer.is_grad_accumulation_step = False
        second = trainer.hook_train_loop(torch.tensor([3.]))
        torch.testing.assert_close(p, torch.tensor([0.9]))
        self.assertEqual(first['loss'], 1.0)
        self.assertEqual(second['loss'], 9.0)

    def test_adamconvrot_buffers_are_visible_to_clipping(self):
        p = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        optimizer = AdamConvRot([p], lr=0.01)
        trainer = make_trainer(p, optimizer, clip=0.1)
        norms = []
        original = optimizer.step

        def step():
            norms.append(p.grad.float().norm().item())
            return original()

        optimizer.step = step
        trainer.hook_train_loop([torch.tensor([10.]), torch.tensor([20.])])
        self.assertEqual(len(norms), 1)
        self.assertLessEqual(norms[0], 0.101)
        self.assertFalse(hasattr(p, '_accum_grad'))


class AdamConvRotTests(unittest.TestCase):
    def test_factory_defaults_and_fused_rejection(self):
        p = nn.Parameter(torch.ones(2))
        optimizer = get_optimizer([p], 'adamconvrot')
        self.assertEqual(optimizer.qtype, 'convrot8')
        self.assertEqual(optimizer.param_groups[0]['eps'], 1e-6)
        for build in (lambda: get_optimizer([p], 'adamconvrot', optimizer_params={'fused': True}),
                      lambda: TrainConfig(optimizer='adamconvrot', optimizer_params={'fused': True})):
            with self.assertRaisesRegex(ValueError, 'OOM recovery'):
                build()

    def test_fp32_fallback_matches_adamw(self):
        a = nn.Parameter(torch.linspace(-1., 1., 31))
        b = nn.Parameter(a.detach().clone())
        ours = AdamConvRot([a], lr=0.01, eps=1e-6, weight_decay=0.01)
        reference = torch.optim.AdamW([b], lr=0.01, eps=1e-6, weight_decay=0.01)
        for _ in range(3):
            for p, opt in ((a, ours), (b, reference)):
                p.square().mean().backward()
                opt.step()
                opt.zero_grad()
        torch.testing.assert_close(a, b)

    def test_reset_discards_private_gradients(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                p = nn.Parameter(torch.ones(8, dtype=dtype))
                optimizer = AdamConvRot([p], lr=0.01)
                p.float().sum().backward()
                optimizer.zero_grad()
                self.assertFalse(hasattr(p, '_accum_grad'))
                optimizer.step()
                self.assertTrue(torch.equal(p, torch.ones_like(p)))
                self.assertEqual(len(optimizer.state), 0)

    def test_quantized_moments_and_checkpoint_resume(self):
        for qtype in ('convrotint2', 'convrotint4', 'convrot8'):
            with self.subTest(qtype=qtype):
                p = nn.Parameter(torch.linspace(-1., 1., 4355).view(65, 67))
                optimizer = AdamConvRot([p], lr=0.001, qtype=qtype)
                optimizer.stochastic_rounding = False
                p.square().mean().backward()
                optimizer.step()
                optimizer.zero_grad()
                self.assertTrue(torch.isfinite(p).all())
                self.assertEqual(optimizer.state[p]['exp_avg'].dtype, torch.uint8)
                with tempfile.TemporaryDirectory() as folder:
                    path = str(Path(folder) / 'optimizer.pt')
                    atomic_save_optimizer_checkpoint(path, optimizer.state_dict(), step=1,
                                                     checkpoint_path='model_000000001.safetensors')
                    state = load_optimizer_checkpoint(path, expected_step=1)
                other = nn.Parameter(p.detach().clone())
                resumed = AdamConvRot([other], lr=0.001, qtype=qtype)
                resumed.stochastic_rounding = False
                resumed.load_state_dict(state)
                for value, opt in ((p, optimizer), (other, resumed)):
                    value.square().mean().backward()
                    opt.step()
                torch.testing.assert_close(p, other)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_fp16_rounding_uses_actual_spacing_at_small_and_large_magnitudes(self):
        p = nn.Parameter(torch.ones(32768, device='cuda', dtype=torch.float16))
        optimizer = AdamConvRot([p])
        for value in (1e-5, -1e-5, 1.0003, -1.0003):
            with self.subTest(value=value):
                source = torch.full(p.shape, value, device='cuda', dtype=torch.float32)
                torch.manual_seed(42)
                optimizer._copy_parameter(p, source)
                # Every output must be one of the two neighboring FP16 values.
                rounded = source.to(torch.float16)
                lower = torch.where(rounded.float() > source,
                                    torch.nextafter(rounded, torch.full_like(rounded, -torch.inf)), rounded)
                upper = torch.where(rounded.float() < source,
                                    torch.nextafter(rounded, torch.full_like(rounded, torch.inf)), rounded)
                self.assertTrue(((p == lower) | (p == upper)).all())
                spacing = (upper.float() - lower.float())[0].item()
                self.assertLess(abs(p.float().mean().item() - value), spacing * 0.02)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_kernel_matches_torch_fallback(self):
        for qtype in ('convrotint2', 'convrotint4', 'convrot8'):
            with self.subTest(qtype=qtype):
                a = nn.Parameter(torch.linspace(-1., 1., 4355, device='cuda').view(65, 67))
                b = nn.Parameter(a.detach().clone())
                fast = AdamConvRot([a], lr=0.001, qtype=qtype)
                fallback = AdamConvRot([b], lr=0.001, qtype=qtype)
                fast.stochastic_rounding = fallback.stochastic_rounding = False
                for i in range(3):
                    grad = torch.sin(torch.arange(a.numel(), device='cuda') + i).view_as(a) * 0.01
                    a.grad, b.grad = grad.clone(), grad.clone()
                    fast.step()
                    with mock.patch('toolkit.optimizers.adamconvrot._triton_available', return_value=False):
                        fallback.step()
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_quantized_steps_all_precisions(self):
        for qtype in ('convrotint2', 'convrotint4', 'convrot8'):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                with self.subTest(qtype=qtype, dtype=dtype):
                    p = nn.Parameter(torch.linspace(-1., 1., 4355, device='cuda', dtype=dtype).view(65, 67))
                    optimizer = AdamConvRot([p], lr=0.001, qtype=qtype)
                    for _ in range(2):
                        p.float().square().mean().backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.isfinite(p).all())
                    self.assertEqual(optimizer.state[p]['step'], 2)


class CheckpointAndUIRequestTests(unittest.TestCase):
    def test_save_and_sample_keep_pending_gradients(self):
        from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
        p = nn.Parameter(torch.ones(1))
        trainer = DiffusionTrainer.__new__(DiffusionTrainer)
        trainer.is_ui_trainer = True
        trainer.step_num = 3
        trainer.progress_bar = None
        trainer.optimizer = torch.optim.SGD([p], lr=0.1)
        trainer._accumulated_microbatches = 1
        trainer.should_save = lambda: True
        trainer.update_db_key = mock.Mock()
        trainer.ui_job_store = SimpleNamespace(consume_sample_request=lambda: True)
        trainer.save = mock.Mock()
        trainer.sample = mock.Mock()
        trainer.ensure_params_requires_grad = mock.Mock()
        trainer.train_config = SimpleNamespace(free_u=False, unload_text_encoder=False)
        for request in (trainer.maybe_save, trainer.maybe_sample):
            with self.subTest(request=request.__name__):
                p.grad = torch.tensor([4.])
                with mock.patch('extensions_built_in.sd_trainer.DiffusionTrainer.flush'):
                    request()
                torch.testing.assert_close(p.grad, torch.tensor([4.]))
        trainer.save.assert_called_once_with(3)
        trainer.sample.assert_called_once_with(3)

    def test_native_feature_models_checkpoint_in_eval_with_input_gradients(self):
        from toolkit.models.sapiens2 import Sapiens2
        from toolkit.models.tipsv2 import VisionTransformer
        from torch.utils.checkpoint import checkpoint
        tiny_arch = dict(embed_dims=32, num_layers=2, num_heads=4,
                         feedforward_channels=64, num_tokenizer_layers=1)
        with mock.patch.dict(Sapiens2.arch_zoo, {'test': tiny_arch}):
            sapiens = Sapiens2(arch='test', img_size=(8, 8), patch_size=4,
                               pos_embed_rope_dtype='fp32', n_storage_tokens=1)
        tips = VisionTransformer(img_size=8, patch_size=4, embed_dim=32,
                                 depth=2, num_heads=4)
        for model, target in ((sapiens, 'toolkit.models.sapiens2.checkpoint'),
                              (tips, 'torch.utils.checkpoint.checkpoint')):
            with self.subTest(model=type(model).__name__):
                model.eval().requires_grad_(False)
                model.enable_gradient_checkpointing()
                x = torch.randn(1, 3, 8, 8, requires_grad=True)
                with mock.patch(target, wraps=checkpoint) as called:
                    result = model(x)
                    output = result[0] if isinstance(result, tuple) else result
                    output.square().mean().backward()
                    self.assertGreater(called.call_count, 0)
                    self.assertFalse(model.training)
                    self.assertTrue(torch.isfinite(x.grad).all())
                    self.assertGreater(x.grad.abs().sum().item(), 0)
                    called.reset_mock()
                    with torch.no_grad():
                        model(x.detach())
                    called.assert_not_called()

    def test_sapiens_optional_rope_augmentation_is_callable(self):
        from toolkit.models.sapiens2 import RopePositionEmbedding
        rope = RopePositionEmbedding(32, num_heads=4, jitter_coords=1.1, rescale_coords=1.2)
        rope.train()
        for value in rope(H=2, W=2):
            self.assertTrue(torch.isfinite(value).all())


class HiDreamTests(unittest.TestCase):
    def test_trainer_preserves_x0_across_loss_targets_and_legacy_dfe(self):
        for loss_target, t0, dfe_version in (('noise', False, None), ('noise', True, None),
                                           ('source', False, None), ('unaugmented', False, None),
                                           ('noise', False, 1), ('noise', False, 2)):
            with self.subTest(loss_target=loss_target, t0=t0, dfe=dfe_version):
                trainer = SDTrainer.__new__(SDTrainer)
                trainer.train_config = TrainConfig(dtype='fp32', loss_target=loss_target,
                                                    t0_loss_target=t0, linear_timesteps=False)
                trainer.device_torch = torch.device('cpu')
                trainer.adapter = None
                trainer.additional_logs = {}
                trainer._record_tensor_stats = trainer._record_monitor_metric = lambda *_: None
                trainer.sd = SimpleNamespace(
                    x0_pred=True, noise_scale=8.0, is_flow_matching=True,
                    prediction_type='epsilon', vae=SimpleNamespace(device='cpu', dtype=torch.float32),
                    get_loss_target=lambda **kw: kw['batch'].latents,
                    encode_images=lambda images: images,
                )
                seen = []

                class LegacyDFE:
                    version = dfe_version

                    def __call__(self, value):
                        seen.append(value.detach().clone())
                        return [value] if self.version == 2 else value

                trainer.dfe = LegacyDFE() if dfe_version is not None else None
                clean = torch.ones(2, 3, 2, 2)
                noise = torch.full_like(clean, 0.25)
                sigma = torch.tensor([0.25, 0.75]).view(2, 1, 1, 1)
                noisy = clean * (1 - sigma) + noise * 8.0 * sigma
                pred = (clean + 0.1).requires_grad_()
                batch = SimpleNamespace(latents=clean, sigmas=sigma, mask_tensor=None,
                                        loss_multiplier_list=[1., 1.], get_is_reg_list=lambda: [False, False],
                                        audio_pred=None, audio_target=None, unaugmented_tensor=clean)
                loss = trainer.calculate_loss(pred, noise, noisy, sigma.flatten() * 1000, batch)
                expected = (pred - clean).square()
                if loss_target in ('source', 'unaugmented'):
                    expected = expected / sigma.square()
                if dfe_version is None:
                    torch.testing.assert_close(loss, expected.mean())
                elif dfe_version == 1:
                    torch.testing.assert_close(seen[0], pred.detach())
                else:
                    torch.testing.assert_close(seen[0][:, :3], noise * 8.0 - clean)
                    torch.testing.assert_close(seen[1][:, :3], (noisy - pred.detach()) / sigma)
                loss.backward()
                self.assertTrue(torch.isfinite(pred.grad).all())

    def test_export_removes_compile_wrappers_and_reloads_plain_weights(self):
        from safetensors.torch import save_file, load_file

        class ExportModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._orig_mod = nn.Module()
                self._orig_mod.layer = nn.Linear(2, 2)
                self._orig_mod.lm_head = nn.Linear(2, 2, bias=False)

            def save_pretrained(self, save_directory, state_dict=None, **kwargs):
                save_file(state_dict or self.state_dict(), str(Path(save_directory) / 'model.safetensors'))

        model = HidreamO1Model.__new__(HidreamO1Model)
        model.model = ExportModel()
        model.tokenizer = SimpleNamespace(save_pretrained=lambda _: None)
        for comfy in (False, True):
            with self.subTest(comfy=comfy), tempfile.TemporaryDirectory() as folder:
                model.is_comfy_weight = comfy
                path = str(Path(folder) / 'export.safetensors') if comfy else folder
                with mock.patch('extensions_built_in.diffusion_models.hidream.hidream_o1_model.unwrap_model', side_effect=lambda x: x):
                    model.save_model(path, {}, torch.float32)
                exported = load_file(path if comfy else str(Path(folder) / 'model.safetensors'))
                fresh = nn.Module()
                fresh.layer = nn.Linear(2, 2)
                if not comfy:
                    fresh.lm_head = nn.Linear(2, 2, bias=False)
                fresh.load_state_dict(exported, strict=True)
                torch.testing.assert_close(fresh.layer.weight, model.model._orig_mod.layer.weight)

    def test_model_flag_and_clean_target_without_loading_weights(self):
        model = HidreamO1Model('cpu', ModelConfig(arch='hidream_o1', name_or_path='unused'))
        self.assertTrue(model.x0_pred)
        clean = torch.ones(2, 3, 2, 2, requires_grad=True)
        target = model.get_loss_target(batch=SimpleNamespace(latents=clean), noise=torch.zeros_like(clean))
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(target, clean)

    def test_scheduler_uses_x0_table_only_for_weighted_x0(self):
        scheduler = CustomFlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(1000)
        timesteps = scheduler.timesteps[[0, 500, 999]]
        for x0, table in ((False, default_weighing_scheme), (True, x0_weighing_scheme)):
            actual = scheduler.get_weights_for_timesteps(timesteps, timestep_type='weighted', x0_pred=x0)
            torch.testing.assert_close(actual, torch.tensor([table[i] for i in (0, 500, 999)]))
        a = scheduler.get_weights_for_timesteps(timesteps)
        b = scheduler.get_weights_for_timesteps(timesteps, x0_pred=True)
        torch.testing.assert_close(a, b)

    def test_mean_flow_accepts_shared_weighting_interface(self):
        scheduler = MeanFlowScheduler()
        scheduler.set_train_timesteps(1000, device='cpu')
        timesteps = scheduler.timesteps[[0, 500]]
        torch.testing.assert_close(scheduler.get_weights_for_timesteps(timesteps, x0_pred=True), torch.ones(2))
        actual = scheduler.get_weights_for_timesteps(timesteps, timestep_type='weighted', x0_pred=True)
        torch.testing.assert_close(actual, torch.tensor([x0_weighing_scheme[i] for i in (0, 500)]))


class MemoryMovementTests(unittest.TestCase):
    def test_plain_parameter_keeps_identity(self):
        layer = nn.Linear(2, 2)
        weight = layer.weight
        _move_own_param(layer, 'weight', weight, torch.device('cpu'))
        self.assertIs(layer.weight, weight)
        self.assertTrue(layer.weight.requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_torchao_resident_weights_move_real_storage_and_preserve_ties(self):
        from torchao.dtypes import to_affine_quantized_intx
        from torchao.quantization.quant_primitives import MappingType

        weight = to_affine_quantized_intx(
            torch.randn(16, 16), MappingType.SYMMETRIC, (1, 16), torch.int8,
        )
        root = nn.Module()
        root.left = nn.Linear(16, 16, bias=False)
        root.right = nn.Linear(16, 16, bias=False)
        root.left.weight = nn.Parameter(weight, requires_grad=False)
        root.right.weight = root.left.weight
        try:
            MemoryManager.attach(root, torch.device('cuda:0'), offload_percent=0.0)
            self.assertEqual(_storage_device(root.left.weight), torch.device('cuda:0'))
            self.assertIs(root.left.weight, root.right.weight)
            self.assertFalse(root.left.weight.requires_grad)
            output = root.left(torch.randn(2, 16, device='cuda'))
            self.assertTrue(torch.isfinite(output).all())
        finally:
            MemoryManager.detach(root)


if __name__ == '__main__':
    unittest.main()
