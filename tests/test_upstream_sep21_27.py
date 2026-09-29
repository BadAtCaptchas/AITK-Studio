"""Behavior regressions for the September 21–27 ports; no model downloads."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

from toolkit.config_modules import ModelConfig
from toolkit.dataloader_mixins import TextEmbeddingFileItemDTOMixin
from toolkit.models.base_model import BaseModel
from toolkit.models.v2._mixin import OstrisModelMixin
from toolkit.unloader import FakeTextEncoder, unload_text_encoder
from toolkit.util import unified_memory
from toolkit.model_registry import resolve_model
from extensions_built_in.diffusion_models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusModel
from extensions_built_in.diffusion_models.qwen_image_2 import QwenImage2Model
from extensions_built_in.diffusion_models.minimax_h3 import MinimaxH3FastModel, MinimaxH3FastV2Model
from extensions_built_in.diffusion_models.ming_image_comfy import MingImageModel
from extensions_built_in.diffusion_models.ming_image_comfy.src.checkpoints import canonical_repo
from extensions_built_in.diffusion_models.ming_image_comfy.src.text_encoder import MingImageTextEncoder
from extensions_built_in.diffusion_models.ming_image_comfy.src.transformer import MingImageTransformer2DModel
from extensions_built_in.diffusion_models.ming_image_comfy.src.pipeline import run_transformer, MingImagePipeline
from extensions_built_in.audio_models.yue2.src.model import merge_nar_lora
from extensions_built_in.audio_models.yue2.src.tokenizer import load_head_state_dict
from extensions_built_in.audio_models.yue2.yue2_model import YuE2AudioModel as YuE2Holder
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from testing.test_base_model_device_state_presets import make_base_model_with_fake_modules
from tests.test_qwen_image_2 import PosteriorMeanVAE


class TinyLoader(torch.nn.Linear, OstrisModelMixin):
    @classmethod
    def aitk_from_config(cls, config):
        return cls(2, 2)


class SeptemberPortsTests(unittest.TestCase):
    def test_qwen_edit_batch_padding_and_caller_controls(self):
        def encode(prompts, **kwargs):
            length = 1 if prompts[0] == 'short' else 3
            return torch.full((1, length, 4), float(length)), None
        encoder = Mock(side_effect=encode)
        owner = SimpleNamespace(device_torch=torch.device('cpu'), pipeline=SimpleNamespace(
            text_encoder=SimpleNamespace(device=torch.device('cpu')), encode_prompt=encoder))
        images = [[torch.rand(3, 32, 64)], [torch.rand(3, 64, 32)]]
        before = [row[0].clone() for row in images]
        with patch('extensions_built_in.diffusion_models.qwen_image.qwen_image_edit_plus.CONDITION_IMAGE_SIZE', 1024):
            result = QwenImageEditPlusModel.get_prompt_embeds(owner, ['short', 'long'], images)
        self.assertEqual([call.args[0] for call in encoder.call_args_list], [['short'], ['long']])
        self.assertEqual(result.text_embeds.shape, (2, 3, 4))
        self.assertEqual(result.attention_mask.tolist(), [[1, 0, 0], [1, 1, 1]])
        for row, original in zip(images, before):
            torch.testing.assert_close(row[0], original)
        encoder.reset_mock()
        with patch('extensions_built_in.diffusion_models.qwen_image.qwen_image_edit_plus.CONDITION_IMAGE_SIZE', 1024):
            result = QwenImageEditPlusModel.get_prompt_embeds(owner, ['short', 'long'], torch.rand(2, 3, 32, 32))
        self.assertEqual(result.text_embeds.shape[0], 2)

    def test_unload_model_owned_encoders_and_aliases_once(self):
        for self_pipeline in (False, True):
            real = torch.nn.Linear(2, 2)
            owner = SimpleNamespace(text_encoder=[real], mllm=real, device_torch='cpu', torch_dtype=torch.float32)
            owner.pipeline = owner if self_pipeline else SimpleNamespace(text_encoder=None, mllm=real)
            with patch('toolkit.unloader.MemoryManager.free', wraps=__import__('toolkit.memory_management', fromlist=['MemoryManager']).MemoryManager.free) as free:
                unload_text_encoder(owner)
                unload_text_encoder(owner)
            self.assertEqual(free.call_count, 1)
            self.assertEqual(real.weight.device.type, 'meta')
            self.assertIsInstance(owner.text_encoder[0], FakeTextEncoder)
            self.assertIs(owner.mllm, owner.text_encoder[0])
            self.assertIs(owner.pipeline.mllm, owner.mllm)

    def test_unloaded_text_encoders_do_not_probe_missing_internals(self):
        for encoders in (FakeTextEncoder('cpu', torch.float32), [FakeTextEncoder('cpu', torch.float32)]):
            owner = make_base_model_with_fake_modules(encoders)
            owner.get_te_has_grad = Mock(side_effect=AssertionError('unloaded encoder inspected'))
            BaseModel.save_device_state(owner)
            state = owner.device_state['text_encoder']
            self.assertFalse((state[0] if isinstance(state, list) else state)['requires_grad'])

    def test_comfy_attention_metadata_is_ignored_without_mutating_input(self):
        original = TinyLoader(2, 2)
        state = dict(original.state_dict())
        state['attn.comfy_attention.config'] = torch.tensor(list(b'{"attention":"comfy_kitchen_int8"}'), dtype=torch.uint8)
        loaded = TinyLoader.load_from_state_dict(state, config={}, dtype=torch.float32)
        torch.testing.assert_close(loaded.weight, original.weight)
        self.assertIn('attn.comfy_attention.config', state)

    def test_qwen_reference_sizing_policy_and_cache_identity(self):
        holder = QwenImage2Model('cpu', ModelConfig(arch='qwen_image_2', name_or_path='test'), dtype='fp32')
        refs = [[torch.rand(1, 4, 32, 64)]]
        prepared = holder._prepare_control_images(refs, target_pixels=64 * 128)[0][0]
        self.assertEqual(prepared.shape, (1, 4, 64, 128))
        self.assertEqual(holder._target_pixels((95, 159)), 64 * 128)
        matched_key = holder.get_text_embedding_space_version()
        holder.model_config.model_kwargs['match_target_res'] = False
        self.assertEqual(holder._prepare_control_images(refs, target_pixels=64 * 128)[0][0].shape, refs[0][0].shape)
        self.assertNotEqual(matched_key, holder.get_text_embedding_space_version())
        self.assertFalse(holder.text_embedding_uses_target_size)
        # Equal token counts do not make distinct reference grids interchangeable.
        holder.vae = PosteriorMeanVAE()
        with self.assertRaisesRegex(ValueError, 'per-reference dimensions'):
            holder.encode_condition_images([[torch.rand(1, 4, 32, 64)], [torch.rand(1, 4, 64, 32)]])

    def test_optional_target_size_does_not_break_legacy_prompt_plugins(self):
        owner = SimpleNamespace(encode_control_in_text_embeddings=True, get_prompt_embeds=lambda prompt, control_images: (prompt, control_images))
        self.assertEqual(BaseModel.encode_prompt(owner, 'caption', control_images='ref', target_size=(64, 64)), (['caption'], 'ref'))
        owner.get_prompt_embeds = lambda prompt, control_images, target_size=None: target_size
        self.assertEqual(BaseModel.encode_prompt(owner, 'caption', control_images='ref', target_size=(64, 96)), (64, 96))

    def test_reference_cache_identity_follows_bucket_and_keeps_text_only_reusable(self):
        item = SimpleNamespace(caption='test', text_embedding_space_version='qwen_image_2_test',
            text_embedding_version=1, control_path='reference.png', encode_control_in_text_embeddings=True,
            text_embedding_uses_target_size=True, crop_width=64, crop_height=96)
        info = TextEmbeddingFileItemDTOMixin.get_text_embedding_info_dict
        first, plain = info(item), info(item, text_only=True)
        item.crop_width = 128
        self.assertNotEqual(first, info(item))
        self.assertEqual(plain, info(item, text_only=True))
        item.text_embedding_uses_target_size = False
        self.assertNotIn('control_target_size', info(item))

    def test_live_and_prior_prompt_kwargs_keep_ordered_references_and_target_size(self):
        owner = object.__new__(SDTrainer)
        owner.sd = SimpleNamespace(encode_control_in_text_embeddings=True, device_torch='cpu', torch_dtype=torch.float32)
        references = [[torch.rand(1, 3, 32, 64), torch.rand(1, 3, 64, 32)]]
        batch = SimpleNamespace(control_tensor_list=references, control_tensor=None,
            file_items=[SimpleNamespace(crop_width=128, crop_height=96)])
        kwargs = owner.get_batch_prompt_kwargs(batch)
        self.assertIs(kwargs['control_images'], references)
        self.assertEqual(kwargs['target_size'], (128, 96))
        batch.control_tensor_list = None
        batch.control_tensor = torch.rand(1, 3, 32, 64).double()
        self.assertEqual(owner.get_batch_prompt_kwargs(batch)['control_images'].dtype, torch.float32)
        owner.sd.encode_control_in_text_embeddings = False
        self.assertEqual(owner.get_batch_prompt_kwargs(batch), {})

    def test_ming_reference_cache_uses_live_crop_and_content_identity(self):
        model = MingImageModel('cpu', ModelConfig(arch='ming_image', name_or_path='test'), dtype='fp32')
        image = torch.rand(3, 16, 32)
        item = SimpleNamespace(control_tensor=image, load_control_image=Mock(), cleanup_control=Mock())
        torch.testing.assert_close(model.load_cached_control_images(item)[0], image.unsqueeze(0))
        item.load_control_image.assert_called_once()
        item.cleanup_control.assert_called_once()
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp) / 'ref.png'
            reference.write_bytes(b'first')
            item = SimpleNamespace(caption='test', text_embedding_space_version=model.get_text_embedding_space_version(),
                text_embedding_version=1, control_path=str(reference), encode_control_in_text_embeddings=True,
                preserve_image_alpha=False, scale_to_width=32, scale_to_height=32,
                crop_x=0, crop_y=0, crop_width=16, crop_height=32, flip_x=False, flip_y=False)
            info = TextEmbeddingFileItemDTOMixin.get_text_embedding_info_dict
            first = info(item)
            item.crop_x = 16
            self.assertNotEqual(first['reference_crop'], info(item)['reference_crop'])
            reference.write_bytes(b'changed')
            self.assertNotEqual(first['ming_reference_contents'], info(item)['ming_reference_contents'])
            item.is_encrypted = True
            self.assertNotIn('ming_reference_contents', info(item))
        model.model_config.model_kwargs['rgba'] = True
        with self.assertRaisesRegex(ValueError, 'requires PNG'):
            model.generate_images([SimpleNamespace(output_ext='jpg')])

    def test_yue_heads_support_both_release_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            state = {'head.weight': torch.randn(2, 3)}
            torch.save({'model': state}, root / 'head.pt')
            save_file(state, root / 'head.safetensors')
            for path in root.iterdir():
                torch.testing.assert_close(load_head_state_dict(str(path))['head.weight'], state['head.weight'])

    def test_yue_safetensors_nar_merge_with_grad_enabled(self):
        linear = lambda: torch.nn.Linear(2, 2, bias=False)
        attention = SimpleNamespace(qkv_proj=torch.nn.Linear(2, 6, bias=False), o_proj=linear())
        mlp = SimpleNamespace(gate_up_proj=torch.nn.Linear(2, 4, bias=False), down_proj=linear())
        model = SimpleNamespace(cfg=SimpleNamespace(num_attention_heads=1, head_dim=2, num_key_value_heads=1, intermediate_size=2),
            nar=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention, mlp=mlp)]), vae2llm=linear(), llm2vae=linear()))
        weights = {}
        for block, names in [('nar_self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj')), ('nar_mlp', ('gate_proj', 'up_proj', 'down_proj'))]:
            for name in names:
                weights[f'layers.0.{block}.{name}.lora_A'] = torch.ones(1, 2)
                weights[f'layers.0.{block}.{name}.lora_B'] = torch.ones(2, 1)
        weights['vae2llm.weight'] = torch.full((2, 2), 7.)
        before = attention.qkv_proj.weight.detach().clone()
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / 'nar.safetensors')
            save_file(weights, path)
            merge_nar_lora(model, path, scale=0.5)
        torch.testing.assert_close(attention.qkv_proj.weight, before + 0.5)
        torch.testing.assert_close(model.nar.vae2llm.weight, torch.full((2, 2), 7.))

    def test_yue_head_cache_keeps_cot_and_distinguishes_equal_filenames(self):
        holder = YuE2Holder('cpu', ModelConfig(arch='yue2', name_or_path='test'), dtype='fp32')
        baseline = holder.get_latent_space_version()
        holder.semantic_head_path = '/first/head.safetensors'
        first = holder.get_latent_space_version()
        holder.semantic_head_path = '/second/head.safetensors'
        self.assertNotEqual(first, holder.get_latent_space_version())
        self.assertNotEqual(first, baseline)
        self.assertTrue(first.endswith('_cot_' + holder.cot))

    def test_fast_v2_has_its_own_defaults_and_keeps_legacy_schedule(self):
        legacy = MinimaxH3FastModel('cpu', ModelConfig(arch='minimax_h3_vsa', name_or_path='test'))
        current = MinimaxH3FastV2Model('cpu', ModelConfig(arch='minimax_h3_vsa_v2', name_or_path='test'))
        self.assertEqual(legacy.vsa_sparsity, 0.9)
        self.assertEqual(current.vsa_sparsity, 0.8)
        self.assertEqual(current.get_train_scheduler().config.shift, 10)
        self.assertEqual(legacy.get_train_scheduler().config.shift, 12)
        self.assertNotEqual(current._dit_component(), legacy._dit_component())
        self.assertIs(resolve_model(current.arch), MinimaxH3FastV2Model)
        self.assertIs(resolve_model('ming_image'), MingImageModel)
        self.assertNotEqual(resolve_model('ming_image_design'), MingImageModel)

    def test_ming_checkpoint_alias_and_vision_metadata(self):
        self.assertEqual(canonical_repo('Kijai/Ming-Image-ComfyUI'), 'Comfy-Org/Ming-Image')
        self.assertEqual(canonical_repo('/local/model'), '/local/model')
        vision = torch.randn(2)
        state = MingImageTextEncoder.convert_state_dict_on_load({'thinker.embed_tokens.weight': torch.randn(2, 3),
            'thinker.lm_head.weight': torch.randn(2, 3), 'tokenizer_json': torch.zeros(1), 'vision.weight': vision})
        self.assertEqual(set(state), {'llm.word_embeddings.weight', 'vision.weight'})
        self.assertIs(state['vision.weight'], vision)

    def test_tiny_ming_forward_backward_and_live_preview(self):
        transformer = MingImageTransformer2DModel(in_channels=4, dim=16, n_layers=1, n_refiner_layers=1,
            n_heads=2, n_kv_heads=2, cap_feat_dim=8, axes_dims=(2, 2, 4), axes_lens=(256, 64, 64))
        noisy = torch.randn(1, 4, 4, 4, requires_grad=True)
        query, direct = [torch.randn(3, 8)], [torch.randn(5, 16)]
        output = run_transformer(transformer, noisy, torch.tensor([500.]), query, direct, [torch.randn(4, 4, 4)])
        self.assertEqual(output.shape, noisy.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(noisy.grad).all())
        holder = MingImageModel('cpu', ModelConfig(arch='ming_image', name_or_path='test'), dtype='fp32')
        holder.model = transformer
        holder.decode_to_images = lambda values: list(values)
        holder.sample_step_hook = True
        holder._emit_sample_step = Mock()
        pe = SimpleNamespace(text_embeds=query, direct_embeds=direct)
        result = MingImagePipeline(holder)(pe, height=32, width=32, num_inference_steps=3, latents=noisy.detach())
        self.assertEqual(len(result), 1)
        self.assertEqual(holder._emit_sample_step.call_count, 2)
        key = holder.get_latent_space_version()
        holder.model_config.model_kwargs['rgba'] = True
        self.assertTrue(holder.preserve_image_alpha)
        self.assertNotEqual(key, holder.get_latent_space_version())

    def test_ming_training_adapter_toggles_without_changing_base_weights(self):
        transformer = MingImageTransformer2DModel(in_channels=4, dim=16, n_layers=1, n_refiner_layers=1,
            n_heads=2, n_kv_heads=2, cap_feat_dim=8, axes_dims=(2, 2, 4), axes_lens=(256, 64, 64))
        projection = transformer.layers[0].attention.to_k
        weights = projection.weight.detach().clone()
        inputs = torch.ones(1, 2, projection.in_features)
        expected = projection(inputs).detach()
        with tempfile.TemporaryDirectory() as temp:
            adapter = Path(temp) / 'training-adapter.safetensors'
            save_file({
                'diffusion_model.layers.0.attention.to_k.lora_A.weight': torch.ones(2, projection.in_features),
                'diffusion_model.layers.0.attention.to_k.lora_B.weight': torch.ones(projection.out_features, 2),
            }, str(adapter))
            holder = MingImageModel('cpu', ModelConfig(arch='ming_image', name_or_path='test', assistant_lora_path=str(adapter)), dtype='fp32')
            holder.load_training_adapter(transformer)
            self.assertFalse(any(parameter.requires_grad for parameter in holder.assistant_lora.parameters()))
            self.assertFalse(torch.allclose(projection(inputs), expected))
            holder.assistant_lora.is_active = False
            torch.testing.assert_close(projection(inputs), expected)
            torch.testing.assert_close(projection.weight, weights)

    def test_unified_memory_detection_gates(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(unified_memory.sys, 'platform', 'linux'), \
             patch.object(torch.version, 'hip', None), patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(is_integrated=True)):
            self.assertTrue(unified_memory.is_unified_memory())
            with patch.dict(os.environ, {'AITK_DISABLE_UNIFIED_MEMORY': '1'}):
                self.assertFalse(unified_memory.is_unified_memory())
            with patch.object(unified_memory.sys, 'platform', 'win32'):
                self.assertFalse(unified_memory.is_unified_memory())
            with patch.object(torch.version, 'hip', 'rocm'):
                self.assertFalse(unified_memory.is_unified_memory())
            with patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(is_integrated=False)):
                self.assertFalse(unified_memory.is_unified_memory())

    def test_unified_module_cpu_suppression_keeps_casts_and_tensor_moves(self):
        original_to, original_cpu = torch.nn.Module.to, torch.nn.Module.cpu
        try:
            with patch.object(unified_memory, '_patched', False), patch.object(unified_memory, 'is_unified_memory', return_value=True), \
                 patch.object(torch.cuda, 'get_device_name', return_value='Test GB10'):
                unified_memory.apply_unified_memory_patches()
                installed = torch.nn.Module.to
                unified_memory.apply_unified_memory_patches()
                self.assertIs(installed, torch.nn.Module.to)
                model = torch.nn.Linear(2, 2)
                self.assertIs(model.to('cpu'), model)
                model.to(device='cpu', dtype=torch.float64)
                self.assertEqual(model.weight.dtype, torch.float64)
                self.assertEqual(torch.ones(1).cpu().device.type, 'cpu')
        finally:
            torch.nn.Module.to, torch.nn.Module.cpu = original_to, original_cpu


if __name__ == '__main__':
    unittest.main()
