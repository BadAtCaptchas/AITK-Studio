"""Qwen Image 2.1 integration checks without model downloads."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from PIL import Image
from torchvision.transforms.functional import to_tensor
from transformers import BatchFeature, Qwen2VLImageProcessor
from types import SimpleNamespace

import torch

from extensions_built_in.diffusion_models import QwenImage2Model
from extensions_built_in.diffusion_models.qwen_image_2.src.pipeline import (
    QwenImage21Pipeline, QwenImage21PromptEncoder, pack_latents, unpack_latents, pad_prompt_batch,
    prepare_condition_image, run_transformer, tensor_to_pil,
)
from extensions_built_in.diffusion_models.qwen_image_2.src.text_encoder import QwenImage21TextEncoder
from extensions_built_in.diffusion_models.qwen_image_2.src.transformer import QwenImage21Transformer2DModel
from extensions_built_in.diffusion_models.qwen_image_2.src.vae import AutoencoderKLQwenImage21
from toolkit.config_modules import ModelConfig, GenerateImageConfig, DatasetConfig
from toolkit.sample_controls import sample_control_paths, validate_control_paths
from toolkit.data_transfer_object.data_loader import FileItemDTO
from extensions_built_in.inference_engine.validation import validate_generation
from toolkit.config_contract import config_contract_errors
from toolkit.model_registry import resolve_model
from toolkit.models.registry import get_arch_entry
from toolkit.models.v2._mixin import OstrisModelMixin
from toolkit.util.comfy_quant_import import import_comfy_quantized_layers


def tiny_transformer():
    return QwenImage21Transformer2DModel(
        in_channels=4, out_channels=4, num_layers=1, num_attention_heads=1,
        attention_head_dim=8, context_in_dim=8, axes_dims_rope=(2, 2, 4),
    )


class QwenImage2Tests(unittest.TestCase):
    def test_registration_and_optional_edit_controls(self):
        self.assertIs(resolve_model('qwen_image_2'), QwenImage2Model)
        self.assertEqual(config_contract_errors({'config': {'process': [{
            'type': 'diffusion_trainer', 'model': {'arch': 'qwen_image_2'},
        }]}}), [])
        entry = get_arch_entry('qwen_image_2')
        self.assertFalse(entry['needs_control_image'])
        self.assertEqual(entry['sample']['guidance_scale'], 3)
        self.assertEqual(entry['model']['qtype'], 'convrot8')

    def test_materialized_convrot_embedding_is_not_a_missing_checkpoint_key(self):
        model = torch.nn.Module()
        model.embed = torch.nn.Embedding(3, 16, device='meta')
        model.proj = torch.nn.Linear(16, 2, bias=False, device='meta')
        state = {
            'embed.weight': torch.arange(-24, 24, dtype=torch.int8).reshape(3, 16),
            'embed.weight_scale': torch.ones(3) * 0.1,
            'embed.comfy_quant': torch.tensor(list(json.dumps({
                'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 16,
            }).encode()), dtype=torch.uint8),
            'proj.weight': torch.ones(2, 16),
        }
        remaining, count = import_comfy_quantized_layers(model, state, orig_dtype=torch.float32)
        self.assertEqual(count, 1)
        self.assertNotIn('embed.weight', remaining)
        OstrisModelMixin._load_state_dict_with_quantized(model, remaining)
        self.assertTrue(all(not p.is_meta for p in model.parameters()))
        self.assertTrue(torch.isfinite(model.proj(model.embed(torch.tensor([0])))).all())

    def test_loader_still_rejects_missing_meta_and_unexpected_parameters(self):
        model = torch.nn.Linear(2, 2, device='meta')
        with self.assertRaisesRegex(ValueError, 'missing'):
            OstrisModelMixin._load_state_dict_with_quantized(model, {'weight': torch.ones(2, 2)})
        model = torch.nn.Linear(2, 2, device='meta')
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            OstrisModelMixin._load_state_dict_with_quantized(model, {
                'weight': torch.ones(2, 2), 'bias': torch.ones(2), 'unknown': torch.ones(1),
            })

    def test_checkpoint_key_conversions(self):
        state = {key: torch.ones(1) for key in (
            'model.layers.0.attn.weight', 'model.embed_tokens.weight',
            'model.norm.weight', 'model.visual.weight',
        )}
        converted = QwenImage21TextEncoder.convert_state_dict_on_load(state)
        self.assertIn('model.language_model.embed_tokens.weight', converted)
        restored = QwenImage21TextEncoder.convert_state_dict_on_save(converted)
        self.assertEqual(set(restored), set(state))
        gate, up = torch.ones(3, 2), torch.full((3, 2), 2.0)
        fused = QwenImage21Transformer2DModel.convert_state_dict_on_load({
            'transformer_blocks.0.img_mlp.gate_layer.weight': gate,
            'transformer_blocks.0.img_mlp.proj.weight': up,
        })
        torch.testing.assert_close(fused['transformer_blocks.0.img_mlp.gate_up.weight'], torch.cat([gate, up]))
        vae = AutoencoderKLQwenImage21.convert_state_dict_on_load({'encoder.conv1.weight': torch.ones(2, 4, 1, 3, 3)})
        self.assertEqual(tuple(vae['encoder.conv_in.weight'].shape), (2, 4, 3, 3))

    def test_reference_resize_alpha_and_intentional_duplicates(self):
        image = torch.rand(1, 4, 90, 170)
        prepared = prepare_condition_image(image, 64 * 64)
        self.assertEqual(prepared.shape[-1] % 32, 0)
        self.assertEqual(prepared.shape[-2] % 32, 0)
        torch.testing.assert_close(prepare_condition_image(image, 64 * 64), prepared)
        transparent = torch.zeros(4, 2, 2)
        self.assertEqual(tensor_to_pil(transparent).getpixel((0, 0)), (255, 255, 255))
        holder = QwenImage2Model('cpu', ModelConfig(arch='qwen_image_2', name_or_path='test'), dtype='fp32')
        self.assertEqual(holder.get_bucket_divisibility(), 32)
        self.assertFalse(holder.use_old_lokr_format)
        self.assertEqual(len(holder._normalize_control_images([[image, image, image + 1]], 1)[0]), 3)

    def test_prompt_padding_and_latent_roundtrip(self):
        embeds, mask, slots = pad_prompt_batch(
            [torch.ones(2, 8), torch.ones(3, 8)],
            [torch.ones(2), torch.ones(3)],
            [torch.tensor([True, False]), torch.tensor([True, False, False])],
            'cpu', torch.float32,
        )
        self.assertEqual(embeds.shape, (2, 3, 8))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertFalse(mask[0, -1])
        self.assertEqual(slots.dtype, torch.bool)
        latent = torch.randn(2, 4, 2, 4)
        torch.testing.assert_close(unpack_latents(pack_latents(latent), 2, 4), latent)

    def test_text_to_image_and_edit_transformer_forward_backward(self):
        model = tiny_transformer()
        latent = torch.randn(1, 4, 2, 2, requires_grad=True)
        prompt = torch.randn(1, 3, 8)
        mask = torch.ones(1, 3, dtype=torch.bool)
        for edit in (False, True):
            slots = torch.tensor([[False, edit, False]])
            output = run_transformer(model, latent, torch.tensor([0.5]), prompt, mask, slots,
                condition_latents=torch.randn(1, 4, 4) if edit else None,
                condition_shapes=[(2, 2)] if edit else [])
            self.assertEqual(output.shape, latent.shape)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.isfinite(latent.grad).all())

    def test_inconsistent_reference_slots_fail_before_inference(self):
        args = (tiny_transformer(), torch.randn(2, 4, 2, 2), torch.ones(2), torch.randn(2, 3, 8), torch.ones(2, 3, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, 'same reference image slot layout'):
            run_transformer(*args, torch.tensor([[False, True, False], [False, False, False]]))
        with self.assertRaisesRegex(ValueError, 'reference image slots'):
            run_transformer(*args, torch.ones(2, 3, dtype=torch.bool))

    def test_rgba_vae_encode_decode(self):
        model = AutoencoderKLQwenImage21(base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[1, 1],
            num_res_blocks=1, temperal_downsample=[False], latents_mean=[0] * 4, latents_std=[1] * 4,
            scale_factor_spatial=2, scale_factor_temporal=1)
        with torch.no_grad():
            inputs = torch.randn(1, 4, 1, 8, 8)
            latent = model.encode(inputs).latent_dist.mode()
            output = model.decode(latent).sample
        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_sampler_reports_progress_and_propagates_cancellation(self):
        events = []
        holder = SimpleNamespace(device_torch=torch.device('cpu'), torch_dtype=torch.float32,
            transformer=tiny_transformer(), get_train_scheduler=QwenImage2Model.get_train_scheduler,
            encode_condition_images=lambda images: (None, []), pad_prompt_embeds=lambda value: value,
            decode_to_images=lambda latents: [latents], sample_step_hook=True,
            _emit_sample_step=lambda latent, index, total: events.append((index, total, latent.clone())))
        prompt = (torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.bool))
        pipeline = QwenImage21Pipeline(holder)
        for guidance_scale in (1.0, 3.0):
            events.clear()
            output = pipeline(prompt, unconditional_embeds=(torch.zeros_like(prompt[0]), *prompt[1:]),
                guidance_scale=guidance_scale, height=32, width=32, num_inference_steps=2)
            self.assertEqual([(index, total) for index, total, _ in events], [(0, 2), (1, 2)])
            self.assertTrue(torch.isfinite(output[0]).all())
        class Cancelled(Exception):
            pass
        def cancel(*args):
            raise Cancelled()
        holder._emit_sample_step = cancel
        with self.assertRaises(Cancelled):
            pipeline(prompt, height=32, width=32, num_inference_steps=2)

    def test_small_references_use_identical_processor_and_vae_grids(self):
        image_processor = Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2,
            size={'shortest_edge': 65536, 'longest_edge': 16777216})
        class Processor:
            def __call__(self, **kwargs):
                result = image_processor(images=kwargs['images'], do_resize=kwargs.get('do_resize', True), return_tensors='pt')
                slots = int(result.image_grid_thw.prod(dim=1).sum()) // 4
                return BatchFeature({'input_ids': torch.full((1, slots), 42),
                    'attention_mask': torch.ones(1, slots), 'pixel_values': result.pixel_values,
                    'image_grid_thw': result.image_grid_thw})
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleNamespace(language_model=SimpleNamespace(norm=torch.nn.Identity()))
                self.device = 'cpu'
            def forward(self, input_ids, **kwargs):
                hidden = self.model.language_model.norm(torch.ones(1, input_ids.shape[1], 8))
                return SimpleNamespace(hidden_states=[hidden])
        encoder = QwenImage21PromptEncoder.__new__(QwenImage21PromptEncoder)
        encoder.text_encoder, encoder.processor = Encoder(), Processor()
        encoder.drop_index, encoder.image_token_id = 0, 42
        for height, width in ((32, 32), (64, 128), (256, 256), (32, 1024)):
            reference = prepare_condition_image(torch.zeros(1, 4, height, width), 1024 ** 2)
            _, _, slots = encoder.encode(['edit'], images=[[tensor_to_pil(reference)]])
            self.assertEqual(int(slots[0].sum()), reference.shape[-2] * reference.shape[-1] // 1024)

    def test_ten_ordered_rgba_references_and_legacy_alias(self):
        holder = QwenImage2Model('cpu', ModelConfig(arch='qwen_image_2', name_or_path='test'), dtype='fp32')
        with TemporaryDirectory() as directory:
            paths = []
            for index in range(10):
                path = Path(directory) / f'{index}.png'
                Image.new('RGBA', (32, 32), (index * 20, 0, 0, index * 20)).save(path)
                paths.append(str(path))
            config = GenerateImageConfig(prompt='edit', output_folder=directory, output_ext='png', ctrl_imgs=paths)
            tensors = holder.load_sample_control_images(config)
            self.assertEqual(len(tensors), 10)
            for index, tensor in enumerate(tensors):
                self.assertEqual(tensor.shape, (1, 4, 32, 32))
                self.assertAlmostEqual(float(tensor[0, 3, 0, 0]), index * 20 / 255, places=6)
            self.assertEqual(tensor_to_pil(tensors[0]).getpixel((0, 0)), (255, 255, 255))
            legacy = GenerateImageConfig(prompt='edit', output_folder=directory, output_ext='png', ctrl_img=paths[0], ctrl_img_2=paths[1])
            self.assertEqual(sample_control_paths(legacy), paths[:2])
            config.ctrl_imgs = [paths[0], paths[0]]
            self.assertEqual(len(holder.load_sample_control_images(config)), 2)
            config.ctrl_imgs = []
            self.assertIsNone(holder.load_sample_control_images(config))
        for invalid in ('abc', ['a'] * 11, [''], [3]):
            with self.assertRaises(ValueError): validate_control_paths(invalid)
        validate_generation({'model': {'arch': 'qwen_image_2'}, 'sample': {'ctrl_imgs': ['a'] * 10}})
        with self.assertRaises(ValueError):
            validate_generation({'model': {'arch': 'flux'}, 'sample': {'ctrl_imgs': ['a']}})

    def test_training_preserves_alpha_for_targets_controls_and_cache_identity(self):
        holder = QwenImage2Model('cpu', ModelConfig(arch='qwen_image_2', name_or_path='test'), dtype='fp32')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'target.png'
            Image.new('RGBA', (64, 64), (100, 20, 30, 64)).save(path)
            config = DatasetConfig(folder_path=directory, control_path=directory, resolution=64, buckets=True)
            item = FileItemDTO(path=str(path), dataset_config=config, sd=holder)
            item.scale_to_width = item.scale_to_height = item.crop_width = item.crop_height = 64
            item.crop_x = item.crop_y = 0
            item.load_and_process_image(lambda image: to_tensor(image) * 2 - 1)
            self.assertEqual(item.tensor.shape, (4, 64, 64))
            self.assertAlmostEqual(float(item.tensor[3, 0, 0]), 64 / 255 * 2 - 1, places=6)
            self.assertEqual(item.control_tensor.shape, (4, 64, 64))
            self.assertEqual(item.get_latent_info_dict()['image_channels'], 4)
            cached_controls = holder.load_cached_control_images(item)
            self.assertEqual(cached_controls[0].shape, (1, 4, 64, 64))
            self.assertAlmostEqual(float(cached_controls[0][0, 3, 0, 0]), 64 / 255, places=6)
            self.assertIsNone(item.control_tensor)
            item.caption = 'edit'
            before = item.get_text_embedding_info_dict()
            item.crop_x += 1
            self.assertNotEqual(before, item.get_text_embedding_info_dict())
            item.crop_x -= 1

            # The encrypted reader returns decoded image memory; no plaintext sidecar is needed.
            item.is_encrypted = True
            item.encrypted_item = object()
            item.encrypted_reader = SimpleNamespace(open_image=Mock(return_value=Image.new('RGBA', (64, 64), (1, 2, 3, 77))))
            item.load_and_process_image(lambda image: to_tensor(image) * 2 - 1, only_load_latents=True)
            self.assertAlmostEqual(float(item.tensor[3, 0, 0]), 77 / 255 * 2 - 1, places=6)
            item.encrypted_reader.open_image.assert_called_once()
            item.is_encrypted = False
            item.preserve_image_alpha = False
            item.load_and_process_image(lambda image: to_tensor(image) * 2 - 1)
            self.assertEqual(item.tensor.shape[0], 3)
            self.assertEqual(item.control_tensor.shape[0], 3)
            self.assertNotIn('image_channels', item.get_latent_info_dict())

    def test_cached_sampling_matches_uncached_with_guidance_and_references(self):
        model = tiny_transformer().eval()
        for editing in (False, True):
            reference = torch.randn(1, 4, 4) if editing else None
            holder = SimpleNamespace(device_torch=torch.device('cpu'), torch_dtype=torch.float32,
                transformer=model, get_train_scheduler=QwenImage2Model.get_train_scheduler,
                encode_condition_images=lambda images: (reference, [(2, 2)] if editing else []),
                pad_prompt_embeds=lambda value: value, decode_to_images=lambda latent: [latent])
            cond = (torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool), torch.tensor([[False, editing, False]]))
            uncond = (torch.zeros_like(cond[0]), *cond[1:])
            latent = torch.randn(1, 4, 2, 2)
            for guidance in (1, 3):
                pipeline = QwenImage21Pipeline(holder)
                kwargs = dict(unconditional_embeds=uncond, height=32, width=32, num_inference_steps=3,
                    guidance_scale=guidance, latents=latent.clone())
                uncached = pipeline(cond, use_kv_cache=False, **kwargs)[0]
                with patch.object(model, 'forward', wraps=model.forward) as forward:
                    cached = pipeline(cond, **kwargs)[0]
                    calls = forward.call_args_list
                    self.assertEqual(calls[0].kwargs['kv_cache_mode'], 'extract')
                    self.assertEqual(calls[-1].kwargs['kv_cache_mode'], 'cached')
                    if guidance > 1:
                        self.assertIsNot(calls[0].kwargs['kv_cache'], calls[1].kwargs['kv_cache'])
                torch.testing.assert_close(cached, uncached, rtol=2e-5, atol=2e-5)
                # A later generation has a fresh prefix even on the same resident model.
                changed = (cond[0] + 0.7, *cond[1:])
                torch.testing.assert_close(pipeline(changed, **kwargs)[0], pipeline(changed, use_kv_cache=False, **kwargs)[0], rtol=2e-5, atol=2e-5)

    def test_official_rewriter_parsing_penalty_and_cancellation(self):
        from extensions_built_in.diffusion_models.qwen_image_2.prompt_rewrite import parse_rewrite, PresencePenalty, CheckCancellation
        self.assertEqual(parse_rewrite('reasoning</think>```json\n{"rewritten_prompt":"a {red} cat", "wh_ratio":"1:1"}\n```'), 'a {red} cat')
        with self.assertRaises(ValueError): parse_rewrite('<think>unfinished')
        with self.assertRaises(ValueError): parse_rewrite('{"rewritten_prompt": 3}')
        scores = PresencePenalty(1.5, 2)(torch.tensor([[1, 2, 3, 3]]), torch.zeros(1, 5))
        self.assertEqual(float(scores[0, 3]), -1.5)
        self.assertEqual(float(scores[0, 1]), 0)
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            CheckCancellation(Mock(side_effect=RuntimeError('cancelled')))(None, None)

    def test_rgba_output_save_retains_alpha_and_rejects_jpeg(self):
        holder = QwenImage2Model('cpu', ModelConfig(arch='qwen_image_2', name_or_path='test', model_kwargs={'rgba': True}), dtype='fp32')
        with TemporaryDirectory() as directory:
            image = holder.image_tensor_to_pil(torch.full((4, 32, 32), -0.5))
            config = GenerateImageConfig(prompt='sticker', output_folder=directory, output_ext='png')
            config.save_image(image)
            with Image.open(config.get_image_path()) as saved:
                self.assertEqual(saved.mode, 'RGBA')
                self.assertEqual(saved.getpixel((0, 0))[3], 64)
            config.output_ext = 'jpg'
            with self.assertRaisesRegex(ValueError, 'PNG'):
                holder.generate_images([config])

    def test_official_rewriter_task_selection_and_cpu_rng_isolation(self):
        from extensions_built_in.diffusion_models.qwen_image_2 import prompt_rewrite as rewrite
        with TemporaryDirectory() as directory:
            system = Path(directory) / 'system_prompt.txt'
            system.write_text('Task-specific instructions', encoding='utf-8')
            image = Path(directory) / 'ref.png'
            Image.new('RGBA', (32, 32), (255, 0, 0, 0)).save(image)
            for paths, task, tokens in (([], 't2i', 16256), ([str(image)], 'edit', 24000)):
                processor = SimpleNamespace(
                    apply_chat_template=Mock(return_value={'input_ids': torch.tensor([[1, 2]])}),
                    tokenizer=SimpleNamespace(eos_token_id=0, decode=Mock(return_value='private reasoning</think>{"rewritten_prompt":"new prompt","wh_ratio":"16:9"}')))
                model = Mock()
                model.eval.return_value = model
                model.generate.return_value = torch.tensor([[1, 2, 3]])
                state = torch.random.get_rng_state().clone()
                with patch.object(rewrite, 'hf_hub_download', return_value=str(system)), \
                     patch.object(rewrite.AutoProcessor, 'from_pretrained', return_value=processor) as load_processor, \
                     patch.object(rewrite.AutoModelForImageTextToText, 'from_pretrained', return_value=model):
                    self.assertEqual(rewrite.rewrite_prompt('original', paths, 42), 'new prompt')
                self.assertEqual(load_processor.call_args.args[0], rewrite.REWRITER_REPOS[task])
                self.assertEqual(model.generate.call_args.kwargs['max_new_tokens'], tokens)
                self.assertEqual(len(model.generate.call_args.kwargs['logits_processor']), int(task == 't2i'))
                self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
                if paths:
                    content = processor.apply_chat_template.call_args.args[0][1]['content']
                    self.assertEqual(content[0]['image'].getpixel((0, 0)), (255, 255, 255))

    def test_dynamic_reference_caches_fail_before_dataset_or_model_loading(self):
        from toolkit.data_loader import AiToolkitDataset
        holder = QwenImage2Model('cpu', ModelConfig(arch='qwen_image_2', name_or_path='test'), dtype='fp32')
        config = DatasetConfig(folder_path='unused', cache_text_embeddings=True, control_from_same_folder=True)
        with self.assertRaisesRegex(ValueError, 'fixed reference'):
            AiToolkitDataset(config, sd=holder)

    def test_rgba_augmentations_transform_alpha_spatially_not_by_color(self):
        holder = QwenImage2Model('cpu', ModelConfig(arch='qwen_image_2', name_or_path='test'), dtype='fp32')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'target.png'
            image = Image.new('RGBA', (64, 64), (90, 30, 50, 64))
            image.paste((90, 30, 50, 200), (32, 0, 64, 64))
            image.save(path)
            config = DatasetConfig(folder_path=directory, control_path=directory, resolution=64, buckets=True,
                replay_transforms=True, augmentations=[{'method': 'HorizontalFlip', 'params': {'p': 1}},
                    {'method': 'RandomBrightnessContrast', 'params': {'brightness_limit': 0.1, 'contrast_limit': 0, 'p': 1}}])
            item = FileItemDTO(path=str(path), dataset_config=config, sd=holder)
            item.scale_to_width = item.scale_to_height = item.crop_width = item.crop_height = 64
            item.crop_x = item.crop_y = 0
            item.load_and_process_image(to_tensor)
            self.assertEqual(item.tensor.shape, (4, 64, 64))
            torch.testing.assert_close(item.tensor[3], item.control_tensor[3])
            self.assertAlmostEqual(float(item.tensor[3, 0, 0]), 200 / 255, places=6)
            self.assertAlmostEqual(float(item.tensor[3, 0, -1]), 64 / 255, places=6)

    def test_comfy_bridge_preserves_reference_array_and_rgba(self):
        from toolkit.comfy.workflows import image_config_to_dict
        from toolkit.comfy.aitk_comfy_nodes.aitk_generate_image import AITKGenerateImage
        references = [str(i) for i in range(10)]
        config = GenerateImageConfig(prompt='test', output_folder='unused', output_ext='png', ctrl_imgs=references)
        self.assertEqual(image_config_to_dict(config)['ctrl_imgs'], references)
        class Model:
            output_rgba = True
            def __init__(self, **kwargs): pass
            @staticmethod
            def get_train_scheduler(): return None
            def load_model(self): pass
            def generate_images(self, configs, **kwargs):
                Image.new('RGBA', (4, 4), (10, 20, 30, 40)).save(configs[0].get_image_path(0, 0))
        with patch('toolkit.util.get_model.get_model_class', return_value=Model):
            result = AITKGenerateImage().generate(json.dumps({
                'model_config': {'arch': 'qwen_image_2', 'name_or_path': 'test', 'low_vram': True},
                'image_config': image_config_to_dict(config)}))[0]
        self.assertEqual(result.shape, (1, 4, 4, 4))
        self.assertAlmostEqual(float(result[0, 0, 0, 3]), 40 / 255, places=6)


if __name__ == '__main__':
    unittest.main()
