"""Offline, weight-free MOSS batching contracts and real Qwen3.5 prefix parity."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

from extensions_built_in.diffusion_models import MossImageModel
from toolkit.config_modules import ModelConfig
from toolkit.models.v2.text_encoders.qwen35 import Qwen35TextOnlyEncoder, Qwen35VLEncoder


def tokenizer():
    backend = Tokenizer(WordLevel({'[UNK]': 0, '[EOS]': 1, 'a': 2, 'b': 3, 'c': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single='$A [EOS]', special_tokens=[('[EOS]', 1)])
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]', pad_token='[EOS]',
        padding_side='left', chat_template='{{ messages[0]["content"] }}',
    )


class RecordingEncoder(torch.nn.Module):
    """Causal deterministic stand-in to observe shapes, arguments and storage."""
    def __init__(self):
        super().__init__()
        self.register_buffer('anchor', torch.zeros(()))
        self.calls = []
        self.outputs = []

    @property
    def device(self):
        return self.anchor.device

    def forward(self, input_ids, use_cache):
        assert use_cache is False
        self.calls.append(input_ids.clone())
        states = input_ids.float().cumsum(-1).unsqueeze(-1).expand(-1, -1, 4).contiguous()
        self.outputs.append(states)
        return SimpleNamespace(last_hidden_state=states)


def model(batch_size=1):
    result = MossImageModel('cpu', ModelConfig(arch='moss_image', name_or_path='unused',
                           model_kwargs={'text_encoder_batch_size': batch_size}), dtype='fp32')
    result.tokenizer = tokenizer()
    result.text_encoder = RecordingEncoder()
    result.max_text_length = 6
    return result


class MossTextBatchingTests(unittest.TestCase):
    def test_option_validation_and_sequential_default(self):
        default = MossImageModel('cpu', ModelConfig(arch='moss_image', name_or_path='unused'), dtype='fp32')
        self.assertEqual(default.text_encoder_batch_size, 1)
        batched = model(2)
        self.assertEqual(default.get_text_embedding_space_version(), batched.get_text_embedding_space_version())
        self.assertEqual(default.get_latent_space_version(), batched.get_latent_space_version())
        for value in (0, -1, True, False, 1.5, '2', None, 33):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'text_encoder_batch_size'):
                model(value)
        self.assertEqual(model(32).text_encoder_batch_size, 32)
        sequential = model()
        sequential.get_prompt_embeds(['a', 'a b'])
        self.assertEqual([tuple(x.shape) for x in sequential.text_encoder.calls], [(1, 2), (1, 3)])

    def test_chunks_lengths_eos_padding_dtype_and_owned_storage(self):
        prompts = ['a', 'b', 'a b c', '', 'a ' * 12, 'c', 'b c']
        sequential, batched = model(), model(3)
        expected = sequential.get_prompt_embeds(prompts).text_embeds
        actual = batched.get_prompt_embeds(prompts).text_embeds
        self.assertEqual([len(x) for x in actual], [2, 2, 4, 1, 6, 2, 3])
        self.assertEqual([tuple(x.shape) for x in batched.text_encoder.calls], [(3, 4), (3, 6), (1, 3)])
        for index, (got, want) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(got, want)
            self.assertEqual(got.dtype, torch.float32)
            if index < 6:
                self.assertEqual(got.untyped_storage().nbytes(), got.numel() * got.element_size())
                self.assertNotIn(got.untyped_storage().data_ptr(),
                                 [x.untyped_storage().data_ptr() for x in batched.text_encoder.outputs])
        self.assertEqual(batched.tokenizer.padding_side, 'left')
        batched.torch_dtype = torch.bfloat16
        self.assertTrue(all(x.dtype == torch.bfloat16 for x in batched.get_prompt_embeds(['a', 'b']).text_embeds))
        self.assertEqual(batched.get_prompt_embeds([]).text_embeds, [])

    def test_single_prompt_keeps_unpadded_path(self):
        item = model(4)
        item.get_prompt_embeds('a')
        self.assertEqual(tuple(item.text_encoder.calls[0].shape), (1, 2))

    def test_encoder_failure_propagates_without_retry(self):
        item = model(2)
        item.text_encoder.forward = Mock(side_effect=RuntimeError('encoder failed'))
        with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
            item.get_prompt_embeds(['a', 'b', 'c'])
        item.text_encoder.forward.assert_called_once()

    def test_image_forward_remains_uncached_and_untruncated(self):
        item = model(2)
        item.text_encoder.config = SimpleNamespace(image_token_id=4)
        item.image_processor = Mock(return_value={
            'image_grid_thw': torch.tensor([[1, 2, 2]]),
            'pixel_values': torch.ones(4, 3),
        })
        item.image_processor.merge_size = 1
        item._prep_vlm_images = Mock(return_value=[])
        item.tokenizer = Mock(return_value={'input_ids': torch.tensor([[4, 4, 4, 4, 2, 1]])})
        item.tokenizer.apply_chat_template.return_value = 'image chat'
        item.text_encoder.forward = Mock(return_value=SimpleNamespace(last_hidden_state=torch.ones(1, 6, 4)))
        self.assertEqual(item._encode_with_images('a', [torch.ones(3, 2, 2)]).shape, (6, 4))
        item.tokenizer.assert_called_once_with('image chat', return_tensors='pt')
        arguments = item.text_encoder.forward.call_args.kwargs
        self.assertFalse(arguments['use_cache'])
        self.assertEqual(arguments['mm_token_type_ids'].tolist(), [[1, 1, 1, 1, 0, 0]])

    def test_image_text_mixed_repeated_calls_order_and_validation(self):
        item = model(2)
        item.is_edit = True
        image = torch.ones(3, 2, 2)
        refs = [image, image]
        item._encode_with_images = Mock(return_value=torch.full((3, 4), -7.))
        for _ in range(2):
            item.get_prompt_embeds(['a', 'b'], [refs])
            self.assertEqual(item._encode_with_images.call_count, 2)
            for call in item._encode_with_images.call_args_list:
                self.assertIs(call.args[1], refs)
            item._encode_with_images.reset_mock()
            expected = item.get_prompt_embeds(['a', 'b c', '']).text_embeds
            actual = item.get_prompt_embeds(['a', 'ref', 'b c', ''], [[], refs, [], []]).text_embeds
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], torch.full((3, 4), -7.))
            torch.testing.assert_close(actual[2], expected[1])
            torch.testing.assert_close(actual[3], expected[2])
            item._encode_with_images.assert_called_once_with('ref', refs)
            item._encode_with_images.reset_mock()
        with self.assertRaisesRegex(ValueError, 'number of prompts'):
            item.get_prompt_embeds(['a', 'b', 'c'], [refs, refs])
        item.is_edit = False
        item.get_prompt_embeds(['a', 'b'], [refs])
        item._encode_with_images.assert_not_called()

    def test_default_preserves_mixed_forward_order(self):
        item = model()
        item.is_edit = True
        events = []
        item._encode_text_prompt = lambda text: events.append(text) or torch.zeros(1, 4)
        item._encode_with_images = lambda text, images: events.append(text) or torch.zeros(1, 4)
        item.get_prompt_embeds(['first', 'image', 'last'], [[], [torch.ones(3, 2, 2)], []])
        self.assertEqual(events, ['first', 'image', 'last'])

    def test_image_placeholder_mismatch_still_rejected(self):
        item = model(2)
        item.text_encoder.config = SimpleNamespace(image_token_id=4)
        item.image_processor = Mock(return_value={'image_grid_thw': torch.tensor([[1, 2, 2]])})
        item.image_processor.merge_size = 1
        item._prep_vlm_images = Mock(return_value=[])
        with self.assertRaisesRegex(ValueError, 'placeholder count'):
            item._encode_with_images('a', [torch.ones(3, 2, 2)])

    def test_real_qwen35_full_and_linear_attention_valid_prefix_parity(self):
        # Optional CUDA kernels must not be selected merely because installed.
        # Patches live only for this CPU fixture and restore production globals.
        with ExitStack() as stack:
            stack.enter_context(torch.random.fork_rng(devices=[]))
            for name in ('causal_conv1d_fn', 'causal_conv1d_update', 'chunk_gated_delta_rule',
                         'fused_recurrent_gated_delta_rule', 'FusedRMSNormGated'):
                if hasattr(qwen, name):
                    stack.enter_context(patch.object(qwen, name, None))
            torch.manual_seed(123)
            config = Qwen3_5TextConfig(
                vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                layer_types=['linear_attention', 'full_attention'],
                linear_num_key_heads=2, linear_num_value_heads=4,
                linear_key_head_dim=8, linear_value_head_dim=8,
                rope_parameters={'rope_type': 'default', 'rope_theta': 10000,
                                 'partial_rotary_factor': 1.0, 'mrope_section': [1, 1, 2]},
            )
            encoder = Qwen35TextOnlyEncoder(config).eval()
            item = model()
            item.text_encoder = encoder
            cases = [(6, ['a']), (6, ['a', 'b']),
                     (6, ['a b c', '', 'a ' * 12, 'c', 'b c']),
                     (80, ['a ' * 70, 'a b c', 'b ' * 120, ''])]
            for max_length, prompts in cases:
                with self.subTest(max_length=max_length, prompts=prompts):
                    item.max_text_length = max_length
                    item.text_encoder_batch_size = 1
                    expected = item.get_prompt_embeds(prompts).text_embeds
                    item.text_encoder_batch_size = 2
                    actual = item.get_prompt_embeds(prompts).text_embeds
                    for got, want in zip(actual, expected):
                        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-4)

            # Exercise the real multimodal tower repeatedly, including transitions
            # from reference images to text-only batches and back to mixed inputs.
            item.tokenizer.add_special_tokens({'additional_special_tokens':
                ['<|vision_start|>', '<|image_pad|>', '<|vision_end|>']})
            ids = item.tokenizer.convert_tokens_to_ids
            vision = Qwen3_5VisionConfig(depth=1, hidden_size=32, intermediate_size=64,
                                        num_heads=4, patch_size=2, temporal_patch_size=1,
                                        spatial_merge_size=1, out_hidden_size=32,
                                        num_position_embeddings=16)
            item.text_encoder = Qwen35VLEncoder(Qwen3_5Config(
                text_config=config.to_dict(), vision_config=vision.to_dict(),
                image_token_id=ids('<|image_pad|>'), vision_start_token_id=ids('<|vision_start|>'),
                vision_end_token_id=ids('<|vision_end|>'),
            )).eval()
            item.is_edit = True
            item.image_processor = Mock(return_value={
                'image_grid_thw': torch.tensor([[1, 2, 2]]), 'pixel_values': torch.randn(4, 12),
            })
            item.image_processor.merge_size = 1
            item._prep_vlm_images = Mock(return_value=[])
            references = [torch.ones(3, 4, 4)]
            item.text_encoder_batch_size = 1
            expected = item.get_prompt_embeds(['a', 'b', 'c'], [[], references, []]).text_embeds
            item.text_encoder_batch_size = 2
            for _ in range(2):
                item.get_prompt_embeds('b', references)
                item.get_prompt_embeds(['a', 'c'])
                actual = item.get_prompt_embeds(['a', 'b', 'c'], [[], references, []]).text_embeds
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-4)


if __name__ == '__main__':
    unittest.main()
