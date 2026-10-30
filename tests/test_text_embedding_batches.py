"""CPU tensor/disk regressions for batched text caching; no weights or GPU jobs."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.dataloader_mixins import TextEmbeddingCachingMixin, TEXT_EMBED_CACHE_BATCH
from toolkit.prompt_utils import PromptEmbeds


def advanced(captions):
    embeds = AdvancedPromptEmbeds(
        tokens=[torch.full((len(c) + 1, 2), float(len(c))) for c in captions],
        pooled=[torch.tensor([float(len(c))]) for c in captions],
        ids=[torch.tensor([len(c)], dtype=torch.int64) for c in captions],
        mask=[torch.ones(len(c) + 1, dtype=torch.bool) for c in captions],
    )
    embeds.frozen_dtype_keys = ['pooled', 'ids', 'mask']
    return embeds


class Encoder:
    device = 'cpu'
    device_torch = torch.device('cpu')
    torch_dtype = torch.float32
    has_multiple_control_images = False
    encode_first_frame_in_text_embeddings = False

    def __init__(self, mode='advanced'):
        self.mode = mode
        self.calls = []
        self.presets = []

    def set_device_state_preset(self, preset):
        self.presets.append(preset)

    def encode_prompt(self, captions, **kwargs):
        self.calls.append((captions, kwargs))
        if self.mode == 'error':
            raise ValueError('encoder failed')
        if self.mode == 'oom' or (self.mode == 'batch_oom' and isinstance(captions, list)):
            raise torch.OutOfMemoryError('test OOM')
        values = captions if isinstance(captions, list) else [captions]
        if self.mode == 'legacy':
            return PromptEmbeds(torch.tensor([[[float(len(c))]] for c in values]))
        result = advanced(values)
        if self.mode == 'bad_split':
            result.pooled = result.pooled[:1]
        return result


class Item:
    encode_control_in_text_embeddings = False
    control_path = None
    control_video_paths = []
    is_video = False
    dopsd_self_ref = False
    is_text_embedding_cached = False

    def __init__(self, root, name, caption):
        self.root, self.name, self.caption = root, name, caption
        self.caption_dop = 'class ' + caption
        self.caption_dopsd = 'reference ' + caption
        self.path = str(root / (name + '.png'))
        self.tensor = None

    def cache(self, suffix):
        return str(self.root / (self.name + '-' + suffix + '.safetensors'))

    def get_text_embedding_path(self, recalculate=False): return self.cache('normal')
    def get_dop_text_embedding_path(self, recalculate=False): return self.cache('dop')
    def get_blank_text_embedding_path(self, recalculate=False): return self.cache('blank')
    def get_dop_blank_text_embedding_path(self, recalculate=False): return self.cache('dop-blank')
    def get_dopsd_text_embedding_path(self, recalculate=False): return self.cache('teacher')
    def get_dopsd_blank_text_embedding_path(self, recalculate=False): return self.cache('teacher-blank')
    def get_dropout_caption(self): return ''
    def get_dop_dropout_caption(self): return 'class'
    def get_dopsd_dropout_caption(self): return 'reference'
    def load_and_process_image(self, *_args, **_kwargs):
        self.tensor = torch.ones((2, 3, 2, 2) if self.is_video else (3, 2, 2))
    def cleanup(self): self.tensor = None


class TextEmbeddingBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_cache(self, items, encoder=None, **options):
        encoder = encoder or Encoder()
        config = dict(diff_output_preservation=False, caption_dropout_rate=0, do_i2v=False)
        config.update(options)
        dataset = SimpleNamespace(file_list=items, sd=encoder, dataset_path=str(self.root),
                                  dataset_config=SimpleNamespace(**config), transform=None)
        TextEmbeddingCachingMixin.cache_text_embeddings(dataset)
        return encoder

    def test_split_concat_disk_round_trip_keeps_frozen_and_integral_dtypes(self):
        original = advanced(['x', 'long caption', ''])
        for count in (None, 3):
            parts = AdvancedPromptEmbeds.split_prompt_embeds(original, count)
            self.assertEqual(len(parts), 3)
            restored = []
            for i, part in enumerate(parts):
                filename = str(self.root / f'{i}.safetensors')
                part.save(filename)
                restored.append(AdvancedPromptEmbeds.load(filename).to(dtype=torch.float16))
            joined = AdvancedPromptEmbeds.concat_prompt_embeds(restored)
            self.assertEqual(joined.frozen_dtype_keys, original.frozen_dtype_keys)
            for key in original.keys():
                for actual, expected in zip(joined[key], original[key]):
                    expected = expected.half() if key == 'tokens' else expected
                    torch.testing.assert_close(actual, expected)
            parts[0].frozen_dtype_keys.append('tokens')
            self.assertNotIn('tokens', original.frozen_dtype_keys)
            self.assertNotIn('tokens', parts[1].frozen_dtype_keys)
        with self.assertRaises(ValueError):
            AdvancedPromptEmbeds.split_prompt_embeds(original, 2)

    def test_bounded_batches_deduplicate_pending_and_published_paths(self):
        items = [Item(self.root, str(i), 'x' * (70 - i)) for i in range(70)]
        items.insert(1, Item(self.root, '0', 'x' * 70))
        items.append(Item(self.root, '0', 'x' * 70))
        encoder = self.run_cache(items)
        sizes = [len(c) if isinstance(c, list) else 1 for c, _ in encoder.calls]
        self.assertEqual(sizes, [16, 16, 16, 16, 6])
        self.assertLessEqual(max(sizes), TEXT_EMBED_CACHE_BATCH)
        self.assertEqual(sum(sizes), 70)
        for captions, kwargs in encoder.calls:
            self.assertEqual(kwargs, {})
            self.assertEqual(captions, sorted(captions, key=len))
        for item in items:
            cached = AdvancedPromptEmbeds.load(item.get_text_embedding_path())
            self.assertEqual(cached.pooled[0].item(), len(item.caption))
            self.assertTrue(item.is_text_embedding_cached)
        reused = self.run_cache(items)
        self.assertEqual(reused.calls, [])
        self.assertEqual(reused.presets, [])

    def test_legacy_result_retries_individually_and_keeps_single_item_input(self):
        encoder = Encoder('legacy')
        items = [Item(self.root, str(i), 'x' * (i + 1)) for i in range(18)]
        self.run_cache(items, encoder)
        self.assertEqual(len(encoder.calls[0][0]), 16)
        self.assertTrue(all(isinstance(c, str) for c, _ in encoder.calls[1:]))
        self.assertEqual(len(encoder.calls), 19)
        for item in items:
            cached = PromptEmbeds.load(item.get_text_embedding_path())
            self.assertEqual(cached.text_embeds.item(), len(item.caption))

    def test_batch_oom_retries_single_items(self):
        items = [Item(self.root, str(i), 'x' * (i + 1)) for i in range(3)]
        encoder = self.run_cache(items, Encoder('batch_oom'))
        self.assertEqual(len(encoder.calls), 4)
        self.assertTrue(all(isinstance(c, str) for c, _ in encoder.calls[1:]))
        self.assertTrue(all(item.is_text_embedding_cached for item in items))

    def test_encoder_errors_single_oom_and_bad_split_do_not_publish_or_mark_cached(self):
        for mode, exception in [('error', ValueError), ('oom', torch.OutOfMemoryError), ('bad_split', ValueError)]:
            with self.subTest(mode=mode):
                items = [Item(self.root, mode + str(i), 'x' * (i + 1)) for i in range(2)]
                encoder = Encoder(mode)
                with self.assertRaises(exception): self.run_cache(items, encoder)
                self.assertTrue(all(not item.is_text_embedding_cached for item in items))
                self.assertTrue(all(not Path(item.get_text_embedding_path()).exists() for item in items))
                self.assertEqual(len(encoder.calls), 2 if mode == 'oom' else 1)

    def test_failed_later_batch_reuses_completed_files_on_retry(self):
        items = [Item(self.root, str(i), 'x' * (i + 1)) for i in range(20)]
        encoder = Encoder()
        encode = encoder.encode_prompt
        def fail_second(captions, **kwargs):
            if encoder.calls:
                raise ValueError('second batch failed')
            return encode(captions, **kwargs)
        encoder.encode_prompt = fail_second
        with self.assertRaisesRegex(ValueError, 'second batch failed'):
            self.run_cache(items, encoder)
        self.assertEqual(sum(Path(item.get_text_embedding_path()).exists() for item in items), 16)
        self.assertTrue(all(not item.is_text_embedding_cached for item in items))
        retry = self.run_cache(items)
        self.assertEqual(len(retry.calls), 1)
        self.assertEqual(len(retry.calls[0][0]), 4)
        self.assertTrue(all(item.is_text_embedding_cached for item in items))

    def test_write_failure_is_propagated_without_advertising_queued_cache(self):
        items = [Item(self.root, str(i), 'x') for i in range(2)]
        with patch.object(AdvancedPromptEmbeds, 'save', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'): self.run_cache(items)
        self.assertTrue(all(not item.is_text_embedding_cached for item in items))

    def test_existing_normal_cache_still_populates_dropout_and_dop(self):
        item = Item(self.root, 'preservation', 'subject')
        advanced([item.caption]).save(item.get_text_embedding_path())
        encoder = self.run_cache([item], diff_output_preservation=True, caption_dropout_rate=0.5)
        self.assertEqual(set(encoder.calls[0][0]), {'', 'class', item.caption_dop})
        for suffix, caption in [('normal', item.caption), ('dop', item.caption_dop), ('blank', ''), ('dop-blank', 'class')]:
            self.assertEqual(AdvancedPromptEmbeds.load(item.cache(suffix)).pooled[0].item(), len(caption))

    def test_reference_video_and_first_frame_paths_remain_unbatched(self):
        reference = Item(self.root, 'ref', 'reference')
        reference.encode_control_in_text_embeddings = True
        reference.control_video_paths = ['reference.mp4']
        first_frame = Item(self.root, 'first', 'first')
        first_frame.is_video = True
        encoder = Encoder()
        encoder.encode_first_frame_in_text_embeddings = True
        self.run_cache([reference, first_frame], encoder, do_i2v=True, caption_dropout_rate=0.5, diff_output_preservation=True)
        self.assertEqual(len(encoder.calls), 8)
        self.assertTrue(all(isinstance(c, str) and 'control_images' in kw for c, kw in encoder.calls))
        self.assertEqual(encoder.calls[0][1]['control_images'], 'reference.mp4')
        torch.testing.assert_close(encoder.calls[4][1]['control_images'], torch.ones(1, 3, 2, 2))
        self.assertIsNone(first_frame.tensor)

    def test_self_reference_teacher_stays_separate_from_plain_student_batch(self):
        item = Item(self.root, 'self', 'student')
        item.dopsd_self_ref = True
        encoder = self.run_cache([item], caption_dropout_rate=0.5)
        teacher = [(c, kw) for c, kw in encoder.calls if kw]
        student = [(c, kw) for c, kw in encoder.calls if not kw]
        self.assertEqual([c for c, _ in teacher], [item.caption_dopsd, 'reference'])
        self.assertEqual(student[0][0], ['', 'student'])
        self.assertTrue(all(isinstance(c, str) for c, _ in teacher))
        self.assertTrue(Path(item.cache('teacher')).exists())
        self.assertTrue(Path(item.cache('teacher-blank')).exists())
