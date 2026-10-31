"""CPU LRU isolation, lifecycle, invalidation and loader-contract regressions."""
import os
import copy
import pickle
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.prompt_utils import PromptEmbeds
from toolkit.text_embedding_cache import TextEmbeddingCache, _clone_cpu, _tensor_bytes, job_text_embedding_cache


class TextEmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def save(self, name='item', value=1):
        path = str(self.root / (name + '.safetensors'))
        PromptEmbeds(torch.full((1, 8, 8), float(value))).save(path)
        return path

    def test_hit_returns_independent_containers_and_tensor_storage(self):
        path = self.save()
        cache = TextEmbeddingCache(4096)
        with patch.object(PromptEmbeds, 'load', wraps=PromptEmbeds.load) as load:
            first = cache.load(path, load)
            first.text_embeds.fill_(9)
            second = cache.load(path, load)
            second.to(dtype=torch.float64)
            third = cache.load(path, load)
            self.assertEqual(load.call_count, 1)
            self.assertTrue(torch.all(third.text_embeds == 1))
            self.assertEqual(third.text_embeds.dtype, torch.float32)
            self.assertNotEqual(second.text_embeds.data_ptr(), third.text_embeds.data_ptr())
            self.assertEqual(cache.retained_bytes, 256)


    def test_private_cpu_byte_copy_preserves_shapes_strides_and_dtypes(self):
        for dtype in [torch.bfloat16, torch.float16, torch.float32, torch.float64, torch.int8, torch.int64, torch.bool, torch.complex64]:
            values = [torch.ones((), dtype=dtype), torch.empty((0, 3), dtype=dtype), torch.ones((4, 5), dtype=dtype).T, torch.ones(10, dtype=dtype)[::2]]
            if dtype == torch.complex64:
                values.append(torch.tensor([1 + 2j, 3 - 4j], dtype=dtype).conj())
            for original in values:
                with self.subTest(dtype=dtype, shape=original.shape):
                    expected = original.clone()
                    result = _clone_cpu(original)
                    self.assertEqual(result.dtype, dtype)
                    self.assertEqual(result.shape, original.shape)
                    self.assertTrue(torch.equal(result, original))
                    result.zero_()
                    self.assertTrue(torch.equal(original, expected))

    def test_advanced_ragged_metadata_and_integer_masks_survive(self):
        path = str(self.root / 'advanced.safetensors')
        embeds = AdvancedPromptEmbeds(tokens=[torch.ones(2, 3), torch.ones(5, 3)], ids=[torch.arange(2), torch.arange(5)])
        embeds.frozen_dtype_keys = ['ids']
        # The cache also accepts ragged in-memory containers; disk files currently
        # represent one batch item, so use a custom loader for this shape.
        Path(path).write_bytes(b'fixture')
        loader = lambda _: embeds.clone()
        cache = TextEmbeddingCache(4096)
        cache.load(path, loader)
        changed = cache.load(path, loader)
        changed.frozen_dtype_keys.append('tokens')
        changed.tokens[0].zero_()
        restored = cache.load(path, loader)
        self.assertEqual(restored.frozen_dtype_keys, ['ids'])
        self.assertEqual([tuple(t.shape) for t in restored.tokens], [(2, 3), (5, 3)])
        restored.to(dtype=torch.bfloat16)
        self.assertEqual(restored.ids[0].dtype, torch.int64)
        self.assertTrue(torch.all(restored.tokens[0] == 1))

    def test_lru_evicts_by_storage_budget_and_counts_views(self):
        paths = [self.save(str(i), i) for i in range(3)]
        cache = TextEmbeddingCache(512)
        with patch.object(PromptEmbeds, 'load', wraps=PromptEmbeds.load) as load:
            for path in paths[:2]: cache.load(path, load)
            cache.load(paths[0], load)
            cache.load(paths[2], load)
            self.assertEqual(cache.retained_bytes, 512)
            cache.load(paths[0], load)
            self.assertEqual(load.call_count, 3)
            cache.load(paths[1], load)
            self.assertEqual(load.call_count, 4)
        backing = torch.zeros(100)
        self.assertEqual(_tensor_bytes([backing[:1], backing[1:2]]), 400)

    def test_tiny_entry_metadata_is_bounded_by_count_as_well_as_bytes(self):
        cache = TextEmbeddingCache(4096, max_entries=2)
        for i in range(3): cache.load(self.save(str(i)), PromptEmbeds.load)
        self.assertEqual(len(cache._entries), 2)
        self.assertEqual(cache.retained_bytes, 512)

    def test_file_replacement_edit_removal_and_failed_load_invalidate(self):
        path = self.save()
        cache = TextEmbeddingCache(4096)
        cache.load(path, PromptEmbeds.load)
        replacement = self.save('replacement', 2)
        old_stat = os.stat(path)
        os.utime(replacement, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        os.replace(replacement, path)
        self.assertEqual(cache.load(path, PromptEmbeds.load).text_embeds[0, 0, 0], 2)
        self.save(value=3)
        self.assertEqual(cache.load(path, PromptEmbeds.load).text_embeds[0, 0, 0], 3)
        Path(path).unlink()
        with self.assertRaises(FileNotFoundError): cache.load(path, PromptEmbeds.load)
        self.assertEqual(cache.retained_bytes, 0)
        self.save(value=4)
        self.assertEqual(cache.load(path, PromptEmbeds.load).text_embeds[0, 0, 0], 4)


    def test_clone_storage_expansion_cannot_exceed_budget(self):
        path = self.save()
        shared = torch.ones(64)
        original = AdvancedPromptEmbeds(left=[shared], right=[shared])
        self.assertEqual(_tensor_bytes(original), 256)
        self.assertEqual(_tensor_bytes(original.clone()), 512)
        cache = TextEmbeddingCache(256)
        cache.load(path, lambda _: original)
        self.assertEqual(cache.retained_bytes, 0)
        cache = TextEmbeddingCache(512)
        cache.load(path, lambda _: original)
        self.assertEqual(cache.retained_bytes, 512)

    def test_file_changed_during_read_is_not_admitted(self):
        path = self.save()
        cache = TextEmbeddingCache(4096)
        def changing_loader(name):
            value = PromptEmbeds.load(name)
            self.save(value=7)
            return value
        cache.load(path, changing_loader)
        self.assertEqual(cache.retained_bytes, 0)
        self.assertEqual(cache.load(path, PromptEmbeds.load).text_embeds[0, 0, 0], 7)

    def test_zero_budget_and_large_file_bypass(self):
        path = self.save()
        for cache in [TextEmbeddingCache(0), TextEmbeddingCache(128), TextEmbeddingCache(4096, max_entry_bytes=100)]:
            with patch.object(PromptEmbeds, 'load', wraps=PromptEmbeds.load) as load:
                cache.load(path, load)
                cache.load(path, load)
                self.assertEqual(load.call_count, 2)
                self.assertEqual(cache.retained_bytes, 0)

    def test_spawn_fork_and_worker_budgets_do_not_retain_parent_entries(self):
        path = self.save()
        cache = TextEmbeddingCache(4096)
        cache.load(path, PromptEmbeds.load)
        spawned = pickle.loads(pickle.dumps(cache))
        self.assertEqual(spawned.retained_bytes, 0)
        self.assertEqual(spawned.max_bytes, 4096)
        with patch('toolkit.text_embedding_cache.os.getpid', return_value=os.getpid() + 100), patch('torch.utils.data.get_worker_info', return_value=SimpleNamespace(num_workers=8)):
            with patch.object(PromptEmbeds, 'load', wraps=PromptEmbeds.load) as load:
                cache.load(path, load)
                cache.load(path, load)
                self.assertEqual(load.call_count, 2)
                self.assertEqual(cache.retained_bytes, 0)

    def test_shared_job_scope_and_explicit_disable(self):
        model = SimpleNamespace()
        with patch.dict(os.environ, {'AITK_TEXT_EMBEDDING_CACHE_MB': '0'}):
            first = job_text_embedding_cache(model)
        self.assertIs(first, job_text_embedding_cache(model))
        self.assertEqual(first.max_bytes, 0)
        self.assertIsNot(first, job_text_embedding_cache(SimpleNamespace()))
        with patch.dict(os.environ, {'AITK_TEXT_EMBEDDING_CACHE_MB': '-1'}):
            with self.assertRaises(ValueError): job_text_embedding_cache(SimpleNamespace())


    def test_sample_deepcopy_shares_job_cache_without_copying_warm_tensors(self):
        path = self.save()
        cache = TextEmbeddingCache(4096)
        original = SimpleNamespace(_cpu_text_embedding_cache=cache)
        with patch.object(PromptEmbeds, 'load', wraps=PromptEmbeds.load) as load:
            for _ in range(3):
                sample = copy.deepcopy(original)
                self.assertIs(sample._cpu_text_embedding_cache, cache)
                result = sample._cpu_text_embedding_cache.load(path, load)
                result.text_embeds.zero_()
            self.assertEqual(load.call_count, 1)

    def test_threaded_reads_never_alias_or_exceed_retained_budget(self):
        paths = [self.save(str(i), i) for i in range(8)]
        cache = TextEmbeddingCache(512)
        def read(index):
            result = cache.load(paths[index % 8], PromptEmbeds.load)
            self.assertEqual(result.text_embeds[0, 0, 0], index % 8)
            result.text_embeds.fill_(-1)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(read, range(40)))
        self.assertLessEqual(cache.retained_bytes, 512)


    def test_anima_extra_integer_fields_and_advanced_disk_metadata(self):
        from extensions_built_in.diffusion_models.anima.anima import AnimaPromptEmbeds
        path = str(self.root / 'anima.safetensors')
        original = AnimaPromptEmbeds(torch.ones(1, 3, 4), torch.arange(3).unsqueeze(0), torch.ones(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.long))
        original.save(path)
        cache = TextEmbeddingCache(4096)
        loaded = cache.load(path, PromptEmbeds.load)
        loaded.t5_input_ids.zero_()
        fresh = cache.load(path, PromptEmbeds.load)
        self.assertIsInstance(fresh, AnimaPromptEmbeds)
        fresh.to(dtype=torch.bfloat16)
        self.assertEqual(fresh.t5_input_ids.dtype, torch.int64)
        self.assertEqual(fresh.t5_input_ids.tolist(), [[0, 1, 2]])
        advanced_path = str(self.root / 'advanced-disk.safetensors')
        advanced = AdvancedPromptEmbeds(tokens=[torch.ones(2, 3)], ids=[torch.arange(2)])
        advanced.frozen_dtype_keys = ['tokens', 'ids']
        advanced.save(advanced_path)
        cache.load(advanced_path, PromptEmbeds.load)
        result = cache.load(advanced_path, PromptEmbeds.load)
        result.to(dtype=torch.bfloat16)
        self.assertEqual(result.tokens[0].dtype, torch.float32)

    def test_dropout_rng_and_dop_teacher_paths_are_unchanged(self):
        from toolkit.dataloader_mixins import TextEmbeddingFileItemDTOMixin
        paths = {key: self.save(key, i) for i, key in enumerate(['normal', 'blank', 'dop', 'dop_blank', 'teacher', 'teacher_blank'])}
        class Item(TextEmbeddingFileItemDTOMixin):
            def get_text_embedding_path(self): return paths['normal']
            def get_blank_text_embedding_path(self): return paths['blank']
            def get_dop_text_embedding_path(self): return paths['dop']
            def get_dop_blank_text_embedding_path(self): return paths['dop_blank']
            def get_dopsd_text_embedding_path(self): return paths['teacher']
            def get_dopsd_blank_text_embedding_path(self): return paths['teacher_blank']
        item = Item()
        item.is_text_embedding_cached = True
        item.dataset_config = SimpleNamespace(caption_dropout_rate=0.5, diff_output_preservation=True)
        item.dopsd_self_ref = True
        item._cpu_text_embedding_cache = TextEmbeddingCache(4096)
        with patch('toolkit.dataloader_mixins.random.random', side_effect=[0.1, 0.9, 0.1, 0.9]) as rng:
            for expected in [1, 0, 1, 0]:
                item.load_prompt_embedding()
                self.assertEqual(item.prompt_embeds.text_embeds[0, 0, 0], expected)
                self.assertEqual(item.dop_prompt_embeds.text_embeds[0, 0, 0], expected + 2)
                self.assertEqual(item.dopsd_prompt_embeds.text_embeds[0, 0, 0], expected + 4)
                item.prompt_embeds.text_embeds.zero_()
                item.cleanup_text_embedding()
            self.assertEqual(rng.call_count, 4)


if __name__ == '__main__':
    unittest.main()
