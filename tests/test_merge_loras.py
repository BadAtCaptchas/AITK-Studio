"""Real CPU safetensors regressions for the selective November 13 port."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from toolkit.basic import flush
from ui_scripts.merge_loras import main


class MergeLorasTests(unittest.TestCase):
    def merge(self, states, strengths=None, dtype=None):
        with tempfile.TemporaryDirectory() as folder:
            entries = []
            for index, state in enumerate(states):
                source = str(Path(folder) / f'{index}.safetensors')
                save_file(state, source, metadata={'format': 'pt', 'version': str(index), 'private': 'omit'})
                entries.append({'path': source, 'strength': strengths[index] if strengths else 1})
            output = str(Path(folder) / 'nested' / 'merged.safetensors')
            args = ['merge_loras', '--loras', json.dumps(entries), '--output', output]
            if dtype:
                args += ['--save_dtype', dtype]
            stdout = io.StringIO()
            with patch('sys.argv', args), contextlib.redirect_stdout(stdout):
                status = main()
            if status:
                return status, None, None, None
            with safe_open(output, framework='pt') as opened:
                metadata = opened.metadata()
            return status, load_file(output), json.loads(stdout.getvalue().splitlines()[-1]), metadata

    def test_same_dtype_and_per_key_defaults(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                state = {'a': torch.tensor([1, 2], dtype=dtype), 'b': torch.tensor([3.25], dtype=torch.float32)}
                status, result, payload, metadata = self.merge([state, state], [0.5, 1])
                self.assertEqual(status, 0)
                self.assertEqual(result['a'].dtype, dtype)
                self.assertEqual(result['b'].dtype, torch.float32)
                torch.testing.assert_close(result['a'], state['a'] * 1.5)
                self.assertEqual(payload['save_dtype'], 'preserve')
                self.assertEqual(payload['num_keys'], 2)
                self.assertEqual(payload['num_loras'], 2)
                self.assertEqual(payload['device'], 'cpu')
                self.assertTrue(payload['ok'])
                self.assertEqual(metadata, {'format': 'pt', 'version': '0'})

    def test_mixed_half_formats_preserve_range_in_both_orders(self):
        a = {'a': torch.tensor([1e10], dtype=torch.bfloat16)}
        b = {'a': torch.tensor([1], dtype=torch.float16)}
        for states in ([a, b], [b, a]):
            _, result, _, _ = self.merge(states)
            self.assertEqual(result['a'].dtype, torch.float32)
            self.assertTrue(torch.isfinite(result['a']).all())
            torch.testing.assert_close(result['a'], a['a'].float() + b['a'].float())

    def test_accumulates_half_in_float32(self):
        states = [{'a': torch.tensor([v], dtype=torch.float16)} for v in (2048, 1, -2048)]
        _, result, _, _ = self.merge(states)
        self.assertEqual(result['a'].item(), 1)

    def test_float64_accumulation_survives_later_float32(self):
        for states in ([torch.float32, torch.float64], [torch.float64, torch.float32]):
            _, result, _, _ = self.merge([
                {'a': torch.tensor([1 + 2**-40 if dtype == torch.float64 else -1], dtype=dtype)}
                for dtype in states
            ])
            self.assertEqual(result['a'].dtype, torch.float64)
            self.assertEqual(result['a'].item(), 2**-40)

    def test_rank_padding_with_mixed_precision(self):
        _, result, _, _ = self.merge([
            {'a': torch.ones((1, 2), dtype=torch.float32)},
            {'a': torch.full((2, 1), 2, dtype=torch.float64)},
        ])
        torch.testing.assert_close(result['a'], torch.tensor([[3, 1], [2, 0]], dtype=torch.float64))

    def test_incompatible_dimensions_fail_without_output(self):
        with contextlib.redirect_stderr(io.StringIO()):
            status, result, _, _ = self.merge([{'a': torch.ones(2)}, {'a': torch.ones(2, 2)}])
        self.assertEqual(status, 2)
        self.assertIsNone(result)

    def test_explicit_dtype_aliases_remain_compatible(self):
        for name, dtype in [('float32', torch.float32), ('fp32', torch.float32),
                            ('float16', torch.float16), ('fp16', torch.float16),
                            ('bfloat16', torch.bfloat16), ('bf16', torch.bfloat16)]:
            with self.subTest(name=name):
                _, result, payload, _ = self.merge([{'a': torch.tensor([1.25], dtype=torch.float64)}], dtype=name)
                self.assertEqual(result['a'].dtype, dtype)
                self.assertEqual(result['a'].item(), 1.25)
                self.assertEqual(payload['save_dtype'], name)

    def test_gc_remains_default_and_opt_out_is_supported(self):
        with patch('toolkit.basic.gc.collect') as collect, \
             patch('torch.cuda.is_available', return_value=False), \
             patch('torch.backends.mps.is_available', return_value=False):
            flush()
            collect.assert_called_once()
            flush(garbage_collect=False)
            collect.assert_called_once()


if __name__ == '__main__':
    unittest.main()
