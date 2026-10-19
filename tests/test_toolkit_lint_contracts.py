"""Behavioral coverage for optional backends and compiled model dispatch."""
import builtins
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

import torch

from toolkit.audio.melbandroformer.model import MelBandRoformer
from toolkit.memory_management import manager_modules
from toolkit.models.v2._mixin import OstrisModelMixin


class QuantizedTensorDetectionTests(unittest.TestCase):
    def test_dense_integer_and_missing_weights(self):
        self.assertFalse(manager_modules._is_quantized_tensor(None))
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.assertFalse(manager_modules._is_quantized_tensor(torch.ones(2, dtype=dtype)))
        for dtype in (torch.int8, torch.uint8, torch.int64):
            with self.subTest(dtype=dtype):
                self.assertTrue(manager_modules._is_quantized_tensor(torch.ones(2, dtype=dtype)))

    def test_native_quantized_tensor_is_recognized_before_wrapper_fallback(self):
        weight = torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0]), 0.1, 0, torch.qint8)
        with mock.patch.object(manager_modules, '_is_ao_quantized_tensor',
                               wraps=manager_modules._is_ao_quantized_tensor) as wrapper_check:
            self.assertTrue(manager_modules._is_quantized_tensor(weight))
        wrapper_check.assert_not_called()

    def test_float_wrappers_are_recognized_by_their_quantization_metadata(self):
        class Params4bit(torch.nn.Parameter):
            """Minimal bitsandbytes parameter protocol without an optional backend."""

        for weight in (Params4bit(torch.ones(2), requires_grad=False), torch.ones(2)):
            if not isinstance(weight, Params4bit):
                weight._scale = torch.tensor(0.1)
            with self.subTest(wrapper=type(weight).__name__):
                self.assertFalse(weight.is_quantized)
                self.assertTrue(weight.dtype.is_floating_point)
                self.assertTrue(manager_modules._is_quantized_tensor(weight))


class CompiledAudioCoreTests(unittest.TestCase):
    def test_repeated_compilation_preserves_outputs_and_gradients(self):
        torch.manual_seed(8)
        model = MelBandRoformer(dim=8, depth=1, num_bands=4, dim_head=4, heads=2,
            time_transformer_depth=1, freq_transformer_depth=1,
            attn_dropout=0.0, ff_dropout=0.0, sample_rate=8000,
            stft_n_fft=64, stft_hop_length=16, stft_win_length=64).eval()
        features = torch.randn(2, 3, model.freq_indices.numel() * 2, requires_grad=True)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        expected = model.core(features)
        expected_grads = torch.autograd.grad(expected.square().mean(), [features, *parameters])
        original_compile = torch.compile
        targets = []

        def compile_target(target, **kwargs):
            targets.append(target)
            return original_compile(target, **kwargs)

        with mock.patch('torch.compile', side_effect=compile_target):
            for _ in range(2):
                self.assertIs(model.compile_core(backend='eager'), model)
                actual_input = features.detach().clone().requires_grad_()
                actual = model.core(actual_input)
                actual_grads = torch.autograd.grad(actual.square().mean(), [actual_input, *parameters])
                torch.testing.assert_close(actual, expected)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(actual_grad, expected_grad)
        self.assertEqual(len(targets), 2)
        for target in targets:
            self.assertIs(target.__self__, model)
            self.assertIs(target.__func__, MelBandRoformer._core)


class ReadaheadTests(unittest.TestCase):
    def test_missing_function_or_flag_does_not_open_a_file(self):
        for advise, flag in ((None, 3), (object(), 3), (mock.Mock(), None)):
            with self.subTest(advise=advise, flag=flag), \
                    mock.patch.object(os, 'posix_fadvise', advise, create=True), \
                    mock.patch.object(os, 'POSIX_FADV_WILLNEED', flag, create=True), \
                    mock.patch.object(os, 'open') as opened, mock.patch.object(os, 'close') as closed:
                OstrisModelMixin._readahead('weights.safetensors')
                opened.assert_not_called()
                closed.assert_not_called()

    def test_open_descriptor_is_closed_on_success_and_advice_failure(self):
        for error in (None, OSError('unsupported filesystem')):
            advise = mock.Mock(side_effect=error)
            with self.subTest(error=error), \
                    mock.patch.object(os, 'posix_fadvise', advise, create=True), \
                    mock.patch.object(os, 'POSIX_FADV_WILLNEED', 0, create=True), \
                    mock.patch.object(os, 'open', return_value=37) as opened, \
                    mock.patch.object(os, 'close') as closed:
                OstrisModelMixin._readahead('weights.safetensors')
                opened.assert_called_once_with('weights.safetensors', os.O_RDONLY)
                advise.assert_called_once_with(37, 0, 0, 0)
                closed.assert_called_once_with(37)

    def test_failed_open_does_not_close_an_unowned_descriptor(self):
        advise = mock.Mock()
        with mock.patch.object(os, 'posix_fadvise', advise, create=True), \
                mock.patch.object(os, 'POSIX_FADV_WILLNEED', 3, create=True), \
                mock.patch.object(os, 'open', side_effect=OSError('missing file')), \
                mock.patch.object(os, 'close') as closed:
            OstrisModelMixin._readahead('weights.safetensors')
            advise.assert_not_called()
            closed.assert_not_called()


class OptionalOrbitBackendTests(unittest.TestCase):
    def test_without_triton_import_is_safe_and_public_calls_fail_clearly(self):
        path = Path(__file__).resolve().parents[1] / 'toolkit/util/orbit_triton.py'
        spec = importlib.util.spec_from_file_location('orbit_without_triton_for_test', path)
        module = importlib.util.module_from_spec(spec)
        original_import = builtins.__import__

        def missing_triton(name, *args, **kwargs):
            if name == 'triton' or name.startswith('triton.'):
                raise ModuleNotFoundError("No module named 'triton'", name='triton')
            return original_import(name, *args, **kwargs)

        with mock.patch('builtins.__import__', side_effect=missing_triton):
            spec.loader.exec_module(module)
        value = torch.zeros(2, 4, dtype=torch.float16)
        permutation, signs = torch.arange(4), torch.ones(4)
        packed, norms, codebook = torch.zeros(4, 2, dtype=torch.uint8), torch.ones(4), torch.zeros(16)
        self.assertFalse(module.is_available())
        self.assertFalse(module.can_use(value))
        for name in ('_rpbh_prepare_forward_kernel', '_fwht_stage_kernel',
                     '_rpbh_finish_inverse_kernel', '_orbit4_forward_kernel', '_orbit4_backward_input_kernel'):
            self.assertIsNone(getattr(module, name))
        calls = (
            lambda: module.rpbh_forward(value, permutation, signs, 4),
            lambda: module.rpbh_inverse(value, permutation, signs, 4),
            lambda: module.linear_forward(value, packed, norms, codebook, None, 4, 4),
            lambda: module.linear_backward_input(value, packed, norms, codebook, 4, 4),
        )
        for call in calls:
            with self.assertRaisesRegex(RuntimeError, 'Triton is not installed'):
                call()


if __name__ == '__main__':
    unittest.main()
