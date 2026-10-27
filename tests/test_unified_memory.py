"""CPU-only regression coverage for integrated NVIDIA detection and module moves."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from toolkit.util import unified_memory


class UnifiedMemoryTests(unittest.TestCase):
    def test_detection_and_opt_out(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(torch.version, 'hip', None), \
             patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(is_integrated=True)):
            for platform in ('linux', 'win32'):
                with self.subTest(platform=platform), patch.object(unified_memory.sys, 'platform', platform):
                    self.assertTrue(unified_memory.is_unified_memory())
                    with patch.dict(os.environ, {'AITK_DISABLE_UNIFIED_MEMORY': '1'}):
                        self.assertFalse(unified_memory.is_unified_memory())
                    with patch.object(torch.cuda, 'is_available', return_value=False):
                        self.assertFalse(unified_memory.is_unified_memory())
                    with patch.object(torch.version, 'hip', 'rocm'):
                        self.assertFalse(unified_memory.is_unified_memory())
                    with patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(is_integrated=False)):
                        self.assertFalse(unified_memory.is_unified_memory())
                    with patch.object(torch.cuda, 'get_device_properties', side_effect=RuntimeError('driver unavailable')):
                        self.assertFalse(unified_memory.is_unified_memory())
            with patch.object(unified_memory.sys, 'platform', 'darwin'):
                self.assertFalse(unified_memory.is_unified_memory())

    def test_patch_keeps_dtype_tensor_moves_and_is_idempotent(self):
        original_to, original_cpu, tensor_cpu = torch.nn.Module.to, torch.nn.Module.cpu, torch.Tensor.cpu
        try:
            with patch.object(unified_memory, '_patched', False), \
                 patch.object(unified_memory, 'is_unified_memory', return_value=True), \
                 patch.object(torch.cuda, 'get_device_name', return_value='RTX Spark'):
                unified_memory.apply_unified_memory_patches()
                installed = torch.nn.Module.to
                unified_memory.apply_unified_memory_patches()
                self.assertIs(torch.nn.Module.to, installed)
                model = torch.nn.Linear(2, 2)
                self.assertIs(model.cpu(), model)
                self.assertIs(model.to('cpu'), model)
                model.to(device='cpu', dtype=torch.float64)
                self.assertEqual(model.weight.dtype, torch.float64)
                self.assertIs(torch.Tensor.cpu, tensor_cpu)
                self.assertEqual(torch.ones(1).cpu().device.type, 'cpu')
        finally:
            torch.nn.Module.to, torch.nn.Module.cpu = original_to, original_cpu
