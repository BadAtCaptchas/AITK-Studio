"""Weight-free regressions for runtime defects exposed by project-wide Pylint."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from toolkit.util.nvfp4_quant import swizzle_nvfp4_scales, unswizzle_nvfp4_scales
from toolkit.util.ostris_quant import OstrisLinear
from toolkit.util.quantize import attach_ara_and_quantize, quantize, quantize_module


class PylintNativeInferenceTests(unittest.TestCase):
    """The native export plugin must remain precise about unknown members."""

    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('pylint') is None:
            bundles = sorted((Path.home() / '.vscode' / 'extensions').glob('ms-python.pylint-*/bundled/libs'))
            if not bundles:
                raise unittest.SkipTest('Install requirements_dev.txt to test native Pylint inference')
            sys.path.append(str(bundles[-1]))

    def lint(self, source, errors_only=True):
        """Run the real project profile on a disposable source file."""
        from pylint.lint import Run
        from pylint.reporters.collecting_reporter import CollectingReporter

        with tempfile.TemporaryDirectory(prefix='aitk-native-lint-') as folder:
            filename = Path(folder) / 'native_exports.py'
            filename.write_text(source, encoding='utf-8')
            options = [f'--rcfile={Path(__file__).resolve().parents[1] / ".pylintrc"}']
            if errors_only:
                options.append('--errors-only')
            reporter = CollectingReporter()
            Run(options + [str(filename)], reporter=reporter, exit=False)
            return reporter.messages

    def test_opencv_classes_retain_real_methods_and_reject_unknown_members(self):
        """Native constructors infer instances without accepting arbitrary methods."""
        good = (
            'import cv2\n'
            'capture = cv2.VideoCapture()\n'
            'capture.isOpened()\n'
            'capture.read()\n'
            'capture.release()\n'
            'writer = cv2.VideoWriter()\n'
            'writer.isOpened()\n'
            'writer.release()\n'
        )
        self.assertEqual(self.lint(good), [])
        messages = self.lint(good + 'capture.not_a_real_capture_method()\ncv2.not_a_real_export()\n')
        self.assertEqual([message.symbol for message in messages], ['no-member', 'no-member'])
        self.assertIn('not_a_real_capture_method', messages[0].msg)
        self.assertIn('not_a_real_export', messages[1].msg)

    def test_torch_native_exports_and_unknown_names(self):
        """Installed top-level and namespace functions work; typos remain errors."""
        good = (
            'import torch\n'
            'import torch.nn.functional as functional\n'
            'value = torch.ones(2, 2)\n'
            'torch.istft(value, 2)\n'
            'torch.linalg.vector_norm(value)\n'
            'torch.fft.fft(value)\n'
            'torch.special.erf(value)\n'
            'functional.scaled_dot_product_attention(value, value, value)\n'
        )
        self.assertEqual(self.lint(good), [])
        messages = self.lint(good + 'torch.not_a_real_export(value)\nfunctional.not_a_real_export(value)\n')
        self.assertEqual([message.symbol for message in messages], ['no-member', 'no-member'])

    def test_errors_only_suppresses_warnings_without_disabling_errors(self):
        """Default lint retains practical warnings; the opt-in mode retains errors."""
        source = 'value = f"constant"\nprint(not_defined)\n'
        default = {message.symbol for message in self.lint(source, errors_only=False)}
        errors = {message.symbol for message in self.lint(source)}
        self.assertIn('f-string-without-interpolation', default)
        self.assertIn('undefined-variable', default)
        self.assertEqual(errors, {'undefined-variable'})

    def test_tensor_native_methods_and_context_manager(self):
        """Parameter methods stay callable and the native context manager is recognized."""
        source = (
            'import torch\n'
            'parameter = torch.nn.Parameter(torch.ones(2))\n'
            'with torch._C.DisableTorchFunctionSubclass():\n'
            '    value = parameter.detach()\n'
        )
        self.assertEqual(self.lint(source), [])

    def test_installed_framework_constructor_and_checkpoint_contracts(self):
        """Generated configuration kwargs and enabled checkpoint calls are real APIs."""
        source = (
            'from transformers.models.clip.configuration_clip import CLIPVisionConfig\n'
            'from transformers.models.qwen2.configuration_qwen2 import Qwen2Config\n'
            'from transformers.models.qwen3.configuration_qwen3 import Qwen3Config\n'
            'from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig\n'
            'from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5VisionConfig\n'
            'from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig\n'
            'from transformers.models.gemma3.configuration_gemma3 import Gemma3Config\n'
            'from diffusers.models.modeling_utils import ModelMixin\n'
            'CLIPVisionConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=1)\n'
            'Qwen2Config(vocab_size=16, hidden_size=8)\n'
            'Qwen3Config(vocab_size=16, hidden_size=8)\n'
            'Qwen3_5TextConfig(vocab_size=16, hidden_size=8, layer_types=["full_attention"])\n'
            'Qwen3_5Config(text_config={}, image_token_id=6)\n'
            'Qwen3_5VisionConfig(depth=1, hidden_size=32, num_heads=4, spatial_merge_size=1)\n'
            'Qwen3VLConfig(text_config={})\n'
            'Gemma3Config(mm_tokens_per_image=16, bos_token_id=2)\n'
            'class Checkpointed(ModelMixin):\n'
            '    def run(self, block, value):\n'
            '        if self.gradient_checkpointing:\n'
            '            return self._gradient_checkpointing_func(block, value)\n'
            '        return block(value)\n'
        )
        self.assertEqual(self.lint(source), [])

    def test_gradio_generated_events_keep_bounded_keyword_signatures(self):
        """Actual event kwargs work without accepting a misspelled keyword."""
        source = (
            'from gradio.components.file import File\n'
            'from gradio.components.button import Button\n'
            'files = File()\n'
            'files.upload(fn=None, inputs=[], outputs=[])\n'
            'files.delete(fn=None)\n'
            'files.clear(fn=None)\n'
            'button = Button()\n'
            'button.click(fn=None, queue=False)\n'
        )
        self.assertEqual(self.lint(source), [])
        messages = self.lint(source + 'button.click(not_a_real_keyword=True)\n')
        self.assertEqual([message.symbol for message in messages], ['unexpected-keyword-arg'])

    def test_lazy_exports_only_add_runtime_declared_names(self):
        """Lazy registry and tokenizer aliases retain their actual canonical targets."""
        from scripts.pylint_inference import _lazy_exports, _model_registry_exports
        aliases = _lazy_exports('transformers', ('T5TokenizerFast', 'Qwen2TokenizerFast'))
        self.assertIn('T5TokenizerFast', aliases.locals)
        self.assertIn('Qwen2TokenizerFast', aliases.locals)
        self.assertNotIn('NotARealTokenizer', aliases.locals)
        self.assertIn('T5Tokenizer as T5TokenizerFast', aliases.as_string())
        registry = _model_registry_exports()
        self.assertIn('MossImageModel', registry.locals)
        self.assertIn('QwenImage2Model', registry.locals)
        self.assertNotIn('NotARealModel', registry.locals)

    def test_triton_extern_return_inference(self):
        """The optional installed rint extern returns a value in kernels."""
        if importlib.util.find_spec('triton') is None:
            self.skipTest('Triton is optional')
        self.assertEqual(self.lint(
            'from triton.language.extra import libdevice\n'
            'def kernel(value):\n'
            '    rounded = libdevice.rint(value)\n'
            '    return rounded\n'
        ), [])

    def test_unavailable_optional_triton_does_not_break_lint(self):
        """Missing packages and unloadable native binaries keep Triton optional."""
        from scripts.pylint_inference import _triton_rint

        for error in (ImportError('missing Triton'), OSError('native runtime unavailable')):
            with self.subTest(error=type(error).__name__):
                with patch('scripts.pylint_inference.importlib.import_module', side_effect=error):
                    self.assertEqual(_triton_rint('triton.language.extra.libdevice').body, [])

    def test_runner_forwards_errors_only_only_when_requested(self):
        """Keep the existing runner defaults and report-format arguments intact."""
        from scripts import lint

        for errors_only in (False, True):
            argv = ['lint.py', '--output-format=json', '--output=report.json']
            if errors_only:
                argv.append('--errors-only')
            result = SimpleNamespace(linter=SimpleNamespace(msg_status=2))
            with (
                patch.object(sys, 'argv', argv),
                patch.object(sys, 'path', list(sys.path)),
                patch.object(lint.subprocess, 'check_output', return_value=b'scripts/lint.py\0'),
                patch('pylint.lint.Run', return_value=result) as run,
            ):
                self.assertEqual(lint.main(), 2)
            options = run.call_args.args[0]
            self.assertEqual('--errors-only' in options, errors_only)
            self.assertIn('--output-format=json', options)
            self.assertIn('--output=report.json', options)
            self.assertIn('scripts.lint', options)
            self.assertEqual(run.call_args.kwargs, {'exit': False})


class QuantizationLintRegressionTests(unittest.TestCase):
    """Exercise quantization paths with tiny CPU layers and in-memory adapters."""

    def test_nvfp4_scale_export_inverts_the_existing_checkpoint_decoder(self):
        """Preserve every scale byte for both aligned and padded tile dimensions."""
        for rows, cols in [(1, 1), (128, 4), (130, 9)]:
            with self.subTest(rows=rows, cols=cols):
                scales = torch.arange(rows * cols).remainder(127).to(torch.uint8)
                scales = scales.reshape(rows, cols).view(torch.float8_e4m3fn)
                restored = unswizzle_nvfp4_scales(swizzle_nvfp4_scales(scales), rows, cols)
                self.assertTrue(torch.equal(restored.view(torch.uint8), scales.view(torch.uint8)))
        with self.assertRaises(ValueError):
            swizzle_nvfp4_scales(torch.zeros(2, 3, 4))

    def test_block_quantization_reuses_matching_packed_weights(self):
        """A second quantization pass must not cast or replace packed buffers."""
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([torch.nn.Linear(128, 16)])
        quantize_module(model, 'uint4', block_names=['blocks'], status_fn=lambda _: None)
        self.assertIsInstance(model.blocks[0], OstrisLinear)
        before = {name: value.clone() for name, value in model.named_buffers()}
        quantize_module(
            model, 'uint4', dtype=torch.bfloat16, block_names=['blocks'], status_fn=lambda _: None
        )
        after = dict(model.named_buffers())
        self.assertEqual(before.keys(), after.keys())
        self.assertTrue(all(torch.equal(value, after[name]) for name, value in before.items()))

    def test_configured_exclusions_do_not_duplicate_quantize_arguments(self):
        """Combine model exclusions and user exclusions without changing the input map."""
        model = torch.nn.Sequential(torch.nn.Linear(128, 16), torch.nn.Linear(16, 16))
        options = {'exclude': ['1']}
        quantize_module(model, 'uint4', exclude=['0'], quantize_kwargs=options, status_fn=lambda _: None)
        self.assertNotIsInstance(model[0], OstrisLinear)
        self.assertNotIsInstance(model[1], OstrisLinear)
        self.assertEqual(options, {'exclude': ['1']})

    def test_leaf_quantization_accepts_explicit_final_placement(self):
        """The streamed extras path supports its advertised placement keyword."""
        layer = torch.nn.Linear(128, 16)
        quantize(layer, weights='uint4', quantize_device='cpu', keep_on_quantize_device=True)
        self.assertIsInstance(layer, OstrisLinear)

    def test_both_adapter_formats_reach_shared_network_and_quantization_setup(self):
        """LoRA and LoKr adapters use locally scoped options after helper extraction."""
        states = [
            {'projection.lora_A.weight': torch.zeros(4, 128)},
            {'lycoris_projection.lokr_w1': torch.zeros(2, 2)},
        ]
        for state in states:
            with self.subTest(keys=list(state)):
                model = torch.nn.Module()
                model.projection = torch.nn.Linear(128, 16)
                model.kept = torch.nn.Linear(128, 16)
                owner = SimpleNamespace(
                    model_config=SimpleNamespace(
                        qtype='uint4', low_vram=False,
                        quantize_kwargs={'include': ['projection'], 'exclude': ['kept']},
                    ),
                    device_torch=torch.device('cpu'), torch_dtype=torch.float32,
                    is_transformer=True, print_and_status_update=lambda _: None,
                )
                network = MagicMock()
                network.unet_loras = []
                with (
                    patch('toolkit.util.quantize.os.path.exists', return_value=True),
                    patch('toolkit.util.quantize.load_file', return_value=state),
                    patch('toolkit.lora_special.LoRASpecialNetwork', return_value=network) as factory,
                ):
                    result = attach_ara_and_quantize(owner, model, 'fixture.safetensors', device='cpu')
                factory.assert_called_once()
                network.apply_to.assert_called_once()
                self.assertIs(owner.accuracy_recovery_adapter, network)
                self.assertIsInstance(model.projection, OstrisLinear)
                self.assertNotIsInstance(model.kept, OstrisLinear)
                self.assertEqual(result.quantized_modules, 1)


if __name__ == '__main__':
    unittest.main()
