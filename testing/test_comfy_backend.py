import ast
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Union
from unittest.mock import patch

from PIL import Image


sys.modules.setdefault('torchaudio', types.SimpleNamespace(save=lambda *_args, **_kwargs: None))
sys.modules.setdefault('av', types.ModuleType('av'))
prompt_utils_module = types.ModuleType('toolkit.prompt_utils')
prompt_utils_module.PromptEmbeds = object
sys.modules.setdefault('toolkit.prompt_utils', prompt_utils_module)
if 'torchao.quantization.quant_primitives' not in sys.modules:
    torchao_module = types.ModuleType('torchao')
    quantization_module = types.ModuleType('torchao.quantization')
    quant_primitives_module = types.ModuleType('torchao.quantization.quant_primitives')
    quant_primitives_module._DTYPE_TO_BIT_WIDTH = {}
    sys.modules.setdefault('torchao', torchao_module)
    sys.modules.setdefault('torchao.quantization', quantization_module)
    sys.modules.setdefault('torchao.quantization.quant_primitives', quant_primitives_module)

from toolkit.comfy.client import ComfyClient
from toolkit.comfy.errors import ComfyConfigError
from toolkit.comfy.install_progress import PROGRESS_PATH_ENV
from toolkit.comfy.runtime import ManagedComfyRuntime
from toolkit.comfy.workflows import build_workflow
from toolkit.config_modules import ComfyConfig, GenerateImageConfig, SampleConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATE_PROCESS_PATH = PROJECT_ROOT / 'jobs' / 'process' / 'GenerateProcess.py'
GENERATE_PAGE_PATH = PROJECT_ROOT / 'ui' / 'src' / 'app' / 'generate' / 'page.tsx'
SETTINGS_API_PATH = PROJECT_ROOT / 'ui' / 'src' / 'app' / 'api' / 'settings' / 'route.ts'
SETTINGS_HOOK_PATH = PROJECT_ROOT / 'ui' / 'src' / 'hooks' / 'useSettings.tsx'
SETTINGS_PAGE_PATH = PROJECT_ROOT / 'ui' / 'src' / 'app' / 'settings' / 'page.tsx'
NEW_JOB_PAGE_PATH = PROJECT_ROOT / 'ui' / 'src' / 'app' / 'jobs' / 'new' / 'page.tsx'
SIMPLE_JOB_PATH = PROJECT_ROOT / 'ui' / 'src' / 'app' / 'jobs' / 'new' / 'SimpleJob.tsx'
JOBS_API_PATH = PROJECT_ROOT / 'ui' / 'src' / 'app' / 'api' / 'jobs' / 'route.ts'
COMFY_PROGRESS_ROUTE_PATH = PROJECT_ROOT / 'ui' / 'src' / 'app' / 'api' / 'jobs' / '[jobID]' / 'comfy-install-progress' / 'route.ts'
COMFY_PROGRESS_SERVER_PATH = PROJECT_ROOT / 'ui' / 'src' / 'server' / 'comfyInstallProgress.ts'
COMFY_PROGRESS_HOOK_PATH = PROJECT_ROOT / 'ui' / 'src' / 'hooks' / 'useJobComfyInstallProgress.tsx'
COMFY_PROGRESS_COMPONENT_PATH = PROJECT_ROOT / 'ui' / 'src' / 'components' / 'ComfyInstallProgress.tsx'
JOB_OVERVIEW_PATH = PROJECT_ROOT / 'ui' / 'src' / 'components' / 'JobOverview.tsx'
JOBS_TABLE_PATH = PROJECT_ROOT / 'ui' / 'src' / 'components' / 'JobsTable.tsx'
START_JOB_PATH = PROJECT_ROOT / 'ui' / 'cron' / 'actions' / 'startJob.ts'
INLINE_GENERATE_ROUTE_PATH = PROJECT_ROOT / 'ui' / 'src' / 'app' / 'api' / 'generate' / 'inline' / 'route.ts'
SETTINGS_COMFY_INSTALL_ROUTE_PATH = PROJECT_ROOT / 'ui' / 'src' / 'app' / 'api' / 'comfy' / 'install' / 'route.ts'
COMFY_MANAGED_INSTALL_SERVER_PATH = PROJECT_ROOT / 'ui' / 'src' / 'server' / 'comfyManagedInstall.ts'
COMFY_INSTALL_SCRIPT_PATH = PROJECT_ROOT / 'scripts' / 'install_comfy.py'


def load_generate_config_class():
    source = GENERATE_PROCESS_PATH.read_text(encoding='utf-8')
    module = ast.parse(source, filename=str(GENERATE_PROCESS_PATH))
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == 'GenerateConfig'
    )
    test_module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(test_module)
    namespace = {
        'Any': Any,
        'ComfyConfig': ComfyConfig,
        'Dict': Dict,
        'List': List,
        'Union': Union,
        'json': json,
        'os': os,
        'random': __import__('random'),
    }
    exec(compile(test_module, str(GENERATE_PROCESS_PATH), 'exec'), namespace)
    return namespace['GenerateConfig']


class ComfyConfigDefaultsTest(unittest.TestCase):
    def test_sample_config_defaults_to_native_without_managed_install(self):
        sample = SampleConfig()

        self.assertEqual(sample.backend, 'native')
        self.assertEqual(sample.comfy.mode, 'external')
        self.assertFalse(sample.comfy.managed_install)

    def test_generate_config_defaults_to_native_without_managed_install(self):
        GenerateConfig = load_generate_config_class()
        generate = GenerateConfig(prompts=['a prompt'])

        self.assertEqual(generate.backend, 'native')
        self.assertEqual(generate.comfy.mode, 'external')
        self.assertFalse(generate.comfy.managed_install)

    def test_managed_runtime_requires_explicit_install_opt_in(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ComfyConfig(mode='managed', root=os.path.join(tmpdir, 'ComfyUI'))
            runtime = ManagedComfyRuntime(config)

            with self.assertRaisesRegex(ComfyConfigError, 'managed_install: true'):
                runtime.ensure_installed()

    def test_managed_runtime_missing_opt_in_writes_install_progress_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = os.path.join(tmpdir, '.comfy_install_progress.json')
            config = ComfyConfig(mode='managed', root=os.path.join(tmpdir, 'ComfyUI'))

            with patch.dict(os.environ, {PROGRESS_PATH_ENV: progress_path}):
                runtime = ManagedComfyRuntime(config)
                with self.assertRaisesRegex(ComfyConfigError, 'managed_install: true'):
                    runtime.ensure_installed()

            with open(progress_path, 'r', encoding='utf-8') as handle:
                progress = json.load(handle)

            self.assertEqual(progress['status'], 'failed')
            self.assertEqual(progress['step'], 'opt_in_required')
            self.assertIn('Enable managed_install', progress['message'])
            self.assertEqual(progress['root'], os.path.join(tmpdir, 'ComfyUI'))


class ComfyWorkflowTest(unittest.TestCase):
    def test_custom_workflow_bindings_update_api_workflow(self):
        workflow = {
            '1': {'class_type': 'CLIPTextEncode', 'inputs': {'text': ''}},
            '2': {'class_type': 'KSampler', 'inputs': {'seed': 0, 'steps': 1, 'cfg': 1.0}},
        }
        comfy = ComfyConfig(
            workflow=workflow,
            bindings={
                'prompt': '1.inputs.text',
                'seed': '2.inputs.seed',
                'sample_steps': '2.inputs.steps',
                'guidance_scale': '2.inputs.cfg',
            },
        )
        gen = GenerateImageConfig(
            prompt='bound prompt',
            seed=123,
            num_inference_steps=12,
            guidance_scale=4.5,
            output_folder='unused',
            output_ext='png',
        )

        bound = build_workflow(comfy, gen, sampler='ddpm', model_config={})

        self.assertEqual(bound['1']['inputs']['text'], 'bound prompt')
        self.assertEqual(bound['2']['inputs']['seed'], 123)
        self.assertEqual(bound['2']['inputs']['steps'], 12)
        self.assertEqual(bound['2']['inputs']['cfg'], 4.5)
        self.assertEqual(workflow['1']['inputs']['text'], '')


class _FakeComfyHandler(BaseHTTPRequestHandler):
    prompt_payloads = []
    free_calls = 0
    interrupt_calls = 0

    @classmethod
    def reset(cls):
        cls.prompt_payloads = []
        cls.free_calls = 0
        cls.interrupt_calls = 0

    def do_GET(self):
        if self.path == '/system_stats':
            self._send_json({'system': {}})
            return
        if self.path == '/object_info':
            self._send_json({'SaveImage': {}, 'AITKGenerateImage': {}})
            return
        if self.path.startswith('/history/test-prompt-id'):
            self._send_json(
                {
                    'test-prompt-id': {
                        'outputs': {
                            '2': {
                                'images': [
                                    {'filename': 'sample.png', 'subfolder': '', 'type': 'output'},
                                ]
                            }
                        }
                    }
                }
            )
            return
        if self.path.startswith('/view'):
            image = Image.new('RGB', (8, 8), color=(32, 64, 96))
            buf = BytesIO()
            image.save(buf, format='PNG')
            payload = buf.getvalue()
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length else '{}'
        payload = json.loads(body)
        if self.path == '/prompt':
            self.prompt_payloads.append(payload)
            self._send_json({'prompt_id': 'test-prompt-id'})
            return
        if self.path == '/free':
            type(self).free_calls += 1
            self._send_json({})
            return
        if self.path == '/interrupt':
            type(self).interrupt_calls += 1
            self._send_json({})
            return
        self.send_error(404)

    def log_message(self, *_args):
        pass

    def _send_json(self, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeComfyServerTest(unittest.TestCase):
    def test_client_submits_prompt_fetches_history_and_saves_image(self):
        _FakeComfyHandler.reset()
        server = HTTPServer(('127.0.0.1', 0), _FakeComfyHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = ComfyClient(f'http://127.0.0.1:{server.server_port}', timeout=2, poll_interval=0.01)
            with tempfile.TemporaryDirectory() as tmpdir:
                gen = GenerateImageConfig(
                    prompt='a saved image',
                    output_path=os.path.join(tmpdir, '[time]_000000000_0.png'),
                    output_ext='png',
                    add_prompt_file=True,
                )

                history = client.prompt_and_wait({'1': {'class_type': 'SaveImage', 'inputs': {}}}, 'client-id')
                saved_paths = client.save_history_images(history, gen)
                client.free()

                self.assertEqual(len(_FakeComfyHandler.prompt_payloads), 1)
                self.assertEqual(len(saved_paths), 1)
                self.assertTrue(os.path.exists(saved_paths[0]))
                self.assertTrue(os.path.exists(gen.get_prompt_path()))
                self.assertEqual(_FakeComfyHandler.free_calls, 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


class GenerateProcessComfyTest(unittest.TestCase):
    def test_comfy_generate_process_skips_native_model_initialization(self):
        source = GENERATE_PROCESS_PATH.read_text(encoding='utf-8')

        self.assertIn("if self.generate_config.backend != 'comfy':", source)
        self.assertIn('self._init_native_model()', source)
        self.assertIn("if self.generate_config.backend == 'comfy':", source)
        self.assertIn('self._run_comfy_generation(sampler_groups)', source)
        self.assertIn('self._run_native_generation(sampler_groups)', source)


class ComfyUiOptInTest(unittest.TestCase):
    def test_ui_defaults_to_native_and_requires_managed_install_toggle(self):
        generate_page = GENERATE_PAGE_PATH.read_text(encoding='utf-8')
        simple_job = SIMPLE_JOB_PATH.read_text(encoding='utf-8')

        self.assertIn("useState<GenerationBackend>('native')", generate_page)
        self.assertIn("backend: 'native' as const", generate_page)
        self.assertIn('label="Install Managed ComfyUI"', generate_page)
        self.assertIn('comfy.managed_install = comfyAutoInstall || comfyManagedInstall', generate_page)
        self.assertIn('label="Install Managed ComfyUI"', simple_job)
        self.assertIn("'config.process[0].sample.comfy.managed_install'", simple_job)

    def test_settings_persist_comfy_auto_install_and_apply_to_jobs(self):
        settings_api = SETTINGS_API_PATH.read_text(encoding='utf-8')
        settings_hook = SETTINGS_HOOK_PATH.read_text(encoding='utf-8')
        settings_page = SETTINGS_PAGE_PATH.read_text(encoding='utf-8')
        generate_page = GENERATE_PAGE_PATH.read_text(encoding='utf-8')
        new_job_page = NEW_JOB_PAGE_PATH.read_text(encoding='utf-8')
        simple_job = SIMPLE_JOB_PATH.read_text(encoding='utf-8')

        self.assertIn('COMFY_AUTO_INSTALL', settings_api)
        self.assertIn('normalizeBooleanSetting(COMFY_AUTO_INSTALL, false)', settings_api)
        self.assertIn("COMFY_AUTO_INSTALL: 'false'", settings_hook)
        self.assertIn('Auto-install managed ComfyUI', settings_page)
        self.assertIn("settings.COMFY_AUTO_INSTALL === 'true'", generate_page)
        self.assertIn('comfyAutoInstall || comfyManagedInstall', generate_page)
        self.assertIn('applyComfyAutoInstallSetting', new_job_page)
        self.assertIn("generationConfig.comfy.managed_install = true", new_job_page)
        self.assertIn('disabled={comfyAutoInstall}', simple_job)

    def test_comfy_install_progress_is_visible_in_job_ui(self):
        jobs_api = JOBS_API_PATH.read_text(encoding='utf-8')
        route = COMFY_PROGRESS_ROUTE_PATH.read_text(encoding='utf-8')
        server = COMFY_PROGRESS_SERVER_PATH.read_text(encoding='utf-8')
        hook = COMFY_PROGRESS_HOOK_PATH.read_text(encoding='utf-8')
        component = COMFY_PROGRESS_COMPONENT_PATH.read_text(encoding='utf-8')
        overview = JOB_OVERVIEW_PATH.read_text(encoding='utf-8')
        jobs_table = JOBS_TABLE_PATH.read_text(encoding='utf-8')
        start_job = START_JOB_PATH.read_text(encoding='utf-8')
        inline_route = INLINE_GENERATE_ROUTE_PATH.read_text(encoding='utf-8')
        generate_page = GENERATE_PAGE_PATH.read_text(encoding='utf-8')

        self.assertIn('AITK_COMFY_INSTALL_PROGRESS_PATH', start_job)
        self.assertIn('AITK_COMFY_INSTALL_PROGRESS_PATH', inline_route)
        self.assertIn('withComfyInstallProgress', jobs_api)
        self.assertIn('getComfyInstallProgress', route)
        self.assertIn("COMFY_INSTALL_PROGRESS_FILE = '.comfy_install_progress.json'", server)
        self.assertIn('/comfy-install-progress', hook)
        self.assertIn('ComfyInstallProgressBand', component)
        self.assertIn('ComfyInstallProgressBand', overview)
        self.assertIn('ComfyInstallProgressInline', jobs_table)
        self.assertIn('Preparing managed ComfyUI, then generating image', generate_page)

    def test_settings_can_start_managed_comfy_install_now(self):
        settings_page = SETTINGS_PAGE_PATH.read_text(encoding='utf-8')
        route = SETTINGS_COMFY_INSTALL_ROUTE_PATH.read_text(encoding='utf-8')
        server = COMFY_MANAGED_INSTALL_SERVER_PATH.read_text(encoding='utf-8')
        script = COMFY_INSTALL_SCRIPT_PATH.read_text(encoding='utf-8')

        self.assertIn('/api/comfy/install', settings_page)
        self.assertIn('Download / Install Now', settings_page)
        self.assertIn('ComfyInstallProgressBand', settings_page)
        self.assertIn('startComfyManagedInstall', route)
        self.assertIn('install_comfy.py', server)
        self.assertIn('AITK_COMFY_INSTALL_PROGRESS_PATH', server)
        self.assertIn('ManagedComfyRuntime', script)
        self.assertIn('managed_install=True', script)


if __name__ == '__main__':
    unittest.main()
