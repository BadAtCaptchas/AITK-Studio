import os
import shutil
import socket
import subprocess
import sys
import time
from typing import Optional

from toolkit.comfy.client import ComfyClient
from toolkit.comfy.errors import ComfyConfigError, ComfyError
from toolkit.comfy.install_progress import ComfyInstallProgressReporter
from toolkit.network_policy import is_offline_mode_enabled
from toolkit.paths import TOOLKIT_ROOT


COMFY_REPO_URL = 'https://github.com/comfyanonymous/ComfyUI.git'


class ExternalComfyRuntime:
    def __init__(self, comfy_config):
        self.comfy_config = comfy_config

    def start(self) -> ComfyClient:
        if not self.comfy_config.server_url:
            raise ComfyConfigError("comfy.server_url is required when comfy.mode is external")
        client = ComfyClient(
            self.comfy_config.server_url,
            timeout=self.comfy_config.timeout,
            poll_interval=self.comfy_config.poll_interval,
        )
        client.system_stats()
        return client

    def close(self):
        pass


class ManagedComfyRuntime:
    def __init__(self, comfy_config):
        self.comfy_config = comfy_config
        self.root = os.path.abspath(
            comfy_config.root
            or os.environ.get('AITK_COMFY_ROOT')
            or os.path.join(TOOLKIT_ROOT, '.aitk_comfy', 'ComfyUI')
        )
        self.process: Optional[subprocess.Popen] = None
        self.server_url: Optional[str] = None
        self.progress = ComfyInstallProgressReporter.from_env()

    def is_installed(self) -> bool:
        return os.path.exists(os.path.join(self.root, 'main.py'))

    def ensure_installed(self):
        self.progress.write(
            status='checking',
            step='detect',
            message='Checking managed ComfyUI install',
            percent=0,
            root=self.root,
        )
        if self.is_installed():
            self.progress.write(
                status='installing',
                step='custom_nodes',
                message='Refreshing AI Toolkit ComfyUI node',
                percent=80,
                root=self.root,
            )
            self._install_custom_nodes_with_progress('Refreshing AI Toolkit ComfyUI node')
            self.progress.write(
                status='ready',
                step='installed',
                message='Managed ComfyUI is installed',
                percent=85,
                root=self.root,
            )
            return
        if not self.comfy_config.managed_install:
            message = (
                "Managed ComfyUI is not installed. Enable managed_install to allow AI Toolkit "
                "to clone and install its trainer-owned ComfyUI."
            )
            self.progress.failed('opt_in_required', message, root=self.root)
            raise ComfyConfigError(
                "Managed ComfyUI is not installed. Set comfy.managed_install: true to allow AI Toolkit "
                "to clone and install its trainer-owned ComfyUI, or use comfy.mode: external with server_url."
            )
        self.install()

    def install(self):
        if is_offline_mode_enabled():
            message = "Managed ComfyUI install is blocked by offline mode"
            self.progress.failed('offline-mode', message, root=self.root)
            raise ComfyConfigError(message)
        os.makedirs(os.path.dirname(self.root), exist_ok=True)
        self._run_install_command(
            ['git', 'clone', COMFY_REPO_URL, self.root],
            step='clone',
            message='Downloading ComfyUI',
            percent=10,
        )
        if self.comfy_config.ref:
            self._run_install_command(
                ['git', '-C', self.root, 'checkout', self.comfy_config.ref],
                step='checkout',
                message=f'Checking out ComfyUI ref {self.comfy_config.ref}',
                percent=25,
            )
        self._run_install_command(
            [sys.executable, '-m', 'venv', self.venv_dir],
            step='venv',
            message='Creating ComfyUI Python environment',
            percent=35,
        )
        self._run_install_command(
            [self.python_exe, '-m', 'pip', 'install', '-r', os.path.join(self.root, 'requirements.txt')],
            step='dependencies',
            message='Installing ComfyUI Python dependencies',
            percent=50,
        )
        self.progress.write(
            status='installing',
            step='custom_nodes',
            message='Installing AI Toolkit ComfyUI node',
            percent=80,
            root=self.root,
        )
        self._install_custom_nodes_with_progress('Installing AI Toolkit ComfyUI node')
        self.progress.write(
            status='ready',
            step='installed',
            message='Managed ComfyUI install complete',
            percent=85,
            root=self.root,
        )

    def install_custom_nodes(self):
        src = os.path.join(TOOLKIT_ROOT, 'toolkit', 'comfy', 'aitk_comfy_nodes')
        dest = os.path.join(self.root, 'custom_nodes', 'aitk_comfy_nodes')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copytree(
            src,
            dest,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'),
        )

    def start(self) -> ComfyClient:
        self.ensure_installed()
        port = _find_free_port()
        self.server_url = f'http://127.0.0.1:{port}'
        self.progress.write(
            status='launching',
            step='launch',
            message='Launching managed ComfyUI',
            percent=90,
            root=self.root,
        )
        env = os.environ.copy()
        env['AITK_TOOLKIT_ROOT'] = TOOLKIT_ROOT
        args = [
            self.python_exe,
            os.path.join(self.root, 'main.py'),
            '--listen',
            '127.0.0.1',
            '--port',
            str(port),
            '--disable-auto-launch',
            '--output-directory',
            os.path.join(self.root, '.aitk_output'),
            '--temp-directory',
            os.path.join(self.root, '.aitk_temp'),
            '--user-directory',
            os.path.join(self.root, '.aitk_user'),
        ]
        self.process = subprocess.Popen(args, cwd=self.root, env=env)
        client = ComfyClient(
            self.server_url,
            timeout=self.comfy_config.timeout,
            poll_interval=self.comfy_config.poll_interval,
        )
        self._wait_until_ready(client)
        self.progress.write(
            status='completed',
            step='ready',
            message='Managed ComfyUI is ready',
            percent=100,
            root=self.root,
        )
        return client

    def close(self):
        if self.process is None:
            return
        try:
            if self.server_url:
                ComfyClient(self.server_url, timeout=5).interrupt()
        except Exception:
            pass
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=15)
        self.process = None

    @property
    def venv_dir(self) -> str:
        return os.path.join(self.root, '.venv')

    @property
    def python_exe(self) -> str:
        if os.name == 'nt':
            return os.path.join(self.venv_dir, 'Scripts', 'python.exe')
        return os.path.join(self.venv_dir, 'bin', 'python')

    def _wait_until_ready(self, client: ComfyClient):
        self.progress.write(
            status='launching',
            step='healthcheck',
            message='Waiting for managed ComfyUI to become ready',
            percent=95,
            root=self.root,
        )
        deadline = time.time() + min(self.comfy_config.timeout, 120)
        last_error = None
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                self.progress.failed(
                    'healthcheck',
                    f"Managed ComfyUI exited early with code {self.process.returncode}",
                    root=self.root,
                )
                raise ComfyError(f"Managed ComfyUI exited early with code {self.process.returncode}")
            try:
                client.system_stats()
                return
            except Exception as e:
                last_error = e
                time.sleep(0.5)
        self.progress.failed(
            'healthcheck',
            f"Managed ComfyUI did not become ready: {last_error}",
            root=self.root,
            error=str(last_error),
        )
        raise ComfyError(f"Managed ComfyUI did not become ready: {last_error}")

    def _run_install_command(self, args, step: str, message: str, percent: float):
        self.progress.write(
            status='installing',
            step=step,
            message=message,
            percent=percent,
            root=self.root,
        )
        try:
            subprocess.check_call(args)
        except Exception as exc:
            self.progress.failed(step, f"{message} failed", root=self.root, error=str(exc))
            raise

    def _install_custom_nodes_with_progress(self, message: str):
        try:
            self.install_custom_nodes()
        except Exception as exc:
            self.progress.failed('custom_nodes', f"{message} failed", root=self.root, error=str(exc))
            raise


def runtime_for_config(comfy_config):
    if comfy_config.mode == 'managed':
        return ManagedComfyRuntime(comfy_config)
    return ExternalComfyRuntime(comfy_config)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]
