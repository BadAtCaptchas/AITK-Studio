"""Bounded, read-only runtime diagnostics. Never installs packages or loads weights."""
import argparse
import importlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def diagnose(arch=None):
    """Report runtime health and repair actions without downloading model weights."""
    checks = []
    def add(name, ok, detail, action=None):
        checks.append(dict(name=name, ok=ok, detail=detail, action=None if ok else action))
    supported = sys.version_info[:2] in {(3, 11), (3, 12)}
    add('Python', supported, f'{platform.python_version()} at {sys.executable}', 'Select a supported isolated Python 3.11 or 3.12 environment.')
    versions = {}
    for module, distribution in [('torch', 'torch'), ('torchvision', 'torchvision'), ('torchaudio', 'torchaudio'),
                                 ('diffusers', 'diffusers'), ('transformers', 'transformers'),
                                 ('safetensors', 'safetensors'), ('psutil', 'psutil'), ('yaml', 'PyYAML')]:
        try:
            importlib.import_module(module)
            versions[module] = importlib.metadata.version(distribution)
            add(module, True, versions[module])
        except Exception as error:
            add(module, False, f'{type(error).__name__}: {error}', 'Repair this interpreter with scripts/install_runtime.py for your hardware profile.')
    device = {'cuda': False, 'mps': False, 'count': 0}
    try:
        import torch
        device = {'cuda': torch.cuda.is_available(), 'mps': torch.backends.mps.is_available(),
                  'count': torch.cuda.device_count(), 'cudaVersion': torch.version.cuda}
        add('Accelerator', device['cuda'] or device['mps'], json.dumps(device), 'Verify the GPU driver and matching Torch profile; CPU-only diagnostics cannot validate training.')
        # Exercise binary operators without model construction.
        from torchvision.ops import nms
        nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5)
        add('Torch binary compatibility', True, 'torchvision CPU operator executed')
    except Exception as error:
        add('Torch binary compatibility', False, f'{type(error).__name__}: {error}', 'Install torch/vision/audio together from the selected profile.')
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True, timeout=30, check=False)
        add('Package dependencies', result.returncode == 0, (result.stdout or result.stderr).strip()[:8192], 'Resolve the reported dependency conflicts in this environment.')
    except Exception as error:
        add('Package dependencies', False, str(error), 'Run python -m pip check using this interpreter.')
    executable = shutil.which('ffmpeg')
    add('FFmpeg', bool(executable), executable or 'Not found on PATH', 'Install FFmpeg and add its executable to the managed app PATH for video/audio workflows.')
    if arch:
        try:
            from toolkit.model_registry import resolve_model
            model = resolve_model(arch)
            add('Selected model import', True, f'{model.__module__}.{model.__name__}')
        except Exception as error:
            add('Selected model import', False, f'{type(error).__name__}: {error}', 'Repair the selected integration before launching this model.')
    return dict(version=1, ready=all(item['ok'] for item in checks), executable=sys.executable,
                prefix=sys.prefix, toolkitRoot=str(ROOT), platform=platform.platform(), versions=versions,
                device=device, checks=checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arch')
    args = parser.parse_args()
    report = diagnose(args.arch)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['ready'] else 1)
