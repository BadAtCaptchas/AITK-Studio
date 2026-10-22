"""Validate Python test ownership and run approved suites in isolated processes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'scripts' / 'python_test_groups.json'
GROUPS = {'lightweight', 'runtime', 'windows-runtime', 'gpu', 'manual'}


def validate_manifest(manifest, inventory):
    """Reject missing ownership, stale entries, and unexplained manual coverage."""
    if not isinstance(manifest, dict) or manifest.get('version') != 1:
        raise ValueError('Expected Python test manifest version 1')
    suites = manifest.get('suites')
    if not isinstance(suites, dict):
        raise ValueError('Expected a suites map')
    errors = []
    for module in sorted(inventory - suites.keys()):
        errors.append(f'Unassigned test suite: {module}')
    for module in sorted(suites.keys() - inventory):
        errors.append(f'Missing test module: {module}')
    for module, entry in suites.items():
        if not isinstance(entry, dict) or entry.get('group') not in GROUPS:
            errors.append(f'Invalid test group for {module}')
        elif entry['group'] in {'manual', 'gpu'} and not str(entry.get('reason', '')).strip():
            errors.append(f'Explain the requirements/exclusion for {module}')
    if errors:
        raise ValueError('\n'.join(errors))
    return suites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate ownership without importing tests')
    parser.add_argument('--group', action='append', choices=sorted(GROUPS - {'manual'}), default=[])
    args = parser.parse_args()
    inventory = {
        '.'.join(path.relative_to(ROOT).with_suffix('').parts)
        for folder in ('tests', 'testing')
        for path in (ROOT / folder).rglob('test_*.py')
    }
    try:
        suites = validate_manifest(json.loads(MANIFEST.read_text(encoding='utf-8')), inventory)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f'Python test ownership: {len(suites)} suites assigned.', flush=True)
    if args.check or not args.group:
        return 0
    environment = {
        **os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1', 'NO_ALBUMENTATIONS_UPDATE': '1',
    }
    if 'gpu' not in args.group:
        # A nonempty sentinel also works with Windows CUDA/NVML discovery.
        environment['CUDA_VISIBLE_DEVICES'] = '-1'
        environment['PYTORCH_NVML_BASED_CUDA_CHECK'] = '0'
    else:
        subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available(), "GPU group requires CUDA"'], check=True)
    failed = []
    for module, entry in suites.items():
        if entry['group'] not in args.group:
            continue
        print(f'\nRunning {module} ({entry["group"]})', flush=True)
        result = subprocess.run(
            [sys.executable, '-B', '-m', 'unittest', '-v', module], cwd=ROOT, env=environment,
            check=False,
        )
        if result.returncode:
            failed.append(module)
    if failed:
        print('Failed suites: ' + ', '.join(failed), file=sys.stderr)
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
