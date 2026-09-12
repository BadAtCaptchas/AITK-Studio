"""Install one coherent runtime profile into the active virtual environment.

No environment is created or replaced. --dry-run checks the same complete inputs
through pip's resolver without installing; --report records pip's resolution.
"""
import argparse
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    'legacy-cu128': ('requirements_torch_legacy_cu128.txt', 'requirements.txt'),
    'blackwell-cu128': ('requirements_torch_blackwell_cu128.txt', 'requirements.txt'),
    'blackwell-cu130': ('requirements_torch_blackwell_cu130.txt', 'requirements.txt'),
    'dgx-cu130': ('requirements_torch_blackwell_cu130.txt', 'dgx_requirements.txt'),
    'macos': ('requirements_torch_macos.txt', 'requirements.txt'),
}


def check_interpreter(version=None, isolated=None):
    """Reject unsupported Python versions and installations outside an environment."""
    version = sys.version_info[:2] if version is None else version
    if version not in {(3, 11), (3, 12)}:
        raise ValueError('Use Python 3.12 (reference) or 3.11 (DGX). Python 3.13+ is not supported by these profiles.')
    if isolated is None:
        isolated = sys.prefix != sys.base_prefix or bool(os.environ.get('CONDA_PREFIX'))
    if not isolated:
        raise ValueError('Activate a virtual environment first. No packages will be installed into system Python.')


def main():
    """Resolve one hardware profile and check the resulting package dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    try:
        check_interpreter()
    except ValueError as error:
        parser.error(str(error))
    if (args.profile == 'macos') != (sys.platform == 'darwin'):
        parser.error('The macos profile is for Apple silicon; CUDA profiles require Linux or Windows.')
    command = [sys.executable, '-m', 'pip', 'install']
    for filename in PROFILES[args.profile]:
        command.extend(['-r', str(ROOT / filename)])
    if args.dry_run:
        command.extend(['--dry-run', '--ignore-installed'])
    if args.report:
        command.extend(['--report', str(args.report)])
    subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)


if __name__ == '__main__':
    main()
