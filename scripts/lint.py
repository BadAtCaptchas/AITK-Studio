"""Run Pylint on every Git-visible Python source file in the project."""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Use the project interpreter and namespace-aware source module names."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-format', default='text')
    parser.add_argument('--output', help='Write the Pylint report to this file')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if importlib.util.find_spec('pylint') is None:
        bundles = sorted((Path.home() / '.vscode' / 'extensions').glob(
            'ms-python.pylint-*/bundled/libs'
        ))
        if not bundles:
            parser.error('Install requirements_dev.txt in the project environment to run Pylint.')
        # Append so editor-vendored dependencies cannot shadow training packages.
        sys.path.append(str(bundles[-1]))
    from pylint.lint import Run

    inventory = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        cwd=ROOT,
    ).decode('utf-8').split('\0')
    files = sorted({name for name in inventory if name.endswith('.py') and (ROOT / name).is_file()})
    # Keep explicit __init__ modules so Pylint does not recursively expand a
    # package into ignored artifacts or misidentify nested namespace packages.
    modules = [name[:-3].replace('/', '.') for name in files]
    options = [f'--rcfile={ROOT / ".pylintrc"}', f'--output-format={args.output_format}']
    if args.output:
        options.append(f'--output={args.output}')
    print(f'Pylint: checking {len(files)} Python source files.', file=sys.stderr, flush=True)
    return Run(options + modules, exit=False).linter.msg_status


if __name__ == '__main__':
    raise SystemExit(main())
