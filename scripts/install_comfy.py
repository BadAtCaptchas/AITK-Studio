import argparse
import os
import sys
from types import SimpleNamespace


sys.path.insert(0, os.getcwd())

from toolkit.network_policy import install_offline_network_guard
from toolkit.comfy.install_progress import PROGRESS_PATH_ENV, ComfyInstallProgressReporter
from toolkit.comfy.runtime import ManagedComfyRuntime

install_offline_network_guard()


def main():
    parser = argparse.ArgumentParser(description="Install the AI Toolkit managed ComfyUI runtime.")
    parser.add_argument("--root", default=None, help="Managed ComfyUI install root.")
    parser.add_argument("--ref", default=None, help="Optional ComfyUI git ref to checkout.")
    parser.add_argument("--progress", default=None, help="Progress JSON path.")
    args = parser.parse_args()

    if args.progress:
        os.environ[PROGRESS_PATH_ENV] = args.progress

    config = SimpleNamespace(
        mode="managed",
        root=args.root,
        ref=args.ref,
        managed_install=True,
    )
    runtime = ManagedComfyRuntime(config)
    reporter = runtime.progress if runtime.progress.enabled else ComfyInstallProgressReporter(args.progress)

    try:
        runtime.ensure_installed()
        reporter.write(
            status="completed",
            step="installed",
            message="Managed ComfyUI install complete",
            percent=100,
            root=runtime.root,
        )
        return 0
    except Exception as exc:
        reporter.failed(
            step="install",
            message="Managed ComfyUI install failed",
            root=getattr(runtime, "root", args.root),
            error=str(exc),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
