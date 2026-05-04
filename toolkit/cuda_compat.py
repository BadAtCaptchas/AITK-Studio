import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


BLACKWELL_MIN_CUDA = (12, 8)
BLACKWELL_INSTALL_COMMAND = (
    "pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 "
    "torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128"
)


def parse_cuda_version(version: Optional[str]) -> Optional[Tuple[int, int]]:
    if not version:
        return None
    match = re.match(r"^\s*(\d+)(?:\.(\d+))?", version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def _format_capability(capability: Sequence[int]) -> str:
    return f"sm_{int(capability[0])}{int(capability[1])}"


def _is_blackwell_capability(capability: Sequence[int]) -> bool:
    return len(capability) >= 2 and int(capability[0]) >= 12


def _get_supported_arch_list(torch_module: Any) -> List[str]:
    get_arch_list = getattr(torch_module.cuda, "get_arch_list", None)
    if get_arch_list is None:
        return []
    try:
        return list(get_arch_list())
    except Exception:
        return []


def get_cuda_compatibility_report(torch_module: Any = None) -> Dict[str, Any]:
    if torch_module is None:
        import torch as torch_module

    report = {
        "torch_version": getattr(torch_module, "__version__", "unknown"),
        "torch_cuda": getattr(getattr(torch_module, "version", None), "cuda", None),
        "arch_list": [],
        "devices": [],
        "problems": [],
    }

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return report

    report["arch_list"] = _get_supported_arch_list(torch_module)
    cuda_version = parse_cuda_version(report["torch_cuda"])
    device_count = cuda.device_count()

    for device_idx in range(device_count):
        name = cuda.get_device_name(device_idx)
        capability = tuple(cuda.get_device_capability(device_idx))
        arch_name = _format_capability(capability)
        device_info = {
            "index": device_idx,
            "name": name,
            "capability": capability,
            "arch": arch_name,
        }
        report["devices"].append(device_info)

        if not _is_blackwell_capability(capability):
            continue

        cuda_too_old = cuda_version is None or cuda_version < BLACKWELL_MIN_CUDA
        arch_missing = bool(report["arch_list"]) and arch_name not in report["arch_list"]
        if cuda_too_old or arch_missing:
            report["problems"].append(
                {
                    "device": device_info,
                    "cuda_too_old": cuda_too_old,
                    "arch_missing": arch_missing,
                }
            )

    return report


def format_cuda_compatibility_error(report: Dict[str, Any]) -> str:
    problem_lines = []
    arch_list = report.get("arch_list") or []
    supported_arches = ", ".join(arch_list) if arch_list else "unknown"
    for problem in report.get("problems", []):
        device = problem["device"]
        problem_lines.append(
            f" - {device['name']} ({device['arch']}) requires a Blackwell-compatible "
            "PyTorch CUDA wheel."
        )

    return (
        "AI Toolkit detected an incompatible PyTorch/CUDA install for Blackwell GPUs.\n"
        f"Installed torch: {report.get('torch_version')}\n"
        f"Installed torch CUDA: {report.get('torch_cuda')}\n"
        f"Compiled CUDA arches: {supported_arches}\n"
        + "\n".join(problem_lines)
        + "\n\nInstall a CUDA 12.8+ PyTorch build with:\n"
        f"{BLACKWELL_INSTALL_COMMAND}\n\n"
        "Set AI_TOOLKIT_SKIP_CUDA_COMPAT_CHECK=1 only if you are using a custom "
        "PyTorch build that you know includes Blackwell sm_120 kernels."
    )


def check_blackwell_cuda_compatibility(torch_module: Any = None):
    if os.environ.get("AI_TOOLKIT_SKIP_CUDA_COMPAT_CHECK", "0") == "1":
        return None

    report = get_cuda_compatibility_report(torch_module)
    if report["problems"]:
        raise RuntimeError(format_cuda_compatibility_error(report))
    return report

