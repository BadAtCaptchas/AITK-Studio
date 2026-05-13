import os
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple


BLACKWELL_MIN_CUDA = (12, 8)
OLDER_GPU_RECOMMENDED_TORCH = (2, 8)
BLACKWELL_RECOMMENDED_TORCH = (2, 10)
HIDREAM_O1_NOT_RECOMMENDED_TORCH = (2, 9)
OLDER_GPU_INSTALL_COMMAND = (
    "pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 "
    "torchaudio==2.8.0 torchcodec==0.7.0 "
    "--index-url https://download.pytorch.org/whl/cu128 "
    "--extra-index-url https://pypi.org/simple"
)
BLACKWELL_INSTALL_COMMAND = (
    "pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 "
    "torchaudio==2.10.0 torchcodec==0.10.0 "
    "--index-url https://download.pytorch.org/whl/cu128 "
    "--extra-index-url https://pypi.org/simple"
)


def parse_cuda_version(version: Optional[str]) -> Optional[Tuple[int, int]]:
    if not version:
        return None
    match = re.match(r"^\s*(\d+)(?:\.(\d+))?", version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def parse_torch_version(version: Optional[str]) -> Optional[Tuple[int, int]]:
    if not version:
        return None
    match = re.match(r"^\s*(\d+)\.(\d+)", version)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def is_hidream_o1_torch_not_recommended(torch_module: Any = None) -> bool:
    if torch_module is None:
        import torch as torch_module
    torch_version = parse_torch_version(getattr(torch_module, "__version__", None))
    return torch_version == HIDREAM_O1_NOT_RECOMMENDED_TORCH


def format_hidream_o1_torch_warning(torch_module: Any = None) -> str:
    if torch_module is None:
        import torch as torch_module
    torch_version = getattr(torch_module, "__version__", "unknown")
    return (
        "HiDream-O1-Image does not recommend PyTorch 2.9.x. "
        f"Installed torch: {torch_version}. "
        "Use the older-GPU stack with torch==2.8.0, or the Blackwell stack "
        "with torch==2.10.0 when your GPU requires Blackwell CUDA kernels."
    )


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
        "warnings": [],
    }

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return report

    report["arch_list"] = _get_supported_arch_list(torch_module)
    cuda_version = parse_cuda_version(report["torch_cuda"])
    torch_version = parse_torch_version(report["torch_version"])
    device_count = cuda.device_count()
    has_blackwell = False

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
        has_blackwell = True

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

    if (
        not has_blackwell
        and torch_version is not None
        and torch_version != OLDER_GPU_RECOMMENDED_TORCH
    ):
        report["warnings"].append(
            {
                "type": "older_gpu_alternate_torch",
                "installed_torch": report["torch_version"],
                "recommended_torch": "2.8.x",
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


def format_cuda_compatibility_warnings(report: Dict[str, Any]) -> List[str]:
    messages = []
    for warning in report.get("warnings", []):
        if warning.get("type") == "older_gpu_alternate_torch":
            messages.append(
                "AI Toolkit detected a non-Blackwell NVIDIA GPU with a PyTorch "
                f"version outside the older-GPU recommendation. Installed torch: "
                f"{warning.get('installed_torch')}. The older-GPU known-good stack is:\n"
                f"{OLDER_GPU_INSTALL_COMMAND}\n"
                "Continuing without blocking training."
            )
    return messages


def check_blackwell_cuda_compatibility(torch_module: Any = None, warn: bool = True):
    if os.environ.get("AI_TOOLKIT_SKIP_CUDA_COMPAT_CHECK", "0") == "1":
        return None

    report = get_cuda_compatibility_report(torch_module)
    if report["problems"]:
        raise RuntimeError(format_cuda_compatibility_error(report))
    if warn:
        for message in format_cuda_compatibility_warnings(report):
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    return report
