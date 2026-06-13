from __future__ import annotations

from typing import Optional, Sequence

import torch

from .block_offload import BlockOffloadManager, infer_block_paths, resolve_block_layers
from .manager import MemoryManager


SUPPORTED_BLOCK_ARCHES = {
    "flux",
    "flux_kontext",
    "flux2",
    "flux2_klein_4b",
    "flux2_klein_9b",
    "asymflux2_klein_9b",
    "wan21",
    "wan21_i2v",
    "wan22_14b",
    "wan22_14b_i2v",
    "wan22_5b",
    "zimage",
    "zeta_chroma",
    "qwen_image",
    "qwen_image_plus",
    "qwen_image_edit_plus",
    "qwen_image_edit",
    "glm_image",
    "hidream",
    "hidream_e1",
    "hidream_o1",
    "ltx2",
    "ltx2.3",
    "nucleus_image",
    "ernie_image",
    "zimage_l2p",
    "ideogram4",
    "prx_pixel",
}


def normalize_arch(arch: Optional[str]) -> Optional[str]:
    if arch is None:
        return None
    return str(arch).split(":", 1)[0]


def is_block_offload_arch_supported(arch: Optional[str]) -> bool:
    return normalize_arch(arch) in SUPPORTED_BLOCK_ARCHES


def get_layer_offloading_backend(model_config) -> str:
    backend = getattr(model_config, "layer_offloading_backend", "block")
    if backend not in {"block", "legacy"}:
        return "block"
    return backend


def is_legacy_layer_offloading(model_config) -> bool:
    return bool(getattr(model_config, "layer_offloading", False)) and get_layer_offloading_backend(model_config) == "legacy"


def resolve_layer_offloading_backend(
    model_config,
    module: torch.nn.Module,
    device: torch.device,
    block_paths: Optional[Sequence[str]] = None,
) -> str:
    requested_backend = get_layer_offloading_backend(model_config)
    if requested_backend == "legacy":
        return "legacy"

    arch = normalize_arch(getattr(model_config, "arch", None))
    if arch is not None and not is_block_offload_arch_supported(arch):
        return "legacy"

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return "legacy"

    paths = list(block_paths or infer_block_paths(module))
    if not paths or not resolve_block_layers(module, paths):
        return "legacy"

    return "block"


def attach_layer_offloading(
    model_or_config,
    module: torch.nn.Module,
    device: torch.device,
    offload_percent: float = 1.0,
    component: str = "model",
    block_paths: Optional[Sequence[str]] = None,
    ignore_modules: Optional[Sequence[torch.nn.Module]] = None,
):
    model_config = getattr(model_or_config, "model_config", model_or_config)
    if not getattr(model_config, "layer_offloading", False) or offload_percent <= 0:
        return None

    backend = resolve_layer_offloading_backend(model_config, module, device, block_paths)
    if backend == "block":
        try:
            manager = BlockOffloadManager.attach(
                module=module,
                device=device,
                offload_fraction=offload_percent,
                block_paths=block_paths,
                ignore_modules=ignore_modules,
            )
        except ValueError as exc:
            label = component or "component"
            print(f"Block layer offloading unavailable for {label}: {exc} Falling back to legacy layer offloading.")
        else:
            module._aitk_layer_offloading_backend = "block"
            return manager

    manager = MemoryManager.attach(
        module=module,
        device=device,
        offload_percent=offload_percent,
        ignore_modules=list(ignore_modules or []),
    )
    module._aitk_layer_offloading_backend = "legacy"
    if component:
        module._aitk_layer_offloading_component = component
    return manager
