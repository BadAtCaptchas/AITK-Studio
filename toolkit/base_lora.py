import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.print import print_acc


DEFAULT_LORA_FILENAME = "pytorch_lora_weights.safetensors"


@dataclass
class LoadedLoraNetwork:
    path: str
    network: LoRASpecialNetwork
    network_type: str
    num_modules: int


@dataclass
class BaseLoraMergeResult:
    path: str
    strength: float
    network_type: str
    num_modules: int


def _is_localish_path(path: str) -> bool:
    return (
        os.path.isabs(path)
        or path.startswith(".")
        or "\\" in path
        or ":" in path
    )


def resolve_lora_path(path: str, label: str = "LoRA") -> str:
    if path is None or not str(path).strip():
        raise ValueError(f"{label} path is required.")

    path = str(path).strip()
    if os.path.isdir(path):
        path = os.path.join(path, DEFAULT_LORA_FILENAME)
        if not os.path.exists(path):
            raise ValueError(f"{label} directory does not contain {DEFAULT_LORA_FILENAME}: {path}")
        return path

    if os.path.exists(path):
        if not path.lower().endswith(".safetensors"):
            raise ValueError(f"{label} must be a .safetensors file: {path}")
        return path

    if _is_localish_path(path):
        raise ValueError(f"{label} path does not exist: {path}")

    parts = path.split("/")
    if len(parts) == 2 and not path.lower().endswith(".safetensors"):
        repo_id = path
        filename = DEFAULT_LORA_FILENAME
    elif len(parts) >= 3:
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
    else:
        raise ValueError(
            f"{label} path must be an existing local .safetensors file, a directory containing "
            f"{DEFAULT_LORA_FILENAME}, or a Hugging Face path like user/repo/file.safetensors."
        )

    if not filename.lower().endswith(".safetensors"):
        raise ValueError(f"{label} Hugging Face filename must end with .safetensors: {filename}")

    print_acc(f"Grabbing {label} from the hub: {path}")
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        raise ValueError(f"Failed to download {label} from {path}: {e}") from e


def _strip_lora_suffix(key: str) -> Optional[str]:
    suffixes = [
        ".lora_A.",
        ".lora_B.",
        ".lora_down.",
        ".lora_up.",
        ".lokr_w1",
        ".lokr_w2",
    ]
    for suffix in suffixes:
        if suffix in key:
            return key.split(suffix)[0]
    return None


def _infer_network_config(state_dict: Dict[str, torch.Tensor]) -> Tuple[NetworkConfig, Dict[str, Any], str]:
    if not state_dict:
        raise ValueError("Base LoRA file is empty.")

    keys = list(state_dict.keys())
    lowered_keys = [key.lower() for key in keys]
    if any("dora" in key or "magnitude" in key for key in lowered_keys):
        raise ValueError("model.base_lora_path only supports mergeable LoRA/LoKr adapters. DoRA is not supported.")

    is_lokr = any("lokr_" in key or ".lokr" in key for key in lowered_keys)
    network_kwargs: Dict[str, Any] = {}
    only_if_contains = []
    for key in keys:
        contains_key = _strip_lora_suffix(key)
        if contains_key is not None:
            if contains_key.startswith("lycoris_"):
                contains_key = contains_key.replace("lycoris_", "", 1)
            if contains_key not in only_if_contains:
                only_if_contains.append(contains_key)

    if is_lokr:
        largest_factor = 0
        for key, value in state_dict.items():
            if "lokr_w1" in key:
                largest_factor = max(largest_factor, int(value.shape[0]))
        if largest_factor <= 0:
            raise ValueError("Could not infer LoKr factor from base LoRA file.")
        config = NetworkConfig(
            type="lokr",
            lokr_full_rank=True,
            lokr_factor=largest_factor,
            transformer_only=False,
        )
    else:
        linear_dim = None
        for key, value in state_dict.items():
            if "lora_A" in key or "lora_down" in key:
                linear_dim = int(value.shape[0])
                break
        if linear_dim is None:
            raise ValueError("Could not infer LoRA rank from base LoRA file.")
        config = NetworkConfig(
            type="lora",
            linear=linear_dim,
            linear_alpha=linear_dim,
            transformer_only=False,
        )

    network_kwargs["only_if_contains"] = only_if_contains
    return config, network_kwargs, config.type


def load_lora_network_for_model(
    base_model: Any,
    model_to_train: Optional[torch.nn.Module] = None,
    path: Optional[str] = None,
    label: str = "LoRA",
    is_assistant_adapter: bool = False,
    is_ara: bool = False,
) -> LoadedLoraNetwork:
    resolved_path = resolve_lora_path(path, label=label)
    model_to_train = model_to_train if model_to_train is not None else base_model.get_model_to_train()

    raw_state_dict = load_file(resolved_path)
    if hasattr(base_model, "convert_lora_weights_before_load"):
        state_dict = base_model.convert_lora_weights_before_load(raw_state_dict)
    else:
        state_dict = raw_state_dict

    network_config, network_kwargs, network_type = _infer_network_config(state_dict)
    if hasattr(base_model, "target_lora_modules"):
        network_kwargs["target_lin_modules"] = base_model.target_lora_modules

    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=model_to_train,
        lora_dim=network_config.linear,
        multiplier=1.0,
        alpha=network_config.linear_alpha,
        train_unet=True,
        train_text_encoder=False,
        network_config=network_config,
        network_type=network_config.type,
        transformer_only=network_config.transformer_only,
        is_transformer=getattr(base_model, "is_transformer", False),
        is_sdxl=getattr(base_model, "is_xl", False) or getattr(base_model, "is_ssd", False),
        is_v2=getattr(base_model, "is_v2", False),
        is_v3=getattr(base_model, "is_v3", False),
        is_pixart=getattr(base_model, "is_pixart", False),
        is_auraflow=getattr(base_model, "is_auraflow", False),
        is_flux=getattr(base_model, "is_flux", False),
        is_lumina2=getattr(base_model, "is_lumina2", False),
        is_ssd=getattr(base_model, "is_ssd", False),
        is_vega=getattr(base_model, "is_vega", False),
        is_assistant_adapter=is_assistant_adapter,
        is_ara=is_ara,
        base_model=base_model,
        **network_kwargs,
    )

    network.apply_to(None, model_to_train, apply_text_encoder=False, apply_unet=True)
    num_modules = len(network.get_all_modules())
    if num_modules == 0:
        raise ValueError(
            f"{label} matched zero trainable modules. Check that the adapter targets this model architecture."
        )

    device = getattr(base_model, "device_torch", torch.device("cpu"))
    dtype = getattr(base_model, "torch_dtype", torch.float32)
    network.force_to(device, dtype=dtype)
    network._update_torch_multiplier()

    # The state dict has already been converted for this model, so keep load_weights from converting twice.
    base_model_ref = network.base_model_ref
    network.base_model_ref = None
    try:
        network.load_weights(state_dict)
    finally:
        network.base_model_ref = base_model_ref

    return LoadedLoraNetwork(
        path=resolved_path,
        network=network,
        network_type=network_type,
        num_modules=num_modules,
    )


def fuse_lora_network(
    loaded: LoadedLoraNetwork,
    strength: float,
    base_model: Any,
    label: str = "Base LoRA",
) -> BaseLoraMergeResult:
    if not math.isfinite(float(strength)):
        raise ValueError(f"{label} strength must be a finite number.")

    network = loaded.network
    if not network.can_merge_in:
        raise ValueError(
            f"{label} cannot be merged into this model. Mergeable LoRA/LoKr adapters must be fused before "
            "quantization or layer offloading, and DoRA/non-mergeable formats are not supported."
        )

    network.merge_in(merge_weight=float(strength))
    if not network.is_merged_in or not network.can_merge_in:
        raise ValueError(f"{label} could not be merged into the loaded base model.")

    network.is_active = False
    network.can_merge_in = False
    try:
        network.force_to("cpu", dtype=getattr(base_model, "torch_dtype", torch.float32))
    except Exception:
        pass

    fused_networks = getattr(base_model, "_fused_base_lora_networks", None)
    if fused_networks is None:
        fused_networks = []
        setattr(base_model, "_fused_base_lora_networks", fused_networks)
    fused_networks.append(network)

    return BaseLoraMergeResult(
        path=loaded.path,
        strength=float(strength),
        network_type=loaded.network_type,
        num_modules=loaded.num_modules,
    )


def fuse_base_lora_into_model(
    base_model: Any,
    model_to_train: Optional[torch.nn.Module] = None,
    path: Optional[str] = None,
    strength: Optional[float] = None,
) -> Optional[BaseLoraMergeResult]:
    model_config = getattr(base_model, "model_config", None)
    path = path if path is not None else getattr(model_config, "base_lora_path", None)
    if path is None or not str(path).strip():
        return None
    strength = strength if strength is not None else getattr(model_config, "base_lora_strength", 1.0)

    loaded = load_lora_network_for_model(
        base_model=base_model,
        model_to_train=model_to_train,
        path=path,
        label="Base LoRA",
        is_ara=True,
    )
    result = fuse_lora_network(
        loaded=loaded,
        strength=float(strength),
        base_model=base_model,
        label="Base LoRA",
    )
    setattr(base_model, "_base_lora_fused", True)
    return result

