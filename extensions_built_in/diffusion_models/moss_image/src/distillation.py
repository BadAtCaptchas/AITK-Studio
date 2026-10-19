"""Frozen, unmerged fp32 Kromoss distillation adapters.

These are part of the selected base model, independent of the user's trainable
LoRA/LoKr. Keeping their tensors outside the transformer's module tree prevents
dtype changes, optimizers and adapter/checkpoint exports from absorbing them.
Forward hooks survive the existing trainable adapters' forward replacement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
import math
import weakref

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from toolkit.util.quantize import dequantize_if_quantized


_ATTRIBUTE = "_moss_distillation_adapter"
_PROJECTIONS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
_SUFFIXES = ("lora_down.weight", "lora_up.weight", "alpha")


@dataclass
class _FrozenDelta:
    down: torch.Tensor
    up: torch.Tensor
    scale: float

    def to(self, device: torch.device) -> None:
        self.down = self.down.to(device=device, dtype=torch.float32).detach()
        self.up = self.up.to(device=device, dtype=torch.float32).detach()

    def apply(self, inputs: torch.Tensor) -> torch.Tensor:
        self.to(inputs.device)
        # Keep the vendor's operation order, including scale before the second
        # matmul. Autocast is disabled by the enclosing hook.
        return (self.scale * (inputs @ self.down.T)) @ self.up.T


def _validated_delta(
    state_dict: Mapping[str, torch.Tensor], prefix: str, in_features: int, out_features: int
) -> _FrozenDelta:
    down, up, alpha = (state_dict[f"{prefix}.{suffix}"] for suffix in _SUFFIXES)
    for suffix, value in zip(_SUFFIXES, (down, up, alpha)):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f"Distillation tensor {prefix}.{suffix} must be float32.")
        if value.device.type == "meta" or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Distillation tensor {prefix}.{suffix} must contain finite values.")
    if down.ndim != 2 or down.shape[0] < 1 or down.shape[1] != in_features:
        raise ValueError(f"Invalid distillation down-projection shape for {prefix}.")
    rank = down.shape[0]
    if up.ndim != 2 or tuple(up.shape) != (out_features, rank):
        raise ValueError(f"Invalid distillation up-projection shape/rank for {prefix}.")
    if alpha.ndim != 0 or float(alpha) <= 0:
        raise ValueError(f"Distillation alpha for {prefix} must be a positive scalar.")
    scale = float(alpha) / rank
    if not math.isfinite(scale):
        raise ValueError(f"Invalid distillation scale for {prefix}.")
    # Own the tensors: a caller retaining or modifying the loaded state dict
    # must not change this frozen base adapter after validation.
    return _FrozenDelta(down.detach().clone(), up.detach().clone(), scale)


class FrozenMossDistillationAdapter:
    """Always-on fixed adapter, attached once to an already prepared transformer.

    ``identity`` is the caller's canonical source identity (path plus file
    stat/hash). A pooled transformer can reuse the same adapter safely; changing
    its identity requires explicitly closing the previous controller first.
    """

    def __init__(self, transformer: nn.Module, identity: str):
        self.identity = identity
        self._transformer = weakref.ref(transformer)
        self._targets: dict[str, tuple[_FrozenDelta, ...]] = {}
        self._handles: list[RemovableHandle] = []
        self._closed = False

    @classmethod
    def attach(
        cls,
        transformer: nn.Module,
        state_dict: Mapping[str, torch.Tensor],
        *,
        identity: str,
    ) -> FrozenMossDistillationAdapter:
        if not isinstance(identity, str) or not identity:
            raise ValueError("A nonempty distillation adapter source identity is required.")
        previous = getattr(transformer, _ATTRIBUTE, None)
        if previous is not None:
            if isinstance(previous, cls) and not previous._closed and previous.identity == identity:
                return previous
            raise ValueError("Transformer already has a different distillation adapter attached.")

        layers = getattr(transformer, "layers", ())
        if not len(layers):
            raise ValueError("Distillation requires a MOSS transformer with DiT layers.")
        expected = {
            f"diffusion_model.layers.{index}.proj.{projection}.{suffix}"
            for index in range(len(layers))
            for projection in _PROJECTIONS
            for suffix in _SUFFIXES
        }
        keys = set(state_dict)
        if keys != expected:
            missing, extra = sorted(expected - keys), sorted(keys - expected)
            raise ValueError(
                "Invalid distillation checkpoint keys: "
                f"missing={missing[:3]}, unexpected={extra[:3]}."
            )

        controller = cls(transformer, identity)
        modules: dict[str, nn.Module] = {}
        # Validate everything before registering hooks, so a malformed file
        # cannot leave an incompletely patched transformer behind.
        for index in range(len(layers)):
            for target in ("qkv", "o", "ffn_up_gate", "ffn_down"):
                name = f"layers.{index}.proj.{target}"
                module = transformer.get_submodule(name)
                in_features = getattr(module, "in_features", 0)
                out_features = getattr(module, "out_features", 0)
                if not isinstance(in_features, int) or not isinstance(out_features, int) or min(in_features, out_features) < 1:
                    raise ValueError(f"Distillation target {name} must be a linear projection.")
                projections = (target,)
                if target == "qkv":
                    if out_features != 3 * in_features:
                        raise ValueError(f"Distillation target {name} has invalid fused q/k/v dimensions.")
                    out_features //= 3
                    projections = ("q", "k", "v")
                controller._targets[name] = tuple(
                    _validated_delta(
                        state_dict,
                        f"diffusion_model.layers.{index}.proj.{projection}",
                        in_features,
                        out_features,
                    )
                    for projection in projections
                )
                modules[name] = module
        try:
            for name, module in modules.items():
                controller._handles.append(module.register_forward_hook(partial(controller._apply, name)))
            # A plain controller, not nn.Module: PyTorch must not register its
            # fixed tensors in parameters/buffers/state_dict.
            setattr(transformer, _ATTRIBUTE, controller)
        except Exception:
            controller.close()
            raise
        return controller

    def _apply(
        self, name: str, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> torch.Tensor:
        # Calling Tensor.dequantize() on ordinary floating activations adds an
        # operation without a backward implementation. Only quantized tensor
        # subclasses require this conversion.
        inputs = dequantize_if_quantized(args[0])
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            inputs = inputs.float()
            parts = [delta.apply(inputs) for delta in self._targets[name]]
            update = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return output + update.to(dtype=output.dtype)

    def to(self, device: torch.device | str) -> FrozenMossDistillationAdapter:
        if self._closed:
            raise RuntimeError("Distillation adapter has been closed.")
        device = torch.device(device)
        if device.type == "meta":
            raise ValueError("Close the distillation adapter when evicting a transformer to meta.")
        for deltas in self._targets.values():
            for delta in deltas:
                delta.to(device)
        return self

    def close(self) -> None:
        """Remove hooks and release fixed weights; safe to call repeatedly."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._targets.clear()
        self._closed = True
        transformer = self._transformer()
        if transformer is not None and getattr(transformer, _ATTRIBUTE, None) is self:
            delattr(transformer, _ATTRIBUTE)


def get_distillation_adapter(transformer: nn.Module) -> FrozenMossDistillationAdapter | None:
    return getattr(transformer, _ATTRIBUTE, None)
