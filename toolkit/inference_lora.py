"""Inference-time LoRA loading for the engine: any common LoRA file format ->
per-module low-rank deltas, applied either as forward hooks (dynamic strength,
removable) or merged into the weights (quantized weights re-quantized with
stochastic rounding so small deltas survive).

Independent of the training network classes on purpose.

Supported key formats (auto-detected per key):
  peft / diffusers   transformer.blocks.0.attn.to_q.lora_A.weight / lora_B
  comfy              diffusion_model.<...>.lora_A|lora_down (+ .alpha)
  kohya / lycoris    lora_unet_<flat_name>.lora_down.weight / lora_up + alpha,
                     lora_te_ / lora_te1_ / lora_te2_ for text encoders
  full diffs         <module>.diff (weight delta), <module>.diff_b (bias delta)
  lokr               <module>.lokr_w1 | lokr_w1_a/lokr_w1_b (+lokr_t1),
                     lokr_w2 | lokr_w2_a/lokr_w2_b (+lokr_t2), alpha;
                     toolkit peft-style dotted keys or old lycoris_<flat> keys
The holder's convert_lora_weights_before_load hook runs first, so archs with
their own conversions (anima, ltx2, hidream_o1) keep working.
"""

import hashlib
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

_A_SUFFIXES = (".lora_A.weight", ".lora_down.weight", ".lora_A.default.weight")
_B_SUFFIXES = (".lora_B.weight", ".lora_up.weight", ".lora_B.default.weight")
_KRON_PARTS = ("lokr_w1", "lokr_w1_a", "lokr_w1_b", "lokr_t1", "lokr_w2", "lokr_w2_a", "lokr_w2_b", "lokr_t2")
_MAGNITUDE_SUFFIXES = (".magnitude", ".lora_magnitude_vector", ".lora_magnitude_vector.weight", ".lora_magnitude_vector.default.weight")
_TE_PREFIXES = {
    "lora_te1_": 0, "lora_te_": 0, "lora_te2_": 1, "lora_te3_": 2,
    "text_encoder.": 0, "text_encoder_2.": 1, "text_encoder_3.": 2,
    "te.": 0, "te1.": 0, "te2.": 1,
}
_DIT_PREFIXES = (
    "model.diffusion_model.", "diffusion_model.", "transformer.", "unet.",
    "lora_unet_", "lora_transformer_", "lycoris_", "base_model.model.",
)


def _cp_weight(t, wa, wb):
    # tucker core t (r1, r2, kh, kw) with wa (r1, p), wb (r2, q) -> (p, q, kh, kw)
    return torch.einsum("ijkl,ip,jr->prkl", t, wa, wb)


def _kron_factors(k: dict):
    """Rebuild the two kron factors (w1: (a, b), w2: (c, d[, kh, kw])) from stored parts."""
    if "lokr_w1" in k:
        w1 = k["lokr_w1"]
    elif "lokr_t1" in k:
        w1 = _cp_weight(k["lokr_t1"], k["lokr_w1_a"], k["lokr_w1_b"])
    else:
        w1 = k["lokr_w1_a"] @ k["lokr_w1_b"]
    if "lokr_w2" in k:
        w2 = k["lokr_w2"]
    elif "lokr_t2" in k:
        w2 = _cp_weight(k["lokr_t2"], k["lokr_w2_a"], k["lokr_w2_b"])
    else:
        w2 = k["lokr_w2_a"] @ k["lokr_w2_b"]
    return w1, w2


def _kron_apply(x: torch.Tensor, k: dict) -> torch.Tensor:
    """x @ kron(w1, w2).T without materializing the full delta (linear only).
    x (..., b*d) viewed as (..., b, d) -> (..., a, c) -> (..., a*c)."""
    w1, w1a, w1b = k.get("lokr_w1"), k.get("lokr_w1_a"), k.get("lokr_w1_b")
    w2, w2a, w2b = k.get("lokr_w2"), k.get("lokr_w2_a"), k.get("lokr_w2_b")
    b = (w1 if w1 is not None else w1b).shape[1]
    d = (w2 if w2 is not None else w2b).shape[1]
    t = x.reshape(*x.shape[:-1], b, d)
    if w2 is not None:
        t = t @ w2.t()
    else:
        t = (t @ w2b.t()) @ w2a.t()
    if w1 is not None:
        t = torch.einsum("...bc,ab->...ac", t, w1)
    else:
        t = torch.einsum("...bc,rb->...rc", t, w1b)
        t = torch.einsum("...rc,ar->...ac", t, w1a)
    return t.reshape(*x.shape[:-1], -1)


def _weight_shape(module):
    # Packed Linear.weight can reconstruct the entire dense weight. Its
    # logical dimensions are already available without decoding anything.
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        return (module.out_features, module.in_features)
    weight = getattr(module, "weight", None)
    return tuple(weight.shape) if weight is not None else None


def _base_weight(module, device, dtype=torch.float32):
    if getattr(module, "is_ostris_quantized", False):
        return module.dequantize_weight(device=device, dtype=dtype).detach()
    from toolkit.util.quantize import dequantize_if_quantized
    weight = dequantize_if_quantized(module.weight).detach()
    if not weight.is_floating_point():
        raise ValueError(f"Magnitude adapters do not support undecoded {type(module).__name__} weights")
    weight = weight.to(device=device, dtype=dtype).reshape(_weight_shape(module))
    scale = getattr(module, "weight_scale", None)
    if scale is not None:
        weight = weight * scale.to(device=device, dtype=dtype).reshape(-1, *([1] * (weight.ndim - 1)))
    return weight


def _apply_weight(module, x, weight, bias=None):
    if weight.ndim == 2:
        return F.linear(x, weight, bias)
    if weight.ndim == 4 and isinstance(module, torch.nn.Conv2d):
        return F.conv2d(x, weight, bias, module.stride, module.padding, module.dilation, module.groups)
    raise ValueError(f"Adapter application does not support {type(module).__name__} with {weight.ndim}D weights")


def _file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LoRAEntry:
    def __init__(self, module: torch.nn.Module, name: str):
        self.module = module
        self.name = name
        self.A: Optional[torch.Tensor] = None  # (r, in)
        self.B: Optional[torch.Tensor] = None  # (out, r)
        self.alpha: Optional[float] = None
        self.diff: Optional[torch.Tensor] = None
        self.diff_b: Optional[torch.Tensor] = None
        self.kron: Optional[Dict[str, torch.Tensor]] = None  # lokr parts by name
        self.magnitude: Optional[torch.Tensor] = None
        self.dora_scale: Optional[torch.Tensor] = None

    @property
    def has_magnitude(self):
        return self.magnitude is not None or self.dora_scale is not None

    @property
    def is_conv(self):
        return isinstance(self.module, torch.nn.Conv2d)

    @property
    def scale(self) -> float:
        if self.alpha is None:
            return 1.0
        if self.A is not None:
            r = self.A.shape[0]
            return float(self.alpha) / r if r else 1.0
        if self.kron is not None:
            # rank lives in the _b factor; both factors full -> scale 1
            rb = self.kron.get("lokr_w1_b", self.kron.get("lokr_w2_b"))
            if rb is None:
                return 1.0
            return float(self.alpha) / rb.shape[0]
        return 1.0

    @property
    def kron_is_linear(self) -> bool:
        return not self.is_conv and self.kron is not None and all(t.dim() == 2 for t in self.kron.values())

    def validate(self):
        shape = _weight_shape(self.module)
        if shape is None or len(shape) not in (2, 4):
            raise ValueError(f"Unsupported adapter target {self.name}: {type(self.module).__name__}")
        if self.is_conv and self.module.padding_mode != "zeros":
            raise ValueError(f"Convolution adapter {self.name} requires zero padding mode")
        if (self.A is None) != (self.B is None):
            raise ValueError(f"Incomplete LoRA factors for {self.name}")
        if self.A is not None:
            if self.A.ndim != len(shape) or self.B.ndim != len(shape):
                raise ValueError(f"LoRA dimensions do not match {self.name}")
            if self.A.shape[0] != self.B.shape[1] or self.A.shape[1] != shape[1] or self.B.shape[0] != shape[0]:
                raise ValueError(f"LoRA factor shapes do not match {self.name}")
            if self.is_conv and (self.module.groups != 1 or tuple(self.B.shape[2:]) != (1, 1)
                                 or tuple(self.A.shape[2:]) != tuple(shape[2:])):
                raise ValueError(f"Convolution LoRA {self.name} requires groups=1 and a 1x1 up projection")
        if self.diff is not None and tuple(self.diff.shape) != shape:
            raise ValueError(f"Full adapter delta shape does not match {self.name}")
        if self.diff_b is not None and tuple(self.diff_b.shape) != (shape[0],):
            raise ValueError(f"Adapter bias shape does not match {self.name}")
        if self.has_magnitude and (self.diff is not None or self.diff_b is not None):
            raise ValueError(f"Magnitude adapter {self.name} cannot also contain full weight/bias deltas")
        if self.magnitude is not None:
            if self.A is None or self.is_conv or self.magnitude.numel() != shape[0] or self.dora_scale is not None:
                raise ValueError(f"DoRA magnitude for {self.name} requires linear LoRA factors and one magnitude per output")
            if getattr(self.module, "is_ostris_quantized", False):
                quantizer = self.module.ostris_quantizer
                if not callable(getattr(quantizer, "_decode_rotated_rows", None)) or not callable(getattr(quantizer, "_runtime_rows", None)):
                    raise ValueError(f"DoRA {self.name} requires a packed backend with a bounded row decoder (for example orbit4)")
        if self.dora_scale is not None:
            if self.kron is None or self.magnitude is not None:
                raise ValueError(f"LoKr magnitude for {self.name} requires Kronecker factors")
            row_shape = (shape[0],) + (1,) * (len(shape) - 1)
            col_shape = (1, shape[1]) + (1,) * (len(shape) - 2)
            if tuple(self.dora_scale.shape) not in (row_shape, col_shape):
                raise ValueError(f"Unsupported LoKr magnitude shape for {self.name}: {tuple(self.dora_scale.shape)}")
            if getattr(self.module, "is_ostris_quantized", False):
                raise ValueError(f"LoKr weight decomposition on packed target {self.name} is unsupported; use standard LoKr")

    def kron_shape(self) -> Optional[Tuple[int, int]]:
        """(out, in) of the rebuilt kron delta, linear parts only."""
        if not self.kron_is_linear:
            return None
        k = self.kron
        a = (k["lokr_w1"] if "lokr_w1" in k else k["lokr_w1_a"]).shape[0]
        b = (k["lokr_w1"] if "lokr_w1" in k else k["lokr_w1_b"]).shape[1]
        c = (k["lokr_w2"] if "lokr_w2" in k else k["lokr_w2_a"]).shape[0]
        d = (k["lokr_w2"] if "lokr_w2" in k else k["lokr_w2_b"]).shape[1]
        return a * c, b * d

    def delta(self, strength: float, device, dtype=torch.float32) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """(weight delta, bias delta) at this strength, full precision."""
        dw = None
        # Magnitude adapters have different multiplier conventions during
        # training. Construct their direction before applying normalization.
        direction_strength = 1.0 if self.dora_scale is not None else strength
        if self.A is not None and self.B is not None:
            A = self.A.to(device=device, dtype=dtype)
            B = self.B.to(device=device, dtype=dtype)
            if A.dim() == 4:  # conv lora: (r, in, kh, kw) @ (out, r, 1, 1)
                dw = torch.einsum("or,rikl->oikl", B.flatten(1), A) * (self.scale * direction_strength)
            else:
                dw = (B @ A) * (self.scale * direction_strength)
        if self.kron is not None:
            w1, w2 = _kron_factors({n: t.to(device=device, dtype=dtype) for n, t in self.kron.items()})
            if w2.dim() == 4:
                w1 = w1[:, :, None, None]
            dk = torch.kron(w1, w2.contiguous()).reshape(_weight_shape(self.module)) * (self.scale * direction_strength)
            dw = dk if dw is None else dw + dk
        if self.diff is not None:
            d = self.diff.to(device=device, dtype=dtype) * strength
            dw = d if dw is None else dw + d
        if self.has_magnitude and strength != 0:
            base = _base_weight(self.module, device, dtype)
            adapted = base + dw
            if self.magnitude is not None:
                norm = adapted.norm(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
                adapted = adapted * (self.magnitude.to(device=device, dtype=dtype).reshape(-1, 1) / norm)
            else:
                axis = 0 if self.dora_scale.shape[0] == adapted.shape[0] else 1
                dims = tuple(index for index in range(adapted.ndim) if index != axis)
                norm = torch.linalg.vector_norm(adapted, dim=dims, keepdim=True) + torch.finfo(dtype).eps
                ratio = self.dora_scale.to(device=device, dtype=dtype) / norm
                adapted = adapted * (1 + strength * (ratio - 1))
            dw = adapted - base
        elif self.has_magnitude:
            dw = torch.zeros_like(dw)
        db = self.diff_b.to(device=device, dtype=dtype) * strength if self.diff_b is not None else None
        return dw, db

    @torch.no_grad()
    def dora_ratio(self, device, strength):
        """Compute only row norms; packed DoRA never reconstructs a full base."""
        from toolkit.models.DoRA import DoRAModule
        down = self.A.to(device=device, dtype=torch.float32)
        up = self.B.to(device=device, dtype=torch.float32) * (self.scale * strength)
        module = self.module
        if getattr(module, "is_ostris_quantized", False):
            from toolkit.util.orbit_quant import rpbh_forward
            quantizer = module.ostris_quantizer
            down = rpbh_forward(down, module.orbit_perm, module.orbit_signs, module.orbit_block)
            rows = quantizer._runtime_rows(module.out_features, module.in_features, torch.float32)
            chunks = (
                (start, min(start + rows, module.out_features), quantizer._decode_rotated_rows(
                    module, start, min(start + rows, module.out_features), torch.float32,
                ).to(device))
                for start in range(0, module.out_features, rows)
            )
        else:
            # Quanto may need to dequantize its base, but the transient full
            # weight is released after norms are computed, never kept in hooks.
            weight = _base_weight(module, module.weight.device)
            chunks = ((start, min(start + 256, up.shape[0]), weight[start:start + 256].to(device))
                      for start in range(0, up.shape[0], 256))
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            norm = DoRAModule._row_norms_from_chunks(chunks, down, up)
        return self.magnitude.to(device=device, dtype=torch.float32).reshape(-1) / norm.clamp_min(torch.finfo(torch.float32).eps)


class InferenceLoRA:
    def __init__(self, path: str, strength: float = 1.0, name: Optional[str] = None):
        self.path = path
        self.local_path: Optional[str] = None  # set by load(): path resolved on disk
        self.strength = float(strength)
        if not math.isfinite(self.strength):
            raise ValueError("Adapter strength must be finite")
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        self.entries: List[LoRAEntry] = []
        self.unmatched: List[str] = []
        self.num_keys = 0  # tensors in the file
        self.num_skipped_keys = 0  # keys with no recognized lora/diff/lokr suffix
        self.convert_error: Optional[str] = None
        self._hooks: List = []
        self.prepend_hooks = False
        self.source_identity = None

    def resolve_source(self, status_fn=None):
        from toolkit.models.v2.resolver import resolve_lora_file
        self.local_path = resolve_lora_file(self.path, status_fn=status_fn)
        self.source_identity = (
            os.path.normcase(os.path.realpath(self.local_path)), _file_digest(self.local_path),
        )
        return self.source_identity

    # ---- loading / key resolution ----
    def load(self, holder, status_fn=None) -> "InferenceLoRA":
        # Persistent base adapters may run as forward hooks. Trainable LoRA
        # forwards run before those hooks; preserve that order after export.
        self.prepend_hooks = bool(getattr(holder, "inference_lora_prepend_hooks", False))

        # a spec path may be a hub reference (org/repo/file.safetensors); it is
        # searched for under the models folder before anything is downloaded.
        # self.path stays as given: it is the spec identity the engine keys its
        # loaded stack on.
        self.resolve_source(status_fn=status_fn)
        if status_fn:
            status_fn(f"Loading LoRA {self.name} from {self.local_path}")
        sd = load_file(self.local_path)
        if _file_digest(self.local_path) != self.source_identity[1]:
            raise ValueError(f"Adapter changed while being loaded: {self.local_path}; retry the request")
        self.num_keys = len(sd)
        convert = getattr(holder, "convert_lora_weights_before_load", None)
        if callable(convert):
            try:
                sd = convert(sd)
            except Exception as e:
                self.convert_error = f"{type(e).__name__}: {e}"
        roots = self._roots(holder)
        grouped: Dict[Tuple[int, str], dict] = {}
        self.num_skipped_keys = 0
        for key, tensor in sd.items():
            base, part = self._split(key)
            if part is None:
                self.num_skipped_keys += 1
                continue
            grouped.setdefault(base, {})[part] = tensor
        self.entries = []
        self.unmatched = []
        for base, parts in grouped.items():
            module, name = self._resolve(base, roots)
            if module is None:
                self.unmatched.append(base)
                continue
            e = LoRAEntry(module, name)
            e.A = parts.get("A")
            e.B = parts.get("B")
            e.diff = parts.get("diff")
            e.diff_b = parts.get("diff_b")
            e.magnitude = parts.get("magnitude")
            e.dora_scale = parts.get("dora_scale")
            kron = {n: parts[n] for n in _KRON_PARTS if n in parts}
            if kron:
                has_w1 = "lokr_w1" in kron or ("lokr_w1_a" in kron and "lokr_w1_b" in kron)
                has_w2 = "lokr_w2" in kron or ("lokr_w2_a" in kron and "lokr_w2_b" in kron)
                if not (has_w1 and has_w2):
                    self.unmatched.append(f"{base} (incomplete lokr parts {sorted(kron)})")
                    continue
                e.kron = kron
            alpha = parts.get("alpha")
            if alpha is not None:
                e.alpha = float(alpha.flatten()[0])
            w_shape = _weight_shape(module)
            if e.A is not None and e.B is not None and e.A.dim() == 2:
                if w_shape is not None and (e.B.shape[0] != w_shape[0] or e.A.shape[1] != w_shape[1]):
                    self.unmatched.append(f"{base} (shape {tuple(e.B.shape[0:1]) + tuple(e.A.shape[1:])} vs {tuple(w_shape)})")
                    continue
            ks = e.kron_shape()
            if ks is not None and w_shape is not None and len(w_shape) == 2 and tuple(ks) != tuple(w_shape):
                self.unmatched.append(f"{base} (lokr shape {ks} vs {tuple(w_shape)})")
                continue
            if e.A is None and e.kron is None and e.diff is None and e.diff_b is None:
                if e.has_magnitude:
                    raise ValueError(f"Magnitude adapter {base} is missing its direction factors")
                continue
            e.validate()
            self.entries.append(e)
        return self

    @staticmethod
    def _split(key: str):
        for suffix in _MAGNITUDE_SUFFIXES:
            if key.endswith(suffix):
                return key[:-len(suffix)], "magnitude"
        if key.endswith(".dora_scale"):
            return key[:-len(".dora_scale")], "dora_scale"
        for suf in _A_SUFFIXES:
            if key.endswith(suf):
                return key[: -len(suf)], "A"
        for suf in _B_SUFFIXES:
            if key.endswith(suf):
                return key[: -len(suf)], "B"
        if key.endswith(".alpha"):
            return key[: -len(".alpha")], "alpha"
        if key.endswith(".diff"):
            return key[: -len(".diff")], "diff"
        if key.endswith(".diff_b"):
            return key[: -len(".diff_b")], "diff_b"
        for part in _KRON_PARTS:
            if key.endswith("." + part):
                return key[: -len(part) - 1], part
        return key, None

    @staticmethod
    def _roots(holder) -> List[Tuple[str, torch.nn.Module, dict, dict]]:
        """(kind, root module, dotted-name map, flat-name map) for the DiT and TEs."""
        roots = []
        dit = getattr(holder, "model", None)
        if isinstance(dit, torch.nn.Module):
            names = dict(dit.named_modules())
            roots.append(("dit", dit, names, {n.replace(".", "_"): n for n in names}))
        tes = getattr(holder, "text_encoder", None)
        tes = tes if isinstance(tes, list) else [tes]
        for i, te in enumerate(tes):
            if isinstance(te, torch.nn.Module) and not type(te).__name__.startswith("Fake"):
                names = dict(te.named_modules())
                roots.append((f"te{i}", te, names, {n.replace(".", "_"): n for n in names}))
        return roots

    @classmethod
    def _resolve(cls, base: str, roots):
        te_index = None
        rest = base
        for pref, idx in _TE_PREFIXES.items():
            if base.startswith(pref):
                te_index = idx
                rest = base[len(pref):]
                break
        if te_index is None:
            for pref in _DIT_PREFIXES:
                if base.startswith(pref):
                    rest = base[len(pref):]
                    break
        candidates = [r for r in roots if (r[0] == f"te{te_index}" if te_index is not None else r[0] == "dit")]
        if not candidates:
            candidates = roots
        for _, root, names, flat in candidates:
            if rest in names:
                return names[rest], rest
            flat_key = rest.replace(".", "_")
            if flat_key in flat:
                n = flat[flat_key]
                return names[n], n
        return None, None

    # ---- hook mode ----
    def attach(self):
        """Forward hooks that add strength * delta(x) to each module's output."""
        self.detach()
        lora = self
        for e in self.entries:
            e.validate()

            def make(entry: LoRAEntry):
                cache = {}

                def hook(module, inputs, output):
                    if lora.strength == 0:
                        return output
                    x = inputs[0] if inputs else None
                    if x is None:
                        return output
                    from toolkit.util.quantize import dequantize_if_quantized
                    x = dequantize_if_quantized(x)
                    key = (x.device, x.dtype, lora.strength if entry.magnitude is not None else None)
                    if key not in cache:
                        cache.clear()
                        cache[key] = (
                            entry.A.to(device=x.device, dtype=x.dtype) if entry.A is not None else None,
                            entry.B.to(device=x.device, dtype=x.dtype) if entry.B is not None else None,
                            entry.diff.to(device=x.device, dtype=x.dtype) if entry.diff is not None else None,
                            entry.diff_b.to(device=x.device, dtype=x.dtype) if entry.diff_b is not None else None,
                            {n: t.to(device=x.device, dtype=x.dtype) for n, t in entry.kron.items()} if entry.kron is not None else None,
                            entry.dora_ratio(x.device, lora.strength) if entry.magnitude is not None else None,
                        )
                    A, B, diff, diff_b, kron, ratio = cache[key]
                    base_output = output[0] if isinstance(output, tuple) else output
                    add = None
                    if entry.dora_scale is not None:
                        # Weight-decomposed LoKr is already a dense operation
                        # during training. Keep its dense delta transient, so
                        # offloading does not retain every layer's GPU weight.
                        with torch.autocast(device_type=x.device.type, enabled=False):
                            dw, db = entry.delta(lora.strength, x.device)
                            add = _apply_weight(module, x.float(), dw, db)
                        if isinstance(output, tuple):
                            return (base_output + add.to(base_output.dtype),) + tuple(output[1:])
                        return base_output + add.to(base_output.dtype)
                    if A is not None and B is not None:
                        if entry.is_conv:
                            down = F.conv2d(x, A, None, module.stride, module.padding, module.dilation)
                            add = F.conv2d(down, B) * (entry.scale * lora.strength)
                        else:
                            add = ((x @ A.t()) @ B.t()) * (entry.scale * lora.strength)
                    if kron is not None:
                        if entry.is_conv:
                            w1, w2 = _kron_factors(kron)
                            if w2.ndim == 4:
                                w1 = w1[:, :, None, None]
                            weight = torch.kron(w1, w2.contiguous()).reshape(_weight_shape(module))
                            dk = _apply_weight(module, x, weight) * (entry.scale * lora.strength)
                        else:
                            dk = _kron_apply(x, kron) * (entry.scale * lora.strength)
                        add = dk if add is None else add + dk
                    if diff is not None:
                        d = _apply_weight(module, x, diff) * lora.strength
                        add = d if add is None else add + d
                    if diff_b is not None:
                        bias_delta = diff_b.view(1, -1, 1, 1) if entry.is_conv else diff_b
                        add = (bias_delta * lora.strength) if add is None else add + bias_delta * lora.strength
                    if add is None:
                        return output
                    if ratio is not None:
                        bias = getattr(module, "bias", None)
                        bias = bias.to(device=x.device, dtype=base_output.dtype) if bias is not None else 0
                        adapted = (base_output - bias + add.to(base_output.dtype)) * ratio.to(base_output.dtype) + bias
                        if isinstance(output, tuple):
                            return (adapted,) + tuple(output[1:])
                        return adapted
                    if isinstance(output, tuple):
                        return (output[0] + add.to(output[0].dtype),) + tuple(output[1:])
                    return output + add.to(output.dtype)

                return hook

            self._hooks.append(e.module.register_forward_hook(make(e), prepend=self.prepend_hooks))
        return self

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ---- merge mode ----
    def validate_merge(self, skip_modules=None):
        from optimum.quanto import QTensor
        if self.strength == 0:
            return
        # Fail before mutating any module if its quantizer has no supported
        # writeback. Hooks remain available for these tensor subclasses.
        for entry in self.entries:
            module = entry.module
            if (skip_modules and id(module) in skip_modules) or getattr(module, "is_ostris_quantized", False):
                continue
            weight = module.weight
            if "torchao" in type(weight).__module__ or getattr(module, "weight_scale", None) is not None:
                raise ValueError(f"Merging into {type(module).__name__} quantized target {entry.name} is unsupported; use hook mode")
            if isinstance(weight, QTensor) and not callable(getattr(module, "freeze", None)):
                raise ValueError(f"Quanto target {entry.name} has no requantization method; use hook mode")

    @torch.no_grad()
    def merge(self, stochastic: bool = True, skip_modules=None) -> int:
        """Add strength * delta into each module's weight. Quantized (Ostris)
        weights: dequantize -> add -> requantize on the stored grid with
        stochastic rounding (int8 backends), nearest otherwise. Plain weights
        in bf16/fp16 on cuda: stochastic rounding via copy_stochastic."""
        from toolkit.optimizers.optimizer_utils import copy_stochastic
        from optimum.quanto import QTensor

        self.validate_merge(skip_modules)
        if self.strength == 0:
            return 0

        merged = 0
        for e in self.entries:
            m = e.module
            if skip_modules and id(m) in skip_modules:
                continue
            w = getattr(m, "weight", None)
            if w is None:
                continue
            device = w.device
            dw, db = e.delta(self.strength, device)
            if dw is not None:
                if getattr(m, "is_ostris_quantized", False):
                    fp = m.dequantize_weight().to(torch.float32) + dw.reshape(w.shape)
                    q = m.ostris_quantizer
                    rq_codes = getattr(q, "requantize_codes_", None)
                    if rq_codes is not None and stochastic:
                        try:
                            rq_codes(m, fp, stochastic=True)
                        except TypeError:
                            rq_codes(m, fp)
                    elif rq_codes is not None:
                        rq_codes(m, fp)
                    else:
                        m.requantize_(fp)
                elif isinstance(w, QTensor):
                    # Quanto cannot copy a dense Tensor into its quantized
                    # wrapper. Reuse the module's qtype/axis/group optimizer.
                    fp = w.dequantize().float() + dw.reshape(w.shape)
                    m.weight = torch.nn.Parameter(fp.to(w.dtype), requires_grad=False)
                    m.freeze()
                elif isinstance(w, torch.nn.Parameter) or isinstance(w, torch.Tensor):
                    fp = w.data.to(torch.float32) + dw.reshape(w.shape)
                    if stochastic and w.is_cuda and w.dtype in (torch.bfloat16, torch.float16):
                        copy_stochastic(w.data, fp)
                    else:
                        w.data.copy_(fp.to(w.dtype))
                else:
                    continue
            if db is not None and getattr(m, "bias", None) is not None:
                m.bias.data.add_(db.to(m.bias.dtype))
            merged += 1
        return merged

    def report(self, max_unmatched: Optional[int] = None) -> str:
        """Multi-line load report: file, key counts, matched/unmatched modules, unmatched names."""
        n_groups = len(self.entries) + len(self.unmatched)
        lines = [
            f"LoRA {self.name}: {self.local_path or self.path}",
            f"  {self.num_keys} keys in file -> {n_groups} modules: {len(self.entries)} matched, {len(self.unmatched)} unmatched"
            + (f", {self.num_skipped_keys} keys ignored (unrecognized suffix)" if self.num_skipped_keys else ""),
        ]
        if self.convert_error:
            lines.append(f"  convert_lora_weights_before_load failed, using raw keys: {self.convert_error}")
        shown = self.unmatched if max_unmatched is None else self.unmatched[:max_unmatched]
        for u in shown:
            lines.append(f"  unmatched: {u}")
        if len(shown) < len(self.unmatched):
            lines.append(f"  ... {len(self.unmatched) - len(shown)} more unmatched")
        return "\n".join(lines)

    def report_line(self, max_unmatched: int = 5) -> str:
        """One-line form of report() for status strips."""
        n_groups = len(self.entries) + len(self.unmatched)
        line = f"LoRA {self.name}: {len(self.entries)}/{n_groups} modules matched ({self.num_keys} keys)"
        if self.unmatched:
            shown = ", ".join(self.unmatched[:max_unmatched])
            more = len(self.unmatched) - max_unmatched
            line += f"; {len(self.unmatched)} unmatched: {shown}" + (f" +{more} more" if more > 0 else "")
        if self.convert_error:
            line += f"; convert failed: {self.convert_error}"
        return line

    def summary(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "strength": self.strength,
            "modules": len(self.entries),
            "unmatched": len(self.unmatched),
        }


class LoRAStack:
    """The LoRAs applied to one holder, in one mode."""

    def __init__(self, holder, mode: str = "hook"):
        self.holder = holder
        self.mode = mode
        self.loras: List[InferenceLoRA] = []

    def key(self) -> str:
        # Loaded identities are immutable snapshots. A wanted stack resolves
        # fresh bytes, including same-size replacements with preserved mtimes.
        sources = [[l.path, l.source_identity or l.resolve_source(), l.strength] for l in self.loras]
        return json.dumps([self.mode, sources], separators=(",", ":"))

    def load(self, specs: List[dict], status_fn=None):
        for spec in specs:
            lora = InferenceLoRA(spec["path"], spec.get("strength", 1.0), spec.get("name")).load(
                self.holder, status_fn=status_fn
            )
            print(f"[AITK] {lora.report()}", flush=True)
            if status_fn:
                status_fn(lora.report_line())
            if not lora.entries:
                raise ValueError(
                    f"LoRA {lora.name} matched no modules of this model "
                    f"({lora.num_keys} keys, {len(lora.unmatched)} unmatched; see the engine job log for the list)"
                )
            self.loras.append(lora)
        return self

    def apply(self, status_fn=None, skip_modules=None):
        by_module = {}
        for lora in self.loras:
            if self.mode == "merge":
                lora.validate_merge(skip_modules)
            for entry in lora.entries:
                previous = by_module.setdefault(id(entry.module), [])
                if previous and (entry.has_magnitude or any(item.has_magnitude for item in previous)):
                    raise ValueError(
                        f"Stacking magnitude-based adapters with other adapters on {entry.name} is unsupported; "
                        "apply one adapter to that module"
                    )
                previous.append(entry)
        touched = set()
        for lora in self.loras:
            if self.mode == "merge":
                n = lora.merge(stochastic=True, skip_modules=skip_modules)
                if status_fn:
                    status_fn(f"Merged LoRA {lora.name} into {n} modules (stochastic rounding)")
            else:
                lora.attach()
            for e in lora.entries:
                touched.add(id(e.module))
        return touched

    def remove(self):
        """Hook mode: detach every hook. Merge mode: cannot un-merge; the
        engine reloads the affected components instead."""
        for lora in self.loras:
            lora.detach()
        self.loras = []

    def set_strengths(self, specs: List[dict], source_identities=None) -> bool:
        """Hook mode only: update strengths in place for the same paths."""
        if self.mode != "hook" or len(specs) != len(self.loras):
            return False
        for lora, spec in zip(self.loras, specs):
            if spec["path"] != lora.path:
                return False
        if source_identities is None:
            source_identities = [InferenceLoRA(spec["path"]).resolve_source() for spec in specs]
        if len(source_identities) != len(self.loras):
            return False
        if any(lora.source_identity != identity for lora, identity in zip(self.loras, source_identities)):
            return False
        strengths = [float(spec.get("strength", 1.0)) for spec in specs]
        if not all(math.isfinite(value) for value in strengths):
            raise ValueError("Adapter strengths must be finite")
        for lora, strength in zip(self.loras, strengths):
            lora.strength = strength
        return True
