"""Explicit sigma schedules for sample generation.

A sample item may carry `sigmas: [1.0, 0.5]`: the exact noise levels the
sampler visits, descending on a 0..1 scale (1 = pure noise). The terminal clean
0.0 is implied and appended by every consumer, so `[1.0, 0.5]` is a 2-step run.
Values above 1 are read on the toolkit's 0..1000 timestep scale. When set, the
schedule replaces the pipeline's own grid (shift, dynamic shifting, terminal
stretch included) and `sample_steps` is ignored.

Pipelines that drive a diffusers-style scheduler are covered without per-model
code: `install_sample_sigma_override` wraps `set_timesteps` on the scheduler
instances the holder can reach
and rewrites `sigmas`/`timesteps` after the original has sized its state.
Pipelines that build their own grid take `sigmas=` directly.
"""

from typing import Callable, List, Optional, Sequence

import math

import torch


def normalize_sample_sigmas(sigmas) -> Optional[List[float]]:
    if sigmas is None:
        return None
    if isinstance(sigmas, (int, float)):
        sigmas = [sigmas]
    values = [float(s) for s in sigmas]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("sample sigmas must be finite")
    if len(values) == 0:
        return None
    if max(values) > 1.0:
        # 0..1000 timestep scale
        values = [v / 1000.0 for v in values]
    values = [min(max(v, 0.0), 1.0) for v in values]
    values = sorted(set(values), reverse=True)
    # the clean terminal is implied; consumers append it
    values = [v for v in values if v > 0.0]
    if len(values) == 0:
        raise ValueError(
            f"sample sigmas must contain at least one value above 0: {sigmas}"
        )
    return values


def sigmas_with_terminal(sigmas: Sequence[float]) -> List[float]:
    return [float(s) for s in sigmas] + [0.0]


def _is_flow_scheduler(scheduler) -> bool:
    config = getattr(scheduler, "config", None)
    if config is not None:
        if getattr(config, "use_flow_sigmas", False):
            return True
        if "use_dynamic_shifting" in config or "shift_terminal" in config:
            return True
    return "FlowMatch" in type(scheduler).__name__


def apply_sigma_override(scheduler, sigmas: Sequence[float]) -> None:
    """Rewrite a scheduler's schedule in place, after its own set_timesteps ran
    (so step-index, solver history and warmup state are already sized)."""
    sigmas = list(sigmas)
    num_train = int(
        getattr(getattr(scheduler, "config", None), "num_train_timesteps", 1000)
    )
    old_ts = getattr(scheduler, "timesteps", None)
    ts_device = old_ts.device if isinstance(old_ts, torch.Tensor) else "cpu"
    ts_dtype = old_ts.dtype if isinstance(old_ts, torch.Tensor) else torch.float32
    old_sig = getattr(scheduler, "sigmas", None)
    sig_device = old_sig.device if isinstance(old_sig, torch.Tensor) else ts_device
    sig_dtype = old_sig.dtype if isinstance(old_sig, torch.Tensor) else torch.float32

    if _is_flow_scheduler(scheduler):
        scheduler.sigmas = torch.tensor(
            sigmas_with_terminal(sigmas), dtype=sig_dtype, device=sig_device
        )
        scheduler.timesteps = (
            torch.tensor(sigmas, dtype=torch.float64) * num_train
        ).to(ts_device, ts_dtype)
    else:
        # alphas_cumprod schedulers index integer timesteps 0..T-1
        t = (
            (torch.tensor(sigmas, dtype=torch.float64) * num_train)
            .round()
            .clamp(0, num_train - 1)
        )
        scheduler.timesteps = t.to(ts_device, ts_dtype)
        alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
        if isinstance(old_sig, torch.Tensor) and alphas_cumprod is not None:
            ac = alphas_cumprod.detach().to("cpu", torch.float64)
            all_sig = ((1.0 - ac) / ac).sqrt()
            idx = t.long().cpu()
            scheduler.sigmas = torch.cat([all_sig[idx], all_sig.new_zeros(1)]).to(
                sig_device, sig_dtype
            )
    if hasattr(scheduler, "num_inference_steps"):
        scheduler.num_inference_steps = len(sigmas)
    if hasattr(scheduler, "_step_index"):
        scheduler._step_index = None
    if hasattr(scheduler, "_begin_index"):
        scheduler._begin_index = None


def install_sample_sigma_override(holder, pipeline, image_configs) -> Callable[[], None]:
    """Override only this sample's scheduler instances, never shared classes.

    Holder scheduler factories are wrapped locally for pipelines that create
    their scheduler per call. Native pipelines pass sigmas explicitly.
    The returned idempotent cleanup restores even instance-owned methods.
    """
    if not any(getattr(c, "sigmas", None) for c in image_configs):
        return lambda: None
    schedulers = []
    for obj in (getattr(pipeline, "scheduler", None),
                getattr(getattr(holder, "pipeline", None), "scheduler", None),
                getattr(holder, "noise_scheduler", None)):
        if obj is not None and hasattr(obj, "set_timesteps") and all(obj is not s for s in schedulers):
            schedulers.append(obj)
    restores = []
    missing = object()
    previous = getattr(holder, "sample_sigmas", missing)

    def unwrap():
        while restores:
            obj, name, own = restores.pop()
            if own is missing:
                delattr(obj, name)
            else:
                setattr(obj, name, own)
        if previous is missing:
            holder.__dict__.pop("sample_sigmas", None)
        else:
            holder.sample_sigmas = previous

    def patch_scheduler(obj):
        if any(target is obj and name == "set_timesteps" for target, name, _ in restores):
            return obj
        original = obj.set_timesteps
        own = obj.__dict__.get("set_timesteps", missing)

        def set_timesteps(*args, **kwargs):
            result = original(*args, **kwargs)
            sigmas = getattr(holder, "sample_sigmas", None)
            if sigmas is not None:
                apply_sigma_override(obj, sigmas)
            return result

        obj.set_timesteps = set_timesteps
        restores.append((obj, "set_timesteps", own))
        return obj

    try:
        for obj in schedulers:
            patch_scheduler(obj)
        for name in ("get_sample_scheduler", "get_train_scheduler"):
            factory = getattr(holder, name, None)
            if not callable(factory):
                continue
            own = holder.__dict__.get(name, missing)

            def wrap_factory(factory):
                def create(*args, **kwargs):
                    return patch_scheduler(factory(*args, **kwargs))
                return create

            setattr(holder, name, wrap_factory(factory))
            restores.append((holder, name, own))
    except BaseException:
        unwrap()
        raise
    return unwrap
