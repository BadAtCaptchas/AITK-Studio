"""Reject invalid updates before they can alter optimizer or checkpoint state."""

from collections import defaultdict
from typing import Iterable

import torch

from toolkit.exceptions import UserFacingError


class TrainingNumericsError(UserFacingError):
    """A loss or gradient is not usable for a training update."""


def require_finite_tensors(
    tensors: Iterable[torch.Tensor], stage: str, *, step=None, accelerator=None,
) -> None:
    """Check with one host synchronization per device, including sparse grads.

    Every distributed rank participates before backward/step so a failed rank
    cannot leave its peers waiting in an autograd collective.
    """
    flags = defaultdict(list)
    for tensor in tensors:
        value = tensor.detach()
        if value.layout == torch.sparse_coo:
            value = value.coalesce().values()
        elif value.layout != torch.strided:
            value = value.values()
        flags[value.device].append(torch.isfinite(value).all())
    failed = any(not bool(torch.stack(values).all()) for values in flags.values())
    if getattr(accelerator, 'num_processes', 1) > 1:
        flag = torch.tensor(int(failed), device=accelerator.device, dtype=torch.int32)
        failed = bool(accelerator.reduce(flag, reduction='sum').item())
    if failed:
        location = f' at training step {step}' if step is not None else ''
        raise TrainingNumericsError(
            f'Non-finite {stage}{location}. Training stopped before an optimizer update. '
            'Resume from the last valid checkpoint after checking the learning rate, '
            'training precision, and input data.'
        )


def require_finite_gradients(optimizer, *, step=None, accelerator=None) -> None:
    """Include stochastic/private accumulation buffers without modifying them."""
    gradients = []
    seen = set()
    for group in optimizer.param_groups:
        for param in group['params']:
            if id(param) in seen:
                continue
            seen.add(id(param))
            for grad in (param.grad, getattr(param, '_accum_grad', None)):
                if grad is not None:
                    gradients.append(grad)
    require_finite_tensors(gradients, 'gradients', step=step, accelerator=accelerator)
