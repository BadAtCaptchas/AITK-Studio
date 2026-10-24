"""Numerical import/cancellation diagnostics, not an image-retention benchmark."""
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from toolkit.lora_continuation import build_source_teacher, inspect_lora, sha256_file, validate_targets
from toolkit.config_modules import NetworkConfig
from toolkit.continuation_assets import pin_json


def pin_base_checkpoint(path, context):
    if not context:
        return
    identity = {'version': 1, 'sha256': sha256_file(path)}
    destination = Path(context['assets']) / 'base-checkpoint.json'
    pin_json(destination, identity, 'The continuation base checkpoint changed since this job started')


def source_probe(base, transformer, context, cancel=False):
    """Same deterministic latents/text/mask/times on both runtime paths.

    Synthetic text features isolate numerical drift without a second encoder or
    base resident in memory. They do not assess prompt adherence or image quality.
    """
    from extensions_built_in.diffusion_models.krea2.src.pipeline import predict_velocity
    inspected = inspect_lora(context['source'], scale=context.get('scale'), expected_sha256=context['sha256'])
    validate_targets(inspected, transformer)
    old_model, old_network = getattr(base, 'model', None), base.network
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    assistant = getattr(base, 'assistant_lora', None)
    old_assistant = assistant.is_active if assistant is not None else False
    old_training = transformer.training
    network = None
    try:
        base.model = transformer
        process = SimpleNamespace(sd=base, network_config=NetworkConfig(type='lora'),
                                  network=SimpleNamespace(can_merge_in=False), device_torch=base.device_torch)
        network = build_source_teacher(process, inspected)
        network.is_active = True
        if assistant is not None:
            assistant.is_active = cancel
        transformer.eval()
        transformer.to(base.device_torch)
        config = transformer.config
        generator = torch.Generator(device='cpu').manual_seed(1739)
        latent = torch.randn(1, config.channels, 16, 16, generator=generator).to(base.device_torch, base.torch_dtype)
        text = torch.randn(1, 8, config.txtlayers * config.txtdim, generator=generator).to(base.device_torch, base.torch_dtype)
        mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], device=base.device_torch)
        with torch.no_grad():
            outputs = [predict_velocity(transformer, latent, torch.tensor([t], device=base.device_torch, dtype=base.torch_dtype), text, mask).float().cpu()
                       for t in (.1, .5, .9)]
        if not all(torch.isfinite(value).all() for value in outputs):
            raise ValueError('Nonfinite source cancellation diagnostic prediction')
        return torch.stack(outputs)
    finally:
        if network is not None:
            for adapter in network.get_all_modules():
                adapter.orig_module_ref().forward = adapter.org_forward
        if assistant is not None:
            assistant.is_active = old_assistant
        transformer.train(old_training)
        base.model, base.network = old_model, old_network
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def drift_report(reference, cancelled):
    delta = cancelled - reference
    return {'version': 1, 'probe': 'synthetic_velocity', 'seed': 1739, 'timesteps': [.1, .5, .9],
            'rmse': delta.square().mean().sqrt().item(), 'max_abs': delta.abs().max().item(),
            'relative_rmse': (delta.square().mean() / reference.square().mean().clamp_min(1e-20)).sqrt().item(),
            'retention_validated': False}


def release_probe_transformer(transformer):
    from toolkit.memory_management import MemoryManager
    from toolkit.basic import flush
    MemoryManager.detach(transformer)
    transformer.to('cpu')
    gc.collect()
    flush()
