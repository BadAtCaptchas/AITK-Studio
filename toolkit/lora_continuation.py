"""Strict, weight-only Krea 2 continuation. No model weights are needed for inspection.

Native exports have unit scaling. PEFT exports without scaling metadata require an
explicit scale; shapes cannot recover alpha. This path never truncates/pads ranks.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file

VERSION = 1
MAX_BYTES = 2 * 1024 ** 3
MODES = ('none', 'replay', 'teacher', 'replay_teacher')
BASES = ('raw', 'turbo')

# Explicit module-name equivalences; unsupported tensors are never discarded.
_TOP = {'img_in': 'first', 'final_layer.linear': 'last.linear',
        'time_embed.linear_1': 'tmlp.0', 'time_embed.linear_2': 'tmlp.2',
        'time_mod_proj': 'tproj.1', 'txt_in.linear_1': 'txtmlp.1',
        'txt_in.linear_2': 'txtmlp.3'}
_LEAF = {'attn.to_q': 'attn.wq', 'attn.to_k': 'attn.wk', 'attn.to_v': 'attn.wv',
         'attn.to_out.0': 'attn.wo', 'attn.to_gate': 'attn.gate',
         'ff.gate': 'mlp.gate', 'ff.up': 'mlp.up', 'ff.down': 'mlp.down'}
_NATIVE = re.compile(r'^(?:first|last\.linear|tmlp\.[02]|tproj\.1|txtmlp\.[13]|txtfusion\.projector|'
                     r'(?:blocks\.\d+|txtfusion\.(?:layerwise_blocks|refiner_blocks)\.\d+)\.'
                     r'(?:attn\.(?:wq|wk|wv|wo|gate)|mlp\.(?:gate|up|down)))$')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def native_module(name):
    # Kohya/Comfy exports flatten native paths. Decode only the supported grammar;
    # replacing all underscores would corrupt layerwise_blocks/refiner_blocks.
    if name.startswith(('lora_unet_', 'lora_transformer_')):
        flat = name.split('_', 2)[2]
        block = re.fullmatch(r'(blocks|txtfusion_(?:layerwise_blocks|refiner_blocks))_(\d+)_(attn|mlp)_(\w+)', flat)
        if block:
            parent = block[1].replace('txtfusion_', 'txtfusion.', 1)
            name = f'{parent}.{block[2]}.{block[3]}.{block[4]}'
        else:
            name = flat.replace('_', '.')
    for prefix in ('base_model.model.', 'diffusion_model.', 'transformer.'):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name in _TOP:
        name = _TOP[name]
    else:
        name = re.sub(r'^transformer_blocks\.', 'blocks.', name)
        name = re.sub(r'^text_fusion\.', 'txtfusion.', name)
        for source, target in _LEAF.items():
            if name.endswith('.' + source):
                name = name[:-len(source)] + target
                break
    if not _NATIVE.fullmatch(name):
        raise ValueError(f'Unsupported Krea 2 denoiser target: {name}')
    return name


@dataclass
class InspectedLora:
    state: dict
    modules: list
    sha256: str
    format: str
    metadata: dict
    size_bytes: int = 0

    def report(self):
        return dict(version=VERSION, status='compatible', sha256=self.sha256,
                    format=self.format, modules=self.modules, tensor_count=len(self.state), size_bytes=self.size_bytes,
                    ranks=sorted({item['rank'] for item in self.modules}),
                    declared_training_base=self.metadata.get('aitk_continuation_training_base'),
                    architecture='krea2', topology_verified=False)

    def network_kwargs(self):
        dims = {('transformer.' + item['name']).replace('.', '$$'): item['rank'] for item in self.modules}
        return dict(modules_dim=dims, modules_alpha=dims.copy(), ignore_if_contains=[],
                    only_if_contains=None, full_if_contains=None)


def inspect_lora(path, *, scale=None, expected_sha256=None):
    path = Path(path)
    if path.suffix.lower() != '.safetensors' or not path.is_file():
        raise ValueError('Continuation requires a local .safetensors file')
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError('LoRA must be nonempty and no larger than 2 GiB')
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Source LoRA checksum changed; inspect the source again')
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0):
        raise ValueError('Declared source scale must be finite and positive')
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        metadata = handle.metadata() or {}
    for flag in ('use_rslora', 'use_dora'):
        if str(metadata.get(flag, '')).lower() not in ('', 'false', '0'):
            raise ValueError(f'Unsupported adapter type: {flag}')
    metadata_alpha = metadata.get('ss_network_alpha')
    if metadata_alpha is not None:
        try:
            metadata_alpha = float(metadata_alpha)
        except (ValueError, TypeError):
            raise ValueError('Invalid source network alpha metadata') from None
        if not math.isfinite(metadata_alpha) or metadata_alpha <= 0:
            raise ValueError('Invalid source network alpha metadata')
    raw = load_file(str(path), device='cpu')
    if not raw or len(raw) > 20000:
        raise ValueError('Empty or excessively large adapter tensor map')
    pairs = {}
    native_export = all(key.startswith('diffusion_model.') for key in raw)
    for key, tensor in raw.items():
        match = re.fullmatch(r'(.+)\.(lora_A(?:\.default)?\.weight|lora_B(?:\.default)?\.weight|lora_down\.weight|lora_up\.weight|alpha)', key)
        if not match:
            raise ValueError(f'Unsupported adapter tensor: {key}')
        name = native_module(match[1])
        kind = match[2]
        part = 'alpha' if kind == 'alpha' else 'A' if kind.startswith(('lora_A', 'lora_down')) else 'B'
        group = pairs.setdefault(name, {})
        if part in group:
            raise ValueError(f'Duplicate normalized tensor: {name}.{part}')
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f'Nonfinite or nonfloating adapter tensor: {key}')
        group[part] = tensor
    state, modules = {}, []
    for name, pair in sorted(pairs.items()):
        if 'A' not in pair or 'B' not in pair:
            raise ValueError(f'Incomplete LoRA pair: {name}')
        down, up = pair['A'], pair['B']
        if down.ndim != 2 or up.ndim != 2 or min(*down.shape, *up.shape) <= 0 or down.shape[0] != up.shape[1]:
            raise ValueError(f'Invalid linear LoRA shapes: {name}')
        rank = down.shape[0]
        alpha = pair.get('alpha')
        if alpha is None and metadata_alpha is not None and not native_export and metadata.get('aitk_continuation_scale') != '1':
            alpha = torch.tensor(metadata_alpha, dtype=torch.float64)
        if alpha is not None:
            if alpha.numel() != 1 or alpha.item() <= 0:
                raise ValueError(f'Invalid alpha: {name}')
            effective_scale = alpha.item() / rank
            if scale is not None and not math.isclose(scale, effective_scale):
                raise ValueError('Declared scale conflicts with tensor alpha')
        elif scale is not None:
            effective_scale = scale
        elif native_export or metadata.get('aitk_continuation_scale') == '1':
            effective_scale = 1.0
        else:
            raise ValueError('Scaling is unknown. Supply the source alpha/rank as an explicit scale (1 for a confirmed unit-scale export).')
        canonical_down = down.float()
        if not torch.isfinite(canonical_down).all():
            raise ValueError(f'Float32 conversion overflow: {name}')
        state[f'transformer.{name}.lora_A.weight'] = canonical_down
        scaled = up.float() * effective_scale
        if not torch.isfinite(scaled).all():
            raise ValueError(f'Scaling overflow: {name}')
        state[f'transformer.{name}.lora_B.weight'] = scaled
        modules.append(dict(name=name, rank=rank, input=down.shape[1], output=up.shape[0], source_scale=effective_scale))
    # Detect replacement during inspection, not merely at launch.
    if sha256_file(path) != digest:
        raise ValueError('Source changed during inspection')
    detected = 'aitk_comfy_native' if native_export else 'kohya_native' if all(key.startswith(('lora_unet_', 'lora_transformer_')) for key in raw) else 'diffusers_mapped'
    return InspectedLora(state, modules, digest, detected, metadata, path.stat().st_size)


def validate_targets(inspected, model):
    targets = dict(model.named_modules())
    for item in inspected.modules:
        module = targets.get(item['name'])
        weight = getattr(module, 'weight', None)
        if weight is None or tuple(weight.shape) != (item['output'], item['input']):
            raise ValueError(f"Source target does not match the selected model: {item['name']}")


def load_strict(network, inspected):
    """Validate every key before changing any parameter. Retain generated alpha buffers."""
    incoming = {key.replace('.lora_A.weight', '.lora_down.weight').replace('.lora_B.weight', '.lora_up.weight'): value
                for key, value in inspected.state.items()}
    incoming = {key.rsplit('.', 2)[0].replace('.', '$$') + '.' + '.'.join(key.rsplit('.', 2)[1:]): value
                for key, value in incoming.items()}
    current = network.state_dict()
    expected = {key for key in current if not key.endswith('.alpha')}
    if set(incoming) != expected:
        raise ValueError(f'Incomplete adapter mapping: missing={sorted(expected - incoming.keys())[:8]}, unexpected={sorted(incoming.keys() - expected)[:8]}')
    if any(value.shape != current[key].shape for key, value in incoming.items()):
        raise ValueError('Source rank or target dimensions differ from constructed adapter')
    # Check the destination precision before load_state_dict can mutate even one
    # parameter. Finite source weights can overflow during a dtype conversion.
    if any(not torch.isfinite(value.to(dtype=current[key].dtype)).all() for key, value in incoming.items()):
        raise ValueError('Source weights overflow the trainable adapter precision')
    network.load_state_dict({**current, **incoming}, strict=True)


def snapshot_file(source, asset_dir, expected_sha256=None):
    source = Path(source)
    digest = sha256_file(source)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError('Continuation asset checksum changed')
    root = Path(asset_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (digest + source.suffix.lower())
    if destination.exists():
        if sha256_file(destination) != digest:
            raise ValueError('Existing continuation asset is corrupt')
        return str(destination), digest
    fd, temporary = tempfile.mkstemp(dir=root, prefix='.copy-')
    try:
        with os.fdopen(fd, 'wb') as out, open(source, 'rb') as inp:
            shutil.copyfileobj(inp, out, 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        if sha256_file(temporary) != digest:
            raise ValueError('Continuation asset changed while copying')
        # Hard-link publication is no-replace, including concurrent callers.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if sha256_file(destination) != digest:
                raise ValueError('Continuation asset publication conflict')
    finally:
        os.unlink(temporary)
    return str(destination), digest


@contextmanager
def source_teacher_context(student, teacher, assistant=None):
    states = [(module, module.is_active, module.training) for module in (student, teacher, assistant) if module is not None]
    try:
        student.is_active = False
        teacher.is_active = True
        teacher.eval()
        if assistant is not None:
            assistant.is_active = False
        with torch.no_grad():
            yield
    finally:
        for module, active, training in states:
            module.is_active = active
            module.train(training)


def preservation_loss(student, teacher, mode, coefficient, mask=1.0):
    if mode not in ('teacher', 'replay_teacher'):
        raise ValueError('Not a teacher preservation mode')
    error = (student.float() - teacher.detach().float()).square()
    return (error * mask).mean() * coefficient


def validate_continuation_config(network, model, train, datasets):
    config = getattr(network, 'continuation', None)
    preservation = getattr(train, 'source_preservation', None) or {'mode': 'none'}
    if not isinstance(preservation, dict):
        raise ValueError('Invalid source preservation configuration')
    if config is None:
        if preservation.get('mode', 'none') != 'none' or any(getattr(d, 'role', 'training') == 'source_preservation' for d in datasets):
            raise ValueError('Source preservation requires an inspected continuation adapter')
        return
    if not isinstance(config, dict) or config.get('version') != VERSION:
        raise ValueError('Unsupported continuation version')
    if config.get('training_base') not in BASES or config.get('deployment_base', 'turbo') not in BASES:
        raise ValueError('Declare the Krea 2 training and deployment bases')
    if not re.fullmatch('[a-f0-9]{64}', str(config.get('source_sha256', ''))):
        raise ValueError('Inspect the source adapter before training')
    if not network.pretrained_lora_path or network.type != 'lora':
        raise ValueError('Continuation requires a source path and standard LoRA')
    if getattr(model, 'arch', '').split(':')[0] != 'krea2' or any(model.model_kwargs.get(key) for key in ('is_edit', 'kv_cache', 'edit', 'kroma')):
        raise ValueError('Continuation supports Krea 2 Raw/Turbo text-to-image only')
    if (getattr(model, 'arch_original', None) or getattr(model, 'arch', '')) not in ('krea2', 'krea2:turbo'):
        raise ValueError('Edit and Kroma continuation are not supported')
    identity = str(model.name_or_path).lower()
    if 'kroma' in identity or (identity in ('krea/krea-2-raw', 'krea/krea-2-turbo') and not identity.endswith(config['training_base'])):
        raise ValueError('Declared training base conflicts with the selected checkpoint')
    if any(getattr(model, key, None) for key in ('base_lora_path', 'lora_path', 'inference_lora_path')):
        raise ValueError('Continuation cannot be combined with a merged base or inference LoRA')
    if config['training_base'] == 'raw' and getattr(model, 'assistant_lora_path', None):
        raise ValueError('Raw continuation does not use the Turbo training adapter')
    if config['training_base'] == 'turbo' and not getattr(model, 'assistant_lora_path', None):
        raise ValueError('Turbo continuation requires its training adapter')
    if getattr(train, 'train_text_encoder', False) or not getattr(train, 'train_unet', True) or getattr(train, 'merge_network_on_save', False):
        raise ValueError('Continuation trains and exports only the denoiser adapter')
    if getattr(train, 'start_step', None) not in (None, 0):
        raise ValueError('Imported continuation starts at zero; resume its own checkpoints normally')
    # Zero is an imported-weight initialization preference, not a checkpoint override.
    train.start_step = None
    if not isinstance(preservation, dict) or preservation.get('mode', 'none') not in MODES:
        raise ValueError('Invalid source preservation mode')
    mode = preservation.get('mode', 'none')
    count = preservation.get('new_batches', 3)
    weight = preservation.get('teacher_weight', 1.0)
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError('Source preservation new_batches must be an integer between 1 and 100')
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError('Source teacher weight must be finite and positive')
    preserve = [d for d in datasets if getattr(d, 'role', 'training') == 'source_preservation']
    if any(d.type != 'image' or d.num_frames != 1 or d.auto_frame_count or d.do_audio or
           d.control_path or d.control_from_same_folder or d.controls or d.unconditional_path for d in datasets):
        raise ValueError('Continuation requires text-to-image datasets without video, control or unconditional images')
    if not any(getattr(d, 'role', 'training') == 'training' and not d.is_reg for d in datasets):
        raise ValueError('Continuation requires new training examples')
    if (mode != 'none') != bool(preserve):
        raise ValueError('Select preservation data when preservation is enabled; remove it when disabled')
    if mode != 'none':
        conflicts = ('diff_output_preservation', 'blank_prompt_preservation', 'sega_distill',
                     'do_prior_divergence', 'do_guidance_loss', 'inverted_mask_prior', 'do_differential_guidance')
        if any(getattr(train, key, False) for key in conflicts) or getattr(train, 'loss_type', '') == 'mean_flow':
            raise ValueError('Source preservation cannot be combined with other teacher/guided losses')
        if any(d.is_reg or d.prior_reg for d in datasets):
            raise ValueError('Source preservation uses its own data role, not regularization datasets')
        if any(d.network_weight != 1 or d.caption_dropout_rate or d.trigger_word or
               d.token_dropout_rate or d.shuffle_tokens or d.random_triggers or d.use_short_captions for d in preserve):
            raise ValueError('Preservation captions must be unchanged: weight 1, caption dropout 0, no injected trigger')
    preservation.update(mode=mode, new_batches=count, teacher_weight=float(weight))
    train.source_preservation = preservation


def prepare_continuation(process):
    """Pin inputs before loading the base, including the Turbo auxiliary adapter."""
    config = getattr(process.network_config, 'continuation', None)
    validate_continuation_config(process.network_config, process.model_config, process.train_config, process.dataset_configs)
    if config is None:
        return None
    if any(getattr(process, key, None) for key in ('adapter_config', 'embed_config', 'decorator_config')):
        raise ValueError('Continuation supports only the imported denoiser adapter')
    if getattr(process, 'watermark_config', None) and process.watermark_config.enabled:
        raise ValueError('Source continuation does not support watermark adapters')
    assets = Path(process.save_root) / '.continuation'
    pinned_source = assets / (config['source_sha256'] + '.safetensors')
    source, digest = snapshot_file(pinned_source if pinned_source.exists() else process.network_config.pretrained_lora_path, assets, config['source_sha256'])
    process.network_config.pretrained_lora_path = source
    inspected = inspect_lora(source, scale=config.get('source_scale'), expected_sha256=digest)
    declared_source_base = inspected.metadata.get('aitk_continuation_training_base')
    runtime_profile = {key: getattr(process.model_config, key, None) for key in (
        'quantize', 'qtype', 'quantize_te', 'qtype_te', 'dtype', 'vae_dtype', 'low_vram',
        'layer_offloading', 'layer_offloading_backend', 'layer_offloading_transformer_percent',
        'layer_offloading_text_encoder_percent')}
    runtime_profile_path = assets / 'runtime-profile.json'
    from toolkit.continuation_assets import pin_json
    pin_json(runtime_profile_path, runtime_profile,
             'Continuation precision or offloading settings changed; create a new job')
    assistant = getattr(process.model_config, 'assistant_lora_path', None)
    if assistant:
        from toolkit.base_lora import resolve_lora_path
        existing_provenance = assets / 'provenance.json'
        if existing_provenance.exists():
            previous = json.loads(existing_provenance.read_text(encoding='utf-8'))
            previous_hash = previous.get('assistant_sha256')
            if isinstance(previous_hash, str) and re.fullmatch('[a-f0-9]{64}', previous_hash):
                assistant = str(assets / (previous_hash + '.safetensors'))
                config['assistant_sha256'] = previous_hash
        path, assistant_hash = snapshot_file(resolve_lora_path(assistant, 'Turbo training adapter'), assets, config.get('assistant_sha256'))
        process.model_config.assistant_lora_path = path
        config['assistant_sha256'] = assistant_hash
    manifests = []
    for dataset in process.dataset_configs:
        if getattr(dataset, 'role', 'training') != 'source_preservation':
            continue
        from toolkit.preservation_manifest import validate_preservation_manifest
        manifest_path = validate_preservation_manifest(dataset.folder_path, digest)
        _, manifest_hash = snapshot_file(manifest_path, assets)
        manifests.append(manifest_hash)
    provenance = dict(version=1, source_sha256=digest, training_base=config['training_base'],
                      source_scale=config.get('source_scale'),
                      deployment_base=config.get('deployment_base', 'turbo'),
                      assistant_sha256=config.get('assistant_sha256'), preservation_manifests=manifests,
                      base=str(process.model_config.name_or_path), qtype=getattr(process.model_config, 'qtype', None),
                      quantize=process.model_config.quantize, preservation=process.train_config.source_preservation)
    provenance_path = assets / 'provenance.json'
    if declared_source_base is not None:
        provenance['source_declared_training_base'] = declared_source_base
    pin_json(provenance_path, provenance,
             'Continuation inputs changed; create a new job instead of changing its source')
    config['preservation_manifests'] = manifests
    process.model_config._continuation_context = dict(assets=str(assets), source=source,
        sha256=digest, scale=config.get('source_scale'), turbo=config['training_base'] == 'turbo')
    process._continuation_provenance = provenance
    return inspected


def build_source_teacher(process, inspected):
    from toolkit.lora_special import LoRASpecialNetwork
    teacher = LoRASpecialNetwork(text_encoder=None, unet=process.sd.get_model_to_train(),
        lora_dim=1, alpha=1, train_unet=True, train_text_encoder=False,
        network_config=process.network_config, network_type='lora', transformer_only=False,
        is_transformer=True, base_model=process.sd, target_lin_modules=process.sd.target_lora_modules,
        **inspected.network_kwargs())
    modules = teacher.text_encoder_loras + teacher.unet_loras
    original_forwards = [(module.org_module[0], module.org_module[0].forward) for module in modules]
    try:
        teacher.apply_to(None, process.sd.get_model_to_train(), False, True)
        load_strict(teacher, inspected)
        teacher.force_to(process.device_torch, dtype=torch.float32)
        teacher.requires_grad_(False)
        teacher.eval()
        teacher.can_merge_in = False
        teacher.multiplier = 1.0
        teacher._update_torch_multiplier()
        teacher.is_active = False
    except BaseException:
        # A failed load/device transfer must not leave half-installed teacher
        # forwards on the shared base (including diagnostic probe failures).
        for module, forward in original_forwards:
            module.forward = forward
        raise
    process.network.can_merge_in = False
    return teacher


def attach_inference_source(base_model):
    """Krea's native generation must actually load the selected source adapter."""
    from toolkit.base_lora import resolve_lora_path
    from toolkit.config_modules import NetworkConfig
    from types import SimpleNamespace
    path = resolve_lora_path(base_model.model_config.lora_path)
    inspected = inspect_lora(path, scale=base_model.model_config.model_kwargs.get('continuation_source_scale'))
    validate_targets(inspected, base_model.get_model_to_train())
    process = SimpleNamespace(sd=base_model, network_config=NetworkConfig(type='lora'),
                              device_torch=base_model.device_torch, network=SimpleNamespace(can_merge_in=False))
    network = build_source_teacher(process, inspected)
    network.is_active = True
    base_model.network = network
    base_model._source_lora_sha256 = inspected.sha256
    base_model.model_config.lora_path = path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--scale', type=float)
    parser.add_argument('--sha256')
    parser.add_argument('--validate-krea2-topology', action='store_true')
    args = parser.parse_args()
    try:
        inspected = inspect_lora(args.path, scale=args.scale, expected_sha256=args.sha256)
        result = inspected.report()
        if args.validate_krea2_topology:
            # Meta construction validates the real topology without allocating/downloading weights.
            import contextlib
            import sys
            with contextlib.redirect_stdout(sys.stderr):
                from extensions_built_in.diffusion_models.krea2.krea2 import KREA2_MMDIT_CONFIG
                from extensions_built_in.diffusion_models.krea2.src.mmdit import SingleStreamDiT, SingleMMDiTConfig
                with torch.device('meta'):
                    model = SingleStreamDiT(SingleMMDiTConfig(**KREA2_MMDIT_CONFIG))
                validate_targets(inspected, model)
            result['topology_verified'] = True
    except (ValueError, OSError, RuntimeError) as error:
        message = str(error)
        result = dict(version=VERSION, status='unsupported', error=message,
                      rejection_reasons=[message], missing_information=['source_scale'] if 'Scaling is unknown' in message else [])
    print(json.dumps(result))


if __name__ == '__main__':
    main()
