"""Validation for Studio's versioned configuration boundary; advanced keys are retained."""
import math
import re
from toolkit.model_registry import CONTRACT


def config_contract_errors(value, device_backend=None):
    """Return configuration errors using the capability contract shared with Studio."""
    if not isinstance(value, dict) or not isinstance(value.get('config'), dict):
        return ['Config requires between 1 and 32 processes.']
    processes = value['config'].get('process')
    if not isinstance(processes, list) or not 1 <= len(processes) <= 32:
        return ['Config requires between 1 and 32 processes.']
    errors = []
    if isinstance(value.get('capability_version'), bool) or value.get('capability_version', 1) != CONTRACT['version']:
        errors.append('Unsupported capability_version.')
    extras = {'processKinds': set(), 'modelArches': set()}
    extensions = value.get('extensions', {})
    if not isinstance(extensions, dict):
        errors.append('extensions must be a namespaced object.')
    else:
        for namespace, declaration in extensions.items():
            if not re.fullmatch(r'[a-z][a-z0-9_-]*\.[a-z][a-z0-9_.-]*', namespace) or not isinstance(declaration, dict):
                errors.append('Invalid extension namespace.')
                continue
            for key, target in extras.items():
                entries = declaration.get(key, [])
                if not isinstance(entries, list) or len(entries) > 32 or any(not isinstance(entry, str) or len(entry) > 100 for entry in entries):
                    errors.append(f'Invalid extension {key}.')
                else:
                    target.update(entries)
    def finite(obj, key, low, high, integer=False):
        if key not in obj:
            return
        number = obj[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not low <= number <= high or integer and number != int(number):
            errors.append(f'{key} is outside the supported numeric range.')
    for process in processes:
        if not isinstance(process, dict) or not isinstance(process.get('type'), str):
            errors.append('Every process requires a type.')
            continue
        if process['type'] not in CONTRACT['processKinds'] and process['type'] not in extras['processKinds']:
            errors.append(f"Unknown process type: {process['type']}.")
        for field in ('train', 'model'):
            if field in process and not isinstance(process[field], dict):
                errors.append(f'{field} must be an object.')
        train = process.get('train')
        if isinstance(train, dict):
            finite(train, 'steps', 1, 1_000_000_000, True)
            finite(train, 'batch_size', 1, 4096, True)
            finite(train, 'lr', 0, 10)
            finite(train, 'gradient_accumulation', 1, 1_000_000, True)
        model = process.get('model')
        if isinstance(model, dict):
            arch = model.get('arch')
            if not isinstance(arch, str) or not arch:
                errors.append('model.arch is required.')
                continue
            base = arch.split(':', maxsplit=1)[0]
            choice = next((item for item in CONTRACT['choices'] if item['name'] == arch), None)
            choice = choice or next((item for item in CONTRACT['choices'] if item['name'] == base), None)
            if not choice and not any(item['arch'] == base for item in CONTRACT['models']) and base not in extras['modelArches']:
                errors.append(f'Unknown model architecture: {arch}.')
            network = process.get('network', {})
            if choice and choice['allowedNetworkTypes'] and isinstance(network, dict) and isinstance(network.get('type'), str) and network['type'] not in choice['allowedNetworkTypes']:
                errors.append(f'{arch} does not support the selected network.')
            backend = device_backend or str(process.get('device', '')).split(':', maxsplit=1)[0]
            if choice and backend and backend not in choice['deviceBackends']:
                errors.append(f'{arch} does not support device backend {backend}.')
            dtype = model.get('dtype', train.get('dtype') if isinstance(train, dict) else process.get('dtype'))
            if choice and dtype is not None and dtype not in choice['precisions']:
                errors.append('Unsupported model precision.')
            if backend == 'mps' and any(model.get(key) in CONTRACT['cudaOnlyQuantization'] for key in ('qtype', 'qtype_te')):
                errors.append('The selected quantization requires a CUDA worker.')
            if choice and model.get('layer_offloading') is True and not choice['offloading']['layers']:
                errors.append(f'{arch} does not support layer offloading.')
            frames = process.get('sample', {}).get('num_frames') if isinstance(process.get('sample'), dict) else None
            if choice and isinstance(frames, (int, float)) and not isinstance(frames, bool) and (frames - 1) % choice['frames']['multipleAfterFirst'] != 0:
                errors.append('Invalid frame count for selected model.')
            if arch.startswith('minimax_h3') and model.get('layer_offloading') is True:
                errors.append('MiniMax H3 does not support layer offloading.')
        sample = process.get('sample')
        if 'sample' in process and not isinstance(sample, dict):
            errors.append('sample must be an object.')
        if isinstance(sample, dict):
            finite(sample, 'num_frames', 1, 100000, True)
            finite(sample, 'fps', 0.001, 1000)
            finite(sample, 'width', 1, 32768, True)
            finite(sample, 'height', 1, 32768, True)
            finite(sample, 'sample_steps', 1, 10000, True)
            finite(sample, 'guidance_scale', 0, 1000)
    return errors
