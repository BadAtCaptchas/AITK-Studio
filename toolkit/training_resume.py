"""Explicitly supported training continuation without replaying prefetched data."""
import copy
from functools import wraps
import hashlib
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

_CROP_FIELDS = ('scale_to_width', 'scale_to_height', 'crop_width', 'crop_height', 'crop_x', 'crop_y')


def capture_rng():
    numpy_state = np.random.get_state()
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed since the checkpoint')
        torch.cuda.set_rng_state_all(state['cuda'])


def preserve_rng(function):
    """Sampling must not change the random stream of the following train step."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        state = capture_rng()
        try:
            return function(*args, **kwargs)
        finally:
            restore_rng(state)
    return wrapped


def _datasets(dataset):
    children = getattr(dataset, 'datasets', None)
    if children is not None:
        return [leaf for child in children for leaf in _datasets(child)]
    return [dataset]


def _signature_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _signature_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_signature_value(item) for item in value]
    if hasattr(value, '__dict__'):
        return {'type': type(value).__qualname__, 'values': _signature_value(vars(value))}
    return str(value)


def loader_resume_reason(loader):
    if loader is None:
        return None
    if type(loader) is not DataLoader or loader.num_workers != 0:
        return 'Data-loader workers/prefetch or distributed loaders are not resumable exactly'
    if not isinstance(loader.sampler, (RandomSampler, SequentialSampler)):
        return 'Custom data samplers do not expose a resumable order'
    if getattr(loader.sampler, 'generator', None) is not None:
        return 'A custom sampler generator is not captured by the training RNG snapshot'
    for dataset in _datasets(loader.dataset):
        config = getattr(dataset, 'dataset_config', None)
        if config is not None:
            if getattr(config, 'batch_load_threads', 1) > 1:
                return 'Threaded batch loading has nondeterministic random ordering'
            if getattr(config, 'poi', None) is not None or getattr(dataset, 'is_pooled', False):
                return 'Dynamic point-of-interest/pooled buckets require an approximate resume'
            if getattr(dataset, 'is_layered', False):
                return 'Layered-source identity requires an approximate resume'
        elif not isinstance(dataset, torch.utils.data.TensorDataset):
            return 'Custom dataset has no checkpoint identity contract'
    return None


def _dataset_identity(dataset):
    config = getattr(dataset, 'dataset_config', None)
    digest = hashlib.sha256(json.dumps(_signature_value(config), sort_keys=True).encode())
    digest.update(json.dumps(_signature_value(getattr(dataset, 'caption_dict', None)), sort_keys=True).encode())
    files = getattr(dataset, 'file_list', None)
    if files is not None:
        for item in files:
            path = str(item.path)
            digest.update(path.encode())
            for source in (path, os.path.splitext(path)[0] + '.' + str(getattr(config, 'caption_ext', 'txt')).lstrip('.')):
                try:
                    info = os.stat(source)
                    digest.update(f'{info.st_size}:{info.st_mtime_ns}'.encode())
                except FileNotFoundError:
                    digest.update(b'missing')
    elif isinstance(dataset, torch.utils.data.TensorDataset):
        for tensor in dataset.tensors:
            digest.update(str((tensor.shape, tensor.dtype)).encode())
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class _SavedIndices:
    def __init__(self, owner):
        self.owner = owner

    def __iter__(self):
        yield from self.owner.indices[self.owner.position:]

    def __len__(self):
        return len(self.owner.indices) - self.owner.position


class ResumableDataIterator:
    """Own the concrete epoch order; num_workers=0 means no hidden lookahead."""
    def __init__(self, loader, state=None):
        reason = loader_resume_reason(loader)
        if reason:
            raise ValueError(reason)
        self.loader = loader
        self.datasets = _datasets(loader.dataset)
        identities = [_dataset_identity(dataset) for dataset in self.datasets]
        if state is not None:
            if state.get('identities') != identities:
                raise ValueError('Dataset files or configuration changed since the checkpoint')
            saved_datasets = state.get('datasets')
            if not isinstance(saved_datasets, list) or len(saved_datasets) != len(self.datasets):
                raise ValueError('Dataset topology changed since the checkpoint')
            mutations = []
            dataset_lengths = []
            for dataset, saved in zip(self.datasets, saved_datasets):
                if not isinstance(saved, dict):
                    raise ValueError('Invalid saved dataset state')
                indices = saved.get('batch_indices')
                if indices is not None:
                    file_count = len(dataset.file_list)
                    if not isinstance(indices, list) or any(not isinstance(batch, list) or
                            any(type(index) is not int or not 0 <= index < file_count for index in batch) for batch in indices):
                        raise ValueError('Invalid saved dataset buckets')
                    mutations.append((dataset, 'batch_indices', copy.deepcopy(indices)))
                dataset_lengths.append(len(indices) if indices is not None else len(dataset))
                crops = saved.get('crops')
                if crops is not None:
                    if not isinstance(crops, list) or len(crops) != len(dataset.file_list):
                        raise ValueError('Invalid saved dataset crop geometry')
                    for item, geometry in zip(dataset.file_list, crops):
                        if not isinstance(geometry, dict) or any(key not in _CROP_FIELDS or type(value) is not int or value < 0 for key, value in geometry.items()):
                            raise ValueError('Invalid saved dataset crop geometry')
                        for key, value in geometry.items():
                            mutations.append((item, key, value))
                bucket_indices = saved.get('buckets')
                if bucket_indices is not None:
                    if not isinstance(bucket_indices, dict) or set(bucket_indices) != set(dataset.buckets):
                        raise ValueError('Dataset bucket topology changed since the checkpoint')
                    for key, indices in bucket_indices.items():
                        if not isinstance(indices, list) or any(type(index) is not int or not 0 <= index < len(dataset.file_list) for index in indices):
                            raise ValueError('Invalid saved dataset bucket order')
                        mutations.append((dataset.buckets[key], 'file_list_idx', list(indices)))
                if saved.get('epoch_num') is not None:
                    if type(saved['epoch_num']) is not int or saved['epoch_num'] < 0:
                        raise ValueError('Invalid saved dataset epoch')
                    mutations.append((dataset, 'epoch_num', saved['epoch_num']))
            self.indices = state.get('indices')
            self.position = state.get('position')
            if (not isinstance(self.indices, list) or type(self.position) is not int
                    or not 0 <= self.position <= len(self.indices)):
                raise ValueError('Invalid saved data position')
            if loader.batch_sampler is not None and any(not isinstance(batch, list) for batch in self.indices):
                raise ValueError('Invalid saved batch order')
            flattened = [index for batch in self.indices for index in batch] if loader.batch_sampler is not None else self.indices
            if any(type(index) is not int or not 0 <= index < sum(dataset_lengths) for index in flattened):
                raise ValueError('Invalid saved data order')
            for target, key, value in mutations:
                setattr(target, key, value)
            if hasattr(loader.dataset, 'cumulative_sizes'):
                loader.dataset.cumulative_sizes = loader.dataset.cumsum(loader.dataset.datasets)
        else:
            self.indices = list(loader.batch_sampler if loader.batch_sampler is not None else loader.sampler)
            self.position = 0
        self.identities = identities
        kwargs = {'num_workers': 0, 'collate_fn': loader.collate_fn, 'pin_memory': loader.pin_memory,
                  'generator': torch.Generator().manual_seed(0)}
        if loader.batch_sampler is None:
            kwargs.update(batch_size=None, sampler=_SavedIndices(self))
        else:
            kwargs['batch_sampler'] = _SavedIndices(self)
        self.iterator = iter(DataLoader(loader.dataset, **kwargs))

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.iterator)
        self.position += 1
        return batch

    def state_dict(self):
        return {'identities': self.identities, 'indices': self.indices, 'position': self.position,
                'datasets': [{'epoch_num': getattr(dataset, 'epoch_num', None),
                              'batch_indices': copy.deepcopy(getattr(dataset, 'batch_indices', None)),
                              'crops': [{key: getattr(item, key) for key in _CROP_FIELDS if hasattr(item, key)}
                                        for item in dataset.file_list] if hasattr(dataset, 'file_list') else None,
                              'buckets': {key: list(bucket.file_list_idx) for key, bucket in dataset.buckets.items()}
                                         if hasattr(dataset, 'buckets') else None}
                             for dataset in self.datasets]}


def training_signature(process):
    values = vars(process.train_config).copy()
    if values.get('do_random_cfg'):
        values.pop('cfg_scale', None)
        values.pop('do_cfg', None)
    # Extending a run is useful, but changes schedule semantics: do not claim exactness.
    configuration = {'train': values, **{key: getattr(process, key, None) for key in
        ('model_config', 'network_config', 'adapter_config', 'embed_config', 'decorator_config')}}
    configuration['cadence'] = {
        'save_every': getattr(getattr(process, 'save_config', None), 'save_every', None),
        'sample_every': getattr(getattr(process, 'sample_config', None), 'sample_every', None),
        'sample_start_step': getattr(getattr(process, 'sample_config', None), 'sample_start_step', None),
    }
    return hashlib.sha256(json.dumps(_signature_value(configuration), sort_keys=True).encode()).hexdigest()


def capture_training_resume(process):
    reasons = list(getattr(process, '_resume_approximate_reasons', []))
    iterators = getattr(process, '_training_data_iterators', None)
    if iterators is None:
        reasons.append('Checkpoint was created outside an initialized training loop')
    if getattr(process.accelerator, 'num_processes', 1) != 1:
        reasons.append('Distributed RNG and data order are not captured')
    if getattr(getattr(process, 'phase_manager', None), 'enabled', False):
        reasons.append('Manual phase transitions require an approximate resume')
    if getattr(process.train_config, 'merge_network_on_save', False):
        reasons.append('Merging the adapter into the base model changes live parameterization')
    if getattr(process.train_config, 'do_paramiter_swapping', False):
        reasons.append('Optimizer parameter swapping has no exact continuation contract')
    if reasons:
        return {'version': 1, 'mode': 'approximate', 'reasons': reasons,
                'preservation_batch_index': getattr(process, '_preservation_batch_index', 0)}
    from toolkit.memory_management import sync_grad_transfers
    sync_grad_transfers()
    parameters = list({id(param): param for group in process.optimizer.param_groups for param in group['params']}.values())
    in_step = bool(getattr(process, '_in_training_step', False))
    scaler = getattr(process.accelerator, 'scaler', None)
    return {'version': 1, 'mode': 'exact', 'signature': training_signature(process),
            'next_step': process.step_num + int(in_step), 'epoch': process.epoch_num,
            'rng': capture_rng(), 'loaders': {key: value.state_dict() if value is not None else None for key, value in iterators.items()},
            'scheduler': process.lr_scheduler.state_dict(), 'scaler': scaler.state_dict() if scaler is not None else None,
            'accelerator_step': getattr(process.accelerator, 'step', 0),
            'grad_accumulation_step': process.grad_accumulation_step + int(in_step),
            'accumulated_microbatches': getattr(process, '_accumulated_microbatches', 0),
            'preservation_batch_index': getattr(process, '_preservation_batch_index', 0),
            'boundary_index': getattr(process, 'current_boundary_index', 0),
            'boundary_steps': getattr(process, 'steps_this_boundary', 0),
            'parameters': [parameter.detach().cpu().clone() for parameter in parameters],
            'gradients': [parameter.grad.detach().cpu().clone() if parameter.grad is not None else None for parameter in parameters],
            'private_gradients': [getattr(parameter, '_accum_grad').detach().cpu().clone() if getattr(parameter, '_accum_grad', None) is not None else None for parameter in parameters]}


def restore_preservation_schedule(process, state):
    """Replay cadence remains resumable even when exact data order is unavailable."""
    if (not isinstance(state, dict) or state.get('version') != 1
            or state.get('mode') not in ('exact', 'approximate')
            or 'preservation_batch_index' not in state
            or getattr(process.train_config, 'start_step', None) is not None):
        return
    index = state['preservation_batch_index']
    if type(index) is not int or index < 0:
        raise ValueError('Invalid saved source preservation counter')
    process._preservation_batch_index = index


def restore_training_resume(process, state):
    if not isinstance(state, dict) or state.get('version') != 1 or state.get('mode') != 'exact':
        raise ValueError('Checkpoint does not contain exact training state')
    if state.get('signature') != training_signature(process):
        raise ValueError('Training configuration changed since the checkpoint')
    if type(state.get('preservation_batch_index', 0)) is not int or state.get('preservation_batch_index', 0) < 0:
        raise ValueError('Invalid saved source preservation counter')
    for key in ('next_step', 'epoch', 'accelerator_step', 'grad_accumulation_step', 'accumulated_microbatches', 'boundary_index', 'boundary_steps'):
        if type(state.get(key)) is not int or state[key] < 0:
            raise ValueError(f'Invalid saved training counter: {key}')
    parameters = list({id(param): param for group in process.optimizer.param_groups for param in group['params']}.values())
    for field in ('parameters', 'gradients', 'private_gradients'):
        saved_values = state.get(field)
        if not isinstance(saved_values, list) or len(saved_values) != len(parameters):
            raise ValueError('Trainable parameter topology changed since the checkpoint')
        for parameter, saved in zip(parameters, saved_values):
            if saved is None and field != 'parameters':
                continue
            if not isinstance(saved, torch.Tensor) or saved.shape != parameter.shape or not torch.isfinite(saved).all():
                raise ValueError(f'Invalid saved training {field}')
            if field != 'private_gradients' and saved.dtype != parameter.dtype:
                raise ValueError('Trainable parameter precision changed since the checkpoint')
    with torch.no_grad():
        for parameter, value, gradient, private in zip(parameters, state['parameters'], state['gradients'], state['private_gradients']):
            parameter.copy_(value)
            parameter.grad = gradient.to(parameter.device) if gradient is not None else None
            if private is not None:
                parameter._accum_grad = private.to(parameter.device)
            elif hasattr(parameter, '_accum_grad'):
                del parameter._accum_grad
    process.lr_scheduler.load_state_dict(state['scheduler'])
    scaler = getattr(process.accelerator, 'scaler', None)
    if state['scaler'] is not None:
        if scaler is None:
            raise ValueError('AMP scaler configuration changed since the checkpoint')
        scaler.load_state_dict(state['scaler'])
    process.accelerator.step = state['accelerator_step']
    process.step_num = process.start_step = state['next_step']
    process.epoch_num = state['epoch']
    process.grad_accumulation_step = state['grad_accumulation_step']
    process._accumulated_microbatches = state['accumulated_microbatches']
    process._preservation_batch_index = state.get('preservation_batch_index', 0)
    process.current_boundary_index = state['boundary_index']
    process.steps_this_boundary = state['boundary_steps']
    restore_rng(state['rng'])
