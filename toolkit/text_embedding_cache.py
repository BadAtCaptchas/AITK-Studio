"""Job-owned, CPU-only prompt cache. Disk remains the source of truth."""
from collections import OrderedDict
import copy
import os
import threading

import torch

DEFAULT_TEXT_EMBEDDING_CACHE_MB = 64
# Cloning large mmap-backed tensors costs more than re-opening their files.
DEFAULT_MAX_ENTRY_BYTES = 1024 * 1024


def _tensor_bytes(value, seen=None):
    """Count retained tensor storages, including ragged/custom prompt containers."""
    if seen is None:
        seen = set()
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError('The prompt cache only accepts CPU tensors')
        storage = value.untyped_storage()
        identity = ('storage', storage.data_ptr(), storage.nbytes())
        if identity in seen:
            return 0
        seen.add(identity)
        return storage.nbytes()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item, seen) for item in value)
    if hasattr(value, '__dict__'):
        return _tensor_bytes(vars(value), seen)
    return 0


def _clone_cpu(value):
    """Copy small frozen CPU tensors without dispatching PyTorch's thread pool.

    safetensors loads dense CPU tensors. A byte view keeps bfloat16, integer and
    complex dtypes exact; NumPy owns a private allocation retained by from_numpy.
    Container metadata is copied too, including model-specific prompt fields.
    """
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError('The prompt cache only accepts CPU tensors')
        if value.numel() == 0:
            return torch.empty(value.shape, dtype=value.dtype, device='cpu', requires_grad=value.requires_grad)
        raw = value.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8).numpy().copy()
        result = torch.from_numpy(raw).view(value.dtype).reshape(value.shape)
        if value.requires_grad:
            result.requires_grad_(True)
        return result
    if isinstance(value, dict):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    if hasattr(value, '__dict__'):
        result = object.__new__(type(value))
        object.__setattr__(result, '__dict__', _clone_cpu(vars(value)))
        return result
    return copy.deepcopy(value)


def _signature(path):
    stat = os.stat(path)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class TextEmbeddingCache:
    """One LRU shared by a job's datasets; returned tensors never alias entries.

    The budget counts tensor storage (not active batch copies or Python object
    overhead); an entry-count cap also bounds small-entry metadata. Child
    DataLoader processes bypass this cache, avoiding a budget
    per worker. Fork/spawn must never inherit warm entries or a locked mutex.
    """
    def __init__(self, max_bytes, max_entry_bytes=DEFAULT_MAX_ENTRY_BYTES, max_entries=1024):
        self.max_bytes = max(0, int(max_bytes))
        self.max_entry_bytes = max(0, int(max_entry_bytes))
        self.max_entries = max(0, int(max_entries))
        self._reset()

    def _reset(self):
        self._pid = os.getpid()
        self._entries = OrderedDict()
        self.retained_bytes = 0
        self._lock = threading.RLock()

    def __deepcopy__(self, memo):
        # Dataset sample/flip copies share this job resource; only the returned
        # embedding containers are mutable per sample. Process pickle is separate.
        memo[id(self)] = self
        return self

    def __getstate__(self):
        return {'max_bytes': self.max_bytes, 'max_entry_bytes': self.max_entry_bytes, 'max_entries': self.max_entries}

    def __setstate__(self, state):
        self.max_bytes = state['max_bytes']
        self.max_entry_bytes = state['max_entry_bytes']
        self.max_entries = state['max_entries']
        self._reset()

    def load(self, path, loader):
        if self._pid != os.getpid():
            self._reset()
        if self.max_bytes == 0 or self.max_entries == 0 or torch.utils.data.get_worker_info() is not None:
            return loader(path)
        path = os.path.realpath(path)
        try:
            signature = _signature(path)
        except OSError:
            with self._lock:
                stale = self._entries.pop(path, None)
                if stale is not None:
                    self.retained_bytes -= stale[2]
            raise
        with self._lock:
            entry = self._entries.pop(path, None)
            if entry is not None:
                old_signature, embeds, size = entry
                if old_signature == signature:
                    self._entries[path] = entry
                    return _clone_cpu(embeds)
                self.retained_bytes -= size
        if signature[2] > self.max_entry_bytes:
            return loader(path)
        # Preserve existing parallel batch-load I/O: no mutex around disk reads.
        embeds = loader(path)
        size = _tensor_bytes(embeds)
        if size == 0 or size > min(self.max_bytes, self.max_entry_bytes):
            return embeds
        try:
            if _signature(path) != signature:
                return embeds
        except OSError:
            return embeds
        # Keep only our private copy; callers may mutate tensors or .to() in place.
        cached = _clone_cpu(embeds)
        # Container clone() may split aliased views into separate storages.
        size = _tensor_bytes(cached)
        if size > min(self.max_bytes, self.max_entry_bytes):
            return embeds
        with self._lock:
            previous = self._entries.pop(path, None)
            if previous is not None:
                self.retained_bytes -= previous[2]
            while self._entries and (self.retained_bytes + size > self.max_bytes or len(self._entries) >= self.max_entries):
                _, (_, _, removed_size) = self._entries.popitem(last=False)
                self.retained_bytes -= removed_size
            self._entries[path] = (signature, cached, size)
            self.retained_bytes += size
        return embeds


def job_text_embedding_cache(model):
    """Attach once to the model's job lifetime, never to process-global state."""
    cache = getattr(model, '_cpu_text_embedding_cache', None) if model is not None else None
    if cache is None:
        raw = os.environ.get('AITK_TEXT_EMBEDDING_CACHE_MB', str(DEFAULT_TEXT_EMBEDDING_CACHE_MB))
        try:
            budget_mb = int(raw)
        except ValueError as error:
            raise ValueError('AITK_TEXT_EMBEDDING_CACHE_MB must be a non-negative integer') from error
        if budget_mb < 0:
            raise ValueError('AITK_TEXT_EMBEDDING_CACHE_MB must be a non-negative integer')
        cache = TextEmbeddingCache(budget_mb * 1024 * 1024)
        if model is not None:
            model._cpu_text_embedding_cache = cache
    return cache
