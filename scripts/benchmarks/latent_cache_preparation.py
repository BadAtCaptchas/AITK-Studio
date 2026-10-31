"""CPU-only startup microbenchmarks; no model, weights or GPU is used.

Run: .venv/bin/python scripts/benchmarks/latent_cache_preparation.py
The encode-wait benchmark uses a stated 10ms owner wait to model overlap;
it does not measure training or GPU encoding throughput.
"""
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from toolkit.latent_cache_preparation import LatentImageStager, decode_latent_image, mark_cached_survivors


class Item:
    def __init__(self):
        self.is_latent_cached = False


def timed(callback):
    start = time.perf_counter()
    callback()
    return time.perf_counter() - start


def main():
    result = {'finalization': [], 'finalization_method': {
        'fixture': 'synthetic identity objects; all survive; one item per pre-existing latent-path group',
        'included': 'complete final flag-setting pass, including old group traversal and membership checks',
        'excluded_shared_work': 'fixture/group creation, cache validation/encoding, failure filtering and bucket remapping',
        'lookup_construction': 'none: the new implementation directly traverses survivors',
        'baseline_repetitions': 1, 'candidate_repetitions': 5,
        'baseline_variance': 'not measured (single baseline sample)',
    }}
    for size in (1000, 10000, 100000):
        items = [Item() for _ in range(size)]
        groups = {index: [item] for index, item in enumerate(items)}
        def original():
            for group in groups.values():
                for item in group:
                    if item in items:
                        item.is_latent_cached = True
        old = timed(original)
        assert all(item.is_latent_cached for item in items)
        samples = []
        for _ in range(5):
            for item in items:
                item.is_latent_cached = False
            samples.append(timed(lambda: mark_cached_survivors(items)))
            assert all(item.is_latent_cached for item in items)
        new = statistics.median(samples)
        result['finalization'].append({'items': size, 'old_seconds': old, 'new_seconds': new,
            'candidate_min_seconds': min(samples), 'candidate_max_seconds': max(samples),
            'speedup': old / new})
    with tempfile.TemporaryDirectory() as directory:
        generator = np.random.default_rng(42)
        items = []
        for i in range(24):
            path = str(Path(directory) / f'{i}.jpg')
            Image.fromarray(generator.integers(0, 256, (1024, 1536, 3), dtype=np.uint8)).save(path, quality=90)
            items.append(SimpleNamespace(path=path, use_alpha_as_mask=False))
        def serial(wait):
            for item in items:
                image = decode_latent_image(item)
                time.sleep(wait)
                image.close()
        def staged(wait):
            with LatentImageStager() as stager:
                stager.submit(items[0])
                for index in range(len(items)):
                    image = stager.take_and_stage(items[index + 1] if index + 1 < len(items) else None)
                    time.sleep(wait)
                    image.close()
        serial(0)  # warm filesystem cache; both variants use identical sources
        result['decode_overlap'] = []
        for wait in (0, 0.01):
            serial_times, staged_times = [], []
            for _ in range(3):
                serial_times.append(timed(lambda: serial(wait)))
                staged_times.append(timed(lambda: staged(wait)))
            old, new = statistics.median(serial_times), statistics.median(staged_times)
            result['decode_overlap'].append({'images': len(items), 'size': [1536, 1024], 'owner_wait_seconds': wait,
                'serial_seconds': old, 'staged_seconds': new, 'speedup': old / new,
                'max_lookahead_images': 1, 'max_lookahead_pixel_bytes': 4 * 4 * 1024 * 1024})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
