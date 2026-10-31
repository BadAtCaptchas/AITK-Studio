"""CPU warm-filesystem microbenchmark; no model weights or training."""
import json
import statistics
import tempfile
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.prompt_utils import PromptEmbeds
from toolkit.text_embedding_cache import TextEmbeddingCache


def measure_interleaved(callbacks, repeats=50, warmup_rounds=10):
    """Warm all paths, then rotate execution order to reduce drift/order bias."""
    names = list(callbacks)
    samples = {name: [] for name in names}
    for index in range(warmup_rounds + repeats):
        offset = index % len(names)
        for name in names[offset:] + names[:offset]:
            start = time.perf_counter_ns()
            callbacks[name]()
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
            if index >= warmup_rounds:
                samples[name].append(elapsed_ms)
    return {
        name: {
            'median_ms': round(statistics.median(values), 4),
            'min_ms': round(min(values), 4),
            'max_ms': round(max(values), 4),
        }
        for name, values in samples.items()
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    with tempfile.TemporaryDirectory() as directory:
        results = []
        for tokens, width in [(77, 768), (128, 1024), (256, 1024), (512, 1024), (512, 4096), (2048, 4096)]:
            path = str(Path(directory) / f'{tokens}.safetensors')
            embeds = AdvancedPromptEmbeds(tokens=[torch.zeros(tokens, width, dtype=torch.bfloat16)], ids=[torch.arange(tokens)])
            embeds.frozen_dtype_keys = ['ids']
            embeds.save(path)
            cache = TextEmbeddingCache(64 * 1024 ** 2)
            uncapped = TextEmbeddingCache(64 * 1024 ** 2, max_entry_bytes=64 * 1024 ** 2)
            uncapped.load(path, PromptEmbeds.load)
            cache.load(path, PromptEmbeds.load)
            timings = measure_interleaved({
                'disk': lambda: PromptEmbeds.load(path),
                'cache': lambda: cache.load(path, PromptEmbeds.load),
                'uncapped': lambda: uncapped.load(path, PromptEmbeds.load),
            })
            results.append({
                'shape': [tokens, width],
                'disk_median_ms': timings['disk']['median_ms'],
                'cache_median_ms': timings['cache']['median_ms'],
                'uncapped_clone_cache_ms': timings['uncapped']['median_ms'],
                'retained_bytes': cache.retained_bytes,
                'timings': timings,
            })
        print(json.dumps({
            'torch_threads': torch.get_num_threads(), 'repeats': 50,
            'warmup_rounds': 10, 'execution_order': 'rotated each round',
            'results': results,
        }, indent=2))
