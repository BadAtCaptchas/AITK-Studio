# Performance controls and measurements

These optimizations preserve Studio's cache formats, remote-worker JSON contracts,
training RNG order, and exact-resume worker settings. Measurements below are CPU
microbenchmarks, not GPU/MPS training-throughput measurements.

## Behavior and controls

- **Charts:** unchanged metric polls retain series identities and data revisions.
  A changed series copies only its own history. Disabled smoothing/trend work is
  skipped; the primary smoothed series is shared with the summary. Bidirectional
  EMA, log-scale filtering, same-step corrections, delayed metrics, restarts,
  deletion reconciliation, and persisted graph settings retain their semantics.
- **Warm latent startup:** finalization marks the authoritative survivor list in
  one pass. Duplicate paths, failed samples, and bucket remaps are unchanged.
- **Dataset listings:** directory membership and sidecar lookup are indexed in
  memory, with fresh directory identity/timestamps and media/caption metadata.
  Metadata concurrency is capped at 16. Membership is forcibly re-enumerated
  after 30 seconds to recover from coarse or missed filesystem timestamps.
  The index admits at most 256 directories and 100,000 entries; snapshots cannot
  retain evicted indexes. Encryption and layered-publication checks remain live.
  Optional `limit` (1–1000) and `offset` reduce metadata work for API clients.
  Galleries still request complete lists, preserving filters and bulk selection;
  this is not a paged gallery UI. Older workers may return their legacy full list.
  Offset pages are not a transactional snapshot across concurrent file edits.
- **Polling:** frontend polling pauses and aborts requests while the document is
  hidden, then refreshes when visible after canceled work settles. Collapsed
  dashboard output suspends log requests while retaining its log/offset.
  Background worker synchronization and hardware streaming are unchanged.
- **CPU text embeddings:** a job shares a small-entry LRU across its datasets.
  `AITK_TEXT_EMBEDDING_CACHE_MB` sets its retained tensor-storage budget (default
  64 MiB; `0` disables it). Entries larger than 1 MiB bypass the cache because
  copying large mmap-backed tensors is slower than reopening their files.
  At most 1,024 entries are retained. The budget excludes active batch copies and
  Python object overhead. Private CPU byte copies preserve tensor dtype and
  container metadata without changing PyTorch's global thread configuration.
  File identity/size/mtime/ctime changes invalidate entries. Dataset sample copies
  share the job cache, but returned tensors are isolated; fork/spawn drops warm
  entries and DataLoader subprocesses bypass caching. No GPU tensors are cached.
- **Cold latent preparation:** `latent_decode_prefetch: false` in a dataset turns
  off the default one-image decode lookahead. One CPU thread decodes ordinary
  still images while the owner processes the previous item. Crops, augmentation,
  RNG, transforms, device transfers, and encoding stay on the owner thread.
  Single-item runs, specialized/reference media, unsupported modes, and sources
  larger than 4,194,304 pixels remain serial. Retained lookahead pixels are at
  most 16 MiB; the owner's image and transient decoder allocations are additional.
  Speculative read failures fall back to normal loading. Atomic publication,
  concurrent-writer reuse, failure filtering, and cleanup remain protected.

## Reproducing measurements

Run from the repository root unless a command changes directories. Use an idle
machine for meaningful latency comparisons. Filesystem benchmarks use temporary
fixtures and warm OS caches; they do not measure cold physical disks, HTTP,
compression, browser rendering, network filesystems, or actual GPU encoding.

```bash
(cd ui && node scripts/chartPerformance.bench.mjs)
(cd ui && node scripts/benchmarkDatasetListing.mjs)
node --test --test-name-pattern='ten-minute request-count benchmark' ui/scripts/pollVisibility.test.mjs
.venv/bin/python scripts/benchmarks/text_embedding_cache.py
.venv/bin/python scripts/benchmarks/latent_cache_preparation.py
```

The chart benchmark compares against commit
`26cfa8a3c9eb18442261c8bf330fb5b8bc3e11d6`; its optional first argument selects a
different baseline. The embedding benchmark uses the current PyTorch thread count
by default; `--threads 1` is available for comparison, not a training recommendation.
The latent benchmark's 10 ms owner wait is explicitly simulated and demonstrates
potential overlap only. Finalization timing measures training startup, not steps
per second. Actual GPU/MPS training remains unvalidated by these benchmarks.

## Recorded CPU measurements (2026-10-31)

Linux workspace, Node 24.19.0, CPU PyTorch with its existing five-thread setting.
Benchmarks were rerun sequentially after validation processes finished. Timings
are machine-specific and should be reproduced on representative datasets.

| Path | Before | After |
| --- | ---: | ---: |
| Unchanged metric merge, 24 × 4,000 points | 6.566 ms | 0.0153 ms |
| One new point in one of those series | 5.975 ms | 0.0161 ms |
| Raw-only chart data construction | 28.823 ms | 12.968 ms |
| Default chart data construction | 28.354 ms | 27.613 ms |
| 1,000-file full listing | 128.23 ms | 28.92 ms cold index / 16.36 ms warm |
| 10,000-file full listing | 1,141.96 ms | 244.14 ms cold index / 185.93 ms warm |
| 1,000-item latent finalization | 3.097 ms | 0.00874 ms |
| 10,000-item latent finalization | 308.247 ms | 0.0864 ms |
| 100,000-item latent finalization | 32.040 s | 0.906 ms |
| Repeated small embedding load, 118,888 retained bytes | 0.1096 ms | 0.0535 ms |
| Repeated small embedding load, 526,336 retained bytes | 0.1144 ms | 0.0669 ms |
| 24 JPEG preparations, 1536 × 1024 | 466.77 ms | 356.93 ms |

Chart timings are five-round medians and exclude React/uPlot rendering. All 24
histories retain their identities on unchanged polls; 23 do on a single-series
update. The default-chart timing is close enough that no general rendering
speedup is claimed.

The listing fixture has 25% caption sidecars and verifies exact payload equality.
For 10,000 items, stat/lstat calls fell from 55,001 to 12,502, and warm directory
reads from one to zero. Optional API pages of 100 items took 48.30 ms with a cold
index and 5.68 ms for the next warm page. These page timings do not represent the
current gallery's first render, since it still requests the complete list.

The polling benchmark uses deterministic time: six consumers over ten hidden
minutes made 906 baseline requests versus six initial requests and zero hidden
requests after the change, then six immediate foreground refreshes. Browser
regressions verify this policy and collapsed-log offset preservation.

Finalization compares one run of the old quadratic loop with a five-run median
of the linear pass. JPEG preparation uses three interleaved runs and seeded
images. Adding an explicitly simulated 10 ms owner wait per image gave
736.42 → 367.36 ms; this is not measured GPU encoding or training throughput.

Embedding measurements use ten warmup rounds followed by 50 interleaved samples,
rotating disk/cache/uncapped-copy order. This matters because sequential runs were
sensitive to warmup and CPU scheduling. Large entries remain uncached: for example,
the ~1 MiB case measured 0.1038 ms disk versus 0.1123 ms through the bypass path,
showing the small stat/check overhead rather than a speedup. Uncapped large tensor
copies were substantially slower and are not enabled. No end-to-end training
speedup is inferred from these load timings.
