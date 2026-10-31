import assert from 'node:assert/strict';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { findDatasetItemsRecursivelyAsync } from '../dist/src/server/datasetImages.js';
import { findExistingCaptionSidecarAsync } from '../dist/src/server/captionFiles.js';
import { listLayeredImages } from '../dist/src/server/layeredImages.js';
import { loadDatasetListing } from './datasetListingFixture.mjs';
const { listDatasetImages, clearDatasetListingCache, mapDatasetItemsBounded } = loadDatasetListing();
const sizes = process.argv.slice(2).map(Number).filter(value => value > 0);
if (!sizes.length) sizes.push(1000, 10000);

// Reproduce the former route's traversal, sidecar probing and unbounded stat fanout.
async function baseline(root) {
  const files = await findDatasetItemsRecursivelyAsync(root);
  await listLayeredImages(root);
  files.sort((a, b) => a.localeCompare(b));
  return Promise.all(files.map(async filename => {
    const stat = await fsp.stat(filename);
    const caption = await findExistingCaptionSidecarAsync(filename);
    const captionStat = caption ? await fsp.stat(caption) : null;
    return { img_path: filename, added_at: stat.birthtime.toISOString(), captioned_at: captionStat?.mtime.toISOString() ?? null, size_bytes: stat.size };
  }));
}
async function measure(label, fn) {
  const methods = ['stat', 'lstat', 'readdir'];
  const originals = Object.fromEntries(methods.map(method => [method, fsp[method]]));
  const calls = Object.fromEntries(methods.map(method => [method, 0]));
  for (const method of methods) fsp[method] = (...args) => { calls[method]++; return originals[method](...args); };
  const start = performance.now();
  try { const value = await fn(); return { label, ms: +(performance.now() - start).toFixed(2), calls, value }; }
  finally { for (const method of methods) fsp[method] = originals[method]; }
}
for (const size of sizes) {
  const root = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-listing-benchmark-'));
  try {
    await mapDatasetItemsBounded(Array.from({ length: size }, (_, i) => i), async i => {
      const filename = path.join(root, String(i).padStart(7, '0'));
      await fsp.writeFile(filename + '.png', 'synthetic metadata benchmark');
      if (i % 4 === 0) await fsp.writeFile(filename + '.txt', 'caption');
    });
    const before = await measure('legacy full listing data', () => baseline(root));
    clearDatasetListingCache();
    const cold = await measure('indexed cold full listing data', () => listDatasetImages(root));
    const warm = await measure('indexed warm full listing data', () => listDatasetImages(root));
    clearDatasetListingCache();
    const first = await measure('optional API first 100 cold', () => listDatasetImages(root, { limit: 100 }));
    const next = await measure('optional API next 100 warm', () => listDatasetImages(root, { offset: 100, limit: 100 }));
    assert.deepEqual(cold.value.images, before.value);
    assert.deepEqual(warm.value.images, before.value);
    console.log(JSON.stringify({ size, method: { repetitions: 1, variance: 'not measured (one sample per path)', fixture: 'real temporary filesystem entries; synthetic PNG metadata placeholders; 25% text sidecars', cold: 'application index cleared; OS cache not flushed', warm: 'same fixture/index retained from immediately preceding full request', order: 'legacy full, indexed cold full, indexed warm full, cold first page, warm next page', temporaryFilesystemRoot: os.tmpdir(), filesystemTypeMagicHex: (await fsp.statfs(root)).type.toString(16), excludes: 'fixture creation, HTTP, compression, image decoding and browser render' }, measurements: [before, cold, warm, first, next].map(({ value: _value, ...result }) => result) }));
  } finally { clearDatasetListingCache(); await fsp.rm(root, { recursive: true, force: true }); }
}
