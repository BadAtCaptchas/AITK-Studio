import assert from 'node:assert/strict';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { loadDatasetListing } from './datasetListingFixture.mjs';
const { listDatasetImages, clearDatasetListingCache, mapDatasetItemsBounded, DATASET_LISTING_CONCURRENCY, datasetListingCacheStats } = loadDatasetListing();

async function fixture(t) {
  clearDatasetListingCache();
  const root = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-listing-'));
  t.after(async () => { clearDatasetListingCache(); await fsp.rm(root, { recursive: true, force: true }); });
  return root;
}

test('bounded map limits filesystem concurrency and preserves order', async () => {
  let active = 0, peak = 0;
  const result = await mapDatasetItemsBounded(Array.from({ length: 100 }, (_, i) => i), async i => {
    peak = Math.max(peak, ++active);
    await new Promise(resolve => setImmediate(resolve));
    active--;
    return i * 2;
  });
  assert.equal(peak, DATASET_LISTING_CONCURRENCY);
  assert.deepEqual(result, Array.from({ length: 100 }, (_, i) => i * 2));
});

test('cached membership tracks external captions, files, directories, and fresh content metadata', async t => {
  const root = await fixture(t);
  const image = path.join(root, 'image.png');
  const caption = path.join(root, 'image.txt');
  await fsp.writeFile(image, 'image');
  let result = await listDatasetImages(root);
  assert.equal(result.images[0].captioned_at, null);
  await fsp.writeFile(caption, 'caption');
  result = await listDatasetImages(root);
  assert.ok(result.images[0].captioned_at);
  const updated = new Date('2025-01-02T03:04:05.000Z');
  await fsp.utimes(caption, updated, updated);
  await fsp.writeFile(image, 'larger image content');
  result = await listDatasetImages(root);
  assert.equal(result.images[0].captioned_at, updated.toISOString());
  assert.equal(result.images[0].size_bytes, 20);
  await fsp.unlink(caption);
  assert.equal((await listDatasetImages(root)).images[0].captioned_at, null);
  await fsp.mkdir(path.join(root, 'nested'));
  await fsp.writeFile(path.join(root, 'nested', 'clip.mp4'), 'video');
  await fsp.writeFile(path.join(root, 'text.txt'), 'standalone');
  await fsp.mkdir(path.join(root, '_controls'));
  await fsp.writeFile(path.join(root, '_controls', 'ignored.png'), 'ignored');
  assert.equal((await listDatasetImages(root)).images.length, 3);
  await fsp.unlink(image);
  assert.equal((await listDatasetImages(root)).images.length, 2);
});

test('warm membership avoids enumeration, page metadata work is bounded to selected items', async t => {
  const root = await fixture(t);
  for (let i = 0; i < 120; i++) await fsp.writeFile(path.join(root, `${String(i).padStart(4, '0')}.png`), 'img');
  await listDatasetImages(root, { limit: 10, compact: true });
  const readdir = fs.promises.readdir, lstat = fs.promises.lstat;
  let directoryReads = 0, fileStats = 0;
  fs.promises.readdir = async (...args) => { if (args[0] === root) directoryReads++; return readdir(...args); };
  fs.promises.lstat = async (...args) => { if (String(args[0]).endsWith('.png')) fileStats++; return lstat(...args); };
  t.after(() => { fs.promises.readdir = readdir; fs.promises.lstat = lstat; });
  const page = await listDatasetImages(root, { offset: 10, limit: 10, compact: true });
  assert.equal(directoryReads, 0);
  assert.equal(fileStats, 10);
  assert.equal(page.total, 120);
  assert.equal(page.next_offset, 20);
  assert.equal(page.root, root + path.sep);
  assert.equal(page.images[0].img_path, '0010.png');
  assert.equal((await listDatasetImages(root, { offset: 120, limit: 10 })).next_offset, null);
});

test('periodic re-enumeration remains available when directory metadata is unchanged', async t => {
  const root = await fixture(t);
  await fsp.writeFile(path.join(root, 'old.png'), 'old');
  await listDatasetImages(root);
  const now = Date.now;
  Date.now = () => now() + 31_000;
  t.after(() => { Date.now = now; });
  const original = fs.promises.readdir;
  let reads = 0;
  fs.promises.readdir = async (...args) => { if (args[0] === root) reads++; return original(...args); };
  t.after(() => { fs.promises.readdir = original; });
  await listDatasetImages(root);
  assert.equal(reads, 1);
});

test('replaced nested directory and media symlinks cannot reuse cached paths', async t => {
  const root = await fixture(t);
  const nested = path.join(root, 'nested'), outside = await fixture(t);
  await fsp.mkdir(nested);
  await fsp.writeFile(path.join(nested, 'safe.png'), 'safe');
  await fsp.writeFile(path.join(outside, 'secret.png'), 'secret');
  await listDatasetImages(root);
  await fsp.rm(nested, { recursive: true });
  try { await fsp.symlink(outside, nested, process.platform === 'win32' ? 'junction' : 'dir'); }
  catch (error) { if (error.code === 'EPERM') { t.skip('Host does not permit symlinks'); return; } throw error; }
  assert.deepEqual((await listDatasetImages(root)).images, []);
});

test('layer publication and manifest changes are never hidden by warm membership caches', async t => {
  const root = await fixture(t);
  const group = {
    format: 'aitk.layered-image', version: 1, id: 'document123', composite: 'art.png', caption: 'art.txt',
    width: 64, height: 64, order: 'bottom-to-top',
    layers: [{ path: '.layers/document123/0.png', name: 'one', caption: '' }],
  };
  const directory = path.join(root, '.layers', group.id);
  await fsp.mkdir(directory, { recursive: true });
  await fsp.writeFile(path.join(root, group.composite), 'image');
  await fsp.writeFile(path.join(root, group.caption), 'caption');
  await fsp.writeFile(path.join(root, group.layers[0].path), 'layer');
  const pending = path.join(root, '.layers', '.pending-document123.json');
  await fsp.writeFile(pending, JSON.stringify(group));
  assert.deepEqual((await listDatasetImages(root)).images, []);
  await fsp.rename(pending, path.join(directory, 'manifest.json'));
  const listed = await listDatasetImages(root);
  assert.equal(listed.images.length, 1);
  assert.equal(listed.images[0].layer_count, 1);
  await fsp.writeFile(path.join(directory, 'manifest.json'), '{}');
  await assert.rejects(listDatasetImages(root), /Invalid layered image manifest/);
});


test('concurrent cold requests share index admission without duplicating cache accounting', async t => {
  const root = await fixture(t);
  await fsp.writeFile(path.join(root, 'a.png'), 'image');
  const original = fs.promises.readdir;
  let reads = 0;
  fs.promises.readdir = async (...args) => {
    if (args[0] === root) { reads++; await new Promise(resolve => setImmediate(resolve)); }
    return original(...args);
  };
  t.after(() => { fs.promises.readdir = original; });
  await Promise.all(Array.from({ length: 32 }, () => listDatasetImages(root)));
  assert.equal(reads, 1);
  assert.deepEqual(datasetListingCacheStats(), { directories: 1, entries: 1, snapshots: 1, unadmittedSnapshotIndexes: 0 });
});

test('oversized unadmitted directories cannot pin indexes through snapshots', async t => {
  const root = await fixture(t);
  await fsp.writeFile(path.join(root, 'a.png'), 'image');
  const original = fs.promises.readdir;
  const entries = Array.from({ length: 100001 }, (_, i) => ({
    name: i === 0 ? 'a.png' : `.ignored-${i}`, isFile: () => true, isDirectory: () => false,
  }));
  fs.promises.readdir = async (...args) => args[0] === root ? entries : original(...args);
  t.after(() => { fs.promises.readdir = original; });
  const result = await listDatasetImages(root, { limit: 1 });
  assert.equal(result.images.length, 1);
  assert.deepEqual(datasetListingCacheStats(), { directories: 0, entries: 0, snapshots: 0, unadmittedSnapshotIndexes: 0 });
});

test('eviction during a refreshed directory read cannot undercount the cache budget', async t => {
  const root = await fixture(t);
  const first = path.join(root, 'first');
  await fsp.mkdir(first);
  await fsp.writeFile(path.join(first, 'a.png'), 'image');
  await listDatasetImages(first);
  const original = fs.promises.readdir, now = Date.now;
  let release, entered;
  const started = new Promise(resolve => { entered = resolve; });
  const gate = new Promise(resolve => { release = resolve; });
  Date.now = () => now() + 31_000;
  fs.promises.readdir = async (...args) => {
    if (args[0] === first) { entered(); await gate; }
    return original(...args);
  };
  t.after(() => { Date.now = now; fs.promises.readdir = original; release(); });
  const refreshing = listDatasetImages(first);
  await started;
  for (let i = 0; i < 257; i++) {
    const folder = path.join(root, `other-${i}`);
    await fsp.mkdir(folder);
    await fsp.writeFile(path.join(folder, 'b.png'), 'image');
    await listDatasetImages(folder);
  }
  release();
  await refreshing;
  const stats = datasetListingCacheStats();
  assert.equal(stats.entries, 256);
  assert.equal(stats.directories, 256);
  assert.equal(stats.unadmittedSnapshotIndexes, 0);
});
