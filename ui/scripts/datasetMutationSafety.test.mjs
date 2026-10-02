import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const { copyDatasetBetweenRoots } = require('../dist/src/server/datasetCopy.js');
const { deleteDatasetFolder } = require('../dist/src/server/datasetDelete.js');
const { writePlainDatasetCaption } = require('../dist/src/server/datasetCaptionWrite.js');

async function workspace(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-dataset-mutation-'));
  const datasets = path.join(root, 'datasets');
  const source = path.join(datasets, 'source');
  const outside = path.join(root, 'outside');
  await fs.mkdir(source, { recursive: true });
  await fs.mkdir(outside);
  t.after(async () => {
    assert.ok(path.resolve(root).startsWith(path.resolve(os.tmpdir()) + path.sep));
    assert.ok(path.basename(root).startsWith('aitk-dataset-mutation-'));
    await fs.rm(root, { recursive: true, force: true });
  });
  return { root, datasets, source, outside };
}

async function link(t, target, filename, directory = true) {
  try {
    await fs.symlink(target, filename, directory ? (process.platform === 'win32' ? 'junction' : 'dir') : 'file');
    return true;
  } catch (error) {
    if (error.code !== 'EPERM' && error.code !== 'EACCES') throw error;
    t.skip('Creating filesystem links is not permitted in this environment');
    return false;
  }
}

test('concurrent dataset copies reserve different directories without deleting a completed copy', { timeout: 10000 }, async t => {
  const { datasets, source } = await workspace(t);
  await fs.writeFile(path.join(source, 'image.png'), 'source image');
  const destination = path.join(datasets, 'copy');
  const mkdir = fs.mkdir;
  let arrivals = 0;
  let releaseBoth;
  let releaseWinner;
  const both = new Promise(resolve => { releaseBoth = resolve; });
  const winner = new Promise(resolve => { releaseWinner = resolve; });
  t.mock.method(fs, 'mkdir', async (target, options) => {
    if (path.resolve(target) === destination && options === undefined) {
      const ordinal = ++arrivals;
      if (arrivals === 2) releaseBoth();
      await both;
      if (ordinal === 2) await winner;
    }
    return mkdir(target, options);
  });
  const request = { datasetPath: source, sourceDatasetsRoot: datasets, destinationDatasetsRoot: datasets, requestedName: 'copy' };
  const attempts = [copyDatasetBetweenRoots(request), copyDatasetBetweenRoots(request)].map(promise =>
    promise.then(result => { releaseWinner(); return result; }),
  );
  const results = await Promise.all(attempts);
  assert.equal(arrivals, 2);
  assert.deepEqual(results.map(result => result.name).sort(), ['copy', 'copy_2']);
  for (const result of results) assert.equal(await fs.readFile(path.join(result.path, 'image.png'), 'utf8'), 'source image');
  assert.equal(await fs.readFile(path.join(source, 'image.png'), 'utf8'), 'source image');
});

test('copy rollback removes only the reserved directory and handles a source-name collision', async t => {
  const { datasets, source } = await workspace(t);
  await fs.writeFile(path.join(source, 'image.png'), 'source image');
  const copied = await copyDatasetBetweenRoots({
    datasetPath: source, sourceDatasetsRoot: datasets, destinationDatasetsRoot: datasets, requestedName: 'source',
  });
  assert.equal(copied.name, 'source_2');
  t.mock.method(fs, 'copyFile', async () => { throw new Error('simulated copy failure'); });
  await assert.rejects(copyDatasetBetweenRoots({
    datasetPath: source, sourceDatasetsRoot: datasets, destinationDatasetsRoot: datasets, requestedName: 'source',
  }), /simulated copy failure/);
  assert.equal(fsSync.existsSync(path.join(datasets, 'source_3')), false);
  assert.equal(await fs.readFile(path.join(copied.path, 'image.png'), 'utf8'), 'source image');
  assert.equal(await fs.readFile(path.join(source, 'image.png'), 'utf8'), 'source image');
});

test('dataset deletion rejects traversal and junction ancestors outside the configured root', async t => {
  const { datasets, outside } = await workspace(t);
  const victim = path.join(outside, 'victim');
  await fs.mkdir(victim);
  await fs.writeFile(path.join(victim, 'keep.txt'), 'outside data');
  if (!(await link(t, outside, path.join(datasets, 'link')))) return;
  await assert.rejects(deleteDatasetFolder(datasets, 'link/victim'), /Invalid dataset path/);
  await assert.rejects(deleteDatasetFolder(datasets, '../outside/victim'), /Invalid dataset path/);
  await assert.rejects(deleteDatasetFolder(datasets, datasets), /Invalid dataset path/);
  assert.equal(await fs.readFile(path.join(victim, 'keep.txt'), 'utf8'), 'outside data');
});

test('dataset deletion supports nested directories and removes in-root aliases without deleting their targets', async t => {
  const { datasets, source } = await workspace(t);
  const nested = path.join(source, 'nested');
  await fs.mkdir(nested);
  await fs.writeFile(path.join(source, 'keep.txt'), 'source');
  assert.equal((await deleteDatasetFolder(datasets, 'source/nested')).deleted, true);
  assert.equal((await deleteDatasetFolder(datasets, 'source/nested')).deleted, false);
  if (!(await link(t, source, path.join(datasets, 'alias')))) return;
  assert.equal((await deleteDatasetFolder(datasets, 'alias')).deleted, true);
  assert.equal(await fs.readFile(path.join(source, 'keep.txt'), 'utf8'), 'source');
});

test('caption writes retain existing extensions and create JSON captions through a configured root alias', async t => {
  const { root, datasets, source } = await workspace(t);
  await fs.writeFile(path.join(source, 'image.png'), 'image');
  await fs.writeFile(path.join(source, 'image.caption'), 'old');
  const result = await writePlainDatasetCaption(datasets, path.join(source, 'image.png'), 'updated');
  assert.equal(result.success, true);
  assert.ok(Number.isFinite(Date.parse(result.captioned_at)));
  assert.equal(await fs.readFile(path.join(source, 'image.caption'), 'utf8'), 'updated');
  await writePlainDatasetCaption(datasets, path.join(source, 'image.caption'), 'text document');
  assert.equal(await fs.readFile(path.join(source, 'image.caption'), 'utf8'), 'text document');
  await fs.writeFile(path.join(source, 'new.png'), 'image');
  const alias = path.join(root, 'configured-root');
  if (!(await link(t, datasets, alias))) return;
  await writePlainDatasetCaption(alias, path.join(alias, 'source', 'new.png'), '{"caption":"new"}');
  assert.equal(await fs.readFile(path.join(source, 'new.json'), 'utf8'), '{"caption":"new"}');
  assert.equal(fsSync.existsSync(path.join(source, 'new.txt')), false);
  await assert.rejects(writePlainDatasetCaption(datasets, path.join(source, 'missing.png'), 'caption'), error => error.status === 404);
});

test('caption writes reject external directory aliases without changing external captions', async t => {
  const { datasets, outside } = await workspace(t);
  await fs.writeFile(path.join(outside, 'image.png'), 'outside image');
  await fs.writeFile(path.join(outside, 'image.txt'), 'outside caption');
  const alias = path.join(datasets, 'outside-alias');
  if (!(await link(t, outside, alias))) return;
  await assert.rejects(writePlainDatasetCaption(datasets, path.join(alias, 'image.png'), 'overwrite'), /Invalid image or caption path/);
  assert.equal(await fs.readFile(path.join(outside, 'image.txt'), 'utf8'), 'outside caption');
});

test('caption writes reject external and dangling sidecar symlinks', async t => {
  const { datasets, source, outside } = await workspace(t);
  await fs.writeFile(path.join(source, 'image.png'), 'image');
  const external = path.join(outside, 'caption.txt');
  await fs.writeFile(external, 'outside caption');
  const sidecar = path.join(source, 'image.txt');
  if (!(await link(t, external, sidecar, false))) return;
  await assert.rejects(writePlainDatasetCaption(datasets, path.join(source, 'image.png'), 'overwrite'), /Invalid image or caption path/);
  assert.equal(await fs.readFile(external, 'utf8'), 'outside caption');
  await fs.unlink(sidecar);
  const missing = path.join(outside, 'missing.txt');
  await fs.symlink(missing, sidecar, 'file');
  await assert.rejects(writePlainDatasetCaption(datasets, path.join(source, 'image.png'), 'overwrite'), /Invalid image or caption path/);
  assert.equal(fsSync.existsSync(missing), false);
});

test('caption aliases cannot bypass encrypted-dataset or layered-asset protections', async t => {
  const { datasets, source } = await workspace(t);
  const encrypted = path.join(datasets, 'encrypted');
  const objects = path.join(encrypted, 'objects');
  await fs.mkdir(objects, { recursive: true });
  await fs.writeFile(path.join(encrypted, '.aitk_encrypted_dataset.json'), '{}');
  await fs.writeFile(path.join(objects, 'image.png'), 'ciphertext');
  const alias = path.join(source, 'alias');
  if (!(await link(t, objects, alias))) return;
  await assert.rejects(writePlainDatasetCaption(datasets, path.join(alias, 'image.png'), 'plaintext'), error => error.status === 403);
  assert.equal(fsSync.existsSync(path.join(objects, 'image.txt')), false);
  const layers = path.join(source, '.layers', 'document');
  await fs.mkdir(layers, { recursive: true });
  await fs.writeFile(path.join(layers, 'layer.png'), 'layer');
  const layerAlias = path.join(source, 'layer-alias');
  await fs.symlink(layers, layerAlias, process.platform === 'win32' ? 'junction' : 'dir');
  await assert.rejects(writePlainDatasetCaption(datasets, path.join(layerAlias, 'layer.png'), 'caption'), /Layered documents/);
  assert.equal(fsSync.existsSync(path.join(layers, 'layer.txt')), false);
});
