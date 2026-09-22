import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { createRequire } from 'node:module';
import sharp from 'sharp';

const rgba = await sharp({
  create: { width: 64, height: 64, channels: 4, background: { r: 30, g: 60, b: 90, alpha: 0.5 } },
})
  .png()
  .toBuffer();

const require = createRequire(import.meta.url);
const { parseLayeredImageManifest, isLayeredImageAssetPath } = require('../dist/src/domain/layeredImages.js');
const {
  listLayeredImages,
  copyLayeredImage,
  saveLayeredCaptions,
  layeredRevision,
  validateLayeredImageAssets,
  findLayeredImageForPath,
} = require('../dist/src/server/layeredImages.js');
const { deletePlainImagePaths } = require('../dist/src/server/imageDelete.js');
const { combineDatasets } = require('../dist/src/server/datasetCombine.js');
const { performPlainDatasetCaptionBulkAction } = require('../dist/src/server/datasetCaptionBulk.js');
const { copyDatasetBetweenRoots } = require('../dist/src/server/datasetCopy.js');
const {
  createDatasetExportArchive,
  extractZipSafely,
  readDatasetExportManifest,
} = require('../dist/src/server/datasetTransfer.js');
const { ENCRYPTED_DATASET_MANIFEST } = require('../dist/src/server/encryptedDatasets.js');
const { readSampleLayers, deleteSampleLayers } = require('../dist/src/server/sampleLayers.js');
const { getSampleLayersUrl } = require('../dist/src/utils/media.js');
const { makeRemoteAssetRef } = require('../dist/src/server/remoteAssets.js');
const { importLayeredDocument } = require('../dist/src/server/layeredImport.js');
const { streamRequestToStagingFile } = require('../dist/src/server/streamedUpload.js');
const { findDatasetItemsRecursively, findDatasetItemsRecursivelyAsync } = require('../dist/src/server/datasetImages.js');

const manifest = (id = 'document123') => ({
  format: 'aitk.layered-image',
  version: 1,
  id,
  composite: 'art.png',
  caption: 'art.txt',
  width: 64,
  height: 64,
  order: 'bottom-to-top',
  layers: [0, 1].map(index => ({
    path: `.layers/${id}/${index}.png`,
    name: `Layer ${index}`,
    caption: `caption ${index}`,
  })),
  source: { format: 'ora', filename: 'art.ora', sha256: 'a'.repeat(64) },
});
async function fixture(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-layer-test-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  return root;
}
async function writeGroup(root, group = manifest()) {
  await fs.mkdir(path.join(root, '.layers', group.id), { recursive: true });
  for (const [name, value] of [
    [group.composite, rgba],
    [group.caption, 'find me'],
    ...group.layers.map(layer => [layer.path, rgba]),
  ])
    await fs.writeFile(path.join(root, name), value);
  await fs.writeFile(path.join(root, '.layers', group.id, 'manifest.json'), JSON.stringify(group));
  return group;
}

test('layer manifests reject traversal, cross-group ownership, nested assets and wrong ordering', () => {
  const group = manifest();
  assert.deepEqual(parseLayeredImageManifest(group), group);
  const manual = { ...group, source: undefined, layers: group.layers.map(({ caption, ...layer }) => layer) };
  assert.equal(parseLayeredImageManifest(manual).layers[0].caption, '');
  assert.equal(parseLayeredImageManifest(manual).source, undefined);
  for (const altered of [
    { ...group, composite: '../art.png' },
    { ...group, composite: '.hidden.png', caption: '.hidden.txt' },
    { ...group, order: 'top-to-bottom' },
    { ...group, width: 9000 },
    { ...group, layers: [{ ...group.layers[0], path: '.layers/another1/0.png' }] },
    { ...group, layers: [{ ...group.layers[0], path: '.layers/document123/nested/0.png' }] },
    { ...group, layers: [group.layers[0], group.layers[0]] },
  ])
    assert.throws(() => parseLayeredImageManifest(altered));
});

test('caption saves preserve layer ownership/order and reject stale revisions', async t => {
  const root = await fixture(t),
    group = await writeGroup(root);
  const manifestPath = path.join(root, '.layers', group.id, 'manifest.json');
  const revision = layeredRevision(await fs.readFile(manifestPath, 'utf8'), 'find me');
  const edits = group.layers.map(layer => ({ name: layer.name + ' edited', caption: 'new caption' }));
  await saveLayeredCaptions(root, group.id, revision, 'new composite caption', edits);
  const [saved] = await listLayeredImages(root);
  assert.deepEqual(
    saved.layers.map(layer => layer.path),
    group.layers.map(layer => layer.path),
  );
  assert.equal(saved.layers[0].caption, 'new caption');
  assert.equal(await fs.readFile(path.join(root, group.caption), 'utf8'), 'new composite caption');
  await assert.rejects(saveLayeredCaptions(root, group.id, revision, 'stale', edits), error => error.status === 409);
});

test('uncommitted import directories stay hidden until the complete manifest is published', async t => {
  const root = await fixture(t),
    group = manifest();
  await fs.mkdir(path.join(root, '.layers', group.id), { recursive: true });
  await fs.writeFile(path.join(root, group.layers[0].path), 'partial');
  assert.deepEqual(await listLayeredImages(root), []);
  await writeGroup(root, group);
  assert.equal((await listLayeredImages(root)).length, 1);
  await fs.unlink(path.join(root, group.layers[0].path));
  await assert.rejects(listLayeredImages(root), { code: 'ENOENT' });
});

test('layer assets must be static RGBA PNGs with the declared full canvas', async t => {
  const root = await fixture(t),
    group = await writeGroup(root);
  await validateLayeredImageAssets(root);
  await assert.rejects(validateLayeredImageAssets(root, [{ ...group, width: 32 }]), /full-canvas/);
  await sharp(rgba).removeAlpha().png().toFile(path.join(root, group.layers[0].path));
  await assert.rejects(validateLayeredImageAssets(root), /RGBA/);
});

test('copy, combine and archive round-trip preserve groups and collision-safe names', async t => {
  const root = await fixture(t);
  const alpha = path.join(root, 'alpha'),
    beta = path.join(root, 'beta');
  await writeGroup(alpha);
  await writeGroup(beta);
  const copy = await copyDatasetBetweenRoots({
    datasetPath: alpha,
    sourceDatasetsRoot: root,
    destinationDatasetsRoot: root,
    requestedName: 'copy',
  });
  assert.equal((await listLayeredImages(copy.path)).length, 1);
  const combined = await combineDatasets(root, { sourceDatasets: ['alpha', 'beta'], outputName: 'combined' });
  const groups = await listLayeredImages(combined.dataset.path);
  assert.equal(groups.length, 2);
  assert.notEqual(groups[0].id, groups[1].id);
  assert.deepEqual(groups.map(group => group.composite).sort(), ['art.png', 'art_2.png']);
  for (const group of groups)
    assert.deepEqual(await fs.readFile(path.join(combined.dataset.path, group.layers[1].path)), rgba);
  const zip = path.join(root, 'dataset.zip'),
    extracted = path.join(root, 'extracted');
  await createDatasetExportArchive('combined', combined.dataset.path, zip);
  await extractZipSafely(zip, extracted);
  await readDatasetExportManifest(extracted);
  assert.equal((await listLayeredImages(path.join(extracted, 'dataset'))).length, 2);
});

test('group copy rejects encrypted and linked destinations, and rolls back collisions', async t => {
  const root = await fixture(t),
    source = path.join(root, 'source'),
    destination = path.join(root, 'dest');
  const group = await writeGroup(source);
  await fs.mkdir(destination);
  await fs.writeFile(path.join(destination, 'art.png'), 'KEEP');
  await assert.rejects(copyLayeredImage(source, group, destination, 'art.png'));
  assert.equal(await fs.readFile(path.join(destination, 'art.png'), 'utf8'), 'KEEP');
  assert.deepEqual(await fs.readdir(path.join(destination, '.layers')), []);
  await fs.writeFile(path.join(destination, ENCRYPTED_DATASET_MANIFEST), '{}');
  await assert.rejects(copyLayeredImage(source, group, destination, 'next.png'), /encrypted/);
  const linked = path.join(root, 'linked');
  await fs.mkdir(linked);
  await fs.symlink(source, path.join(linked, '.layers'), process.platform === 'win32' ? 'junction' : 'dir');
  await assert.rejects(copyLayeredImage(source, group, linked, 'next.png'), /destination/);
});

test('group publication keeps staged and uncommitted root assets out of dataset discovery', async t => {
  for (const supportsLinks of [true, false]) await t.test(`hard links ${supportsLinks}`, async t => {
    const root = await fixture(t), source = path.join(root, 'source'), destination = path.join(root, 'dest');
    const group = await writeGroup(source);
    await fs.mkdir(destination);
    const originalCopy = fs.copyFile, originalLink = fs.link;
    let published = 0;
    const assertHidden = async () => {
      assert.deepEqual(findDatasetItemsRecursively(destination), []);
      assert.deepEqual(await findDatasetItemsRecursivelyAsync(destination), []);
      assert.deepEqual(await listLayeredImages(destination), []);
    };
    t.mock.method(fs, 'copyFile', async (...args) => {
      await originalCopy(...args);
      if (path.basename(args[1]) === '.composite.tmp') {
        await assert.rejects(fs.access(path.join(destination, group.composite)), { code: 'ENOENT' });
        await assertHidden();
      }
      if (!supportsLinks && path.dirname(args[1]) === destination) {
        published += 1;
        await assertHidden();
      }
    });
    t.mock.method(fs, 'link', async (...args) => {
      if (!supportsLinks) throw Object.assign(new Error('hard links unavailable'), { code: 'ENOTSUP' });
      await originalLink(...args);
      published += 1;
      await assertHidden();
    });
    const copied = await copyLayeredImage(source, group, destination, 'art.png');
    assert.equal(published, 2);
    assert.deepEqual(await findDatasetItemsRecursivelyAsync(destination), [path.join(destination, 'art.png')]);
    assert.deepEqual(findDatasetItemsRecursively(destination), [path.join(destination, 'art.png')]);
    assert.equal((await listLayeredImages(destination))[0].id, copied.id);
    assert.deepEqual((await fs.readdir(path.join(destination, '.layers', copied.id))).sort(), ['000.png', '001.png', 'manifest.json']);
  });
});

test('cancellation after root publication removes the complete pending group', async t => {
  const root = await fixture(t), source = path.join(root, 'source'), destination = path.join(root, 'dest');
  const group = await writeGroup(source);
  await fs.mkdir(destination);
  const controller = new AbortController(), originalLink = fs.link;
  t.mock.method(fs, 'link', async (...args) => {
    await originalLink(...args);
    if (args[1] === path.join(destination, group.composite)) controller.abort();
  });
  await assert.rejects(copyLayeredImage(source, group, destination, 'art.png', controller.signal), error => error.status === 499);
  assert.deepEqual(await fs.readdir(destination), ['.layers']);
  assert.deepEqual(await fs.readdir(path.join(destination, '.layers')), []);
});

test('move and delete operate on complete groups; individual assets are protected', async t => {
  const root = await fixture(t),
    source = path.join(root, 'source'),
    group = await writeGroup(source);
  await assert.rejects(
    deletePlainImagePaths([path.join(source, group.layers[0].path)], root, path.join(root, 'training')),
    /composite/,
  );
  await assert.rejects(
    performPlainDatasetCaptionBulkAction(root, {
      datasetName: 'source',
      action: 'move',
      imgPaths: [path.join(source, group.layers[0].path)],
      query: 'find',
    }),
    /composite/,
  );
  const moved = await performPlainDatasetCaptionBulkAction(root, {
    datasetName: 'source',
    action: 'move',
    imgPaths: [path.join(source, group.composite)],
    query: 'find',
  });
  assert.equal(moved.moved, 1);
  assert.deepEqual(await listLayeredImages(source), []);
  const destination = path.join(root, moved.destinationName),
    [target] = await listLayeredImages(destination);
  const deleted = await deletePlainImagePaths(
    [path.join(destination, target.composite)],
    root,
    path.join(root, 'training'),
  );
  assert.equal(deleted.deleted, 1);
  assert.deepEqual(await listLayeredImages(destination), []);
});

test('case variants of the reserved layer directory cannot bypass mutation guards', async t => {
  const root = await fixture(t), source = path.join(root, 'source'), group = await writeGroup(source);
  for (const directory of ['.layers', '.LAYERS', '.Layers']) {
    const asset = path.join(source, directory, group.id, '0.png');
    assert.equal(isLayeredImageAssetPath(asset), true);
    assert.equal(isLayeredImageAssetPath(asset.replaceAll(path.sep, '\\')), true);
    await assert.rejects(deletePlainImagePaths([asset], root, path.join(root, 'training')), /composite/);
    await assert.rejects(performPlainDatasetCaptionBulkAction(root, {
      datasetName: 'source', action: 'remove_words', imgPaths: [asset], query: 'find',
    }), /composite/);
  }
  assert.equal(isLayeredImageAssetPath(path.join(source, 'art.layers.png')), false);
  assert.equal((await listLayeredImages(source)).length, 1);
});

test('Windows case-variant composite requests retain complete group ownership', { skip: process.platform !== 'win32' }, async t => {
  const root = await fixture(t), source = path.join(root, 'source'), group = await writeGroup(source);
  const requested = path.join(root, 'SOURCE', 'ART.PNG');
  assert.equal((await findLayeredImageForPath(requested, root)).manifest.id, group.id);
  const result = await deletePlainImagePaths([requested], root, path.join(root, 'training'));
  assert.equal(result.deleted, 1);
  assert.equal(result.failed, 0);
  assert.deepEqual(await listLayeredImages(source), []);
  for (const asset of [group.composite, group.caption, ...group.layers.map(layer => layer.path)])
    await assert.rejects(fs.access(path.join(source, asset)), { code: 'ENOENT' });
});

test('case-sensitive filesystems keep distinct composite spellings separate', { skip: process.platform === 'win32' }, async t => {
  const root = await fixture(t), source = path.join(root, 'source'), group = await writeGroup(source);
  const separate = path.join(source, 'ART.PNG');
  try { await fs.writeFile(separate, rgba, { flag: 'wx' }); }
  catch (error) { if (error.code === 'EEXIST') return t.skip('Filesystem is case insensitive'); throw error; }
  assert.equal(await findLayeredImageForPath(separate, root), null);
  const result = await deletePlainImagePaths([separate], root, path.join(root, 'training'));
  assert.equal(result.deleted, 1);
  assert.equal((await listLayeredImages(source))[0].id, group.id);
});

test('dataset junctions cannot redirect layered deletion outside the configured root', async t => {
  const root = await fixture(t),
    datasets = path.join(root, 'datasets'),
    outside = path.join(root, 'outside');
  const group = await writeGroup(outside);
  await fs.mkdir(datasets);
  const linked = path.join(datasets, 'linked');
  await fs.symlink(outside, linked, process.platform === 'win32' ? 'junction' : 'dir');
  const result = await deletePlainImagePaths(
    [path.join(linked, group.composite)],
    datasets,
    path.join(root, 'training'),
  );
  assert.equal(result.failed, 1);
  assert.deepEqual(await fs.readFile(path.join(outside, group.composite)), rgba);
});

test('sample layers preserve order, reject traversal, and clean up with their composite', async t => {
  const root = await fixture(t),
    group = await writeGroup(root);
  const composite = path.join(root, group.composite),
    sidecar = composite + '.layers.json';
  await fs.writeFile(
    sidecar,
    JSON.stringify({ version: 1, order: 'bottom-to-top', layers: group.layers.map(layer => layer.path) }),
  );
  assert.deepEqual(
    (await readSampleLayers(composite)).layers,
    group.layers.map(layer => layer.path),
  );
  await fs.writeFile(sidecar, JSON.stringify({ version: 1, order: 'bottom-to-top', layers: ['../outside.png'] }));
  await assert.rejects(readSampleLayers(composite), /path/);
  await fs.writeFile(
    sidecar,
    JSON.stringify({ version: 1, order: 'bottom-to-top', layers: group.layers.map(layer => layer.path) }),
  );
  await deleteSampleLayers(composite);
  await assert.rejects(fs.stat(sidecar), { code: 'ENOENT' });
  await assert.rejects(fs.stat(path.join(root, group.layers[0].path)), { code: 'ENOENT' });
  assert.deepEqual(await fs.readFile(composite), rgba);
});

test('sample layer URLs retain central job identity for remote samples', () => {
  assert.equal(getSampleLayersUrl('/api/jobs/job1/samples/art.png'), '/api/jobs/job1/sample-layers?sample=art.png');
  assert.equal(
    getSampleLayersUrl(makeRemoteAssetRef('central', 'img', '/api/jobs/worker-job/samples/art.png')),
    '/api/jobs/central/sample-layers?sample=art.png',
  );
  assert.equal(getSampleLayersUrl('/api/jobs/job1/samples/..%2Fother.png'), null);
  assert.equal(getSampleLayersUrl('/api/jobs/job1/samples/art.jpg'), null);
});

test('Windows sample casing preserves self identity and detects aliased shared layers', { skip: process.platform !== 'win32' }, async t => {
  const root = await fixture(t), group = await writeGroup(root);
  const sidecar = path.join(root, 'art.png.layers.json');
  await fs.writeFile(sidecar, JSON.stringify({ version: 1, order: 'bottom-to-top', layers: group.layers.map(layer => layer.path) }));
  const uppercase = path.join(root, 'ART.PNG');
  assert.equal((await readSampleLayers(uppercase)).paths.length, 2);
  const other = path.join(root, 'other.png.LAYERS.JSON');
  await fs.writeFile(other, JSON.stringify({
    version: 1, order: 'bottom-to-top', layers: group.layers.map(layer => layer.path.replace(group.id, group.id.toUpperCase())),
  }));
  await assert.rejects(deleteSampleLayers(uppercase), /shared/);
  await fs.unlink(other);
  await deleteSampleLayers(uppercase);
  await assert.rejects(fs.access(sidecar), { code: 'ENOENT' });
  for (const layer of group.layers) await assert.rejects(fs.access(path.join(root, layer.path)), { code: 'ENOENT' });
});

test('cancelled imports never start a converter and unsafe document names are rejected', async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    importLayeredDocument('absent', 'absent', 'art.ora', controller.signal),
    error => error.status === 499,
  );
  for (const name of ['../art.psd', 'folder/art.ora', 'art.exe'])
    await assert.rejects(importLayeredDocument('absent', 'absent', name), /PSD|OpenRaster/);
});

test('cancelled and timed-out converter work is isolated and removed before publication', async t => {
  const root = await fixture(t);
  for (const mode of ['cancel', 'timeout']) {
    const controller = new AbortController();
    let staging;
    await assert.rejects(
      importLayeredDocument('input.ora', root, 'art.ora', controller.signal, async (_input, output) => {
        staging = output;
        assert.notEqual(output, root);
        const group = await writeGroup(output);
        if (mode === 'timeout') throw Object.assign(new Error('Converter timed out'), { status: 408 });
        controller.abort();
        return { id: group.id, warnings: [] };
      }),
      error => (mode === 'cancel' ? error.status === 499 : error.status === 408),
    );
    assert.deepEqual(await fs.readdir(root), []);
    await assert.rejects(fs.stat(staging), { code: 'ENOENT' });
  }
});

test('supervised import publishes a validated group only after conversion succeeds', async t => {
  const root = await fixture(t);
  let staging;
  const result = await importLayeredDocument('input.ora', root, 'art.ora', undefined, async (_input, output) => {
    staging = output;
    const group = await writeGroup(output);
    assert.deepEqual(await fs.readdir(root), []);
    return { id: group.id, warnings: ['flattened group'] };
  });
  const [published] = await listLayeredImages(root);
  assert.equal(published.id, result.id);
  assert.deepEqual(result.warnings, ['flattened group']);
  await assert.rejects(fs.stat(staging), { code: 'ENOENT' });
});

test('only one converter can run at a time', async t => {
  const root = await fixture(t);
  let release, announce;
  const started = new Promise(resolve => {
    announce = resolve;
  });
  const gate = new Promise(resolve => {
    release = resolve;
  });
  const first = importLayeredDocument('input.ora', root, 'art.ora', undefined, async (_input, output) => {
    announce();
    await gate;
    const group = await writeGroup(output);
    return { id: group.id };
  });
  await started;
  try {
    await assert.rejects(importLayeredDocument('input.ora', root, 'art.ora'), error => error.status === 429);
  } finally {
    release();
    await first;
  }
});

test('cancelled or oversized streamed document uploads remove staging files', async t => {
  const root = await fixture(t);
  const controller = new AbortController();
  controller.abort();
  const cancelled = new Request('http://localhost/import', {
    method: 'POST',
    body: 'partial',
    signal: controller.signal,
  });
  await assert.rejects(streamRequestToStagingFile(cancelled, root, { maxBytes: 128, prefix: 'layered' }));
  const oversized = new Request('http://localhost/import', { method: 'POST', body: 'too large' });
  await assert.rejects(streamRequestToStagingFile(oversized, root, { maxBytes: 2, prefix: 'layered' }));
  assert.deepEqual(await fs.readdir(root), []);
});

test('archive import rejects a complete-looking group that points outside its own assets', async t => {
  const root = await fixture(t),
    source = path.join(root, 'source'),
    group = await writeGroup(source);
  const zip = path.join(root, 'export.zip'),
    extract = path.join(root, 'extract');
  await createDatasetExportArchive('source', source, zip);
  await extractZipSafely(zip, extract);
  const filename = path.join(extract, 'dataset', '.layers', group.id, 'manifest.json');
  await fs.writeFile(filename, JSON.stringify({ ...group, layers: [{ ...group.layers[0], path: '../outside.png' }] }));
  await assert.rejects(readDatasetExportManifest(extract), /target/);
});
