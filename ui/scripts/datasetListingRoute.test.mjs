import assert from 'node:assert/strict';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';
import util from 'node:util';
import ts from 'typescript';
import test from 'node:test';
import { createRequire } from 'node:module';
import { loadDatasetListing } from './datasetListingFixture.mjs';
const require = createRequire(import.meta.url);

function route(overrides = {}) {
  const code = ts.transpileModule(fs.readFileSync(new URL('../src/app/api/datasets/listImages/route.ts', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const dependencies = {
    fs, zlib, util,
    '@/server/commandInput': { readJsonCommand: request => request.json(), withCommandBoundary: value => value },
    '@/utils/obsoleteWorkspaceGuard': { assertGlobalPayload: value => value },
    '@/server/encryptedDatasets': { isEncryptedDatasetFolder: () => false, resolveDatasetFolder: path.join },
    '@/server/remoteClient': { isLocalWorker: value => value === 'local' },
    '@/server/remoteDatasetAssetAccess': { makeSignedRemoteDatasetAssetRef: (worker, type, filename) => `signed:${worker}:${type}:${filename}` },
    '@/server/datasetListing': { listDatasetImages: () => assert.fail('Unexpected plain listing') },
    '@/server/remoteCaptionSecurity': { resolveDatasetDirectoryInsideRoot: () => assert.fail('Unexpected local resolution') },
    '@/server/datasetScope': { DatasetScopeError: class extends Error {}, resolveDatasetScope: () => assert.fail('Unexpected local scope') },
    ...overrides,
  };
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency: ${name}`);
    return dependencies[name];
  }, module, module.exports);
  return body => module.exports.POST(new Request('http://localhost/api/datasets/listImages', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  }));
}

test('legacy remote workers may ignore pagination while signed operational references remain unchanged', async () => {
  let forwarded;
  const post = route({ '@/server/remoteClient': {
    isLocalWorker: () => false, getRemoteWorker: async () => ({ id: 'remote' }),
    remoteJson: async (_worker, _url, options) => {
      forwarded = JSON.parse(options.body);
      return { root: '/remote/data/', images: [{ img_path: 'a.png', size_bytes: 12 }, { img_path: 'b.png' }] };
    },
  } });
  const response = await post({ datasetName: 'images', worker_id: 'remote', compact: true, limit: 1, offset: 0 });
  assert.deepEqual(forwarded, { datasetName: 'images', compact: true, limit: 1, offset: 0 });
  const result = await response.json();
  assert.equal(result.images.length, 2); // Older worker contract is not silently truncated.
  assert.equal(result.images[0].img_path, 'signed:remote:img:/remote/data/a.png');
  assert.equal(result.images[0].size_bytes, 12);
  assert.equal(result.root, undefined);
});

test('invalid pagination rejects before worker or filesystem work', async () => {
  const post = route();
  for (const body of [{ limit: 0 }, { limit: 1001 }, { limit: '10' }, { offset: -1 }, { offset: 0.5 }]) {
    assert.equal((await post({ datasetName: 'images', ...body })).status, 400);
  }
});

test('encrypted manifests bypass cached plain listings after canonical scope validation', async t => {
  const root = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-listing-route-'));
  const folder = path.join(root, 'images');
  await fsp.mkdir(folder);
  t.after(() => fsp.rm(root, { recursive: true, force: true }));
  let canonicalChecked = false;
  const manifest = { encrypted: 'opaque fixture' };
  const post = route({
    '@/server/encryptedDatasets': { resolveDatasetFolder: path.join, isEncryptedDatasetFolder: () => { assert.equal(canonicalChecked, true); return true; }, readEncryptedManifest: async () => manifest },
    '@/server/remoteCaptionSecurity': { resolveDatasetDirectoryInsideRoot: async (requested, scope) => { assert.equal(requested, folder); assert.equal(scope, root); canonicalChecked = true; return folder; } },
    '@/server/datasetScope': { DatasetScopeError: class extends Error {}, resolveDatasetScope: async () => ({ datasetsRoot: root }) },
  });
  assert.deepEqual(await (await post({ datasetName: 'images', limit: 10 })).json(), { encrypted: true, manifest, images: [] });
});


test('canonical listing reads retain configured operational paths for caption and layered consumers', async t => {
  const root = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-listing-alias-'));
  t.after(() => fsp.rm(root, { recursive: true, force: true }));
  const canonicalRoot = path.join(root, 'physical-datasets');
  const configuredRoot = path.join(root, 'configured-datasets');
  const folder = path.join(canonicalRoot, 'images');
  await fsp.mkdir(path.join(folder, '.layers', 'document123'), { recursive: true });
  try { await fsp.symlink(canonicalRoot, configuredRoot, process.platform === 'win32' ? 'junction' : 'dir'); }
  catch (error) { if (error.code === 'EPERM') { t.skip('Host does not permit directory aliases'); return; } throw error; }
  const group = {
    format: 'aitk.layered-image', version: 1, id: 'document123', composite: 'art.png', caption: 'art.txt',
    width: 64, height: 64, order: 'bottom-to-top',
    layers: [{ path: '.layers/document123/0.png', name: 'one', caption: '' }],
  };
  await fsp.writeFile(path.join(folder, 'art.png'), 'image');
  await fsp.writeFile(path.join(folder, 'art.txt'), 'caption survives configured path alias');
  await fsp.writeFile(path.join(folder, group.layers[0].path), 'layer');
  await fsp.writeFile(path.join(folder, '.layers', group.id, 'manifest.json'), JSON.stringify(group));
  const listing = loadDatasetListing();
  t.after(() => listing.clearDatasetListingCache());
  const scope = { DatasetScopeError: class extends Error {}, resolveDatasetScope: async () => ({ datasetsRoot: configuredRoot }) };
  const post = route({
    '@/server/datasetScope': scope,
    '@/server/datasetListing': listing,
    '@/server/remoteCaptionSecurity': require('../dist/src/server/remoteCaptionSecurity.js'),
  });
  // Exercise the real caption handler's configured-root containment and sidecar read.
  const captionCode = ts.transpileModule(fs.readFileSync(new URL('../src/app/api/caption/get/route.ts', import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const captionDependencies = {
    path,
    '@/server/commandInput': { readJsonCommand: request => request.json(), withCommandBoundary: value => value },
    '@/utils/obsoleteWorkspaceGuard': { assertGlobalPayload: value => value },
    '@/server/encryptedDatasets': require('../dist/src/server/encryptedDatasets.js'),
    '@/server/remoteClient': {},
    '@/server/captionFiles': require('../dist/src/server/captionFiles.js'),
    '@/utils/remoteDatasetRefs': require('../dist/src/utils/remoteDatasetRefs.js'),
    '@/utils/captionQuality': require('../dist/src/utils/captionQuality.js'),
    '@/server/datasetScope': scope,
  };
  const captionModule = { exports: {} };
  new Function('require', 'module', 'exports', captionCode)(name => {
    assert.ok(Object.hasOwn(captionDependencies, name), `Unexpected caption dependency ${name}`);
    return captionDependencies[name];
  }, captionModule, captionModule.exports);
  for (const compact of [true, false]) {
    const response = await post({ datasetName: 'images', compact, responseRoot: '/ignored-client-input' });
    assert.equal(response.status, 200);
    const result = await response.json();
    assert.equal(result.images[0].layer_count, 1);
    const operationalPath = compact ? result.root + result.images[0].img_path : result.images[0].img_path;
    assert.equal(operationalPath, path.join(configuredRoot, 'images', 'art.png'));
    assert.notEqual(operationalPath, await fsp.realpath(operationalPath));
    const caption = await captionModule.exports.POST(new Request('http://localhost/api/caption/get', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ imgPath: operationalPath }),
    }));
    assert.equal(caption.status, 200);
    assert.equal(await caption.text(), 'caption survives configured path alias');
  }
  // A dataset alias escaping the configured root still fails canonical validation.
  const outside = path.join(root, 'outside');
  await fsp.mkdir(outside);
  await fsp.symlink(outside, path.join(canonicalRoot, 'escape'), process.platform === 'win32' ? 'junction' : 'dir');
  const escaped = await post({ datasetName: 'escape', compact: true });
  assert.equal(escaped.status, 500);
  assert.equal((await escaped.json()).images, undefined);
});
