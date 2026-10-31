import assert from 'node:assert/strict';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import zlib from 'node:zlib';
import util from 'node:util';
import ts from 'typescript';
import test from 'node:test';

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
