import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import test from 'node:test';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const captions = require('../dist/src/server/randomPromptCaptions.js');
const commandInput = require('../dist/src/server/commandInput.js');
const scopeModule = require('../dist/src/server/datasetScope.js');
const encrypted = require('../dist/src/server/encryptedDatasets.js');
const refs = require('../dist/src/utils/remoteDatasetRefs.js');
const obsolete = require('../dist/src/utils/obsoleteWorkspaceGuard.js');
const { RemoteClientError } = require('../dist/src/server/remoteClient.js');

function loadSource(relative, dependencies) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const loaded = { exports: {} };
  new Function('require', 'module', 'exports', code)(
    name => {
      assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency: ${name}`);
      return dependencies[name];
    },
    loaded,
    loaded.exports,
  );
  return loaded.exports;
}

function fixture(t, remoteResult) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'random-prompt-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const calls = [];
  const scope = { ...scopeModule, resolveDatasetScope: async () => ({ datasetsRoot: root, trainingRoot: root }) };
  const remote = loadSource('../src/server/remoteRandomPrompt.ts', {
    './commandInput': commandInput,
    './datasetScope': scope,
    './randomPromptCaptions': captions,
    '../utils/remoteDatasetRefs': refs,
    './remoteClient': {
      RemoteClientError,
      getRemoteWorker: async id => ({ id }),
      withoutRemoteRedirects: init => ({ ...init, redirect: 'manual' }),
      remoteJson: async (worker, route, init) => {
        calls.push({ worker, route, ...init, body: JSON.parse(init.body) });
        if (remoteResult instanceof Error) throw remoteResult;
        return typeof remoteResult === 'function' ? remoteResult(worker) : remoteResult;
      },
    },
  });
  const plain = loadSource('../src/server/plainDatasetCommand.ts', {
    'fs/promises': fs.promises,
    path,
    './commandInput': commandInput,
    './encryptedDatasets': encrypted,
    './datasetScope': scope,
    './datasetDecryption': {
      DATASET_MIGRATION_MESSAGE: 'Decrypt legacy datasets first',
      DATASET_MIGRATION_REQUIRED: 'DATASET_MIGRATION_REQUIRED',
    },
  });
  const { POST } = loadSource('../src/app/api/datasets/randomPrompt/route.ts', {
    '@/server/plainDatasetCommand': plain,
    '@/server/commandInput': commandInput,
    '@/utils/obsoleteWorkspaceGuard': obsolete,
    'next/server': require('next/server'),
    fs,
    path,
    crypto: require('node:crypto'),
    '@/server/datasetScope': scope,
    '@/server/randomPromptCaptions': captions,
    '@/server/remoteRandomPrompt': remote,
    '@/server/encryptedDatasets': encrypted,
  });
  return {
    root,
    calls,
    post: datasets =>
      POST(
        new Request('http://localhost/api/datasets/randomPrompt', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ datasets }),
        }),
      ),
  };
}

const remoteSample = {
  prompt: 'remote caption',
  source: 'caption',
  datasetPath: '/worker/datasets/cats',
  mediaPath: '/worker/datasets/cats/one.png',
  captionPath: '/worker/datasets/cats/one.txt',
  candidateCount: 9,
  scannedMediaCount: 10,
  skippedDatasets: [],
};

test('remote-only selection requests captions on the owning worker with caption settings', async t => {
  const state = fixture(t, remoteSample);
  const folderPath = refs.makeRemoteDatasetRef('worker-1', 'cats & dogs');
  const response = await state.post([{ folderPath, captionExt: 'caption', defaultCaption: 'fallback' }]);
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { ...remoteSample, datasetPath: folderPath });
  assert.deepEqual(state.calls[0], {
    worker: { id: 'worker-1' },
    route: '/api/datasets/randomPrompt',
    method: 'POST',
    redirect: 'manual',
    body: { datasets: [{ folderPath: 'cats & dogs', captionExt: 'caption', defaultCaption: 'fallback' }] },
  });
});

test('mixed local and remote datasets retain candidate weighting and aggregate counts', async t => {
  const state = fixture(t, remoteSample);
  fs.mkdirSync(path.join(state.root, 'local'));
  fs.writeFileSync(path.join(state.root, 'local', 'one.png'), 'image');
  fs.writeFileSync(path.join(state.root, 'local', 'one.txt'), 'local caption');
  const folderPath = refs.makeRemoteDatasetRef('worker-1', 'cats');
  t.mock.method(Math, 'random', () => 0.8);
  const response = await state.post([{ folderPath: 'local' }, { folderPath }]);
  const result = await response.json();
  assert.equal(response.status, 200);
  assert.equal(result.prompt, 'remote caption');
  assert.equal(result.candidateCount, 10);
  assert.equal(result.scannedMediaCount, 11);
});

test('datasets on multiple workers all contribute captions', async t => {
  const state = fixture(t, worker => ({ ...remoteSample, prompt: worker.id }));
  const response = await state.post(
    ['worker-1', 'worker-2'].map(id => ({ folderPath: refs.makeRemoteDatasetRef(id, 'cats') })),
  );
  const result = await response.json();
  assert.equal(response.status, 200);
  assert.deepEqual(
    state.calls.map(call => call.worker.id),
    ['worker-1', 'worker-2'],
  );
  assert.equal(result.candidateCount, 18);
});

test('empty remote captions do not prevent a local default caption', async t => {
  const state = fixture(
    t,
    new RemoteClientError('empty', 404, JSON.stringify({ scannedMediaCount: 2, skippedDatasets: [] })),
  );
  fs.mkdirSync(path.join(state.root, 'local'));
  fs.writeFileSync(path.join(state.root, 'local', 'one.png'), 'image');
  const response = await state.post([
    { folderPath: refs.makeRemoteDatasetRef('worker-1', 'cats') },
    { folderPath: 'local', defaultCaption: 'fallback' },
  ]);
  const result = await response.json();
  assert.equal(response.status, 200);
  assert.equal(result.prompt, 'fallback');
  assert.equal(result.source, 'default_caption');
  assert.equal(result.scannedMediaCount, 3);
});

test('remote failures, malformed responses and encrypted datasets fail clearly', async t => {
  t.mock.method(console, 'error', () => {});
  for (const [remoteResult, status] of [
    [new RemoteClientError('offline', 503), 502],
    [new RemoteClientError('missing route', 404, '<html>Not found</html>'), 502],
    [new RemoteClientError('legacy', 410), 410],
    [{ ...remoteSample, candidateCount: '9' }, 502],
    [{ ...remoteSample, prompt: 17 }, 502],
    [null, 502],
  ]) {
    const state = fixture(t, remoteResult);
    const response = await state.post([{ folderPath: refs.makeRemoteDatasetRef('worker-1', 'cats') }]);
    assert.equal(response.status, status);
  }
});

test('invalid selections cannot forward traversal or nested remote references', async t => {
  t.mock.method(console, 'error', () => {});
  const state = fixture(t, remoteSample);
  for (const folderPath of [
    refs.makeRemoteDatasetRef('worker-1', '../other'),
    refs.makeRemoteDatasetRef('worker-1', 'C:\\private'),
    refs.makeRemoteDatasetRef('worker-1', refs.makeRemoteDatasetRef('worker-2', 'cats')),
    'aitk-dataset://remote/broken',
  ])
    assert.equal((await state.post([{ folderPath }])).status, 400);
  assert.equal((await state.post([null])).status, 400);
  assert.equal(state.calls.length, 0);
});

test('the job form includes remote selections and still excludes encrypted datasets', () => {
  const source = ts.createSourceFile(
    'SimpleJob.tsx',
    fs.readFileSync(new URL('../src/app/jobs/new/SimpleJob.tsx', import.meta.url), 'utf8'),
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX,
  );
  const names = new Set(['randomPromptDatasets', 'accessibleRandomPromptDatasets', 'canImportRandomPrompt']);
  const declarations = [];
  function visit(node) {
    if (ts.isVariableDeclaration(node) && names.has(node.name.getText(source)))
      declarations.push(`const ${node.getText(source)};`);
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.equal(declarations.length, 3);
  const code = ts.transpileModule(declarations.join('\n'), {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const folderPath = refs.makeRemoteDatasetRef('worker-1', 'cats');
  const evaluate = new Function(
    'useMemo',
    'jobConfig',
    'datasetOptions',
    'defaultDatasetConfig',
    'encryptedKeyRefreshKey',
    'getRememberedEncryptedDatasetKey',
    'parseRemoteDatasetRef',
    `${code}\nreturn { accessibleRandomPromptDatasets, canImportRandomPrompt };`,
  );
  for (const encrypted of [false, true]) {
    const result = evaluate(
      fn => fn(),
      { config: { process: [{ datasets: [{ folder_path: folderPath }] }] } },
      [{ value: folderPath, source: 'remote', encrypted }],
      { folder_path: '/placeholder' },
      0,
      () => null,
      refs.parseRemoteDatasetRef,
    );
    assert.equal(result.canImportRandomPrompt, !encrypted);
    assert.equal(result.accessibleRandomPromptDatasets.length, encrypted ? 0 : 1);
  }
});
