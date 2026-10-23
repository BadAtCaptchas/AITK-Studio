import assert from 'node:assert/strict';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import * as timers from 'node:timers/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { createRequire } from 'node:module';
import test, { after } from 'node:test';
import ts from 'typescript';

const root = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-copy-recovery-'));
const oldPath = process.env.AITK_SQLITE_PATH, oldProvider = process.env.AITK_DB_PROVIDER;
process.env.AITK_SQLITE_PATH = path.join(root, 'test.sqlite');
process.env.AITK_DB_PROVIDER = 'sqlite';
await promisify(execFile)(process.execPath, ['scripts/prepare-db.mjs'], { env: process.env, timeout: 60000 });
const require = createRequire(import.meta.url);
const { db, disconnectDb } = require('../dist/src/server/db.js');
after(async () => {
  await disconnectDb();
  if (oldPath === undefined) delete process.env.AITK_SQLITE_PATH; else process.env.AITK_SQLITE_PATH = oldPath;
  if (oldProvider === undefined) delete process.env.AITK_DB_PROVIDER; else process.env.AITK_DB_PROVIDER = oldProvider;
  assert.ok(path.resolve(root).startsWith(path.resolve(os.tmpdir()) + path.sep));
  assert.ok(path.basename(root).startsWith('aitk-copy-recovery-'));
  await fsp.rm(root, { recursive: true, force: true });
});

function load(relative, dependencies) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const loaded = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
    return dependencies[name];
  }, loaded, loaded.exports);
  return loaded.exports;
}
const commandInput = require('../dist/src/server/commandInput.js');
function copyModule(overrides = {}) {
  const dependencies = {
    fs, 'fs/promises': fsp, path, crypto, './db': { db }, './commandInput': commandInput,
    './plainDatasetCommand': require('../dist/src/server/plainDatasetCommand.js'),
    './remoteCaptionSecurity': require('../dist/src/server/remoteCaptionSecurity.js'),
    './layeredImages': require('../dist/src/server/layeredImages.js'),
    './publishDatasetDirectory': require('../dist/src/server/publishDatasetDirectory.js'),
    './processLease': load('../src/server/processLease.ts', {
      os, crypto, 'timers/promises': timers, './db': { db },
      './processBirth': { getProcessBirth: async () => 1000, processBirthMatches: (left, right) => left === right },
    }),
    ...overrides,
  };
  if (fs.existsSync(new URL('../src/server/datasetCopyRecovery.ts', import.meta.url)))
    dependencies['./datasetCopyRecovery'] = load('../src/server/datasetCopyRecovery.ts', dependencies);
  return load('../src/server/datasetCopy.ts', dependencies);
}
async function fixture() {
  const datasets = await fsp.mkdtemp(path.join(root, 'datasets-'));
  const source = path.join(datasets, 'source');
  await fsp.mkdir(source);
  await fsp.writeFile(path.join(source, 'image.png'), 'synthetic image');
  await fsp.writeFile(path.join(source, 'image.txt'), 'synthetic caption');
  return { datasets, source, request: { datasetPath: source, sourceDatasetsRoot: datasets,
    destinationDatasetsRoot: datasets, requestedName: 'copy', idempotencyKey: crypto.randomUUID() } };
}

test('a keyed copy recovers the published identity after a lost response and SQLite reconnect', async () => {
  const f = await fixture();
  const first = await copyModule().copyDatasetBetweenRoots(f.request);
  await disconnectDb();
  const replay = await copyModule().copyDatasetBetweenRoots(f.request);
  assert.deepEqual(replay, first);
  assert.equal(await fsp.readFile(path.join(replay.path, 'image.txt'), 'utf8'), 'synthetic caption');
  assert.equal(fs.existsSync(path.join(f.datasets, 'copy_2')), false);
});

test('same-key concurrent copies publish only one complete destination', async () => {
  const f = await fixture();
  const results = await Promise.all([
    copyModule().copyDatasetBetweenRoots(f.request), copyModule().copyDatasetBetweenRoots(f.request),
  ]);
  assert.deepEqual(results[0], results[1]);
  assert.equal(await fsp.readFile(path.join(results[0].path, 'image.png'), 'utf8'), 'synthetic image');
});

test('one copy key cannot be rebound to a changed source, name, or suffix', async () => {
  const f = await fixture();
  const copy = copyModule().copyDatasetBetweenRoots;
  await copy(f.request);
  const other = path.join(f.datasets, 'other');
  await fsp.mkdir(other);
  for (const change of [{ datasetPath: other }, { requestedName: 'another' }, { suffix: 'another' }])
    await assert.rejects(copy({ ...f.request, ...change }), error => error.status === 409 && error.code === 'COPY_KEY_CONFLICT');
});

test('a deleted keyed result is not recreated by retrying its command', async () => {
  const f = await fixture();
  const copy = copyModule().copyDatasetBetweenRoots;
  const result = await copy(f.request);
  assert.ok(path.resolve(result.path).startsWith(path.resolve(f.datasets) + path.sep));
  await fsp.rm(result.path, { recursive: true });
  await assert.rejects(copy(f.request), error => error.status === 410 && error.code === 'COPY_RESULT_MISSING');
  assert.equal(fs.existsSync(result.path), false);
});

test('a completed copy replays after its source is deleted and copying a copy preserves user files', async () => {
  const f = await fixture();
  await fsp.writeFile(path.join(f.source, '.aitk-copy-owner.json'), 'unrelated user metadata');
  const copy = copyModule().copyDatasetBetweenRoots;
  const first = await copy(f.request);
  const second = await copy({ ...f.request, datasetPath: first.path, idempotencyKey: crypto.randomUUID() });
  assert.equal(await fsp.readFile(path.join(second.path, '.aitk-copy-owner.json'), 'utf8'), 'unrelated user metadata');
  assert.ok(path.resolve(f.source).startsWith(path.resolve(f.datasets) + path.sep));
  await fsp.rm(f.source, { recursive: true });
  assert.deepEqual(await copy(f.request), first);
});

test('ready staging recovers across reconnect before atomic publication', async () => {
  const f = await fixture();
  const copy = copyModule({ './publishDatasetDirectory': { publishDatasetDirectory: async () => {
    throw new Error('Synthetic exit before publication');
  } } }).copyDatasetBetweenRoots;
  await assert.rejects(copy(f.request), /Synthetic exit/);
  assert.equal(fs.existsSync(path.join(f.datasets, 'copy')), false);
  await disconnectDb();
  const result = await copyModule().copyDatasetBetweenRoots(f.request);
  assert.equal(result.name, 'copy');
  assert.equal(await fsp.readFile(path.join(result.path, 'image.txt'), 'utf8'), 'synthetic caption');
});

test('a published payload recovers when receipt completion was interrupted, even without its source', async t => {
  const f = await fixture();
  const compareAndSwap = db.runtime.compareAndSwap.bind(db.runtime);
  let lost = false;
  const mock = t.mock.method(db.runtime, 'compareAndSwap', async (key, version, value) => {
    if (key.startsWith('dataset-copy-command:') && value?.state === 'completed' && !lost) {
      lost = true;
      throw new Error('Synthetic exit after publication');
    }
    return compareAndSwap(key, version, value);
  });
  await assert.rejects(copyModule().copyDatasetBetweenRoots(f.request), /Synthetic exit/);
  mock.mock.restore();
  assert.equal(await fsp.readFile(path.join(f.datasets, 'copy', 'image.txt'), 'utf8'), 'synthetic caption');
  assert.ok(path.resolve(f.source).startsWith(path.resolve(f.datasets) + path.sep));
  await fsp.rm(f.source, { recursive: true });
  await disconnectDb();
  const result = await copyModule().copyDatasetBetweenRoots(f.request);
  assert.equal(result.name, 'copy');
  assert.equal(fs.existsSync(path.join(f.datasets, 'copy_2')), false);
});

test('a partial copy is rebuilt in owned staging without publishing incomplete data', async t => {
  const f = await fixture();
  const original = fsp.copyFile.bind(fsp);
  let count = 0;
  const mock = t.mock.method(fsp, 'copyFile', async (...args) => {
    if (++count === 2) throw new Error('Synthetic interrupted copy');
    return original(...args);
  });
  await assert.rejects(copyModule().copyDatasetBetweenRoots(f.request), /Synthetic interrupted/);
  mock.mock.restore();
  assert.equal(fs.existsSync(path.join(f.datasets, 'copy')), false);
  const result = await copyModule().copyDatasetBetweenRoots(f.request);
  assert.deepEqual((await fsp.readdir(result.path)).sort(), ['image.png', 'image.txt']);
  assert.equal((await fsp.readdir(f.datasets)).filter(name => name.startsWith('.aitk-copy-')).length, 0);
});

test('publication cannot replace or adopt an unrelated destination created after reservation', async () => {
  const f = await fixture();
  const publish = require('../dist/src/server/publishDatasetDirectory.js').publishDatasetDirectory;
  const copy = copyModule({ './publishDatasetDirectory': { publishDatasetDirectory: async (source, destination) => {
    await fsp.mkdir(destination);
    await publish(source, destination);
  } } }).copyDatasetBetweenRoots;
  await assert.rejects(copy(f.request));
  assert.deepEqual(await fsp.readdir(path.join(f.datasets, 'copy')), []);
  await assert.rejects(copyModule().copyDatasetBetweenRoots(f.request), error => error.code === 'COPY_DESTINATION_CONFLICT');
  assert.deepEqual(await fsp.readdir(path.join(f.datasets, 'copy')), []);
});

test('unkeyed copies remain independent and different keys safely reserve separate destinations', async () => {
  const f = await fixture();
  const copy = copyModule().copyDatasetBetweenRoots;
  const keyed = await Promise.all([copy(f.request), copy({ ...f.request, idempotencyKey: crypto.randomUUID() })]);
  assert.deepEqual(keyed.map(value => value.name).sort(), ['copy', 'copy_2']);
  const first = await copy({ ...f.request, idempotencyKey: undefined });
  const second = await copy({ ...f.request, idempotencyKey: undefined });
  assert.notEqual(first.path, second.path);
});

test('invalid keys, encrypted sources, and unsafe layer/source links cannot publish copies', async t => {
  const f = await fixture();
  const copy = copyModule().copyDatasetBetweenRoots;
  for (const key of [null, '', 'short', 'unsafe/key', 'a'.repeat(129)])
    await assert.rejects(copy({ ...f.request, idempotencyKey: key }), error => error.status === 400);
  await fsp.writeFile(path.join(f.source, '.aitk_encrypted_dataset.json'), '{}');
  await assert.rejects(copy(f.request), error => error.code === 'ENCRYPTED_DATASET_MIGRATION_REQUIRED');
  await fsp.unlink(path.join(f.source, '.aitk_encrypted_dataset.json'));
  const outside = await fsp.mkdtemp(path.join(root, 'outside-'));
  try { await fsp.symlink(outside, path.join(f.source, '.layers'), process.platform === 'win32' ? 'junction' : 'dir'); }
  catch (error) {
    if (!['EPERM', 'EACCES'].includes(error.code)) throw error;
    t.diagnostic('Filesystem link creation unavailable; encrypted/key checks passed.');
    return;
  }
  await assert.rejects(copy(f.request), /symbolic link|junction/i);
  assert.equal(fs.existsSync(path.join(f.datasets, 'copy')), false);
  assert.deepEqual(await fsp.readdir(outside), []);
});

test('source changes during copying cannot publish a mixed snapshot', async t => {
  const f = await fixture();
  const original = fsp.copyFile.bind(fsp);
  const mock = t.mock.method(fsp, 'copyFile', async (...args) => {
    await original(...args);
    await fsp.writeFile(path.join(f.source, 'added.txt'), 'synthetic new file');
  });
  await assert.rejects(copyModule().copyDatasetBetweenRoots(f.request), /source changed/);
  mock.mock.restore();
  assert.equal(fs.existsSync(path.join(f.datasets, 'copy')), false);
  const result = await copyModule().copyDatasetBetweenRoots(f.request);
  assert.equal(await fsp.readFile(path.join(result.path, 'added.txt'), 'utf8'), 'synthetic new file');
  const receipts = await db.runtime.list('dataset-copy-command:', 1000);
  assert.doesNotMatch(JSON.stringify(receipts), /synthetic new file|synthetic caption|synthetic image|image\.txt/);
});
