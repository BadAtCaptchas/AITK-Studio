import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import {
  collectTrainingExportFiles, findLatestCheckpoint, renameImportedTrainingFiles,
  refreshImportedLatestCheckpoint, shouldIncludeTrainingExportPath,
} from '../dist/src/server/trainingJobTransfer.js';
import { createRemoteTrainingJobBundle } from '../dist/src/server/trainingJobBundle.js';
import { extractZipSafely } from '../dist/src/server/safeArchive.js';
const require = createRequire(import.meta.url);

async function fixture(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-checkpoint-export-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  return root;
}
async function generation(root, step, members, primary = Object.keys(members)[0], extra = {}) {
  const relative = `.aitk-checkpoints/${String(step).padStart(9, '0')}-test`;
  const directory = path.join(root, relative);
  const files = [];
  for (const [name, content] of Object.entries(members)) {
    await fs.mkdir(path.dirname(path.join(directory, name)), { recursive: true });
    await fs.writeFile(path.join(directory, name), content);
    files.push({ path: name, size: Buffer.byteLength(content), sha256: createHash('sha256').update(content).digest('hex') });
  }
  await fs.writeFile(path.join(directory, 'checkpoint.json'), JSON.stringify({
    version: 1, step, created_ns: 1_791_234_567_890_000_000 + step * 1000,
    primary, files, model_files: files.map(file => file.path).filter(name => name !== 'optimizer.pt'),
    optimizer: 'optimizer.pt' in members ? 'optimizer.pt' : null, ...extra,
  }));
  return relative;
}

test('latest exports one complete generation, both WAN halves and its optimizer; aliases use canonical bytes', async t => {
  const root = await fixture(t);
  await generation(root, 10, { 'job_000010.safetensors': 'old', 'optimizer.pt': 'old optimizer' });
  const latest = await generation(root, 20, {
    'job_000020_high_noise.safetensors': 'high', 'job_000020_low_noise.safetensors': 'low', 'optimizer.pt': 'paired',
  });
  await fs.writeFile(path.join(root, 'optimizer.pt'), 'unrelated');
  await fs.writeFile(path.join(root, 'job_000010.safetensors'), 'latest mtime but stale');
  await fs.mkdir(path.join(root, '.aitk-checkpoints/.tmp-writing'), { recursive: true });
  await fs.writeFile(path.join(root, '.aitk-checkpoints/.tmp-writing/partial.safetensors'), 'partial');
  await fs.mkdir(path.join(root, '.checkpoint-export-inflight'));
  await fs.writeFile(path.join(root, '.checkpoint-export-inflight/partial'), 'partial');
  await fs.writeFile(path.join(root, 'log.txt'), 'log');
  const result = await collectTrainingExportFiles(root, 30, 'latest');
  assert.equal(result.step, 20);
  assert.equal(result.relativePath, `${latest}/job_000020_high_noise.safetensors`);
  assert.equal(result.optimizerIncluded, true);
  assert.deepEqual(result.files.map(file => file.relativePath).sort(), [
    `${latest}/checkpoint.json`, `${latest}/job_000020_high_noise.safetensors`,
    `${latest}/job_000020_low_noise.safetensors`, `${latest}/optimizer.pt`,
    'job_000020_high_noise.safetensors', 'job_000020_low_noise.safetensors', 'optimizer.pt', 'log.txt',
  ].sort());
  const optimizer = result.files.find(file => file.relativePath === 'optimizer.pt');
  assert.equal(await fs.readFile(optimizer.sourcePath, 'utf8'), 'paired');
  assert.deepEqual(await findLatestCheckpoint(root, 30), { relativePath: result.relativePath, step: 20 });
});

test('all exports whole verified generations, full-model primary directories and only latest portable optimizer', async t => {
  const root = await fixture(t);
  const old = await generation(root, 10, { 'job_000010.safetensors': 'old', 'optimizer.pt': 'old optimizer' });
  const latest = await generation(root, 20, {
    'job_000020/transformer/model.safetensors': 'weights', 'job_000020/model_index.json': '{}', 'optimizer.pt': 'new optimizer',
  }, 'job_000020');
  const result = await collectTrainingExportFiles(root, 20, 'all');
  assert.equal(result.relativePath, `${latest}/job_000020`);
  assert.ok(result.files.some(file => file.relativePath === `${old}/optimizer.pt`));
  assert.ok(result.files.some(file => file.relativePath === `${latest}/optimizer.pt`));
  assert.ok(result.files.some(file => file.relativePath === 'job_000020/transformer/model.safetensors'));
  assert.equal(result.files.filter(file => file.relativePath === 'optimizer.pt').length, 1);
  assert.equal(await fs.readFile(result.files.find(file => file.relativePath === 'optimizer.pt').sourcePath, 'utf8'), 'new optimizer');
});

test('same-size corruption falls back as a whole generation and all-invalid committed exports fail closed', async t => {
  const root = await fixture(t);
  const old = await generation(root, 10, { 'job_000010.safetensors': 'old', 'optimizer.pt': 'paired' });
  const corrupt = await generation(root, 20, { 'job_000020.safetensors': 'new', 'optimizer.pt': 'wrong' });
  await fs.writeFile(path.join(root, corrupt, 'job_000020.safetensors'), 'bad');
  const result = await collectTrainingExportFiles(root, 20, 'latest');
  assert.equal(result.step, 10);
  assert.ok(result.warnings.some(warning => warning.includes('corrupt')));
  assert.ok(result.files.every(file => !file.relativePath.includes('000020')));
  assert.equal(await fs.readFile(result.files.find(file => file.relativePath === 'optimizer.pt').sourcePath, 'utf8'), 'paired');
  await fs.writeFile(path.join(root, old, 'job_000010.safetensors'), 'bad');
  await fs.writeFile(path.join(root, 'job_000030.safetensors'), 'portable must not rescue failed commit');
  await assert.rejects(collectTrainingExportFiles(root, 30, 'latest'), /No valid committed checkpoint/);
});

test('generation validation rejects traversal, unsupported versions, unlisted members, and linked roots', async t => {
  for (const [label, extra] of [
    ['version', { version: 2 }], ['traversal', { primary: '../outside' }], ['optimizer', { optimizer: null }],
  ]) {
    const root = await fixture(t);
    await generation(root, 10, { 'job.safetensors': 'weights', 'optimizer.pt': 'paired' }, 'job.safetensors', extra);
    await assert.rejects(collectTrainingExportFiles(root, 10, 'all'), /No valid committed checkpoint/, label);
  }
  const root = await fixture(t);
  const relative = await generation(root, 10, { 'job.safetensors': 'weights' });
  await fs.writeFile(path.join(root, relative, 'unlisted'), 'unlisted');
  await assert.rejects(collectTrainingExportFiles(root, 10, 'all'), /No valid committed checkpoint/);
  const linked = await fixture(t);
  await fs.symlink(path.join(root, '.aitk-checkpoints'), path.join(linked, '.aitk-checkpoints'), process.platform === 'win32' ? 'junction' : 'dir');
  await assert.rejects(collectTrainingExportFiles(linked, 10, 'all'), /cannot be a link/);
});

test('import identity preserves generation hashes and source prefix; refresh rejects traversal', async t => {
  const root = await fixture(t);
  const relative = await generation(root, 10, { 'source_000010.safetensors': 'weights', 'optimizer.pt': 'paired' });
  const marker = await fs.readFile(path.join(root, relative, 'checkpoint.json'), 'utf8');
  await fs.writeFile(path.join(root, '.checkpoint_names.json'), JSON.stringify(['older']));
  await fs.writeFile(path.join(root, 'source_000010.safetensors'), 'weights');
  await renameImportedTrainingFiles(root, 'source', 'target');
  await refreshImportedLatestCheckpoint(root, `training/${relative}/source_000010.safetensors`, 'source', 'target');
  assert.deepEqual(JSON.parse(await fs.readFile(path.join(root, '.checkpoint_names.json'), 'utf8')), ['older', 'source', 'target']);
  assert.equal(await fs.readFile(path.join(root, relative, 'checkpoint.json'), 'utf8'), marker);
  assert.equal((await collectTrainingExportFiles(root, 10, 'latest')).step, 10);
  assert.equal(await fs.readFile(path.join(root, 'target_000010.safetensors'), 'utf8'), 'weights');
  await assert.rejects(refreshImportedLatestCheckpoint(root, 'training/../../outside', 'source', 'target'), /escapes/);
});

test('legacy split checkpoints are selected by training step and both halves are exported', async t => {
  const root = await fixture(t);
  for (const name of ['job_000020_high_noise.safetensors', 'job_000020_low_noise.safetensors', 'job_000010.safetensors'])
    await fs.writeFile(path.join(root, name), name);
  const result = await collectTrainingExportFiles(root, 30, 'latest');
  assert.equal(result.step, 20);
  assert.deepEqual(result.files.map(file => file.relativePath).sort(), ['job_000020_high_noise.safetensors', 'job_000020_low_noise.safetensors']);
  for (const relative of ['.aitk-checkpoints/.tmp-inflight/a', '.checkpoint-export-inflight/a', '.checkpoint-old-export-backup', '.checkpoint-write-one.tmp'])
    assert.equal(shouldIncludeTrainingExportPath('', relative), false, relative);
});

test('legacy full-model directories retain their primary and optimizer, and canceled scans stop immediately', async t => {
  const root = await fixture(t);
  await fs.mkdir(path.join(root, 'job_000010/transformer'), { recursive: true });
  await fs.writeFile(path.join(root, 'job_000010/aitk_meta.yaml'), 'training_info: {step: 10}');
  await fs.writeFile(path.join(root, 'job_000010/transformer/model.safetensors'), 'weights');
  await fs.writeFile(path.join(root, 'optimizer.pt'), 'paired');
  const result = await collectTrainingExportFiles(root, 30, 'latest');
  assert.equal(result.relativePath, 'job_000010');
  assert.equal(result.step, 10);
  assert.equal(result.optimizerIncluded, true);
  assert.deepEqual(result.files.map(file => file.relativePath).sort(), [
    'job_000010/aitk_meta.yaml', 'job_000010/transformer/model.safetensors', 'optimizer.pt',
  ]);
  await generation(root, 20, { 'job_000020.safetensors': 'weights' });
  const canceled = new Error('canceled');
  await assert.rejects(collectTrainingExportFiles(root, 30, 'latest', () => { throw canceled; }), error => error === canceled);
});

test('all skips corrupt generations and a generation without optimizer cannot borrow a stale portable optimizer', async t => {
  const root = await fixture(t);
  const good = await generation(root, 10, { 'job_000010.safetensors': 'weights' });
  const bad = await generation(root, 20, { 'job_000020.safetensors': 'weights', 'optimizer.pt': 'unrelated' });
  await fs.writeFile(path.join(root, bad, 'job_000020.safetensors'), 'corrupt');
  await fs.writeFile(path.join(root, 'optimizer.pt'), 'unrelated');
  const result = await collectTrainingExportFiles(root, 30, 'all');
  assert.equal(result.relativePath, `${good}/job_000010.safetensors`);
  assert.equal(result.optimizerIncluded, false);
  assert.ok(result.files.every(file => !file.relativePath.includes('000020') && !file.relativePath.endsWith('optimizer.pt')));
});

test('remote-start zip keeps immutable generation and portable pair together with the source storage identity', async t => {
  const root = await fixture(t);
  const storage = 'job-source-id';
  const folder = path.join(root, storage);
  await fs.mkdir(folder);
  const relative = await generation(folder, 20, {
    [`${storage}_000020_high_noise.safetensors`]: 'high',
    [`${storage}_000020_low_noise.safetensors`]: 'low', 'optimizer.pt': 'paired',
  });
  const settings = require('../dist/src/server/settings.js');
  const paths = require('../dist/src/server/trainingPaths.js');
  const { db } = require('../dist/src/server/db.js');
  const originals = {
    getTrainingFolder: settings.getTrainingFolder, getDatasetsRoot: settings.getDatasetsRoot,
    getJobTrainingRoot: paths.getJobTrainingRoot, findById: db.jobs.findById,
  };
  t.after(() => {
    settings.getTrainingFolder = originals.getTrainingFolder;
    settings.getDatasetsRoot = originals.getDatasetsRoot;
    paths.getJobTrainingRoot = originals.getJobTrainingRoot;
    db.jobs.findById = originals.findById;
  });
  settings.getTrainingFolder = async () => root;
  settings.getDatasetsRoot = async () => root;
  paths.getJobTrainingRoot = async () => root;
  db.jobs.findById = async () => ({
    id: 'source-id', storage_key: storage, name: 'Display Name', job_type: 'train', status: 'stopped', step: 25,
    job_config: JSON.stringify({ config: { name: storage, process: [] } }), gpu_ids: '0',
  });
  const bundle = await createRemoteTrainingJobBundle('source-id', { includeDatasets: false, checkpointMode: 'latest' });
  const extracted = path.join(root, 'extracted');
  await extractZipSafely(bundle.zipPath, extracted);
  const manifest = JSON.parse(await fs.readFile(path.join(extracted, 'manifest.json'), 'utf8'));
  const job = JSON.parse(await fs.readFile(path.join(extracted, 'job.json'), 'utf8'));
  assert.equal(job.storage_key, storage);
  assert.equal(manifest.training.latestCheckpointPath, `training/${relative}/${storage}_000020_high_noise.safetensors`);
  assert.equal(manifest.training.latestCheckpointStep, 20);
  assert.equal(await fs.readFile(path.join(extracted, 'training/optimizer.pt'), 'utf8'), 'paired');
  assert.equal(await fs.readFile(path.join(extracted, `training/${storage}_000020_low_noise.safetensors`), 'utf8'), 'low');
  assert.equal((await collectTrainingExportFiles(path.join(extracted, 'training'), 25, 'latest')).step, 20);
});
