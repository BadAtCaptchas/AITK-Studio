import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import childProcess from 'node:child_process';
import util from 'node:util';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import ts from 'typescript';
import { loadTrainingSource, trainingFixture, migration } from './trainingMigrationFixture.mjs';

const domain = loadTrainingSource('domain/loraContinuation.ts');
const health = loadTrainingSource('utils/workerHealth.ts');
const workspaces = loadTrainingSource('domain/trainingWorkspaces.ts');
const commandInput = { isRecord: value => !!value && typeof value === 'object' && !Array.isArray(value) };
function load(relative, dependencies) {
  const code = ts.transpileModule(fs.readFileSync(new URL('../src/' + relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const result = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
    return dependencies[name];
  }, result, result.exports);
  return result.exports;
}
function fixture(mode = 'none') {
  const config = trainingFixture(), p = config.config.process[0];
  p.model = { ...p.model, arch: 'krea2', name_or_path: 'krea/Krea-2-Raw' };
  p.network.continuation = { version: 1, source_sha256: 'a'.repeat(64), training_base: 'raw', deployment_base: 'turbo' };
  p.network.pretrained_lora_path = '/source.safetensors';
  p.train.source_preservation = mode === 'none' ? { mode } : { mode, new_batches: 3, teacher_weight: 1 };
  if (mode !== 'none') p.datasets.push({ ...p.datasets[0], folder_path: '/preserved', role: 'source_preservation', caption_dropout_rate: 0 });
  return config;
}

const continuationSetup = load('app/loras/continue/continuationConfig.ts', {
  '@/app/jobs/new/jobConfig': loadTrainingSource('app/jobs/new/jobConfig.ts'),
  '@/app/jobs/new/utils': loadTrainingSource('app/jobs/new/utils.ts'),
  '@/utils/hooks': loadTrainingSource('utils/hooks.tsx'),
});

test('dedicated continuation defaults and base changes retain source topology and preservation settings', () => {
  const defaults = continuationSetup.defaultContinuationConfig().config.process[0];
  assert.equal(defaults.model.arch, 'krea2');
  assert.equal(defaults.model.name_or_path, 'krea/Krea-2-Raw');
  assert.equal(defaults.network.continuation.training_base, 'raw');
  assert.deepEqual(defaults.train.source_preservation, { mode: 'none' });
  assert.throws(() => continuationSetup.initializeContinuationConfig(trainingFixture()), /Krea 2/);
  const config = fixture('replay_teacher');
  const before = structuredClone(config);
  const turbo = continuationSetup.changeContinuationBase(config, 'krea2:turbo');
  const process = turbo.config.process[0];
  assert.equal(process.model.name_or_path, 'krea/Krea-2-Turbo');
  assert.match(process.model.assistant_lora_path, /krea2_turbo_training_adapter/);
  assert.equal(process.network.continuation.training_base, 'turbo');
  assert.equal(process.network.pretrained_lora_path, before.config.process[0].network.pretrained_lora_path);
  assert.equal(process.network.linear, before.config.process[0].network.linear);
  assert.deepEqual(process.train.source_preservation, before.config.process[0].train.source_preservation);
  assert.equal(process.datasets[1].caption_dropout_rate, 0);
  assert.equal(process.datasets[1].role, 'source_preservation');
  assert.deepEqual(config, before);
  const raw = continuationSetup.changeContinuationBase(turbo, 'krea2').config.process[0];
  assert.equal(raw.model.name_or_path, 'krea/Krea-2-Raw');
  assert.equal(raw.model.assistant_lora_path, undefined);
  assert.equal(raw.network.continuation.training_base, 'raw');
  assert.deepEqual(domain.continuationConfigErrors(raw), []);
});

test('all four preservation modes have explicit data contracts', () => {
  for (const mode of domain.preservationModes) {
    const p = fixture(mode).config.process[0];
    assert.deepEqual(domain.continuationConfigErrors(p), []);
    p.model.model_kwargs = { edit: true };
    assert.match(domain.continuationConfigErrors(p).join(' '), /text-to-image/);
  }
  const p = fixture('teacher').config.process[0];
  p.train.source_preservation.new_batches = 1.2;
  p.datasets[1].shuffle_tokens = true;
  assert.equal(domain.continuationConfigErrors(p).length, 2);
  delete p.network.continuation;
  assert.match(domain.continuationConfigErrors(p).join(' '), /requires an inspected/);
});

test('continuation has a separate editor route and draft namespace for creation, editing and cloning', () => {
  const source = fixture();
  assert.equal(workspaces.trainingWorkspaceForConfig(source), 'continuation');
  assert.equal(workspaces.trainingJobEditorPath(JSON.stringify(source)), '/loras/continue');
  assert.equal(workspaces.trainingJobEditorPath(JSON.stringify(trainingFixture())), '/jobs/new');
  assert.equal(workspaces.trainingJobEditorPath('invalid'), '/jobs/new');
  for (const query of ['', 'id=existing', 'cloneId=original']) {
    const params = new URLSearchParams(query);
    assert.notEqual(workspaces.trainingWorkspaceDraftKey('training', params), workspaces.trainingWorkspaceDraftKey('continuation', params));
  }
});

test('preservation cannot silently replace curated captions with empty conditioning', () => {
  const p = fixture('replay').config.process[0];
  for (const caption_ext of ['', null, 'json']) {
    p.datasets[1].caption_ext = caption_ext;
    assert.match(domain.continuationConfigErrors(p).join(' '), /curated .txt captions/);
  }
  p.datasets[1].caption_ext = 'txt';
  p.train.unload_text_encoder = true;
  p.train.cache_text_embeddings = false;
  assert.match(domain.continuationConfigErrors(p).join(' '), /Cache preservation text embeddings/);
  p.train.cache_text_embeddings = true;
  delete p.datasets[1].cache_text_embeddings;
  assert.deepEqual(domain.continuationConfigErrors(p), []);
  p.datasets[1].cache_text_embeddings = false;
  assert.match(domain.continuationConfigErrors(p).join(' '), /Cache preservation text embeddings/);
});

test('migration clears source initialization and preservation datasets without changing the source', () => {
  const original = fixture('replay_teacher'), before = structuredClone(original);
  const next = migration.migrateTrainingModel(original, 'zimage').config.config.process[0];
  assert.equal(next.network.continuation, undefined);
  assert.equal(next.network.pretrained_lora_path, undefined);
  assert.equal(next.train.source_preservation, undefined);
  assert.equal(next.datasets.length, 1);
  assert.deepEqual(original, before);
});

test('inspection identity changes with source, scale, model and worker but not inspection results', () => {
  const config = fixture(), p = config.config.process[0];
  const original = domain.continuationSourceKey(p, 'local');
  p.network.continuation.source_sha256 = 'b'.repeat(64);
  assert.equal(domain.continuationSourceKey(p, 'local'), original);
  assert.notEqual(domain.continuationSourceKey(p, 'worker-A'), original);
  for (const change of [p => p.network.pretrained_lora_path = '/replacement.safetensors',
    p => p.network.continuation.source_scale = 2, p => p.model.name_or_path = '/another-base',
    p => delete p.network.continuation]) {
    const next = structuredClone(p); change(next);
    assert.notEqual(domain.continuationSourceKey(next, 'local'), original);
  }
});

test('worker capabilities reject old, malformed and incomplete implementations', () => {
  for (const capability of [undefined, { version: 1 }, { version: 1, models: ['krea2'], modes: domain.preservationModes }])
    assert.equal(health.parseWorkerHealth({ ok: true, loraContinuation: capability }).loraContinuationVersion, undefined);
  assert.equal(health.parseWorkerHealth({ ok: true, loraContinuation: {
    version: 1, models: ['krea2', 'krea2:turbo'], modes: domain.preservationModes,
  } }).loraContinuationVersion, 1);
});

async function environment(t, overrides = {}) {
  const root = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-continuation-'));
  t.after(() => fsp.rm(root, { recursive: true, force: true }));
  const assets = load('server/loraContinuation.ts', {
    fs, 'fs/promises': fsp, path, crypto, child_process: childProcess, util,
    '../paths': { TOOLKIT_ROOT: path.resolve('..') }, './pythonPath': { getToolkitPythonPath: () => 'python' },
    './settings': { getDataRoot: async () => root, getTrainingFolder: async () => root, getModelsRoot: async () => root },
    './commandInput': commandInput, '../domain/loraContinuation': domain,
    '../utils/remoteLoraRefs': loadTrainingSource('utils/remoteLoraRefs.ts'),
    ...overrides,
  });
  return { root, assets };
}

test('an existing failed continuation passes start validation using its pinned source and unchanged saved config', async t => {
  let inspectedPath;
  const config = fixture('replay');
  const process = config.config.process[0];
  const execFile = Object.assign(() => assert.fail('Expected promisified inspection'), {
    [util.promisify.custom]: async (_python, args) => {
      inspectedPath = args[args.indexOf('--path') + 1];
      assert.equal(args[args.indexOf('--sha256') + 1], process.network.continuation.source_sha256);
      return { stdout: JSON.stringify({ version: 1, status: 'compatible', sha256: process.network.continuation.source_sha256,
        format: 'aitk_comfy_native', tensor_count: 2, ranks: [1], architecture: 'krea2', topology_verified: true, size_bytes: 100,
        modules: [{ name: 'blocks.0.attn.wq', rank: 1, input: 6144, output: 6144, source_scale: 1 }], declared_training_base: null }), stderr: '' };
    },
  });
  const { root, assets } = await environment(t, { child_process: { execFile } });
  const job = path.join(root, 'already-created-job');
  const pinned = path.join(job, '.continuation', process.network.continuation.source_sha256 + '.safetensors');
  await fsp.mkdir(path.dirname(pinned), { recursive: true });
  await fsp.writeFile(pinned, 'pinned source fixture');
  process.network.pretrained_lora_path = path.join(root, 'no-longer-present.safetensors');
  const saved = JSON.stringify(config);
  await assets.validateContinuationSources(config, job, 'local');
  assert.equal(inspectedPath, pinned);
  assert.equal(JSON.stringify(config), saved);
});

test('remote source references stay bound to their worker and snapshot verified worker files', async t => {
  const { root, assets } = await environment(t);
  const refs = loadTrainingSource('utils/remoteLoraRefs.ts');
  const source = path.join(root, 'worker-source.safetensors');
  await fsp.writeFile(source, 'worker source fixture');
  const config = fixture(), p = config.config.process[0];
  p.network.pretrained_lora_path = refs.makeRemoteLoraRef('worker-A', source);
  p.network.continuation.source_sha256 = await assets.hashContinuationAsset(source);
  assert.deepEqual(refs.parseRemoteLoraRef(p.network.pretrained_lora_path), { workerID: 'worker-A', path: source });
  assert.throws(() => refs.parseRemoteLoraRef('aitk-lora://remote/bad%2Fid/file'), /Invalid/);
  await assert.rejects(assets.validateContinuationSources(config, undefined, 'worker-B'), /another worker/);
  await assert.rejects(assets.collectContinuationAssets(config, undefined, 'local'), /selected execution worker/);
  const extraction = path.join(root, 'extract');
  await fsp.mkdir(extraction);
  const asset = { archivePath: `training/.continuation/${p.network.continuation.source_sha256}.safetensors`,
    configPath: 'config.process[0].network.pretrained_lora_path', sha256: p.network.continuation.source_sha256,
    size: (await fsp.stat(source)).size, workerSource: source };
  await assets.rewriteImportedContinuationAssets(config, [asset], extraction, path.join(root, 'job'));
  assert.equal(await fsp.readFile(path.join(extraction, asset.archivePath), 'utf8'), 'worker source fixture');
  await fsp.writeFile(source, 'replaced');
  await assert.rejects(assets.rewriteImportedContinuationAssets(config, [asset], extraction, path.join(root, 'job')), /changed since inspection/);
});

test('remote uploads stream verified bytes, reject bad acknowledgements, and clean interrupted transfers', async t => {
  const { root, assets } = await environment(t);
  const streamed = load('server/streamedUpload.ts', { crypto, fs, 'fs/promises': fsp, path, stream: { Readable, Transform }, 'stream/promises': { pipeline } });
  const refs = loadTrainingSource('utils/remoteLoraRefs.ts');
  let outcome = 'ok', calls = 0;
  const opened = [];
  const upload = load('server/remoteLoraUpload.ts', {
    fs: { ...fs, createReadStream: (...args) => { const stream = fs.createReadStream(...args); opened.push(stream); return stream; } },
    stream: { Readable }, './commandInput': commandInput, '../utils/remoteLoraRefs': refs,
    './streamedUpload': { ...streamed, cleanupStagedUpload: async file => {
      assert.ok(opened.every(stream => stream.closed), 'Close the transfer handle before deleting staging');
      await streamed.cleanupStagedUpload(file);
    } }, './loraContinuation': assets,
    './loraLibrary': { getUploadedLoraRoot: async () => root, readSafetensorsMetadataStrict: async () => ({}) },
    './remoteClient': {
      getRemoteWorker: async id => ({ id }), fetchWorkerHealth: async () => ({ loraContinuationVersion: outcome === 'old' ? undefined : 1 }),
      remoteJson: async (worker, endpoint, init) => {
        calls++;
        if (outcome === 'early') throw new Error('Connection interrupted before stream opened');
        assert.equal(worker.id, 'worker-A');
        assert.equal(endpoint, '/api/generate/loras/upload');
        assert.equal(init.headers.has('x-aitk-worker-id'), false);
        assert.equal(init.headers.has('authorization'), false);
        const body = Buffer.from(await new Response(init.body).arrayBuffer());
        assert.equal(body.toString(), 'fixture source bytes');
        if (outcome === 'interrupted') throw new Error('Connection interrupted');
        return { lora: { path: '/worker/uploaded.safetensors' }, sha256: outcome === 'bad' ? 'b'.repeat(64) : crypto.createHash('sha256').update(body).digest('hex') };
      },
    },
  });
  const request = () => new Request('http://controller/upload', { method: 'POST', headers: {
    'x-aitk-file-name': 'source.safetensors', 'x-aitk-worker-id': 'worker-A', Authorization: 'do-not-forward',
  }, body: 'fixture source bytes' });
  const result = await upload.uploadLoraToWorker(request(), 'worker-A');
  assert.equal(refs.parseRemoteLoraRef(result.lora.path).workerID, 'worker-A');
  for (outcome of ['bad', 'interrupted', 'early', 'old']) {
    await assert.rejects(upload.uploadLoraToWorker(request(), 'worker-A'), /checksum|interrupted|Update/);
    assert.deepEqual(await fsp.readdir(root), []);
  }
  assert.equal(calls, 4);
  const aborted = new AbortController();
  aborted.abort();
  outcome = 'ok';
  await assert.rejects(upload.uploadLoraToWorker(new Request('http://controller/upload', { method: 'POST', body: 'bytes', signal: aborted.signal }), 'worker-A'), /abort/i);
  assert.deepEqual(await fsp.readdir(root), []);
});

test('an upload failure cannot delete a published LoRA already reused by another request', async t => {
  const { root, assets } = await environment(t);
  const library = load('server/loraLibrary.ts', { fs, crypto, path, './settings': { getDataRoot: async () => root } });
  const streamed = load('server/streamedUpload.ts', { crypto, fs, 'fs/promises': fsp, path,
    stream: { Readable, Transform }, 'stream/promises': { pipeline } });
  let published, failSidecar;
  const publication = new Promise(resolve => { published = resolve; });
  const sidecar = new Promise((_resolve, reject) => { failSidecar = reject; });
  t.mock.method(console, 'error', () => {});
  const route = load('app/api/generate/loras/upload/route.ts', {
    fs, '@/server/remoteLoraUpload': {}, '@/server/loraContinuation': assets, '@/domain/loraContinuation': domain,
    '@/server/streamedUpload': streamed, '@/server/loraLibrary': { ...library,
      readSafetensorsMetadataStrict: async () => ({}), buildUploadedLoraEntry: async file => ({ path: file }),
      writeUploadedLoraSidecar: async () => { published(); await sidecar; },
    },
  });
  const request = () => new Request('http://localhost/upload', { method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream', 'X-AITK-File-Name': 'source.safetensors' }, body: 'fixture' });
  const first = route.POST(request());
  await publication;
  const second = await route.POST(request());
  assert.equal(second.status, 200);
  const reused = await second.json();
  assert.equal(reused.reused, true);
  failSidecar(new Error('Injected sidecar failure'));
  assert.equal((await first).status, 400);
  assert.equal(await fsp.readFile(reused.lora.path, 'utf8'), 'fixture');
});

test('sidecar publication leaves complete JSON on concurrent writes and preserves it after failure', async t => {
  const { root } = await environment(t);
  const deps = { fs, crypto, path, './settings': {} };
  const library = load('server/loraLibrary.ts', deps);
  const file = path.join(root, 'source.safetensors');
  const variants = [{ originalFilename: 'first.safetensors' }, { originalFilename: 'longer-second.safetensors' }];
  await Promise.all(variants.map(value => library.writeUploadedLoraSidecar(file, value)));
  const publishedPath = path.join(root, (await fsp.readdir(root))[0]);
  const original = await fsp.readFile(publishedPath, 'utf8');
  assert.ok(variants.some(value => value.originalFilename === JSON.parse(original).originalFilename));
  let lockedAttempts = 0;
  const failing = load('server/loraLibrary.ts', { ...deps, fs: { ...fs, promises: { ...fsp,
    rename: async () => { lockedAttempts++; throw Object.assign(new Error('Injected publication failure'), { code: 'EPERM' }); },
  } } });
  await assert.rejects(failing.writeUploadedLoraSidecar(file, {}), /publication failure/);
  assert.equal(lockedAttempts, 5, 'Permanent locks must fail after bounded retries');
  assert.equal(await fsp.readFile(publishedPath, 'utf8'), original);
  assert.equal((await fsp.readdir(root)).length, 1);
  let attempts = 0;
  const transient = load('server/loraLibrary.ts', { ...deps, fs: { ...fs, promises: { ...fsp,
    rename: async (...args) => {
      if (++attempts < 3) throw Object.assign(new Error('Temporary destination lock'), { code: 'EPERM' });
      return fsp.rename(...args);
    },
  } } });
  await transient.writeUploadedLoraSidecar(file, variants[0]);
  assert.equal(attempts, 3);
  assert.deepEqual(JSON.parse(await fsp.readFile(publishedPath, 'utf8')), variants[0]);
});

test('source and pinned auxiliary transfer are checksum verified and rewritten independently of original paths', async t => {
  const { root, assets } = await environment(t);
  const config = fixture(), p = config.config.process[0];
  const source = path.join(root, 'source.safetensors');
  await fsp.writeFile(source, 'source fixture');
  p.network.pretrained_lora_path = source;
  p.network.continuation.source_sha256 = await assets.hashContinuationAsset(source);
  const job = path.join(root, 'job');
  await fsp.mkdir(path.join(job, '.continuation'), { recursive: true });
  const auxiliary = path.join(root, 'auxiliary.safetensors');
  await fsp.writeFile(auxiliary, 'auxiliary fixture');
  const auxiliaryHash = await assets.hashContinuationAsset(auxiliary);
  await fsp.copyFile(auxiliary, path.join(job, '.continuation', auxiliaryHash + '.safetensors'));
  await fsp.writeFile(path.join(job, '.continuation', 'provenance.json'), JSON.stringify({ source_sha256: p.network.continuation.source_sha256, assistant_sha256: auxiliaryHash }));
  p.model.assistant_lora_path = 'owner/repo/adapter.safetensors';
  const bundle = await assets.collectContinuationAssets(config, job);
  assert.equal(bundle.assets.length, 2);
  p.network.continuation.assistant_sha256 = 'b'.repeat(64);
  await assert.rejects(assets.collectContinuationAssets(config, job), /training-adapter checksum changed/);
  delete p.network.continuation.assistant_sha256;
  const extraction = path.join(root, 'extraction');
  for (const file of bundle.files) {
    const target = path.join(extraction, file.archivePath);
    await fsp.mkdir(path.dirname(target), { recursive: true });
    await fsp.copyFile(file.sourcePath, target);
  }
  const target = path.join(root, 'target-job');
  const unchanged = structuredClone(config);
  await assert.rejects(assets.rewriteImportedContinuationAssets(config, [bundle.assets[0], bundle.assets[0]], extraction, target), /duplicate/);
  assert.deepEqual(config, unchanged, 'Invalid later mappings cannot partially rewrite the config');
  p.network.continuation.assistant_sha256 = 'b'.repeat(64);
  const wrongAuxiliary = structuredClone(config);
  await assert.rejects(assets.rewriteImportedContinuationAssets(config, bundle.assets, extraction, target), /Auxiliary hash/);
  assert.deepEqual(config, wrongAuxiliary);
  delete p.network.continuation.assistant_sha256;
  await assets.rewriteImportedContinuationAssets(config, bundle.assets, extraction, target);
  assert.equal(p.network.pretrained_lora_path, path.join(target, '.continuation', p.network.continuation.source_sha256 + '.safetensors'));
  assert.equal(p.network.continuation.assistant_sha256, auxiliaryHash);
  await fsp.appendFile(path.join(extraction, bundle.assets[0].archivePath), 'corrupt');
  await assert.rejects(assets.rewriteImportedContinuationAssets(config, bundle.assets, extraction, target), /checksum/);
  await assert.rejects(assets.rewriteImportedContinuationAssets(config, undefined, extraction, target), /missing/);
  await assert.rejects(assets.rewriteImportedContinuationAssets(config, [bundle.assets[0], bundle.assets[0]], extraction, target));
  await fsp.writeFile(source, 'replaced source');
  await assert.rejects(assets.collectContinuationAssets({ ...config, config: { process: [{ ...p, network: { ...p.network, pretrained_lora_path: source } }] } }), /changed/);
});

test('curation uses validated prompt metadata, snapshots hashes and retries without overwriting', async t => {
  const { root, assets } = await environment(t);
  const sample = path.join(root, 'sample.png');
  await fsp.writeFile(sample, 'image fixture');
  await fsp.writeFile(sample + '.txt', 'unsafe caption --seed 42 --steps 8');
  const recorded = { prompt: 'A plain caption.', source_lora_sha256: 'a'.repeat(64), source_base: 'krea/Krea-2-Turbo', seed: 42,
    media_sha256: await assets.hashContinuationAsset(sample) };
  const publication = load('server/preservationPublication.ts', { 'fs/promises': fsp, path,
    './commandInput': commandInput, '../domain/loraContinuation': domain, './loraContinuation': assets });
  let publishCount = 0, releasePublications;
  const concurrentPublications = new Promise(resolve => { releasePublications = resolve; });
  const service = load('server/preservationDataset.ts', {
    'fs/promises': fsp, path, crypto, './db': { db: { jobs: { findById: async () => ({ id: 'job', job_type: 'generate', worker_id: 'local' }) } } },
    './datasetScope': { resolveDatasetScope: async () => ({ datasetsRoot: root }) },
    './jobSamples': { resolveJobSampleFile: async (_job, name) => name === 'sample.png' ? { path: sample, stat: await fsp.stat(sample) } : null },
    './sampleMetadata': { readSampleMetadata: async () => structuredClone(recorded) },
    './preservationPublication': publication,
    './publishDatasetDirectory': { publishDatasetDirectory: async (from, to) => {
      if (++publishCount <= 2) {
        if (publishCount === 2) releasePublications();
        await concurrentPublications;
      }
      await fsp.rename(from, to);
    } },
    './loraContinuation': assets, './commandInput': commandInput, '../domain/loraContinuation': domain,
    './remoteClient': { isLocalWorker: () => true }, '../utils/remoteDatasetRefs': {},
  });
  const request = { job_id: 'job', name: 'preserved', request_id: crypto.randomUUID(), samples: ['sample.png'] };
  const [first, concurrent] = await Promise.all([service.createPreservationDataset(request), service.createPreservationDataset(request)]);
  assert.deepEqual(concurrent, first);
  assert.equal(await fsp.readFile(path.join(first.path, '0000.txt'), 'utf8'), recorded.prompt);
  const manifest = JSON.parse(await fsp.readFile(path.join(first.path, '.aitk-preservation.json'), 'utf8'));
  assert.equal(manifest.examples[0].image_sha256, await assets.hashContinuationAsset(sample));
  assert.deepEqual(manifest.evaluation_examples, []);
  assert.deepEqual(await service.createPreservationDataset(request), first);
  await assert.rejects(service.createPreservationDataset({ ...request, request_id: crypto.randomUUID() }), /already in use/);
  await assert.rejects(service.createPreservationDataset({ ...request, samples: ['../sample.png'] }), /Provide/);
  await assert.rejects(service.createPreservationDataset({ ...request, name: 'con ' }), /another dataset name/);
  await fsp.writeFile(path.join(first.path, '0000.png'), 'changed after publication');
  await assert.rejects(service.createPreservationDataset(request), /dataset changed/);
  recorded.media_sha256 = 'b'.repeat(64);
  await assert.rejects(service.createPreservationDataset({ ...request, name: 'mismatched-image' }), /changed during curation/);
  recorded.source_lora_sha256 = undefined;
  await assert.rejects(service.createPreservationDataset({ ...request, name: 'no-provenance' }), /provenance/);
  assert.equal((await fsp.readdir(root)).some(name => name.startsWith('.aitk-preservation-')), false);
});
