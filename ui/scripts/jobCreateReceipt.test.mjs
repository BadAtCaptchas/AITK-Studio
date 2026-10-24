import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import crypto from 'node:crypto';
import * as timers from 'node:timers/promises';
import test from 'node:test';
import ts from 'typescript';

function load(relative, dependencies = {}) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const loaded = { exports: {} };
  new Function('require', 'module', 'exports', 'console', code)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
    return dependencies[name];
  }, loaded, loaded.exports, { error() {} });
  return loaded.exports;
}
const obsoleteWorkspaceGuard = load('../src/utils/obsoleteWorkspaceGuard.ts');
const commandInput = load('../src/server/commandInput.ts', { '../utils/obsoleteWorkspaceGuard': obsoleteWorkspaceGuard });
const configContract = load('../src/domain/configContract.ts', {
  './loraContinuation': load('../src/domain/loraContinuation.ts'),
  '../utils/trainingPhases': load('../src/utils/trainingPhases.ts'),
  './modelCapabilities.json': JSON.parse(fs.readFileSync(new URL('../src/domain/modelCapabilities.json', import.meta.url), 'utf8')),
});
const jobIdentity = load('../src/utils/jobIdentity.ts');
const validationConfig = load('../src/utils/validationConfig.ts');

function payload(patch = {}) {
  return { id: null, idempotencyKey: 'caption-create-command', name: 'new_caption', gpu_ids: '0', job_type: 'caption',
    job_ref: 'synthetic-dataset', job_config: { job: 'extension', config: { process: [{ type: 'OpenRouterCaptioner', caption: { caption_prompt: 'synthetic-private-prompt' } }] } }, ...patch };
}

function fixture() {
  const records = new Map(), jobs = new Map();
  let created = 0, updated = 0, prepared = 0, workerReads = 0;
  let failAfterCreate = false, failReceiptWrite = false, failReceiptCreation = false, busyLease = false, authenticated = true;
  let trainingRoot = 'D:\\synthetic-training-root';
  const db = {
    runtime: {
      get: async key => structuredClone(records.get(key) ?? null),
      compareAndSwap: async (key, version, value) => {
        if (failReceiptCreation && key.startsWith('job-create-command:') && version === null) return false;
        if (failReceiptWrite && value?.state === 'completed') { failReceiptWrite = false; return false; }
        if ((records.get(key)?.version ?? null) !== version) return false;
        records.set(key, { key, version: (version ?? 0) + 1, value: structuredClone(value) }); return true;
      },
      delete: async (key, version) => records.get(key)?.version === version && records.delete(key),
    },
    jobs: {
      maxQueuePosition: async () => jobs.size * 1000,
      findById: async id => structuredClone(jobs.get(id) ?? null),
      findByName: async name => structuredClone([...jobs.values()].find(job => job.name === name) ?? null),
      create: async data => {
        if ([...jobs.values()].some(job => job.id === data.id || job.name === data.name)) throw Object.assign(new Error('Duplicate synthetic job'), { code: 'P2002' });
        const job = { ...data, id: data.id || crypto.randomUUID(), status: 'stopped' };
        jobs.set(job.id, structuredClone(job)); created++;
        if (failAfterCreate) { failAfterCreate = false; throw new Error('Synthetic failure after insert'); }
        return structuredClone(job);
      },
      updateIf: async () => { updated++; assert.fail('creation replay must never update a job'); },
    },
    workerNodes: { findById: async id => { workerReads++; return id === 'worker-1' ? { id, enabled: true } : null; } },
  };
  const lease = load('../src/server/processLease.ts', {
    os, crypto, 'timers/promises': timers, './db': { db },
    './processBirth': { getProcessBirth: async () => 1000, processBirthMatches: (a, b) => a === b },
  });
  const receipt = () => load('../src/server/jobCreateCommand.ts', {
      crypto, './db': { db }, './commandInput': commandInput,
      './processLease': { ...lease, withProcessLease: (...args) => busyLease ? Promise.reject(new lease.LeaseBusyError()) : lease.withProcessLease(...args) },
    });
  // Actual paths helper; only settings lookup is synthetic.
  const paths = load('../src/server/trainingPaths.ts', {
    './commandInput': commandInput, './settings': { getTrainingFolder: async () => { prepared++; return trainingRoot; } },
  });
  const route = () => load('../src/app/api/jobs/route.ts', {
    '@/domain/configContract': configContract, '@/utils/obsoleteWorkspaceGuard': obsoleteWorkspaceGuard,
    '@/server/commandInput': commandInput, '@/utils/jobIdentity': jobIdentity,
    '@/utils/validationConfig': validationConfig,
    '@/helpers/basic': { isMac: () => false }, '@/utils/authSession': { isRequestAuthenticated: async () => authenticated },
    '@/server/db': { db }, '@/server/jobAttempts': {}, '@/server/jobCreateCommand': receipt(),
    '@/server/remoteClient': { isLocalWorker: id => id === 'local' },
    '@/server/comfyInstallProgress': {}, '@/server/hfDownloadProgress': {}, '@/server/jobsApiList': {},
    '@/server/remoteDatasetPaths': {}, '@/server/trainingPaths': paths,
  });
  return {
    jobs, records, db, lease, receipt,
    post: body => route().POST(new Request('http://localhost/api/jobs', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
    })),
    get created() { return created; }, get updated() { return updated; }, get prepared() { return prepared; }, get workerReads() { return workerReads; },
    loseInsertResult: () => { failAfterCreate = true; }, loseReceiptWrite: () => { failReceiptWrite = true; },
    loseReceiptCreation: () => { failReceiptCreation = true; }, denyAuth: () => { authenticated = false; },
    holdLease: () => { busyLease = true; }, changeTrainingRoot: () => { trainingRoot = 'E:\\different-synthetic-root'; },
  };
}

test('lost create response replays the same saved identity across new request module instances', async () => {
  const f = fixture();
  const first = await f.post(payload());
  assert.equal(first.status, 200);
  const saved = await first.json();
  const replay = await f.post(payload());
  assert.equal(replay.status, 200);
  assert.equal((await replay.json()).id, saved.id);
  assert.equal(f.created, 1);
  assert.equal(f.updated, 0);
  assert.equal(f.prepared, 1);
});

test('create-before-receipt failure recovers the assigned id without repeating insertion', async () => {
  const f = fixture();
  f.loseInsertResult();
  assert.equal((await f.post(payload())).status, 500);
  assert.equal(f.created, 1);
  const saved = [...f.jobs.values()][0];
  const replay = await f.post(payload());
  assert.equal(replay.status, 200);
  assert.equal((await replay.json()).id, saved.id);
  assert.equal(f.created, 1);
});

test('matching concurrent create requests share one row and identity', async () => {
  const f = fixture();
  const responses = await Promise.all([f.post(payload()), f.post(payload()), f.post(payload())]);
  assert.deepEqual(responses.map(response => response.status), [200, 200, 200]);
  const saved = await Promise.all(responses.map(response => response.json()));
  assert.equal(new Set(saved.map(job => job.id)).size, 1);
  assert.equal(f.created, 1);
});

test('same key rejects different normalized input without modifying the earlier job', async () => {
  const f = fixture();
  const saved = await (await f.post(payload())).json();
  const response = await f.post(payload({ name: 'changed_caption' }));
  assert.equal(response.status, 409);
  assert.equal((await response.json()).code, 'CREATE_KEY_CONFLICT');
  assert.deepEqual(f.jobs.get(saved.id), saved);
  assert.equal(f.created, 1);
});

test('a deleted completed result is not recreated or adopted by name', async () => {
  const f = fixture();
  const saved = await (await f.post(payload())).json();
  f.jobs.delete(saved.id);
  f.jobs.set('unrelated-id', { ...saved, id: 'unrelated-id' });
  const replay = await f.post(payload());
  assert.equal(replay.status, 410);
  assert.equal((await replay.json()).code, 'CREATE_RESULT_DELETED');
  assert.equal(f.created, 1);
});

test('changed saved fields block recovery before the caller can start an unexpected configuration', async () => {
  for (const patch of [
    { name: 'renamed-caption' }, { gpu_ids: '1' }, { worker_id: 'worker-1' }, { job_ref: 'different-source' },
    { job_type: 'train' }, { job_config: '{"config":{"process":[{"type":"OpenRouterCaptioner"}]}}' },
    { storage_key: 'different-output' },
  ]) {
    const f = fixture();
    const saved = await (await f.post(payload())).json();
    f.jobs.set(saved.id, { ...saved, ...patch });
    const replay = await f.post(payload());
    assert.equal(replay.status, 409);
    assert.equal((await replay.json()).code, 'CREATE_RESULT_CHANGED');
    assert.equal(f.created, 1);
    assert.equal(f.updated, 0);
    assert.deepEqual(f.jobs.get(saved.id), { ...saved, ...patch });
  }
});

test('prepared crash recovery also rejects a changed or unrelated row occupying its assigned id', async () => {
  const f = fixture();
  f.loseInsertResult();
  assert.equal((await f.post(payload())).status, 500);
  const [saved] = f.jobs.values();
  f.jobs.set(saved.id, { ...saved, name: 'unrelated-job' });
  const response = await f.post(payload());
  assert.equal(response.status, 409);
  assert.equal((await response.json()).code, 'CREATE_RESULT_CHANGED');
  assert.equal(f.created, 1);
});

test('normal runtime status and queue progress do not invalidate the saved configuration receipt', async () => {
  const f = fixture();
  const saved = await (await f.post(payload())).json();
  f.jobs.set(saved.id, { ...saved, status: 'running', queue_position: 9000, step: 12 });
  const response = await f.post(payload());
  assert.equal(response.status, 200);
  assert.equal((await response.json()).status, 'running');
  assert.equal(f.created, 1);
});

test('a receipt without an expected result cannot adopt an existing row', async () => {
  const f = fixture();
  const saved = await (await f.post(payload())).json();
  const receipt = [...f.records.values()].find(row => row.key.startsWith('job-create-command:'));
  receipt.value.state = 'prepared';
  receipt.value.resultDigest = null;
  const response = await f.post(payload());
  assert.equal(response.status, 409);
  assert.equal((await response.json()).code, 'CREATE_RESULT_CHANGED');
  assert.deepEqual(f.jobs.get(saved.id), saved);
});

test('malformed persisted receipts fail closed without replacing their assigned identity', async () => {
  for (const patch of [{ state: 'unknown' }, { version: 2 }, { inputDigest: 'bad' }, { jobID: '../invalid' }, { resultDigest: 'bad' }]) {
    const f = fixture();
    await f.post(payload());
    const receipt = [...f.records.values()].find(row => row.key.startsWith('job-create-command:'));
    receipt.value = { ...receipt.value, ...patch };
    const before = structuredClone(receipt);
    const response = await f.post(payload());
    assert.equal(response.status, 500);
    assert.equal((await response.json()).code, 'CREATE_RECEIPT_INVALID');
    assert.deepEqual(f.records.get(receipt.key), before);
    assert.equal(f.created, 1);
  }
});

test('job id combined with a creation key is rejected before mutation', async () => {
  const f = fixture();
  const response = await f.post(payload({ id: 'existing-id' }));
  assert.equal(response.status, 400);
  assert.equal((await response.json()).code, 'INVALID_COMMAND');
  assert.equal(f.created, 0);
  assert.equal(f.updated, 0);
});

test('receipt contains only digest, assigned identity, version, and state', async () => {
  const f = fixture();
  const saved = await (await f.post(payload())).json();
  const receipts = [...f.records.values()].filter(row => row.key.startsWith('job-create-command:'));
  assert.equal(receipts.length, 1);
  assert.deepEqual(Object.keys(receipts[0].value).sort(), ['inputDigest', 'jobID', 'resultDigest', 'state', 'version']);
  assert.match(receipts[0].value.inputDigest, /^[a-f0-9]{64}$/);
  assert.match(receipts[0].value.resultDigest, /^[a-f0-9]{64}$/);
  assert.equal(receipts[0].value.jobID, saved.id);
  assert.equal(receipts[0].value.state, 'completed');
  assert.doesNotMatch(JSON.stringify(receipts), /synthetic-private-prompt|synthetic-dataset|synthetic-training-root|new_caption/);
});

test('reordered JSON properties and normalized defaults replay the same request', async () => {
  const f = fixture();
  const original = payload();
  const saved = await (await f.post(original)).json();
  const reordered = { ...original, worker_id: ' local ', job_config: {
    config: { process: [{ caption: { caption_prompt: 'synthetic-private-prompt' }, type: 'OpenRouterCaptioner' }] },
    capability_version: 1, job: 'extension',
  } };
  f.changeTrainingRoot();
  const replay = await f.post(reordered);
  assert.equal(replay.status, 200);
  assert.equal((await replay.json()).id, saved.id);
  assert.equal(f.prepared, 1, 'recovery must not regenerate runtime paths from changed settings');
});

test('completion receipt write failure remains recoverable without a second insert', async () => {
  const f = fixture();
  f.loseReceiptWrite();
  const first = await f.post(payload());
  assert.equal(first.status, 500);
  assert.equal((await first.json()).code, 'CREATE_RECEIPT_WRITE_FAILED');
  const retry = await f.post(payload());
  assert.equal(retry.status, 200);
  assert.equal(f.created, 1);
});

test('a held creation lease returns a retryable busy result without creating anything', async () => {
  const f = fixture();
  f.holdLease();
  const response = await f.post(payload());
  assert.equal(response.status, 409);
  assert.equal((await response.json()).code, 'CREATE_BUSY');
  assert.equal(f.created, 0);
});

test('failure to reserve a receipt returns busy without performing preparation or insertion', async () => {
  const f = fixture();
  f.loseReceiptCreation();
  const response = await f.post(payload());
  assert.equal(response.status, 409);
  assert.equal((await response.json()).code, 'CREATE_BUSY');
  assert.equal(f.created, 0);
  assert.equal(f.prepared, 0);
});

test('unrelated name collision keeps conflict and can never be adopted by the receipt', async () => {
  const f = fixture();
  const { idempotencyKey: _unused, ...ordinary } = payload();
  const saved = await (await f.post(ordinary)).json();
  for (let attempt = 0; attempt < 2; attempt++) assert.equal((await f.post(payload())).status, 409);
  assert.equal(f.created, 1);
  assert.deepEqual(f.jobs.get(saved.id), saved);
});

test('requests without a key retain ordinary creation and duplicate-name behavior', async () => {
  const f = fixture();
  const { idempotencyKey: _unused, ...ordinary } = payload();
  assert.equal((await f.post(ordinary)).status, 200);
  assert.equal((await f.post(ordinary)).status, 409);
  assert.equal(f.records.size, 0);
});

test('invalid keys, unsafe configs, removed workspaces, and unknown workers cannot claim receipts', async () => {
  const f = fixture();
  for (const idempotencyKey of ['', 'short', '../outside', 1, null, 'x'.repeat(129)]) {
    assert.equal((await f.post(payload({ idempotencyKey }))).status, 400);
  }
  const invalidBodies = [
    payload({ project_id: 'retired' }), payload({ worker_id: 'unknown' }),
    payload({ job_config: { config: { process: [{ type: 'OpenRouterCaptioner', caption: { base_url: 'forbidden' } }] } } }),
    payload({ job_config: { config: { process: [{ type: 'sd_trainer', train: { auto_train: true } }] } } }),
  ];
  for (const body of invalidBodies) assert.equal((await f.post(body)).status, 400);
  assert.equal(f.created, 0);
  assert.equal(f.records.size, 0);
});

test('new remote creation still does not contact a worker or require its model locally', async () => {
  const f = fixture();
  const input = payload({ worker_id: 'worker-1', job_type: 'train', job_config: { config: { process: [
    { type: 'sd_trainer', model: { arch: 'flux', name_or_path: 'fixture/not-downloaded' } },
  ] } } });
  assert.equal((await f.post(input)).status, 200);
  assert.equal((await f.post(input)).status, 200);
  assert.equal(f.created, 1);
  assert.equal(f.prepared, 0);
});

test('unauthorized creation cannot read or claim a receipt', async () => {
  const f = fixture();
  f.denyAuth();
  const previous = process.env.AI_TOOLKIT_AUTH;
  process.env.AI_TOOLKIT_AUTH = 'synthetic-test-auth';
  try {
    assert.equal((await f.post(payload())).status, 401);
    assert.equal(f.created, 0);
    assert.equal(f.records.size, 0);
  } finally {
    if (previous === undefined) delete process.env.AI_TOOLKIT_AUTH;
    else process.env.AI_TOOLKIT_AUTH = previous;
  }
});
