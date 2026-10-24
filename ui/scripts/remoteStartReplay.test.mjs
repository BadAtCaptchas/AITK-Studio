import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import crypto from 'node:crypto';
import * as timers from 'node:timers/promises';
import test from 'node:test';
import ts from 'typescript';
import * as loraContinuation from '../dist/src/server/loraContinuation.js';

function load(relative, dependencies = {}) {
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

const commandInput = load('../src/server/commandInput.ts', {
  '../utils/obsoleteWorkspaceGuard': load('../src/utils/obsoleteWorkspaceGuard.ts'),
});

function fixture() {
  const records = new Map();
  let job = { id: 'remote-caption', worker_id: 'remote-worker', status: 'stopped', attempt_id: null,
    job_type: 'caption', updated_at: new Date(), job_config: '{"config":{"process":[]}}' };
  let writes = 0, preflights = 0;
  let encrypted = false, invalidConfig = false;
  const db = {
    runtime: {
      get: async key => structuredClone(records.get(key) ?? null),
      compareAndSwap: async (key, version, value) => {
        if ((records.get(key)?.version ?? null) !== version) return false;
        records.set(key, { key, version: (version ?? 0) + 1, value: structuredClone(value) }); return true;
      },
      delete: async (key, version) => records.get(key)?.version === version && records.delete(key),
    },
    jobs: {
      findById: async id => id === job.id ? structuredClone(job) : null,
      updateIf: async (id, expected, patch) => {
        if (id !== job.id || job.attempt_id !== expected.attempt_id || expected.status !== job.status || +expected.updated_at !== +job.updated_at) return null;
        writes++;
        job = { ...job, ...patch, updated_at: new Date(+job.updated_at + 1) };
        return structuredClone(job);
      },
    },
  };
  const lease = load('../src/server/processLease.ts', {
    os, crypto, 'timers/promises': timers, './db': { db },
    './processBirth': { getProcessBirth: async () => 1000, processBirthMatches: (a, b) => a === b },
  });
  const operations = load('../src/server/operations.ts', { crypto, './db': { db }, './processLease': lease, './commandInput': commandInput });
  const jobStart = load('../src/server/jobStart.ts', {
    './loraContinuation': loraContinuation,
    '../domain/configContract': { configContractErrors: () => { preflights++; return invalidConfig ? ['Invalid synthetic config'] : []; } },
    'fs/promises': {}, './pythonPath': {}, './db': { db }, './operations': operations,
    './trainingJobBundle': {}, './remoteClient': { isLocalWorker: () => false }, './remoteCaptionDispatch': {},
    './datasetDecryption': { DATASET_MIGRATION_REQUIRED: 'ENCRYPTED_DATASET_MIGRATION_REQUIRED', DATASET_MIGRATION_MESSAGE: 'Decrypt legacy datasets' },
    './encryptedDatasets': { getEncryptedDatasetsForJobConfig: async () => encrypted ? [{ path: 'synthetic-encrypted', name: 'fixture' }] : [] },
    './encryptedDatasetSecrets': {}, './secureRemoteCaptionJobs': {}, './remoteDatasetSync': {},
    '../../cron/actions/startJob': {}, './queueCoordination': {}, './processLease': lease,
  });
  // Re-load the acceptance module per call: replay must depend on persisted state,
  // not a same-process promise map or closure from the original HTTP request.
  const remote = () => load('../src/server/remoteStartOperation.ts', {
    crypto, './db': { db }, './jobStart': jobStart, './operations': operations, './operationKeys': {},
  });
  return {
    remote, operations, records, db, jobStart,
    accept: (key = 'durable-start-key', jobID = job.id) => remote().acceptRemoteStart(jobID, { idempotencyKey: key }),
    get job() { return structuredClone(job); }, get writes() { return writes; }, get preflights() { return preflights; },
    setJob: patch => { job = { ...job, ...patch }; },
    setEncrypted: () => { encrypted = true; }, setInvalid: () => { invalidConfig = true; },
  };
}

for (const state of ['completed', 'failed', 'canceled']) {
  test(`same-key ${state} remote-start receipt replays while its accepted job remains active`, async () => {
    const f = fixture();
    const first = await f.accept();
    await f.operations.updateOperation(first.id, { state, result: { jobID: 'remote-caption' }, error: state === 'failed' ? 'Recorded failure' : null });
    f.setJob({ status: 'running' });
    const beforeJob = f.job, beforeWrites = f.writes, beforePreflights = f.preflights;
    const replay = await f.accept();
    assert.equal(replay.id, first.id);
    assert.equal(replay.state, state);
    assert.deepEqual(f.job, beforeJob);
    assert.equal(f.writes, beforeWrites);
    assert.equal(f.preflights, beforePreflights, 'historical acknowledgement is not a fresh start');
  });
}

test('old terminal key returns its own receipt instead of attaching to a newer active command', async () => {
  const f = fixture();
  const first = await f.accept();
  await f.operations.updateOperation(first.id, { state: 'completed' });
  f.setJob({ status: 'stopped' });
  const newer = await f.accept('newer-start-key');
  const beforeJob = f.job, beforeWrites = f.writes;
  const replay = await f.accept();
  assert.equal(replay.id, first.id);
  assert.equal(replay.state, 'completed');
  assert.equal((await f.operations.activeOperationForResource('job:remote-caption')).id, newer.id);
  assert.deepEqual(f.job, beforeJob);
  assert.equal(f.writes, beforeWrites);
});

test('foreign job and mismatched-kind receipts are rejected before preflight or job mutation', async () => {
  for (const mismatch of ['job', 'kind']) {
    const f = fixture();
    const first = await f.accept();
    await f.operations.updateOperation(first.id, { state: 'completed' });
    if (mismatch === 'job') f.setJob({ id: 'different-job', status: 'stopped', attempt_id: null });
    else {
      const row = f.records.get(`operation:${first.id}`);
      row.value.input = { kind: 'dataset-decrypt', root: 'fixture-root', sourceName: 'source', outputName: 'plain' };
    }
    const beforeWrites = f.writes, beforePreflights = f.preflights;
    await assert.rejects(f.accept(), error => error.status === 409 && error.payload.code === 'IDEMPOTENCY_KEY_CONFLICT');
    assert.equal(f.writes, beforeWrites);
    assert.equal(f.preflights, beforePreflights);
  }
});

test('an active job still rejects a fresh or absent command key after the prior operation ended', async () => {
  const f = fixture();
  const first = await f.accept();
  await f.operations.updateOperation(first.id, { state: 'completed' });
  f.setJob({ status: 'running' });
  for (const request of [{}, { idempotencyKey: 'different-start-key' }]) {
    await assert.rejects(f.remote().acceptRemoteStart('remote-caption', request), error => error.status === 409 && error.payload.code === 'JOB_ACTIVE');
  }
});

test('active remote-start attachment remains available without preparing or claiming the job again', async () => {
  const f = fixture();
  const first = await f.accept();
  for (const request of [{}, { idempotencyKey: 'durable-start-key' }, { idempotencyKey: 'another-key' }]) {
    const beforeWrites = f.writes, beforePreflights = f.preflights;
    const replay = await f.remote().acceptRemoteStart('remote-caption', request);
    assert.equal(replay.id, first.id);
    assert.equal(f.writes, beforeWrites);
    assert.equal(f.preflights, beforePreflights);
  }
});

test('needs-keys recovery still revalidates and requeues the same owned operation', async () => {
  const f = fixture();
  const first = await f.accept();
  await f.operations.updateOperation(first.id, { state: 'needs-keys' });
  f.records.delete(`operation-pending:${first.id}`);
  const beforePreflights = f.preflights;
  const retry = await f.accept();
  assert.equal(retry.id, first.id);
  assert.equal(retry.state, 'queued');
  assert.equal(f.preflights, beforePreflights + 1);
  assert.equal(f.records.get(`operation-pending:${first.id}`).value, first.id);
});

test('needs-keys replay keeps encrypted migration and config validation guards', async () => {
  for (const guard of ['encrypted', 'config']) {
    const f = fixture();
    const first = await f.accept();
    await f.operations.updateOperation(first.id, { state: 'needs-keys' });
    if (guard === 'encrypted') f.setEncrypted();
    else f.setInvalid();
    const beforeJob = f.job;
    await assert.rejects(f.accept(), error => error.payload.code === (guard === 'encrypted' ? 'ENCRYPTED_DATASET_MIGRATION_REQUIRED' : 'INVALID_CONFIG'));
    assert.equal((await f.operations.getOperation(first.id)).state, 'needs-keys');
    assert.deepEqual(f.job, beforeJob);
  }
});

test('a prior needs-keys command cannot reclaim a resource now held by another active command', async () => {
  const f = fixture();
  const first = await f.accept();
  await f.operations.updateOperation(first.id, { state: 'needs-keys' });
  const newerID = f.operations.operationID('remote-start', 'newer-owner');
  const prior = await f.operations.getOperation(first.id);
  f.records.set(`operation:${newerID}`, { key: `operation:${newerID}`, version: 1, value: { ...prior, id: newerID, state: 'running' } });
  const resource = f.records.get('operation-resource:job:remote-caption');
  resource.value = newerID;
  f.setJob({ attempt_id: newerID, status: 'remote-starting' });
  const beforeJob = f.job;
  await assert.rejects(f.accept(), error => error.status === 409 && error.payload.code === 'JOB_ACTIVE');
  assert.deepEqual(f.job, beforeJob);
  assert.equal((await f.operations.getOperation(first.id)).state, 'needs-keys');
});

test('execution still rejects changed config before reclaiming an unowned operation', async () => {
  const f = fixture();
  const first = await f.accept();
  f.setJob({ status: 'stopped', attempt_id: null, job_config: '{"config":{"process":[]},"changed":true}' });
  const beforeJob = f.job;
  await assert.rejects(f.remote().executeRemoteStart(first), /configuration changed/);
  assert.deepEqual(f.job, beforeJob);
});

test('start route rejects invalid keys and unauthorized requests before remote acceptance', async () => {
  const f = fixture();
  let authenticated = true, accepted = 0;
  const { POST } = load('../src/app/api/jobs/[jobID]/start/route.ts', {
    '@/server/commandInput': commandInput, '@/server/jobStart': f.jobStart, '@/server/db': { db: f.db },
    '@/server/remoteClient': { isLocalWorker: () => false },
    '@/server/remoteStartOperation': { acceptRemoteStart: async () => { accepted++; assert.fail('invalid request accepted'); } },
    '@/server/remoteStartProgress': {}, '@/server/startCommand': {},
    '@/utils/authSession': { isRequestAuthenticated: async () => authenticated },
  });
  const post = body => POST(new Request('http://localhost/api/jobs/remote-caption/start', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  }), { params: Promise.resolve({ jobID: 'remote-caption' }) });
  for (const idempotencyKey of ['', 'short', '../not-valid', 42, 'x'.repeat(129)]) {
    assert.equal((await post({ idempotencyKey })).status, 400);
  }
  authenticated = false;
  assert.equal((await post({ idempotencyKey: 'valid-command-key' })).status, 401);
  assert.equal(accepted, 0);
});
