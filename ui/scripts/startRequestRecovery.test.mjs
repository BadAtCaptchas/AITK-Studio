import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

function load(relative, dependencies = {}) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const loadedModule = { exports: {} };
  new Function('require', 'module', 'exports', 'console', 'window', code)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
    return dependencies[name];
  }, loadedModule, loadedModule.exports, { log() {}, error() {} }, { setTimeout: callback => { callback(); return 0; } });
  return loadedModule.exports;
}
const commandInput = load('../src/server/commandInput.ts', {
  '../utils/obsoleteWorkspaceGuard': load('../src/utils/obsoleteWorkspaceGuard.ts'),
});
function client(apiClient) {
  return load('../src/utils/jobs.ts', { '@/utils/api': { apiClient }, '@/utils/media': {} });
}
function responseError(status, code, message = 'Actionable API reason') {
  return { response: { status, data: { error: message, code }, config: { token: 'synthetic-private-value' } },
    config: { request: 'synthetic-private-value' } };
}
function progress(patch = {}) {
  return { startID: 'start-operation', jobID: 'caption', status: 'queued', message: 'Queued remote start',
    percent: 0, datasetName: null, bytesProcessed: 0, bytesTotal: 0, warnings: [], error: null,
    remoteJobID: null, createdAt: '', updatedAt: '', ...patch };
}

function startFixture() {
  let revision = Date.now();
  let job = { id: 'caption', worker_id: 'local', gpu_ids: '0', job_type: 'caption', status: 'stopped',
    attempt_id: null, job_config: '{"config":{"process":[]}}', updated_at: new Date(++revision) };
  const queue = { id: 1, gpu_ids: '0', is_running: false };
  const records = new Map();
  let enqueues = 0;
  const db = {
    runtime: {
      get: async key => structuredClone(records.get(key) ?? null),
      compareAndSwap: async (key, version, value) => {
        if ((records.get(key)?.version ?? null) !== version) return false;
        records.set(key, { version: (version ?? 0) + 1, value: structuredClone(value) }); return true;
      },
    },
    jobs: { findById: async () => structuredClone(job), maxQueuePosition: async () => 0,
      updateIf: async (_id, expected, patch) => {
        if (job.attempt_id !== expected.attempt_id || job.status !== expected.status || +job.updated_at !== +expected.updated_at) return null;
        if (patch.status === 'queued') enqueues++;
        job = { ...job, ...patch, updated_at: new Date(++revision) }; return structuredClone(job);
      },
    },
    queues: { findByGpuIds: async () => structuredClone(queue) },
  };
  class LeaseBusyError extends Error {}
  const lease = { withProcessLease: async (_key, work) => work(), LeaseBusyError };
  const jobStart = load('../src/server/jobStart.ts', {
    '../domain/configContract': { configContractErrors: () => [] }, 'fs/promises': {},
    './pythonPath': { assertPythonRuntimeReady: async () => {} }, './db': { db }, './operations': {},
    './trainingJobBundle': {}, './remoteClient': { isLocalWorker: id => id === 'local' }, './remoteCaptionDispatch': {},
    './datasetDecryption': {}, './encryptedDatasets': { getEncryptedDatasetsForJobConfig: async () => [] },
    './encryptedDatasetSecrets': {}, './secureRemoteCaptionJobs': { isAnyRemoteOllamaCaptionJob: () => false },
    './remoteDatasetSync': {}, '../../cron/actions/startJob': {},
    './queueCoordination': { withLocalQueueTransition: async (_ids, work) => work() }, './processLease': lease,
  });
  const startCommand = load('../src/server/startCommand.ts', {
    './db': { db }, './processLease': lease, './jobStart': jobStart, './commandInput': commandInput,
  });
  const route = load('../src/app/api/jobs/[jobID]/start/route.ts', {
    '@/server/commandInput': commandInput, '@/server/jobStart': jobStart, '@/server/db': { db },
    '@/server/remoteClient': { isLocalWorker: id => id === 'local' }, '@/server/remoteStartOperation': {},
    '@/server/remoteStartProgress': {}, '@/utils/authSession': { isRequestAuthenticated: async () => true },
    '@/server/startCommand': startCommand,
  });
  const keys = [];
  let dropNext = true;
  const jobs = client({ post: async (_url, body) => {
    keys.push(body.idempotencyKey);
    const result = await route.POST(new Request('http://localhost/api/jobs/caption/start', {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
    }), { params: Promise.resolve({ jobID: 'caption' }) });
    const data = await result.json();
    if (!result.ok) throw { response: { status: result.status, data } };
    if (dropNext) { dropNext = false; throw new Error('Synthetic response lost after commit'); }
    return { data };
  } });
  return { jobs, keys, get job() { return job; }, get queue() { return queue; }, get enqueues() { return enqueues; },
    complete: () => { job = { ...job, status: 'completed', attempt_id: 'finished-attempt', updated_at: new Date(++revision) }; },
  };
}

for (const completed of [false, true]) {
  test(`caller-owned start key recovers a lost acknowledgement after the job is ${completed ? 'completed' : 'queued'}`, async () => {
    const f = startFixture();
    const options = { idempotencyKey: 'caption-start-command' };
    await assert.rejects(f.jobs.startJob('caption', undefined, options), /response lost/);
    assert.equal(f.job.status, 'queued');
    assert.equal(f.queue.is_running, false);
    if (completed) f.complete();
    await f.jobs.startJob('caption', undefined, options);
    assert.deepEqual(f.keys, ['caption-start-command', 'caption-start-command']);
    assert.equal(f.enqueues, 1, 'recovery must not start completed work a second time');
    assert.equal(f.job.status, completed ? 'completed' : 'queued');
  });
}

test('independent starts without a caller-owned key keep independent command identities', async () => {
  const requests = [];
  const jobs = client({ post: async (_url, body) => { requests.push(body); return { data: { id: 'caption', status: 'queued' } }; } });
  await jobs.startJob('caption');
  await jobs.startJob('caption');
  assert.notEqual(requests[0].idempotencyKey, requests[1].idempotencyKey);
  for (const request of requests) assert.match(request.idempotencyKey, /^[a-zA-Z0-9_-]{8,128}$/);
});

for (const [label, error, outcome, status, code] of [
  ['transport failure', new Error('Connection lost'), 'unknown', undefined, undefined],
  ['server failure', responseError(503, 'SERVER_FAILURE'), 'unknown', 503, 'SERVER_FAILURE'],
  ['live start command', responseError(409, 'START_BUSY'), 'unknown', 409, 'START_BUSY'],
  ['validation rejection', responseError(400, 'INVALID_CONFIG'), 'rejected', 400, 'INVALID_CONFIG'],
  ['authentication rejection', responseError(401, undefined), 'rejected', 401, undefined],
  ['missing job rejection', responseError(404, undefined), 'rejected', 404, undefined],
  ['active job conflict', responseError(409, 'JOB_ACTIVE'), 'rejected', 409, 'JOB_ACTIVE'],
]) {
  test(`${label} reports ${outcome} start outcome without retaining transport objects`, async () => {
    const jobs = client({ post: async () => { throw error; } });
    let caught;
    try { await jobs.startJob('caption'); } catch (reason) { caught = reason; }
    assert.equal(jobs.isStartJobError(caught), true);
    assert.equal(caught.name, 'StartJobError');
    assert.equal(caught.outcome, outcome);
    assert.equal(caught.status, status);
    assert.equal(caught.code, code);
    assert.equal(caught.message, error instanceof Error ? error.message : 'Actionable API reason');
    assert.equal('response' in caught, false);
    assert.equal('config' in caught, false);
    assert.equal('cause' in caught, false);
    assert.doesNotMatch(JSON.stringify(caught), /synthetic-private-value/);
  });
}

test('a failed progress poll remains ambiguous even when the read itself has a definitive HTTP status', async () => {
  for (const status of [401, 404, 503]) {
    const jobs = client({
      post: async () => ({ data: { startID: 'start-operation', progress: progress() } }),
      get: async () => { throw responseError(status, 'PROGRESS_READ_FAILED'); },
    });
    await assert.rejects(jobs.startJob('caption'), error => jobs.isStartJobError(error) && error.outcome === 'unknown' && error.status === status);
  }
});

test('observed remote completion succeeds and observed remote failure is a definite result', async () => {
  for (const status of ['completed', 'failed']) {
    const observations = [];
    const jobs = client({
      post: async () => ({ data: { startID: 'start-operation', progress: progress() } }),
      get: async () => ({ data: progress({ status, error: status === 'failed' ? 'Remote preflight rejected' : null }) }),
    });
    const result = jobs.startJob('caption', undefined, { onRemoteStartProgress: value => observations.push(value.status) });
    if (status === 'completed') await result;
    else await assert.rejects(result, error => jobs.isStartJobError(error) && error.outcome === 'rejected' && error.message === 'Remote preflight rejected');
    assert.deepEqual(observations, ['queued', status]);
  }
});

test('invalid or mismatched success acknowledgement cannot be treated as completed start', async () => {
  for (const data of [null, {}, { id: 'different-job', status: 'queued' }, { startID: 'start-operation', progress: null }]) {
    const jobs = client({ post: async () => ({ data }) });
    await assert.rejects(jobs.startJob('caption'), error => jobs.isStartJobError(error) && error.outcome === 'unknown');
  }
});

test('invalid or mismatched progress is ambiguous rather than an endless poll or another command result', async () => {
  for (const data of [null, {}, progress({ status: 'unexpected' }), progress({ startID: 'different-command', status: 'completed' })]) {
    const jobs = client({
      post: async () => ({ data: { startID: 'start-operation', progress: progress() } }),
      get: async () => ({ data }),
    });
    await assert.rejects(jobs.startJob('caption'), error => jobs.isStartJobError(error) && error.outcome === 'unknown');
  }
});

test('restart-from-scratch retains its request contract and never forwards the caption start key', async () => {
  const requests = [];
  const jobs = client({ post: async (url, body) => { requests.push({ url, body }); return { data: { id: 'caption', status: 'queued' } }; } });
  await jobs.restartJobFromScratch('caption', undefined, { idempotencyKey: 'caption-start-command', background: true });
  assert.deepEqual(requests, [{ url: '/api/jobs/caption/restart-from-scratch', body: { background: false } }]);
});
