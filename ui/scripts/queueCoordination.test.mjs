import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import crypto from 'node:crypto';
import test from 'node:test';
import ts from 'typescript';

// Load current source with isolated dependencies: these races need no real DB,
// Python runtime, child processes, or training folders.
function loadSource(relative, dependencies) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const loadedModule = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency: ${name}`);
    return dependencies[name];
  }, loadedModule, loadedModule.exports);
  return loadedModule.exports;
}

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

function fixture({ running = true, missingQueue = false, gpuIds = '0' } = {}) {
  const records = new Map();
  let revision = Date.now();
  let job = {
    id: 'job', worker_id: 'local', gpu_ids: gpuIds, status: 'stopped', attempt_id: null,
    job_config: '{"config":{"process":[]}}', updated_at: new Date(++revision), return_to_queue: false,
  };
  let queue = missingQueue ? null : { id: 1, worker_id: 'local', gpu_ids: gpuIds, is_running: running };
  const launches = [];
  const contended = deferred();
  const retry = deferred();
  let pauseRetries = false;
  const db = {
    runtime: {
      get: async key => structuredClone(records.get(key) ?? null),
      compareAndSwap: async (key, version, value) => {
        if ((records.get(key)?.version ?? null) !== version) return false;
        records.set(key, { version: (version ?? 0) + 1, value: structuredClone(value) });
        return true;
      },
      delete: async (key, version) => records.get(key)?.version === version && records.delete(key),
    },
    jobs: {
      maxQueuePosition: async () => 0,
      findById: async () => structuredClone(job),
      list: async ({ status }) => (Array.isArray(status) ? status : [status]).includes(job.status) ? [structuredClone(job)] : [],
      findFirst: async ({ status }) => job.status === status ? structuredClone(job) : null,
      updateIf: async (_id, expected, patch) => {
        if (job.attempt_id !== expected.attempt_id || (expected.status && job.status !== expected.status)) return null;
        if (expected.updated_at && +job.updated_at !== +expected.updated_at) return null;
        job = { ...job, ...patch, updated_at: new Date(++revision) };
        return structuredClone(job);
      },
    },
    queues: {
      list: async () => queue ? [structuredClone(queue)] : [],
      findByGpuIds: async ids => ids === queue?.gpu_ids ? structuredClone(queue) : null,
      update: async (_id, patch) => { queue = { ...queue, ...patch }; return structuredClone(queue); },
      create: async input => {
        assert.equal(queue, null, 'queue must not be created twice');
        queue = { id: 1, worker_id: 'local', ...input };
        return structuredClone(queue);
      },
    },
  };
  const lease = loadSource('../src/server/processLease.ts', {
    os, crypto, './db': { db },
    './processBirth': { getProcessBirth: async () => 1000, processBirthMatches: (a, b) => a === b },
  });
  const coordination = loadSource('../src/server/queueCoordination.ts', {
    './db': { db }, './processLease': lease,
    'timers/promises': { setTimeout: async () => {
      contended.resolve();
      if (pauseRetries) await retry.promise;
    } },
  });
  const caption = { isAnyRemoteOllamaCaptionJob: () => false };
  const start = loadSource('../src/server/jobStart.ts', {
    '../domain/configContract': {}, 'fs/promises': {}, './pythonPath': {}, './db': { db }, './operations': {},
    './trainingJobBundle': {}, './remoteClient': { isLocalWorker: id => id === 'local' }, './remoteCaptionDispatch': {},
    './encryptedDatasets': {}, './encryptedDatasetSecrets': {}, './secureRemoteCaptionJobs': caption,
    './remoteDatasetSync': {}, '../../cron/actions/startJob': {}, './queueCoordination': coordination, './processLease': lease,
  });
  const scheduler = loadSource('../cron/actions/processQueue.ts', {
    '../../src/server/db': { db }, '../../src/utils/jobIdentity': { devicesOverlap: (a, b) => a === b },
    '../../src/server/jobProcess': { reconcileLocalJobProcess: async value => value },
    '../../src/server/secureRemoteCaptionJobs': caption, '../../src/server/queueCoordination': coordination,
    '../../src/server/processLease': lease, './startJob': { __esModule: true, default: async id => launches.push(id) },
  }).default;
  const prepared = () => ({
    job: structuredClone(job), jobID: job.id, jobConfig: JSON.parse(job.job_config),
    requiredEncryptedDatasets: [], encryptedKeysForLaunch: [], useDurableEncryptedKeys: false,
  });
  return {
    db, scheduler, launches, coordination, prepared,
    enqueue: options => start.startPreparedJob(prepared(), options),
    get job() { return job; }, get queue() { return queue; },
    setJob: patch => { job = { ...job, ...patch }; },
    holdRetries: () => { pauseRetries = true; return { contended: contended.promise, release: retry.resolve }; },
  };
}

test('enqueue-and-start cannot be lost behind an in-flight empty-queue decision', { timeout: 5000 }, async () => {
  const f = fixture();
  const emptyRead = deferred(), releaseEmptyRead = deferred();
  f.db.jobs.findFirst = async () => { emptyRead.resolve(); await releaseEmptyRead.promise; return null; };
  const retries = f.holdRetries();
  const cron = f.scheduler();
  await emptyRead.promise;
  const request = f.enqueue({ startQueue: true });
  await retries.contended;
  assert.equal(f.job.status, 'stopped', 'enqueue must wait until the scheduler decision is complete');
  releaseEmptyRead.resolve();
  await cron;
  retries.release();
  await request;
  assert.equal(f.job.status, 'queued');
  assert.equal(f.queue.is_running, true);
});

test('explicit queue start wins after an already-running empty-queue decision', { timeout: 5000 }, async () => {
  const f = fixture();
  const emptyRead = deferred(), releaseEmptyRead = deferred();
  f.db.jobs.findFirst = async () => { emptyRead.resolve(); await releaseEmptyRead.promise; return null; };
  const retries = f.holdRetries();
  const cron = f.scheduler();
  await emptyRead.promise;
  const request = f.coordination.setLocalQueueRunning('0', true);
  await retries.contended;
  releaseEmptyRead.resolve();
  await cron;
  retries.release();
  await request;
  assert.equal(f.queue.is_running, true);
});

test('a queue stop serializes after an in-flight start', { timeout: 5000 }, async () => {
  const f = fixture({ running: false });
  const writeStarted = deferred(), releaseWrite = deferred();
  const update = f.db.queues.update;
  f.db.queues.update = async (id, patch) => {
    if (patch.is_running) { writeStarted.resolve(); await releaseWrite.promise; }
    return update(id, patch);
  };
  const retries = f.holdRetries();
  const start = f.coordination.setLocalQueueRunning('0', true);
  await writeStarted.promise;
  const stop = f.coordination.setLocalQueueRunning('0', false);
  await retries.contended;
  releaseWrite.resolve();
  await start;
  retries.release();
  await stop;
  assert.equal(f.queue.is_running, false);
});

test('cron re-reads queue state instead of acting on its stale list snapshot', async () => {
  const f = fixture();
  f.setJob({ status: 'running' });
  f.db.queues.list = async () => [{ ...f.queue, is_running: false }];
  await f.scheduler();
  assert.equal(f.job.return_to_queue, false, 'an old paused snapshot must not stop current work');

  await f.coordination.setLocalQueueRunning('0', false);
  f.setJob({ status: 'queued' });
  f.db.queues.list = async () => [{ ...f.queue, is_running: true }];
  await f.scheduler();
  assert.deepEqual(f.launches, [], 'an old running snapshot must not launch work from a stopped queue');
});

test('ordinary enqueue preserves paused queues and running queues', async () => {
  for (const options of [{ missingQueue: true }, { running: false }, { running: true }]) {
    const f = fixture(options);
    await f.enqueue();
    assert.equal(f.job.status, 'queued');
    assert.equal(f.queue.is_running, options.running === true);
  }
});

test('cron does not claim work when the queue stops during Python preflight', async () => {
  const f = fixture();
  f.setJob({ status: 'queued' });
  const preflight = deferred(), releasePreflight = deferred();
  const start = loadSource('../cron/actions/startJob.ts', {
    '../../src/server/db': { db: f.db }, '../../src/server/jobProcess': {},
    '../../src/server/jobAttempts': { claimJobAttempt: async () => assert.fail('stopped queue must not claim a job') },
    '../../src/server/queueCoordination': f.coordination, '../../src/server/processLease': {},
    '../../src/utils/jobIdentity': {}, '../../src/server/inferenceToken': {}, child_process: {}, path: {}, fs: {},
    '../paths': {}, '../../src/server/tensorboard': {}, '../../src/server/pythonPath': {
      assertPythonRuntimeReady: async () => { preflight.resolve(); await releasePreflight.promise; },
    },
    '../../src/server/trainingPaths': {}, '../../src/server/hfTokenEnv': {}, '../../src/server/encryptedDatasets': {},
    '../../src/server/encryptedDatasetSecrets': {}, '../../src/server/secureRemoteCaptionJobs': {},
    '../../src/server/remoteOllamaWorkers': {}, '../../src/server/networkPolicy': {},
    '../../src/server/telemetry': {}, '../../src/utils/telemetry': {},
  });
  const launch = start.startJobNow('job', { requireRunningQueue: true });
  await preflight.promise;
  await f.coordination.setLocalQueueRunning('0', false);
  releasePreflight.resolve();
  assert.equal(await launch, false);
  assert.equal(f.job.status, 'queued');
});

const { startInferenceEngine } = loadSource('../src/utils/startInferenceEngine.ts', {});

test('live inference starts a new or stopped queue using the engine job GPU', async () => {
  for (const missingQueue of [true, false]) {
    const f = fixture({ missingQueue, running: false, gpuIds: '2' });
    await startInferenceEngine({ id: 'job', gpu_ids: '2' }, {
      startJob: id => { assert.equal(id, 'job'); return f.enqueue(); },
      startQueue: (gpuIds, workerID) => {
        assert.equal(gpuIds, '2'); assert.equal(workerID, 'local');
        return f.coordination.setLocalQueueRunning(gpuIds, true);
      },
    });
    await f.scheduler();
    assert.equal(f.queue.is_running, true);
    assert.deepEqual(f.launches, ['job']);
  }
});

test('live inference does not start a queue when the job start was rejected', async () => {
  await assert.rejects(startInferenceEngine({ id: 'engine', gpu_ids: '2' }, {
    startJob: async () => { throw new Error('job rejected'); },
    startQueue: async () => assert.fail('queue must not start after a rejected job'),
  }), /job rejected/);
});
