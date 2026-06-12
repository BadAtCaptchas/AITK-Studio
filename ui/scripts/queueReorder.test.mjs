import assert from 'node:assert/strict';
import test from 'node:test';
import { QueueReorderError, reorderQueueJobs } from '../dist/src/server/queueReorder.js';

function makeJob(overrides = {}) {
  return {
    id: 'job-1',
    name: 'queue-job',
    worker_id: 'local',
    remote_job_id: null,
    remote_sync_at: null,
    remote_error: null,
    gpu_ids: '0',
    job_config: '{}',
    created_at: new Date(),
    updated_at: new Date(),
    status: 'queued',
    stop: false,
    return_to_queue: false,
    step: 0,
    info: '',
    speed_string: '',
    queue_position: 1000,
    pid: null,
    job_type: 'train',
    job_ref: null,
    save_now: false,
    ...overrides,
  };
}

function makeDeps(initialJobs) {
  const jobs = new Map(initialJobs.map(job => [job.id, { ...job }]));
  const calls = {
    updates: [],
    remoteJson: [],
    getRemoteWorker: [],
  };
  const deps = {
    findJobById: async jobID => jobs.get(jobID) || null,
    updateJob: async (jobID, data) => {
      calls.updates.push({ jobID, data });
      const next = { ...jobs.get(jobID), ...data };
      jobs.set(jobID, next);
      return next;
    },
    getRemoteWorker: async workerID => {
      calls.getRemoteWorker.push(workerID);
      return {
        id: workerID,
        name: 'Remote Worker',
        base_url: 'https://worker.example',
        api_token: 'token',
        enabled: true,
        last_status: 'ready',
        last_error: null,
        last_checked_at: null,
        capabilities: '{}',
        gpus: '[]',
        created_at: new Date(),
        updated_at: new Date(),
      };
    },
    remoteJson: async (worker, routePath, init) => {
      calls.remoteJson.push({ worker, routePath, init });
      return { ok: true };
    },
    isLocalWorker: workerID => !workerID || workerID === 'local',
  };
  return { deps, calls, jobs };
}

test('reorders queued local jobs with stable queue positions', async () => {
  const { deps, calls, jobs } = makeDeps([
    makeJob({ id: 'job-a', queue_position: 1000 }),
    makeJob({ id: 'job-b', queue_position: 2000 }),
    makeJob({ id: 'job-c', queue_position: 3000 }),
  ]);

  await reorderQueueJobs('0', ['job-c', 'job-a', 'job-b'], 'local', deps);

  assert.deepEqual(calls.updates, [
    { jobID: 'job-c', data: { queue_position: 1000 } },
    { jobID: 'job-a', data: { queue_position: 2000 } },
    { jobID: 'job-b', data: { queue_position: 3000 } },
  ]);
  assert.equal(jobs.get('job-c').queue_position, 1000);
  assert.equal(jobs.get('job-a').queue_position, 2000);
  assert.equal(jobs.get('job-b').queue_position, 3000);
});

test('rejects duplicate job ids', async () => {
  const { deps, calls } = makeDeps([makeJob({ id: 'job-a' })]);

  await assert.rejects(
    () => reorderQueueJobs('0', ['job-a', 'job-a'], 'local', deps),
    error => error instanceof QueueReorderError && /duplicates/i.test(error.message),
  );
  assert.equal(calls.updates.length, 0);
});

test('rejects jobs outside the requested lane', async () => {
  const { deps, calls } = makeDeps([
    makeJob({ id: 'job-a', gpu_ids: '0' }),
    makeJob({ id: 'job-b', gpu_ids: '1' }),
  ]);

  await assert.rejects(
    () => reorderQueueJobs('0', ['job-a', 'job-b'], 'local', deps),
    error => error instanceof QueueReorderError && /worker\/GPU queue/i.test(error.message),
  );
  assert.equal(calls.updates.length, 0);
});

test('rejects non-queued jobs', async () => {
  const { deps, calls } = makeDeps([
    makeJob({ id: 'job-a', status: 'queued' }),
    makeJob({ id: 'job-b', status: 'running' }),
  ]);

  await assert.rejects(
    () => reorderQueueJobs('0', ['job-a', 'job-b'], 'local', deps),
    error => error instanceof QueueReorderError && /Only queued/i.test(error.message),
  );
  assert.equal(calls.updates.length, 0);
});

test('forwards remote reorders with remote job ids before updating local mirrors', async () => {
  const { deps, calls } = makeDeps([
    makeJob({ id: 'local-a', worker_id: 'worker-1', remote_job_id: 'remote-a' }),
    makeJob({ id: 'local-b', worker_id: 'worker-1', remote_job_id: 'remote-b' }),
  ]);

  await reorderQueueJobs('0', ['local-b', 'local-a'], 'worker-1', deps);

  assert.deepEqual(calls.getRemoteWorker, ['worker-1']);
  assert.equal(calls.remoteJson.length, 1);
  assert.equal(calls.remoteJson[0].routePath, '/api/queue/0/reorder');
  assert.deepEqual(JSON.parse(calls.remoteJson[0].init.body), {
    worker_id: 'local',
    job_ids: ['remote-b', 'remote-a'],
  });
  assert.deepEqual(calls.updates, [
    { jobID: 'local-b', data: { queue_position: 1000 } },
    { jobID: 'local-a', data: { queue_position: 2000 } },
  ]);
});
