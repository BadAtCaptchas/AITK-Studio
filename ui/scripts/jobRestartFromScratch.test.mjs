import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { JobStartError } from '../dist/src/server/jobStart.js';
import {
  restartTrainingJobFromScratch,
  TrainingJobRestartError,
} from '../dist/src/server/trainingJobRestart.js';

function makeJob(overrides = {}) {
  return {
    id: 'job-1',
    name: 'scratch-job',
    worker_id: 'local',
    remote_job_id: null,
    remote_sync_at: null,
    remote_error: null,
    gpu_ids: '0',
    job_config: JSON.stringify({ config: { process: [{ train: { steps: 1000 } }] } }),
    created_at: new Date(),
    updated_at: new Date(),
    status: 'stopped',
    stop: false,
    return_to_queue: false,
    step: 123,
    info: 'old info',
    speed_string: '1.2 it/s',
    queue_position: 10,
    pid: 9876,
    job_type: 'train',
    job_ref: null,
    save_now: true,
    ...overrides,
  };
}

function makePrepared(job, overrides = {}) {
  return {
    jobID: job.id,
    job,
    jobConfig: JSON.parse(job.job_config),
    requiredEncryptedDatasets: [],
    encryptedKeysForLaunch: [],
    useDurableEncryptedKeys: false,
    ...overrides,
  };
}

function makeDeps(job, overrides = {}) {
  const calls = {
    updates: [],
    removedPaths: [],
    deleteMetricsForJob: [],
    startPreparedJob: [],
    remoteJson: [],
    ensureQueueRunning: [],
    syncRemoteJob: [],
    prepareJobStart: [],
    assertPreparedJobCanStart: [],
  };
  let currentJob = job;
  const deps = {
    findJobById: async () => currentJob,
    updateJob: async (_jobID, data) => {
      calls.updates.push(data);
      currentJob = { ...currentJob, ...data };
      return currentJob;
    },
    getTrainingFolder: async () => '',
    pathExists: () => false,
    rmPath: async targetPath => {
      calls.removedPaths.push(targetPath);
    },
    deleteMetricsForJob: async jobID => {
      calls.deleteMetricsForJob.push(jobID);
    },
    prepareJobStart: async (jobID, encryptedDatasetKeys, durableEncryptedDatasetKeys) => {
      calls.prepareJobStart.push({ jobID, encryptedDatasetKeys, durableEncryptedDatasetKeys });
      return makePrepared(currentJob);
    },
    assertPreparedJobCanStart: async prepared => {
      calls.assertPreparedJobCanStart.push(prepared.job.id);
    },
    startPreparedJob: async (prepared, options) => {
      calls.startPreparedJob.push({ prepared, options });
      return { ...prepared.job, status: 'queued', info: options.queueInfo || prepared.job.info };
    },
    getRemoteWorker: async workerID => ({
      id: workerID,
      name: 'Remote',
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
    }),
    remoteJson: async (_worker, routePath, init) => {
      calls.remoteJson.push({ routePath, init });
      return {};
    },
    syncRemoteJob: async remoteJob => {
      calls.syncRemoteJob.push(remoteJob.id);
      return { ...remoteJob, status: 'queued', info: 'remote restarted' };
    },
    ensureQueueRunning: async (gpuIds, workerId) => {
      calls.ensureQueueRunning.push({ gpuIds, workerId });
    },
    ...overrides,
  };
  return { deps, calls };
}

test('restart from scratch rejects non-training jobs', async () => {
  const { deps, calls } = makeDeps(makeJob({ job_type: 'caption' }));

  await assert.rejects(
    () => restartTrainingJobFromScratch('job-1', {}, deps),
    error => error instanceof TrainingJobRestartError && error.status === 400,
  );
  assert.equal(calls.prepareJobStart.length, 0);
  assert.equal(calls.removedPaths.length, 0);
});

test('restart from scratch rejects active jobs', async () => {
  for (const status of ['running', 'stopping']) {
    const { deps, calls } = makeDeps(makeJob({ status }));

    await assert.rejects(
      () => restartTrainingJobFromScratch('job-1', {}, deps),
      error => error instanceof TrainingJobRestartError && error.status === 409,
    );
    assert.equal(calls.prepareJobStart.length, 0);
    assert.equal(calls.removedPaths.length, 0);
  }
});

test('restart from scratch deletes only the resolved job training folder', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-restart-'));
  const jobFolder = path.join(root, 'scratch-job');
  const siblingFolder = path.join(root, 'other-job');
  fs.mkdirSync(jobFolder, { recursive: true });
  fs.mkdirSync(siblingFolder, { recursive: true });
  fs.writeFileSync(path.join(jobFolder, 'checkpoint.safetensors'), 'old');
  fs.writeFileSync(path.join(siblingFolder, 'keep.txt'), 'keep');

  const { deps, calls } = makeDeps(makeJob(), {
    getTrainingFolder: async () => root,
    pathExists: targetPath => fs.existsSync(targetPath),
    rmPath: async (targetPath, options) => {
      calls.removedPaths.push(targetPath);
      await fs.promises.rm(targetPath, options);
    },
  });

  try {
    await restartTrainingJobFromScratch('job-1', {}, deps);

    assert.deepEqual(calls.removedPaths, [path.resolve(root, 'scratch-job')]);
    assert.equal(fs.existsSync(jobFolder), false);
    assert.equal(fs.existsSync(path.join(siblingFolder, 'keep.txt')), true);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('restart from scratch resets progress fields before starting', async () => {
  const { deps, calls } = makeDeps(makeJob());

  await restartTrainingJobFromScratch('job-1', {}, deps);

  assert.deepEqual(calls.updates[0], {
    step: 0,
    speed_string: '',
    pid: null,
    stop: false,
    return_to_queue: false,
    save_now: false,
    status: 'queued',
    info: 'Restarting job from scratch...',
  });
  assert.equal(calls.startPreparedJob.length, 1);
  assert.equal(calls.startPreparedJob[0].prepared.job.step, 0);
  assert.deepEqual(calls.startPreparedJob[0].options, {
    startQueue: true,
    queueInfo: 'Restarted from scratch and queued',
  });
});

test('restart from scratch clears provider metrics for local jobs', async () => {
  const { deps, calls } = makeDeps(makeJob());

  await restartTrainingJobFromScratch('job-1', {}, deps);

  assert.deepEqual(calls.deleteMetricsForJob, ['job-1']);
});

test('restart from scratch does not delete anything when encrypted validation fails', async () => {
  const { deps, calls } = makeDeps(makeJob(), {
    prepareJobStart: async () => {
      throw new JobStartError({ error: 'decryption key required' }, 409);
    },
  });

  await assert.rejects(
    () => restartTrainingJobFromScratch('job-1', {}, deps),
    error => error instanceof JobStartError && error.status === 409,
  );
  assert.equal(calls.removedPaths.length, 0);
  assert.equal(calls.deleteMetricsForJob.length, 0);
  assert.equal(calls.updates.length, 0);
  assert.equal(calls.startPreparedJob.length, 0);
});

test('restart from scratch forwards existing remote training jobs', async () => {
  const job = makeJob({ worker_id: 'worker-1', remote_job_id: 'remote-1' });
  const { deps, calls } = makeDeps(job);

  const restarted = await restartTrainingJobFromScratch('job-1', {}, deps);

  assert.equal(restarted.info, 'remote restarted');
  assert.equal(calls.removedPaths.length, 0);
  assert.equal(calls.deleteMetricsForJob.length, 0);
  assert.equal(calls.updates.length, 0);
  assert.equal(calls.remoteJson.length, 1);
  assert.equal(calls.remoteJson[0].routePath, '/api/jobs/remote-1/restart-from-scratch');
  assert.deepEqual(calls.ensureQueueRunning, [{ gpuIds: '0', workerId: 'worker-1' }]);
  assert.deepEqual(calls.syncRemoteJob, ['job-1']);
});
