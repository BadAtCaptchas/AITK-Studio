import assert from 'node:assert/strict';
import test from 'node:test';

import { listJobsForJobsApi } from '../dist/src/server/jobsApiList.js';

function makeJob(overrides = {}) {
  const now = new Date();
  return {
    id: 'job-1',
    name: 'Job 1',
    project_id: null,
    worker_id: 'local',
    remote_job_id: null,
    remote_sync_at: null,
    remote_error: null,
    gpu_ids: '0',
    job_config: '{}',
    created_at: now,
    updated_at: now,
    status: 'queued',
    stop: false,
    return_to_queue: false,
    step: 0,
    info: '',
    speed_string: '',
    queue_position: 0,
    pid: null,
    job_type: 'caption',
    job_ref: null,
    save_now: false,
    ...overrides,
  };
}

test('local-only jobs API listing returns only local jobs without remote sync', async () => {
  const calls = [];
  const localJob = makeJob({ id: 'local-job', worker_id: 'local' });
  const legacyLocalJob = makeJob({ id: 'legacy-local-job', worker_id: '' });
  const mirrorJob = makeJob({ id: 'mirror-job', worker_id: 'worker-1', remote_job_id: 'remote-job' });

  const jobs = await listJobsForJobsApi(
    { jobType: 'caption', localOnly: true },
    {
      listJobs: async options => {
        calls.push(['list', options]);
        return [localJob, mirrorJob, legacyLocalJob];
      },
      discoverRemoteJobs: async () => {
        throw new Error('local-only listing must not discover remote jobs');
      },
      syncRemoteJobs: async () => {
        throw new Error('local-only listing must not sync remote mirrors');
      },
    },
  );

  assert.deepEqual(calls, [['list', { job_type: 'caption', project_id: null }]]);
  assert.deepEqual(jobs.map(job => job.id), ['local-job', 'legacy-local-job']);
});

test('jobs API listing can include active project jobs for queue surfaces', async () => {
  const calls = [];
  const globalStopped = makeJob({ id: 'global-stopped', job_type: 'train', status: 'stopped' });
  const globalQueued = makeJob({ id: 'global-queued', job_type: 'train', status: 'queued' });
  const projectRunning = makeJob({
    id: 'project-running',
    job_type: 'train',
    project_id: 'project-1',
    status: 'running',
  });
  const projectStopped = makeJob({
    id: 'project-stopped',
    job_type: 'train',
    project_id: 'project-1',
    status: 'stopped',
  });
  const discoveredJobIds = new Set(['project-running']);

  const jobs = await listJobsForJobsApi(
    { jobType: 'train', includeProjectActive: true },
    {
      discoverRemoteJobs: async jobType => {
        calls.push(['discover', jobType]);
        return discoveredJobIds;
      },
      listJobs: async options => {
        calls.push(['list', options]);
        if (Array.isArray(options.status)) {
          return [globalQueued, projectRunning, projectStopped].filter(job => options.status.includes(job.status));
        }
        return [globalStopped, globalQueued];
      },
      syncRemoteJobs: async (listedJobs, alreadySyncedJobIds) => {
        calls.push(['sync', listedJobs.map(job => job.id), alreadySyncedJobIds]);
        return listedJobs;
      },
    },
  );

  assert.deepEqual(jobs.map(job => job.id), ['global-stopped', 'global-queued', 'project-running']);
  assert.deepEqual(calls[0], ['discover', 'train']);
  assert.deepEqual(calls[1], ['list', { job_type: 'train', project_id: null }]);
  assert.deepEqual(calls[2], ['list', { job_type: 'train', status: ['queued', 'running', 'stopping'] }]);
  assert.deepEqual(calls[3][0], 'sync');
  assert.deepEqual(calls[3][1], ['global-stopped', 'global-queued', 'project-running']);
  assert.equal(calls[3][2], discoveredJobIds);
});

test('local-only active project listing filters merged jobs without remote sync', async () => {
  const calls = [];
  const localGlobalJob = makeJob({ id: 'local-global', worker_id: 'local', status: 'queued' });
  const localProjectJob = makeJob({
    id: 'local-project',
    project_id: 'project-1',
    worker_id: 'local',
    status: 'running',
  });
  const remoteProjectJob = makeJob({
    id: 'remote-project',
    project_id: 'project-1',
    worker_id: 'worker-1',
    status: 'running',
  });

  const jobs = await listJobsForJobsApi(
    { jobType: 'caption', localOnly: true, includeProjectActive: true },
    {
      listJobs: async options => {
        calls.push(['list', options]);
        if (Array.isArray(options.status)) {
          return [localProjectJob, remoteProjectJob];
        }
        return [localGlobalJob];
      },
      discoverRemoteJobs: async () => {
        throw new Error('local-only listing must not discover remote jobs');
      },
      syncRemoteJobs: async () => {
        throw new Error('local-only listing must not sync remote mirrors');
      },
    },
  );

  assert.deepEqual(calls, [
    ['list', { job_type: 'caption', project_id: null }],
    ['list', { job_type: 'caption', status: ['queued', 'running', 'stopping'] }],
  ]);
  assert.deepEqual(jobs.map(job => job.id), ['local-global', 'local-project']);
});

test('jobs API listing still discovers before syncing remote mirrors', async () => {
  const calls = [];
  const localJob = makeJob({ id: 'local-job' });
  const discoveredJobIds = new Set(['mirror-job']);

  const jobs = await listJobsForJobsApi(
    { jobType: 'training', localOnly: false },
    {
      discoverRemoteJobs: async jobType => {
        calls.push(['discover', jobType]);
        return discoveredJobIds;
      },
      listJobs: async options => {
        calls.push(['list', options]);
        return [localJob];
      },
      syncRemoteJobs: async (listedJobs, alreadySyncedJobIds) => {
        calls.push(['sync', listedJobs.map(job => job.id), alreadySyncedJobIds]);
        return listedJobs;
      },
    },
  );

  assert.deepEqual(jobs.map(job => job.id), ['local-job']);
  assert.deepEqual(calls[0], ['discover', 'training']);
  assert.deepEqual(calls[1], ['list', { job_type: 'training', project_id: null }]);
  assert.deepEqual(calls[2][0], 'sync');
  assert.deepEqual(calls[2][1], ['local-job']);
  assert.equal(calls[2][2], discoveredJobIds);
});

test('jobs API listing returns cached remote mirrors when remote polling is skipped', async () => {
  const calls = [];
  const localJob = makeJob({ id: 'local-job', worker_id: 'local' });
  const cachedMirrorJob = makeJob({ id: 'cached-mirror', worker_id: 'worker-1', remote_job_id: 'remote-job' });

  const jobs = await listJobsForJobsApi(
    { jobType: 'training', localOnly: false },
    {
      discoverRemoteJobs: async jobType => {
        calls.push(['discover', jobType]);
        return new Set();
      },
      listJobs: async options => {
        calls.push(['list', options]);
        return [localJob, cachedMirrorJob];
      },
      syncRemoteJobs: async listedJobs => {
        calls.push(['sync', listedJobs.map(job => job.id)]);
        return listedJobs;
      },
    },
  );

  assert.deepEqual(jobs.map(job => job.id), ['local-job', 'cached-mirror']);
  assert.deepEqual(calls, [
    ['discover', 'training'],
    ['list', { job_type: 'training', project_id: null }],
    ['sync', ['local-job', 'cached-mirror']],
  ]);
});
