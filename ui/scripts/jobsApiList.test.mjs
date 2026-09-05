import assert from 'node:assert/strict';
import test from 'node:test';

import { listJobsForJobsApi } from '../dist/src/server/jobsApiList.js';

function makeJob(overrides = {}) {
  const now = new Date();
  return {
    id: 'job-1',
    name: 'Job 1',
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

  assert.deepEqual(calls, [['list', { job_type: 'caption' }]]);
  assert.deepEqual(
    jobs.map(job => job.id),
    ['local-job', 'legacy-local-job'],
  );
});

test('obsolete listing arguments are refused before discovery or storage reads', async () => {
  for (const options of [{ scope: 'all' }, { project_id: 'old' }, { includeProjectActive: true }]) {
    const unexpected = async () => {
      assert.fail('No side effects for obsolete scope');
    };
    await assert.rejects(
      () =>
        listJobsForJobsApi(options, {
          listJobs: unexpected,
          discoverRemoteJobs: unexpected,
          syncRemoteJobs: unexpected,
        }),
      /removed/,
    );
  }
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

  assert.deepEqual(
    jobs.map(job => job.id),
    ['local-job'],
  );
  assert.deepEqual(calls[0], ['discover', 'training']);
  assert.deepEqual(calls[1], ['list', { job_type: 'training' }]);
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

  assert.deepEqual(
    jobs.map(job => job.id),
    ['local-job', 'cached-mirror'],
  );
  assert.deepEqual(calls, [
    ['discover', 'training'],
    ['list', { job_type: 'training' }],
    ['sync', ['local-job', 'cached-mirror']],
  ]);
});

test('legacy scoped mirrors never enter the global listing', async () => {
  const global = makeJob({ id: 'global' });
  const legacy = makeJob({ id: 'old', project_id: 'former' });
  const jobs = await listJobsForJobsApi({ localOnly: true }, { listJobs: async () => [global, legacy] });
  assert.deepEqual(
    jobs.map(job => job.id),
    ['global'],
  );
});
