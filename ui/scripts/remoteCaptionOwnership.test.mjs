import assert from 'node:assert/strict';
import test from 'node:test';
import { db } from '../dist/src/server/db.js';
import { installMemoryRuntime } from './memoryRuntimeFixture.mjs';
import { syncRemoteCaptionResultForJob } from '../dist/src/server/remoteCaptionResults.js';
import { withProcessLease } from '../dist/src/server/processLease.js';

test('completed caption publication recovers an interrupted maintenance claim without stealing a live merger', async () => {
  const fixture = installMemoryRuntime(), original = { ...db.jobs };
  let job = { id: 'caption-recovery', status: 'editing', attempt_id: 'old-merge', pid: null, worker_id: 'remote',
    remote_job_id: 'remote-job', updated_at: new Date(), job_config: JSON.stringify({ config: {
      remote_caption: { version: 1, downloadStatus: 'merged' }, process: [] } }) };
  db.jobs.findById = async () => structuredClone(job);
  db.jobs.updateIf = async (_id, expected, patch) => {
    if (expected.attempt_id !== job.attempt_id || expected.status !== job.status) return null;
    job = { ...job, ...patch, updated_at: new Date() }; return structuredClone(job);
  };
  try {
    await db.runtime.compareAndSwap('caption-result-owner:' + job.id, null, job.attempt_id);
    await withProcessLease('caption-result:' + job.id, async () => {
      assert.equal((await syncRemoteCaptionResultForJob(job)).status, 'editing');
      assert.equal(job.attempt_id, 'old-merge');
    });
    assert.equal((await syncRemoteCaptionResultForJob(job)).status, 'completed');
    assert.equal(await db.runtime.get('caption-result-owner:' + job.id), null);
  } finally { Object.assign(db.jobs, original); fixture.restore(); }
});
