import assert from 'node:assert/strict';
import test from 'node:test';
import { listJobsForJobsApi, decodeJobCursor } from '../dist/src/server/jobsApiList.js';
const rows = Array.from({ length: 10000 }, (_, index) => ({ id: String(10000-index).padStart(6, '0'), name: 'Run', created_at: new Date('2026-01-01'),
  worker_id: index % 2 ? 'remote' : 'local', status: index % 10 ? 'completed' : 'running',
  job_type: 'train', job_config: JSON.stringify({ config: { process: [{ train: { steps: 1000 }, datasets: [{ folder_path: 'private dataset' }] }] } }) }));
function fixture() {
  const calls = [];
  return { calls, deps: { runtimeGet: async () => null, listJobs: async options => {
    calls.push(options);
    return rows.filter(job => (!options.worker_id || options.worker_id === job.worker_id) &&
      (!options.status || options.status.includes(job.status)) && (!options.before || job.id < options.before.id)).slice(0, options.limit);
  } } };
}
test('10,000 jobs are read in capped stable pages without remote or filesystem work', async () => {
  const { calls, deps } = fixture();
  const first = await listJobsForJobsApi({ limit: '20' }, deps);
  const next = await listJobsForJobsApi({ limit: '20', cursor: first.nextCursor }, deps);
  assert.equal(first.jobs.length, 20); assert.equal(calls[0].limit, 21);
  assert.equal(new Set([...first.jobs, ...next.jobs].map(job => job.id)).size, 40);
  assert.ok(!JSON.stringify(first).includes('private dataset'));
  assert.equal(decodeJobCursor(first.nextCursor).id, first.jobs.at(-1).id);
});
test('active/local filters are pushed into the provider', async () => {
  const { calls, deps } = fixture();
  const page = await listJobsForJobsApi({ view: 'active', localOnly: true }, deps);
  assert.ok(page.jobs.every(job => job.status === 'running' && job.worker_id === 'local'));
  assert.equal(calls[0].worker_id, 'local'); assert.ok(calls[0].status.includes('starting'));
});
test('bad cursors, excessive limits, and obsolete scopes fail before reads', async () => {
  const deps = { listJobs: async () => assert.fail('unexpected read'), runtimeGet: async () => null };
  for (const options of [{ scope: 'all' }, { cursor: 'invalid' }, { limit: '10000' }, { view: 'unknown' }]) await assert.rejects(listJobsForJobsApi(options, deps));
});

test('full-configuration worker pages respect the transport byte budget and retain a continuation cursor', async () => {
  const large = rows.slice(0, 10).map(job => ({ ...job, job_config: 'x'.repeat(2 * 1024 ** 2) }));
  const result = await listJobsForJobsApi({ limit: '10', summary: false }, { runtimeGet: async () => null, listJobs: async () => large });
  assert.equal(result.jobs.length, 3);
  assert.ok(Buffer.byteLength(JSON.stringify(result)) < 8 * 1024 ** 2);
  assert.equal(decodeJobCursor(result.nextCursor).id, result.jobs.at(-1).id);
});
