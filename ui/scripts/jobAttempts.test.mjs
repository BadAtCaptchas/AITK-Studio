import assert from 'node:assert/strict';
import { test } from 'node:test';
import { db } from '../dist/src/server/db.js';
import { installMemoryRuntime } from './memoryRuntimeFixture.mjs';
import { claimJobAttempt, claimJobMaintenance } from '../dist/src/server/jobAttempts.js';
import { finishObservedJobProcess } from '../dist/src/server/jobProcess.js';
import { deviceIds, jobStorageKey, validJobName } from '../dist/src/utils/jobIdentity.js';

function fixture() {
  installMemoryRuntime();
  const jobs = new Map();
  let revision = Date.now();
  const job = (id, gpu_ids = '0') => {
    const value = {
      id,
      name: id,
      storage_key: `store-${id}`,
      worker_id: 'local',
      gpu_ids,
      job_config: '{"config":{"process":[]}}',
      status: 'stopped',
      attempt_id: null,
      pid: null,
      updated_at: new Date(++revision),
      stop: false,
      return_to_queue: false,
    };
    jobs.set(id, value);
    return structuredClone(value);
  };
  db.jobs.findById = async id => structuredClone(jobs.get(id) ?? null);
  db.jobs.list = async options =>
    [...jobs.values()]
      .filter(
        row =>
          !options.status || (Array.isArray(options.status) ? options.status : [options.status]).includes(row.status),
      )
      .map(row => structuredClone(row));
  db.jobs.updateIf = async (id, expected, patch) => {
    const current = jobs.get(id);
    if (
      !current ||
      current.attempt_id !== expected.attempt_id ||
      (expected.status &&
        !(Array.isArray(expected.status) ? expected.status : [expected.status]).includes(current.status)) ||
      (expected.updated_at && +new Date(expected.updated_at) !== +new Date(current.updated_at))
    )
      return null;
    const updated = { ...current, ...patch, updated_at: new Date(++revision) };
    jobs.set(id, updated);
    return structuredClone(updated);
  };
  return { job, jobs };
}

test('simultaneous starts accept one attempt and one overlapping device owner', async () => {
  const { job } = fixture();
  const first = job('one', '0,1');
  const claims = await Promise.all([claimJobAttempt(first), claimJobAttempt(first)]);
  assert.equal(claims.filter(Boolean).length, 1);
  assert.equal(await claimJobAttempt(job('overlap', '1,2')), null);
  assert.ok(await claimJobAttempt(job('independent', '2')));
});

test('maintenance and start cannot both own a job; old process exit cannot overwrite a newer attempt', async () => {
  const { job, jobs } = fixture();
  const original = job('job');
  const [maintenance, start] = await Promise.all([
    claimJobMaintenance(original, 'deleting'),
    claimJobAttempt(original),
  ]);
  assert.equal([maintenance, start].filter(Boolean).length, 1);
  jobs.set('job', { ...jobs.get('job'), attempt_id: 'newer', status: 'running', pid: 999 });
  assert.equal(await finishObservedJobProcess(start || maintenance, 'error', 'old exit'), null);
  assert.equal(jobs.get('job').status, 'running');
});

test('reported completion cannot release devices while the actual process is alive', async () => {
  const { job, jobs } = fixture();
  const owned = await claimJobAttempt(job('one'));
  jobs.set('one', { ...owned, status: 'completed', pid: process.pid });
  assert.equal(await claimJobAttempt(jobs.get('one')), null);
  assert.equal(await claimJobAttempt(job('other')), null);
});

test('portable identities preserve launch device order and storage across rename', () => {
  assert.deepEqual(deviceIds('01,0,1'), ['1', '0']);
  assert.equal(jobStorageKey({ name: 'new', storage_key: 'old' }), 'old');
  for (const name of ['CON', 'aux.txt', 'name.', 'name ', '../escape']) assert.equal(validJobName(name), false);
});
