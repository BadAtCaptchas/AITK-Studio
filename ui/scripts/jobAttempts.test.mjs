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


test('dataset relinking previews exact references and rejects active or stale jobs independently', async t => {
  const fs = await import('node:fs/promises');
  const os = await import('node:os');
  const path = await import('node:path');
  const { relinkDecryptedDataset, replaceDatasetReferences } = await import('../dist/src/server/datasetRelinking.js');
  const { enqueueOperation, updateOperation } = await import('../dist/src/server/operations.js');
  const { flushCache } = await import('../dist/src/server/settings.js');
  const { job, jobs } = fixture();
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-relink-'));
  const oldGet = db.settings.get, auth = process.env.AI_TOOLKIT_AUTH;
  process.env.AI_TOOLKIT_AUTH = 'test-only';
  db.settings.get = async key => ({ key, value: root });
  flushCache();
  t.after(async () => { db.settings.get = oldGet; if (auth === undefined) delete process.env.AI_TOOLKIT_AUTH; else process.env.AI_TOOLKIT_AUTH = auth; flushCache(); await fs.rm(root, { recursive: true, force: true }); });
  const source = path.join(root, 'locked'), destination = path.join(root, 'plain');
  await fs.mkdir(source); await fs.mkdir(destination);
  const operation = await enqueueOperation({ kind: 'dataset-decrypt', root, sourceName: 'locked', outputName: 'plain' });
  await assert.rejects(relinkDecryptedDataset(operation.id, 'local'), /Complete dataset decryption/);
  await updateOperation(operation.id, { state: 'completed', result: { sourceName: 'locked', datasetName: 'plain', itemCount: 1, captionCount: 1, renamed: [] } });
  for (const id of ['idle', 'queued', 'active', 'stale']) {
    job(id);
    jobs.set(id, { ...jobs.get(id), status: id === 'active' ? 'running' : id === 'queued' ? 'queued' : 'stopped', step: 42, job_config: JSON.stringify({ config: { process: [{ datasets: [{ folder_path: [source, source + '-other'], encrypted: true }], sample: { prompt: source } }] } }) });
  }
  const preview = await relinkDecryptedDataset(operation.id, 'local');
  assert.equal(preview.jobs.length, 4);
  for (const candidate of preview.jobs) assert.deepEqual(candidate.replacements, [{ field: 'config.process[0].datasets[0].folder_path[0]', from: source, to: destination }]);
  jobs.set('stale', { ...jobs.get('stale'), updated_at: new Date(Date.now() + 10000) });
  const applied = await relinkDecryptedDataset(operation.id, 'local', preview.jobs.map(value => ({ id: value.id, revision: value.revision })));
  assert.deepEqual(applied.outcomes.filter(value => value.success).map(value => value.id), ['idle', 'queued']);
  assert.equal(jobs.get('queued').status, 'stopped');
  assert.equal(jobs.get('active').status, 'running');
  assert.equal(jobs.get('idle').step, 42);
  assert.equal(jobs.get('idle').storage_key, 'store-idle');
  const updated = JSON.parse(jobs.get('idle').job_config).config.process[0];
  assert.deepEqual(updated.datasets[0].folder_path, [destination, source + '-other']);
  assert.equal(updated.sample.prompt, source);
  assert.equal(updated.datasets[0].encrypted, undefined);
  const remoteConfig = { config: { process: [{ datasets: [{ folder_path: ['aitk-dataset://remote/w/old', 'aitk-dataset://remote/other/old'] }] }] } };
  await replaceDatasetReferences(remoteConfig, async value => value === 'aitk-dataset://remote/w/old' ? 'aitk-dataset://remote/w/new' : null);
  assert.deepEqual(remoteConfig.config.process[0].datasets[0].folder_path, ['aitk-dataset://remote/w/new', 'aitk-dataset://remote/other/old']);

  const remoteClient = (await import('../dist/src/server/remoteClient.js')).default;
  let remoteRevision = 'revision-1', failRemote = false, applies = 0;
  t.mock.method(remoteClient, 'getRemoteWorker', async () => ({ id: 'worker', base_url: 'https://worker.invalid' }));
  t.mock.method(remoteClient, 'remoteJson', async (_worker, route, init) => {
    assert.equal(init.redirect, 'manual');
    if (route.startsWith('/api/operations/')) return { kind: 'dataset-decrypt', state: 'completed', result: { sourceName: 'locked', datasetName: 'plain' } };
    const body = JSON.parse(init.body);
    if (body.action === 'preview') return { jobs: [{ id: 'remote-job', revision: remoteRevision, replacements: [{ field: 'config.process[0].datasets[0].folder_path', from: '/worker/datasets/locked', to: '/worker/datasets/plain' }] }] };
    applies++;
    assert.equal(body.jobs[0].revision, remoteRevision);
    return { outcomes: [{ id: 'remote-job', success: !failRemote }] };
  });
  job('mirror');
  jobs.set('mirror', { ...jobs.get('mirror'), worker_id: 'worker', remote_job_id: 'remote-job', status: 'queued', step: 73, job_config: JSON.stringify({ config: { process: [{ datasets: [{ folder_path: '/worker/datasets/locked' }] }] } }) });
  let remotePreview = await relinkDecryptedDataset(operation.id, 'worker');
  assert.equal(remotePreview.jobs[0].remoteRevision, remoteRevision);
  remoteRevision = 'revision-2';
  let remoteApply = await relinkDecryptedDataset(operation.id, 'worker', remotePreview.jobs);
  assert.equal(remoteApply.outcomes[0].success, false); assert.equal(applies, 0);
  assert.equal(jobs.get('mirror').status, 'stopped');
  remotePreview = await relinkDecryptedDataset(operation.id, 'worker'); failRemote = true;
  remoteApply = await relinkDecryptedDataset(operation.id, 'worker', remotePreview.jobs);
  assert.equal(remoteApply.outcomes[0].success, false);
  assert.match(jobs.get('mirror').job_config, /locked/);
  remotePreview = await relinkDecryptedDataset(operation.id, 'worker'); failRemote = false;
  remoteApply = await relinkDecryptedDataset(operation.id, 'worker', remotePreview.jobs);
  assert.equal(remoteApply.outcomes[0].success, true);
  assert.match(jobs.get('mirror').job_config, /datasets\/plain/); assert.equal(jobs.get('mirror').step, 73);
});
