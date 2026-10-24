import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { migration, trainingFixture } from './trainingMigrationFixture.mjs';

function loadSource(relative, dependencies = {}) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const evaluatedModule = { exports: {} };
  new Function('require', 'module', 'exports', code)(
    name => {
      assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency: ${name}`);
      return dependencies[name];
    },
    evaluatedModule,
    evaluatedModule.exports,
  );
  return evaluatedModule.exports;
}

const obsoleteWorkspaceGuard = loadSource('../src/utils/obsoleteWorkspaceGuard.ts');
const commandInput = loadSource('../src/server/commandInput.ts', {
  '../utils/obsoleteWorkspaceGuard': obsoleteWorkspaceGuard,
});
const configContract = loadSource('../src/domain/configContract.ts', {
  './loraContinuation': loadSource('../src/domain/loraContinuation.ts'),
  '../utils/trainingPhases': loadSource('../src/utils/trainingPhases.ts'),
  './modelCapabilities.json': JSON.parse(
    fs.readFileSync(new URL('../src/domain/modelCapabilities.json', import.meta.url), 'utf8'),
  ),
});
const jobIdentity = loadSource('../src/utils/jobIdentity.ts');
const validationConfig = loadSource('../src/utils/validationConfig.ts');
const workerTrainingRoot = 'D:\\worker-training';
const trainingPaths = loadSource('../src/server/trainingPaths.ts', {
  './settings': { getTrainingFolder: async () => workerTrainingRoot },
  './commandInput': commandInput,
});

// Exercise the POST handler and real input validation without accessing the user's DB.
function fixture(existing = null) {
  const created = [];
  const updated = [];
  const lookedUp = [];
  const remoteRequests = [];
  const worker = { id: 'worker-1', enabled: true };
  const { POST } = loadSource('../src/app/api/jobs/route.ts', {
    '@/domain/configContract': configContract,
    '@/utils/obsoleteWorkspaceGuard': obsoleteWorkspaceGuard,
    '@/server/commandInput': commandInput,
    '@/utils/jobIdentity': jobIdentity,
    '@/utils/validationConfig': validationConfig,
    '@/helpers/basic': { isMac: () => false },
    '@/utils/authSession': { isRequestAuthenticated: async () => true },
    '@/server/db': {
      db: {
        workerNodes: { findById: async () => worker },
        jobs: {
          maxQueuePosition: async () => 2000,
          findByName: async name => (existing?.name === name ? existing : null),
          findById: async id => {
            lookedUp.push(id);
            return existing?.id === id ? existing : null;
          },
          create: async data => {
            const job = { ...data, id: 'new-job' };
            created.push(job);
            return job;
          },
          updateIf: async (id, condition, data) => {
            updated.push({ id, condition, data });
            return { ...existing, ...data, id };
          },
        },
      },
    },
    '@/server/jobAttempts': { claimJobMaintenance: async job => ({ ...job, status: 'editing' }) },
    '@/server/jobCreateCommand': { createJobWithReceipt: async (key, _input, prepare) => {
      assert.equal(key, undefined);
      const job = { ...await prepare(), id: 'new-job' };
      created.push(job);
      return job;
    } },
    '@/server/remoteClient': {
      isLocalWorker: id => !id || id === 'local',
      getRemoteWorker: async () => worker,
      remoteJson: async (_worker, route, init) => {
        remoteRequests.push({ route, body: JSON.parse(init.body) });
        return { name: 'new_run', gpu_ids: '0' };
      },
    },
    '@/server/comfyInstallProgress': {},
    '@/server/hfDownloadProgress': {},
    '@/server/jobsApiList': {},
    '@/server/remoteDatasetPaths': { rewriteSameWorkerRemoteDatasetRefsForWorker: async config => config },
    '@/server/trainingPaths': trainingPaths,
  });
  return {
    created,
    updated,
    lookedUp,
    remoteRequests,
    post: body => POST(new Request('http://localhost/api/jobs', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    })),
  };
}

function payload(jobType = 'train') {
  return {
    name: 'new_run',
    worker_id: 'local',
    gpu_ids: '0',
    job_type: jobType,
    job_config: {
      job: 'extension',
      config: {
        name: 'new_run',
        process: [{ type: jobType === 'caption' ? 'OpenRouterCaptioner' : 'sd_trainer' }],
      },
    },
  };
}

for (const jobType of ['train', 'caption']) {
  for (const idFields of [{}, { id: null }]) {
    test(`creates a ${jobType} job with ${Object.hasOwn(idFields, 'id') ? 'a null' : 'an omitted'} id`, async () => {
      const state = fixture();
      const body = { ...payload(jobType), ...idFields };
      const response = await state.post(body);
      assert.equal(response.status, 200);
      const saved = await response.json();
      assert.equal(saved.id, 'new-job');
      assert.equal(saved.name, body.name);
      assert.equal(saved.job_type, jobType);
      assert.equal(saved.queue_position, 3000);
      assert.deepEqual(JSON.parse(saved.job_config), {
        ...body.job_config,
        capability_version: 1,
        config: { ...body.job_config.config, process: [{ ...body.job_config.config.process[0], training_folder: workerTrainingRoot }] },
      });
      assert.equal(state.created.length, 1);
      assert.deepEqual(state.lookedUp, []);
      assert.deepEqual(state.updated, []);
    });
  }
}

test('saving rejects removed automatic training before local or remote writes', async () => {
  for (const train of [
    { auto_train: true, steps: 20 }, { auto_learn: true, steps: 20 },
    { steps: 20, phases: [{ steps: 20, auto_advance: {} }] },
    { steps: 20, phases: [{ steps: 20, auto_advance: null }] },
  ]) {
    for (const existing of [null, { id: 'mirror', name: 'old_run', status: 'stopped', worker_id: 'worker-1', remote_job_id: 'remote-job' }]) {
      const state = fixture(existing);
      const body = { ...payload(), id: existing?.id, worker_id: existing?.worker_id || 'local' };
      body.job_config.config.process[0].train = train;
      const response = await state.post(body);
      assert.equal(response.status, 400);
      assert.match((await response.json()).error, /no longer supported/);
      assert.deepEqual(state.created, []);
      assert.deepEqual(state.updated, []);
      assert.deepEqual(state.remoteRequests, []);
    }
  }
});

test('a string id updates the existing job without creating another job', async () => {
  const state = fixture({ id: 'existing-job', name: 'old_run', status: 'stopped', worker_id: 'local' });
  const response = await state.post({ ...payload(), id: 'existing-job' });
  assert.equal(response.status, 200);
  const saved = await response.json();
  assert.equal(saved.id, 'existing-job');
  assert.equal(saved.name, 'new_run');
  assert.equal(saved.storage_key, 'old_run');
  assert.deepEqual(state.lookedUp, ['existing-job']);
  assert.equal(state.updated.length, 1);
  assert.deepEqual(state.created, []);
});

test('an unknown string id returns 404 without creating a job', async () => {
  const state = fixture();
  const response = await state.post({ ...payload(), id: 'missing-job' });
  assert.equal(response.status, 404);
  assert.deepEqual(await response.json(), { error: 'Job not found' });
  assert.deepEqual(state.created, []);
  assert.deepEqual(state.updated, []);
});

test('new remote jobs save before their model is downloaded and without contacting the worker', async () => {
  const state = fixture();
  const body = { ...payload(), worker_id: 'worker-1' };
  body.job_config.config.process[0].model = { arch: 'flux', name_or_path: 'fixture/not-downloaded-yet' };
  body.job_config.config.process[0].datasets = [{ folder_path: 'aitk-dataset://remote/worker-1/cats', resolution: [512] }];
  const response = await state.post(body);
  assert.equal(response.status, 200);
  assert.equal(state.created.length, 1);
  assert.deepEqual(state.remoteRequests, []);
  assert.equal(JSON.parse((await response.json()).job_config).config.process[0].model.name_or_path, 'fixture/not-downloaded-yet');
});

test('non-string, non-null ids remain invalid and cannot create or update jobs', async () => {
  for (const id of [0, 123, false, true, [], {}, ['existing-job']]) {
    const state = fixture();
    const response = await state.post({ ...payload(), id });
    assert.equal(response.status, 400);
    assert.deepEqual(await response.json(), { error: 'Invalid job id' });
    assert.deepEqual(state.created, []);
    assert.deepEqual(state.updated, []);
    assert.deepEqual(state.lookedUp, []);
  }
});

test('worker edits retain the imported training root instead of a controller path', async () => {
  const originalRoot = 'E:\\previous-worker-output';
  const existing = {
    id: 'remote-job', name: 'old_run', status: 'stopped', worker_id: 'local',
    job_config: JSON.stringify({ config: { process: [{ training_folder: originalRoot }] } }),
  };
  for (const incomingRoot of ['C:\\controller-output', undefined]) {
    const state = fixture(existing);
    const body = { ...payload(), id: existing.id };
    body.job_config.config.process[0].training_folder = incomingRoot;
    body.job_config.config.process[0].train = { steps: 7000 };
    const response = await state.post(body);
    assert.equal(response.status, 200);
    const saved = await response.json();
    assert.equal(await trainingPaths.getJobTrainingRoot(saved), originalRoot);
    assert.equal(JSON.parse(saved.job_config).config.process[0].train.steps, 7000);
  }
});

test('new local jobs and jobs moved from a remote worker use the configured local training root', async () => {
  for (const existing of [null, {
    id: 'remote-mirror', name: 'old_run', status: 'stopped', worker_id: 'worker-1', remote_job_id: 'remote-job',
    job_config: JSON.stringify({ config: { process: [{ training_folder: '/remote/output' }] } }),
  }]) {
    const state = fixture(existing);
    const body = { ...payload(), id: existing?.id };
    body.job_config.config.process[0].training_folder = '/another-machine/output';
    const response = await state.post(body);
    assert.equal(response.status, 200);
    assert.equal(await trainingPaths.getJobTrainingRoot(await response.json()), workerTrainingRoot);
    assert.deepEqual(state.remoteRequests, []);
  }
});

test('remote edits omit controller runtime paths and preserve the local mirror config', async () => {
  const state = fixture({ id: 'mirror', name: 'old_run', status: 'stopped', worker_id: 'worker-1', remote_job_id: 'remote-job' });
  const body = { ...payload(), id: 'mirror', worker_id: 'worker-1' };
  body.job_config.config.process = [
    { type: 'sd_trainer', training_folder: 'C:\\controller-output', sqlite_db_path: 'C:\\controller.db', train: { steps: 7000 } },
    { type: 'sd_trainer', training_folder: 'C:\\controller-output', sqlite_db_path: 'C:\\controller.db' },
  ];
  const response = await state.post(body);
  assert.equal(response.status, 200);
  assert.equal(state.remoteRequests.length, 1);
  const sent = state.remoteRequests[0].body;
  assert.equal(sent.id, 'remote-job');
  for (const process of sent.job_config.config.process) {
    assert.equal(Object.hasOwn(process, 'training_folder'), false);
    assert.equal(Object.hasOwn(process, 'sqlite_db_path'), false);
  }
  assert.equal(sent.job_config.config.process[0].train.steps, 7000);
  assert.deepEqual(JSON.parse((await response.json()).job_config).config.process, body.job_config.config.process);
});

test('migrated local and remote jobs use creation without inheriting or updating the running source', async () => {
  for (const worker_id of ['local', 'worker-1']) {
    const source = trainingFixture();
    source.config.process[0].name = 'original-storage';
    source.config.process[0].train.start_step = 700;
    source.config.process[0].network.pretrained_lora_path = 'source.safetensors';
    const existing = { id: 'source-job', name: source.config.name, status: 'running', step: 700, worker_id, remote_job_id: 'source-remote', storage_key: 'original-storage', job_config: JSON.stringify(source) };
    const before = structuredClone(existing);
    const state = fixture(existing);
    const migrated = migration.migrateTrainingModel(source, 'zimage').config;
    migrated.config.name = 'migrated_run';
    const response = await state.post({ name: migrated.config.name, job_config: migrated, worker_id, gpu_ids: '0' });
    assert.equal(response.status, 200);
    const saved = await response.json();
    assert.equal(saved.id, 'new-job');
    for (const key of ['step', 'status', 'storage_key', 'attempt_id', 'remote_job_id']) assert.equal(state.created[0][key], undefined, `${key} must use new-job defaults`);
    const config = JSON.parse(saved.job_config);
    assert.equal(config.config.process[0].train.start_step, undefined);
    assert.equal(config.config.process[0].network.pretrained_lora_path, undefined);
    jobIdentity.applyJobOutputIdentity(config, { name: saved.name, storage_key: 'job-new-id' });
    assert.equal(config.config.process[0].name, 'job-new-id');
    assert.deepEqual(state.updated, []);
    assert.deepEqual(state.remoteRequests, []);
    assert.deepEqual(existing, before);
  }
});
