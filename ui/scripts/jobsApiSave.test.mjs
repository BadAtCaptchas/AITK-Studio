import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

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
  './modelCapabilities.json': JSON.parse(
    fs.readFileSync(new URL('../src/domain/modelCapabilities.json', import.meta.url), 'utf8'),
  ),
});
const jobIdentity = loadSource('../src/utils/jobIdentity.ts');
const validationConfig = loadSource('../src/utils/validationConfig.ts');

// Exercise the POST handler and real input validation without accessing the user's DB.
function fixture(existing = null) {
  const created = [];
  const updated = [];
  const lookedUp = [];
  const { POST } = loadSource('../src/app/api/jobs/route.ts', {
    '@/domain/configContract': configContract,
    '@/utils/obsoleteWorkspaceGuard': obsoleteWorkspaceGuard,
    '@/server/commandInput': commandInput,
    '@/utils/jobIdentity': jobIdentity,
    '@/utils/validationConfig': validationConfig,
    'next/server': { NextResponse: Response },
    '@/helpers/basic': { isMac: () => false },
    '@/utils/authSession': { isRequestAuthenticated: async () => true },
    '@/server/db': {
      db: {
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
    '@/server/remoteClient': { isLocalWorker: id => id === 'local' },
    '@/server/comfyInstallProgress': {},
    '@/server/hfDownloadProgress': {},
    '@/server/jobsApiList': {},
    '@/server/remoteDatasetPaths': {},
  });
  return {
    created,
    updated,
    lookedUp,
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
      assert.deepEqual(JSON.parse(saved.job_config), { ...body.job_config, capability_version: 1 });
      assert.equal(state.created.length, 1);
      assert.deepEqual(state.lookedUp, []);
      assert.deepEqual(state.updated, []);
    });
  }
}

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
