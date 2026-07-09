import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { setupProjectWorkspace } from '../dist/src/server/projectSetup.js';
import { PROJECT_FOLDERS } from '../dist/src/server/projects.js';

function rootsFor(root) {
  return PROJECT_FOLDERS.reduce(
    (roots, zone) => {
      roots[zone] = path.join(root, zone);
      return roots;
    },
    { root },
  );
}

async function createWorkspace(t, name) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), `aitk-project-setup-${name}-`));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  return root;
}

async function ensureProjectZones(root) {
  await Promise.all(PROJECT_FOLDERS.map(zone => fs.mkdir(path.join(root, zone), { recursive: true })));
}

function project(id, slug, root) {
  const now = new Date('2026-01-01T00:00:00.000Z');
  return {
    id,
    slug,
    name: slug,
    description: '',
    badge_asset: null,
    root_path: root,
    storage_root_path: path.dirname(root),
    lifecycle_state: 'active',
    archived_at: null,
    revision: 1,
    operation_started_at: null,
    operation_error: null,
    home_worker_id: 'local',
    home_instance_id: 'test-instance',
    created_at: now,
    updated_at: now,
  };
}

function job(id, projectID, overrides = {}) {
  const now = new Date('2026-01-01T00:00:00.000Z');
  return {
    id,
    name: id,
    project_id: projectID,
    worker_id: 'worker-old',
    remote_job_id: 'remote-old',
    remote_sync_at: now,
    remote_error: 'old remote state',
    gpu_ids: '0',
    job_config: '{}',
    created_at: now,
    updated_at: now,
    status: 'running',
    stop: true,
    return_to_queue: true,
    step: 281,
    info: 'active state',
    speed_string: '3.49 sec/iter',
    queue_position: 9,
    pid: 4321,
    job_type: 'train',
    job_ref: null,
    save_now: true,
    ...overrides,
  };
}

function createDeps({ projects, jobs = [], failCreateAt = null }) {
  const projectByID = new Map(projects.map(item => [item.id, item]));
  const jobStore = new Map(jobs.map(item => [item.id, item]));
  const createInputs = [];
  const operationErrors = new Map();
  let createdCount = 0;

  return {
    jobStore,
    createInputs,
    operationErrors,
    deps: {
      async getProjectRoots(item) {
        return rootsFor(item.root_path);
      },
      async resolveProject(identifier) {
        const found = projectByID.get(identifier) || [...projectByID.values()].find(item => item.slug === identifier);
        if (!found) throw new Error(`Project not found: ${identifier}`);
        return found;
      },
      async listProjectJobs(projectID) {
        return [...jobStore.values()].filter(item => item.project_id === projectID);
      },
      async createJob(input) {
        createInputs.push({ ...input });
        createdCount += 1;
        if (failCreateAt === createdCount) throw new Error('injected job database failure');
        const created = job(`new-job-${createdCount}`, input.project_id, input);
        jobStore.set(created.id, created);
        return created;
      },
      async deleteJob(jobID) {
        const found = jobStore.get(jobID) || null;
        jobStore.delete(jobID);
        return found;
      },
      async updateProjectOperationError(projectID, error) {
        operationErrors.set(projectID, error);
        return projectByID.get(projectID) || null;
      },
    },
  };
}

async function sha256(filePath) {
  return createHash('sha256').update(await fs.readFile(filePath)).digest('hex');
}

async function expectMissing(filePath) {
  await assert.rejects(() => fs.access(filePath), error => error?.code === 'ENOENT');
}

test('blank setup completes without reading, copying, or changing runtime state', async t => {
  const root = await createWorkspace(t, 'blank');
  const destination = project('destination', 'blank-project', path.join(root, 'destination'));
  await ensureProjectZones(destination.root_path);
  const sentinel = path.join(destination.root_path, 'cache', 'existing.cache');
  await fs.writeFile(sentinel, 'preserve');
  let dependencyCalls = 0;
  const throwingDeps = new Proxy(
    {},
    {
      get() {
        dependencyCalls += 1;
        return async () => {
          throw new Error('blank setup must not use dependencies');
        };
      },
    },
  );

  const result = await setupProjectWorkspace(destination, { mode: 'blank' }, throwingDeps);

  assert.deepEqual(result, {
    mode: 'blank',
    status: 'completed',
    copiedFiles: 0,
    copiedBytes: 0,
    clonedJobs: 0,
    error: null,
  });
  assert.equal(dependencyCalls, 0);
  assert.equal(await fs.readFile(sentinel, 'utf8'), 'preserve');
});

test('import preserves file hashes while excluding cache and transient runtime files', async t => {
  const root = await createWorkspace(t, 'import');
  const sourceRoot = path.join(root, 'source');
  const destination = project('destination', 'imported-project', path.join(root, 'destination'));
  await ensureProjectZones(destination.root_path);
  await fs.mkdir(path.join(sourceRoot, 'datasets', 'portraits'), { recursive: true });
  await fs.mkdir(path.join(sourceRoot, 'models'), { recursive: true });
  await fs.mkdir(path.join(sourceRoot, 'runs'), { recursive: true });
  await fs.mkdir(path.join(sourceRoot, 'output'), { recursive: true });
  await fs.mkdir(path.join(sourceRoot, 'cache'), { recursive: true });
  const datasetBytes = Buffer.from([0, 255, 2, 18, 77, 4, 0, 201]);
  await fs.writeFile(path.join(sourceRoot, 'datasets', 'portraits', 'sample.bin'), datasetBytes);
  await fs.writeFile(path.join(sourceRoot, 'models', 'portrait.safetensors'), 'model-contents');
  await fs.writeFile(path.join(sourceRoot, 'runs', 'checkpoint.json'), '{"step":281}');
  await fs.writeFile(path.join(sourceRoot, 'output', 'preview.png'), Buffer.from([137, 80, 78, 71]));
  await fs.writeFile(path.join(sourceRoot, 'runs', 'active.lock'), 'lock');
  await fs.writeFile(path.join(sourceRoot, 'runs', 'trainer.pid'), '1234');
  await fs.writeFile(path.join(sourceRoot, 'runs', 'upload.part'), 'partial');
  await fs.writeFile(path.join(sourceRoot, 'runs', 'download.partial'), 'partial');
  await fs.writeFile(path.join(sourceRoot, 'runs', '.hf_download_progress.json'), 'progress');
  await fs.writeFile(path.join(sourceRoot, 'cache', 'compiled.bin'), 'cache');
  const store = createDeps({ projects: [destination] });
  const previousAuth = process.env.AI_TOOLKIT_AUTH;
  process.env.AI_TOOLKIT_AUTH = 'test-only';
  t.after(() => {
    if (previousAuth === undefined) delete process.env.AI_TOOLKIT_AUTH;
    else process.env.AI_TOOLKIT_AUTH = previousAuth;
  });

  const result = await setupProjectWorkspace(
    destination,
    { mode: 'import', importRoot: sourceRoot },
    store.deps,
  );

  assert.equal(result.status, 'completed');
  assert.equal(result.copiedFiles, 4);
  assert.equal(result.clonedJobs, 0);
  assert.equal(
    await sha256(path.join(destination.root_path, 'datasets', 'portraits', 'sample.bin')),
    await sha256(path.join(sourceRoot, 'datasets', 'portraits', 'sample.bin')),
  );
  assert.equal(
    await sha256(path.join(destination.root_path, 'models', 'portrait.safetensors')),
    await sha256(path.join(sourceRoot, 'models', 'portrait.safetensors')),
  );
  assert.equal(await fs.readFile(path.join(destination.root_path, 'runs', 'checkpoint.json'), 'utf8'), '{"step":281}');
  assert.deepEqual(await fs.readFile(path.join(destination.root_path, 'outputs', 'preview.png')), Buffer.from([137, 80, 78, 71]));
  await expectMissing(path.join(destination.root_path, 'runs', 'active.lock'));
  await expectMissing(path.join(destination.root_path, 'runs', 'trainer.pid'));
  await expectMissing(path.join(destination.root_path, 'runs', 'upload.part'));
  await expectMissing(path.join(destination.root_path, 'runs', 'download.partial'));
  await expectMissing(path.join(destination.root_path, 'runs', '.hf_download_progress.json'));
  await expectMissing(path.join(destination.root_path, 'cache', 'compiled.bin'));
  assert.equal(store.operationErrors.get(destination.id), null);
});

test('clone creates stopped jobs with new identities and rewrites project-root paths', async t => {
  const root = await createWorkspace(t, 'clone');
  const source = project('source-project', 'source', path.join(root, 'source'));
  const destination = project('destination-project', 'destination', path.join(root, 'destination'));
  await ensureProjectZones(source.root_path);
  await ensureProjectZones(destination.root_path);
  await fs.mkdir(path.join(source.root_path, 'datasets', 'portraits'), { recursive: true });
  await fs.writeFile(path.join(source.root_path, 'datasets', 'portraits', 'sample.bin'), Buffer.from([1, 9, 8, 4]));
  await fs.writeFile(path.join(source.root_path, 'cache', 'do-not-copy.bin'), 'cache');
  await fs.writeFile(path.join(source.root_path, 'runs', 'active.tmp'), 'runtime');
  const sourceConfig = {
    config: {
      process: [
        {
          training_folder: path.join(source.root_path, 'runs'),
          output_folder: path.join(source.root_path, 'outputs', 'portrait-run'),
          datasets: [
            {
              folder_path: path.join(source.root_path, 'datasets', 'portraits'),
              control_path: [path.join(source.root_path, 'assets', 'controls')],
            },
          ],
          external_url: 'https://example.test/model',
          portable_ref: 'aitk-project://source-project/models/base.safetensors',
          prompt_with_nul: 'portrait\u0000prompt',
        },
      ],
    },
  };
  const sourceJobs = [
    job('source-job-completed', source.id, {
      name: 'completed run',
      status: 'completed',
      job_config: JSON.stringify(sourceConfig),
      job_ref: path.join(source.root_path, 'configs', 'completed.yaml'),
    }),
    job('source-job-running', source.id, {
      name: 'active run',
      status: 'running',
      job_config: JSON.stringify(sourceConfig),
    }),
  ];
  const store = createDeps({ projects: [source, destination], jobs: sourceJobs });

  const result = await setupProjectWorkspace(
    destination,
    { mode: 'clone', cloneFromProjectID: source.id },
    store.deps,
  );

  assert.equal(result.status, 'completed');
  assert.equal(result.clonedJobs, 2);
  assert.equal(result.copiedFiles, 1);
  assert.equal(
    await sha256(path.join(destination.root_path, 'datasets', 'portraits', 'sample.bin')),
    await sha256(path.join(source.root_path, 'datasets', 'portraits', 'sample.bin')),
  );
  await expectMissing(path.join(destination.root_path, 'cache', 'do-not-copy.bin'));
  await expectMissing(path.join(destination.root_path, 'runs', 'active.tmp'));

  const clonedJobs = [...store.jobStore.values()].filter(item => item.project_id === destination.id);
  assert.deepEqual(clonedJobs.map(item => item.id), ['new-job-1', 'new-job-2']);
  assert.ok(clonedJobs.every(item => !sourceJobs.some(sourceJob => sourceJob.id === item.id)));
  assert.ok(store.createInputs.every(input => !Object.hasOwn(input, 'id')));
  for (const cloned of clonedJobs) {
    assert.equal(cloned.project_id, destination.id);
    assert.equal(cloned.worker_id, 'local');
    assert.equal(cloned.remote_job_id, null);
    assert.equal(cloned.remote_sync_at, null);
    assert.equal(cloned.remote_error, null);
    assert.equal(cloned.status, 'stopped');
    assert.equal(cloned.stop, false);
    assert.equal(cloned.return_to_queue, false);
    assert.equal(cloned.queue_position, 0);
    assert.equal(cloned.pid, null);
    assert.equal(cloned.save_now, false);
    assert.equal(cloned.speed_string, '');
    const config = JSON.parse(cloned.job_config);
    const processConfig = config.config.process[0];
    assert.equal(processConfig.training_folder, path.join(destination.root_path, 'runs'));
    assert.equal(processConfig.output_folder, path.join(destination.root_path, 'outputs', 'portrait-run'));
    assert.equal(processConfig.datasets[0].folder_path, path.join(destination.root_path, 'datasets', 'portraits'));
    assert.deepEqual(processConfig.datasets[0].control_path, [path.join(destination.root_path, 'assets', 'controls')]);
    assert.equal(processConfig.external_url, 'https://example.test/model');
    assert.equal(processConfig.portable_ref, 'aitk-project://source-project/models/base.safetensors');
    assert.equal(processConfig.prompt_with_nul, 'portrait\u0000prompt');
  }
  assert.equal(clonedJobs[0].job_ref, path.join(destination.root_path, 'configs', 'completed.yaml'));
  assert.equal(clonedJobs[0].info, 'Cloned from completed run');
  assert.equal(clonedJobs[1].info, 'Cloned run; resume when ready');
});

test('copy failure rolls back every copied zone while preserving untouched cache state', async t => {
  const root = await createWorkspace(t, 'copy-rollback');
  const sourceRoot = path.join(root, 'source');
  const destination = project('destination', 'destination', path.join(root, 'destination'));
  await ensureProjectZones(destination.root_path);
  await fs.mkdir(path.join(sourceRoot, 'datasets'), { recursive: true });
  await fs.mkdir(path.join(sourceRoot, 'models'), { recursive: true });
  await fs.writeFile(path.join(sourceRoot, 'datasets', 'copied-first.bin'), 'first');
  await fs.writeFile(path.join(sourceRoot, 'models', 'collision.bin'), 'source');
  await fs.writeFile(path.join(destination.root_path, 'models', 'collision.bin'), 'destination');
  await fs.writeFile(path.join(destination.root_path, 'cache', 'untouched.bin'), 'cache');
  const store = createDeps({ projects: [destination] });
  const previousAuth = process.env.AI_TOOLKIT_AUTH;
  process.env.AI_TOOLKIT_AUTH = 'test-only';
  t.after(() => {
    if (previousAuth === undefined) delete process.env.AI_TOOLKIT_AUTH;
    else process.env.AI_TOOLKIT_AUTH = previousAuth;
  });

  const result = await setupProjectWorkspace(
    destination,
    { mode: 'import', importRoot: sourceRoot },
    store.deps,
  );

  assert.equal(result.status, 'failed');
  assert.match(result.error, /exist/i);
  await expectMissing(path.join(destination.root_path, 'datasets', 'copied-first.bin'));
  await expectMissing(path.join(destination.root_path, 'models', 'collision.bin'));
  assert.equal(await fs.readFile(path.join(destination.root_path, 'cache', 'untouched.bin'), 'utf8'), 'cache');
  assert.equal(store.operationErrors.get(destination.id), result.error);
});

test('job clone failure deletes newly created jobs and rolls back copied files', async t => {
  const root = await createWorkspace(t, 'job-rollback');
  const source = project('source-project', 'source', path.join(root, 'source'));
  const destination = project('destination-project', 'destination', path.join(root, 'destination'));
  await ensureProjectZones(source.root_path);
  await ensureProjectZones(destination.root_path);
  await fs.writeFile(path.join(source.root_path, 'assets', 'reference.txt'), 'reference');
  const sourceJobs = [job('source-job-1', source.id), job('source-job-2', source.id)];
  const store = createDeps({ projects: [source, destination], jobs: sourceJobs, failCreateAt: 2 });

  const result = await setupProjectWorkspace(
    destination,
    { mode: 'clone', cloneFromProjectID: source.id },
    store.deps,
  );

  assert.equal(result.status, 'failed');
  assert.match(result.error, /injected job database failure/);
  assert.equal(store.createInputs.length, 2);
  assert.deepEqual(
    [...store.jobStore.values()].map(item => item.id).sort(),
    ['source-job-1', 'source-job-2'],
  );
  await expectMissing(path.join(destination.root_path, 'assets', 'reference.txt'));
  assert.equal(store.operationErrors.get(destination.id), result.error);
});
