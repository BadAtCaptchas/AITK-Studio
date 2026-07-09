import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import sqlite3 from 'sqlite3';
import test from 'node:test';

function openDb(filename) {
  return new sqlite3.Database(filename);
}

function run(db, sql, params = []) {
  return new Promise((resolve, reject) => {
    db.run(sql, params, error => (error ? reject(error) : resolve()));
  });
}

function get(db, sql, params = []) {
  return new Promise((resolve, reject) => {
    db.get(sql, params, (error, row) => (error ? reject(error) : resolve(row)));
  });
}

function all(db, sql, params = []) {
  return new Promise((resolve, reject) => {
    db.all(sql, params, (error, rows) => (error ? reject(error) : resolve(rows)));
  });
}

function close(db) {
  return new Promise((resolve, reject) => db.close(error => (error ? reject(error) : resolve())));
}

async function fileHash(filename) {
  return createHash('sha256').update(await fs.readFile(filename)).digest('hex');
}

async function createDirectoryLink(target, link) {
  await fs.symlink(target, link, process.platform === 'win32' ? 'junction' : 'dir');
}

test('project lifecycle migration and operations preserve registered workspaces', async () => {
  const tempRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-project-lifecycle-'));
  const sqlitePath = path.join(tempRoot, 'legacy.db');
  const projectsRoot = path.join(tempRoot, 'projects-a');
  const legacyRoot = path.join(projectsRoot, 'legacy');
  const legacyFile = path.join(legacyRoot, 'notes', 'keep.txt');
  await fs.mkdir(path.dirname(legacyFile), { recursive: true });
  await fs.writeFile(legacyFile, 'preserve me');
  const hashBefore = await fileHash(legacyFile);

  const sqlite = openDb(sqlitePath);
  await run(sqlite, 'CREATE TABLE Settings (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL UNIQUE, value TEXT NOT NULL)');
  await run(
    sqlite,
    "CREATE TABLE Project (id TEXT PRIMARY KEY NOT NULL, slug TEXT NOT NULL UNIQUE, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', badge_asset TEXT, root_path TEXT NOT NULL DEFAULT '', created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)",
  );
  await run(
    sqlite,
    "CREATE TABLE Job (id TEXT PRIMARY KEY NOT NULL, name TEXT NOT NULL, project_id TEXT, worker_id TEXT NOT NULL DEFAULT 'local', remote_job_id TEXT, remote_sync_at DATETIME, remote_error TEXT, gpu_ids TEXT NOT NULL, job_config TEXT NOT NULL, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)",
  );
  await run(sqlite, 'INSERT INTO Settings (key, value) VALUES (?, ?)', ['PROJECTS_ENABLED', 'true']);
  await run(sqlite, 'INSERT INTO Settings (key, value) VALUES (?, ?)', ['PROJECTS_FOLDER', projectsRoot]);
  await run(
    sqlite,
    'INSERT INTO Project (id, slug, name, description, root_path, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
    ['legacy-project', 'legacy', 'Legacy', 'kept', legacyRoot, '2025-01-02T03:04:05.000Z', '2025-01-03T03:04:05.000Z'],
  );
  await run(
    sqlite,
    'INSERT INTO Job (id, name, project_id, worker_id, remote_job_id, remote_sync_at, gpu_ids, job_config, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
    [
      'legacy-job',
      'legacy-job',
      'legacy-project',
      'worker-legacy',
      'remote-legacy-job',
      '2025-01-05T02:00:00.000Z',
      '0',
      '{"config":{"name":"legacy-job"}}',
      '2025-01-04T03:04:05.000Z',
      '2025-01-05T03:04:05.000Z',
    ],
  );
  await close(sqlite);

  const prepared = spawnSync(process.execPath, ['scripts/prepare-db.mjs'], {
    cwd: path.resolve('.'),
    encoding: 'utf8',
    env: {
      ...process.env,
      AITK_DB_PROVIDER: 'sqlite',
      AITK_SQLITE_PATH: sqlitePath,
      DATABASE_URL: `file:${sqlitePath.replace(/\\/g, '/')}`,
    },
  });
  assert.equal(prepared.status, 0, `${prepared.stdout}\n${prepared.stderr}`);

  const migrated = openDb(sqlitePath);
  const projectColumns = new Set((await all(migrated, 'PRAGMA table_info(Project)')).map(column => column.name));
  for (const column of [
    'storage_root_path',
    'lifecycle_state',
    'archived_at',
    'revision',
    'operation_started_at',
    'operation_error',
    'home_worker_id',
    'home_instance_id',
  ]) {
    assert.equal(projectColumns.has(column), true, `missing Project.${column}`);
  }
  const legacyProject = await get(migrated, 'SELECT * FROM Project WHERE id = ?', ['legacy-project']);
  const legacyJob = await get(migrated, 'SELECT * FROM Job WHERE id = ?', ['legacy-job']);
  const legacyJobReplica = await get(migrated, 'SELECT * FROM JobReplica WHERE job_id = ?', ['legacy-job']);
  assert.equal(legacyProject.root_path, legacyRoot);
  assert.equal(legacyProject.storage_root_path, projectsRoot);
  assert.equal(legacyProject.lifecycle_state, 'active');
  assert.equal(legacyProject.revision, 1);
  assert.ok(legacyProject.home_instance_id);
  assert.equal(legacyJob.project_id, 'legacy-project');
  assert.equal(legacyJobReplica.worker_id, 'worker-legacy');
  assert.equal(legacyJobReplica.remote_job_id, 'remote-legacy-job');
  assert.equal(await fileHash(legacyFile), hashBefore);
  await close(migrated);

  const oldProvider = process.env.AITK_DB_PROVIDER;
  const oldSqlitePath = process.env.AITK_SQLITE_PATH;
  const oldAuth = process.env.AI_TOOLKIT_AUTH;
  process.env.AITK_DB_PROVIDER = 'sqlite';
  process.env.AITK_SQLITE_PATH = sqlitePath;
  process.env.AI_TOOLKIT_AUTH = 'projects-lifecycle-test';
  const [{ db, disconnectDb }, projects, settings] = await Promise.all([
    import('../dist/src/server/db.js'),
    import('../dist/src/server/projects.js'),
    import('../dist/src/server/settings.js'),
  ]);

  try {
    const first = await projects.createProject({ name: 'First Project' });
    assert.equal(first.lifecycle_state, 'active');
    assert.equal(first.revision, 1);
    const firstRoots = await projects.getProjectRoots(first);
    assert.equal(firstRoots.root, path.join(projectsRoot, 'first-project'));
    assert.equal(JSON.parse(await fs.readFile(path.join(firstRoots.root, projects.PROJECT_MANIFEST_FILE), 'utf8')).project_id, first.id);

    const nextProjectsRoot = path.join(tempRoot, 'projects-b');
    await db.settings.upsert('PROJECTS_FOLDER', nextProjectsRoot);
    settings.flushCache();
    assert.equal((await projects.getProjectRoots(first)).root, firstRoots.root);
    const second = await projects.createProject({ name: 'Second Project' });
    const secondRoots = await projects.getProjectRoots(second);
    assert.equal(secondRoots.root, path.join(nextProjectsRoot, 'second-project'));

    const job = await db.jobs.create({
      name: 'first-job',
      project_id: first.id,
      gpu_ids: '0',
      job_config: '{}',
      status: 'running',
    });
    await assert.rejects(
      () => projects.archiveProject(first.id, first.revision),
      error => error?.code === 'PROJECT_BUSY' && error.status === 409,
    );
    await db.jobs.update(job.id, { status: 'stopped' });
    const archived = await projects.archiveProject(first.slug, first.revision);
    assert.equal(archived.lifecycle_state, 'archived');
    await assert.rejects(
      () => projects.resolveProject(first.id, { intent: 'write' }),
      error => error?.code === 'PROJECT_ARCHIVED' && error.status === 409,
    );
    const restored = await projects.restoreProject(first.id, archived.revision);
    assert.equal(restored.lifecycle_state, 'active');
    const archivedAgain = await projects.archiveProject(first.id, restored.revision);
    const purgePreview = await projects.getProjectPurgePreview(first.id);
    assert.equal(purgePreview.can_purge, true);
    const purged = await projects.purgeProject(first.id, {
      expected_revision: archivedAgain.revision,
      confirmation: archivedAgain.slug,
      scope: 'project_and_all_data',
    });
    assert.equal(purged.purged.deleted_jobs, 1);
    assert.equal(await db.projects.findById(first.id), null);
    await assert.rejects(() => fs.access(firstRoots.root));

    await fs.writeFile(path.join(secondRoots.outputs, 'result.txt'), 'result');
    const secondArchived = await projects.archiveProject(second.id, second.revision);
    const relocationStorage = path.join(tempRoot, 'relocated');
    const relocationPreview = await projects.getProjectRelocatePreview(second.id, {
      destination_storage_root: relocationStorage,
      mode: 'copy',
    });
    assert.equal(relocationPreview.can_relocate, true);
    const relocation = await projects.relocateProject(second.id, {
      destination_storage_root: relocationStorage,
      mode: 'copy',
      expected_revision: secondArchived.revision,
    });
    assert.equal(relocation.project.root_path, path.join(relocationStorage, second.slug));
    assert.equal(await fs.readFile(path.join(relocation.project.root_path, 'outputs', 'result.txt'), 'utf8'), 'result');
    assert.equal(await fs.readFile(path.join(secondRoots.outputs, 'result.txt'), 'utf8'), 'result');

    const localInstanceID = await projects.getAITKInstanceID();
    const remoteOwnedActive = await projects.createProject({ name: 'Remote Owned Active' });
    const remoteActive = await db.projects.update(remoteOwnedActive.id, {
      home_worker_id: 'worker-remote',
      home_instance_id: 'remote-instance-id',
    });
    await assert.rejects(
      () => projects.archiveProject(remoteActive.id, remoteActive.revision),
      error => error?.code === 'PROJECT_REPLICA_READ_ONLY' && error.status === 409,
    );
    const localAgain = await db.projects.update(remoteActive.id, {
      home_worker_id: 'local',
      home_instance_id: localInstanceID,
    });
    const locallyArchived = await projects.archiveProject(localAgain.id, localAgain.revision);
    const remoteArchived = await db.projects.update(locallyArchived.id, {
      home_worker_id: 'worker-remote',
      home_instance_id: 'remote-instance-id',
    });
    await assert.rejects(
      () => projects.restoreProject(remoteArchived.id, remoteArchived.revision),
      error => error?.code === 'PROJECT_REPLICA_READ_ONLY' && error.status === 409,
    );
    await assert.rejects(
      () => projects.getProjectPurgePreview(remoteArchived.id),
      error => error?.code === 'PROJECT_REPLICA_READ_ONLY' && error.status === 409,
    );
    await assert.rejects(
      () =>
        projects.purgeProject(remoteArchived.id, {
          expected_revision: remoteArchived.revision,
          confirmation: remoteArchived.slug,
          scope: 'project_and_all_data',
        }),
      error => error?.code === 'PROJECT_REPLICA_READ_ONLY' && error.status === 409,
    );
    await assert.rejects(
      () =>
        projects.relocateProject(remoteArchived.id, {
          destination_storage_root: path.join(tempRoot, 'remote-relocation'),
          mode: 'copy',
          expected_revision: remoteArchived.revision,
        }),
      error => error?.code === 'PROJECT_REPLICA_READ_ONLY' && error.status === 409,
    );
    assert.equal((await db.projects.findById(remoteArchived.id)).lifecycle_state, 'archived');

    const overlapStorage = path.join(tempRoot, 'overlap-projects');
    await db.settings.upsert('PROJECTS_FOLDER', overlapStorage);
    settings.flushCache();
    const overlapProject = await projects.createProject({ name: 'Overlap Project' });
    const overlapRoots = await projects.getProjectRoots(overlapProject);
    await fs.writeFile(path.join(overlapRoots.outputs, 'must-remain.txt'), 'do not delete');
    const overlapArchived = await projects.archiveProject(overlapProject.id, overlapProject.revision);
    const nestedDestination = path.join(overlapRoots.root, 'nested-relocation');
    const overlapPreview = await projects.getProjectRelocatePreview(overlapArchived.id, {
      destination_storage_root: nestedDestination,
      mode: 'move',
    });
    assert.equal(overlapPreview.can_relocate, false);
    assert.equal(overlapPreview.blockers.some(blocker => blocker.code === 'PROJECT_PATH_OVERLAP'), true);
    await assert.rejects(
      () =>
        projects.relocateProject(overlapArchived.id, {
          destination_storage_root: nestedDestination,
          mode: 'move',
          expected_revision: overlapArchived.revision,
          confirmation: overlapArchived.slug,
        }),
      error => error?.code === 'PROJECT_ROOT_INVALID' && error.status === 409,
    );
    assert.equal(await fs.readFile(path.join(overlapRoots.outputs, 'must-remain.txt'), 'utf8'), 'do not delete');

    await fs.mkdir(nestedDestination, { recursive: true });
    const nestedDestinationAlias = path.join(tempRoot, 'nested-destination-alias');
    await createDirectoryLink(nestedDestination, nestedDestinationAlias);
    const canonicalOverlapPreview = await projects.getProjectRelocatePreview(overlapArchived.id, {
      destination_storage_root: nestedDestinationAlias,
      mode: 'copy',
    });
    assert.equal(canonicalOverlapPreview.can_relocate, false);
    assert.equal(canonicalOverlapPreview.blockers.some(blocker => blocker.code === 'PROJECT_PATH_OVERLAP'), true);
    await fs.unlink(nestedDestinationAlias);

    const redirectedTrash = path.join(tempRoot, 'redirected-trash');
    await fs.mkdir(redirectedTrash, { recursive: true });
    const trashLink = path.join(overlapStorage, '.aitk-trash');
    await createDirectoryLink(redirectedTrash, trashLink);
    await assert.rejects(
      () => projects.getProjectPurgePreview(overlapArchived.id),
      error => error?.code === 'PROJECT_ROOT_INVALID' && error.status === 409,
    );
    await fs.unlink(trashLink);

    const moveProject = await projects.createProject({ name: 'Safe Move Project' });
    const moveRoots = await projects.getProjectRoots(moveProject);
    await fs.writeFile(path.join(moveRoots.notes, 'move-me.txt'), 'move safely');
    const moveArchived = await projects.archiveProject(moveProject.id, moveProject.revision);
    const moveDestination = path.join(tempRoot, 'safe-move-destination');
    const moved = await projects.relocateProject(moveArchived.id, {
      destination_storage_root: moveDestination,
      mode: 'move',
      expected_revision: moveArchived.revision,
      confirmation: moveArchived.slug,
    });
    assert.equal(await fs.readFile(path.join(moved.project.root_path, 'notes', 'move-me.txt'), 'utf8'), 'move safely');
    await assert.rejects(() => fs.access(moveRoots.root));

    const stagingRedirectRoot = path.join(tempRoot, 'redirected-staging');
    const relocationWithRedirect = path.join(tempRoot, 'relocation-with-redirect');
    await fs.mkdir(stagingRedirectRoot, { recursive: true });
    await fs.mkdir(relocationWithRedirect, { recursive: true });
    const stagingLink = path.join(relocationWithRedirect, '.aitk-staging');
    await createDirectoryLink(stagingRedirectRoot, stagingLink);
    await assert.rejects(
      () =>
        projects.getProjectRelocatePreview(overlapArchived.id, {
          destination_storage_root: relocationWithRedirect,
          mode: 'copy',
        }),
      error => error?.code === 'PROJECT_ROOT_INVALID' && error.status === 409,
    );
    await fs.unlink(stagingLink);

    const copySafetyProject = await projects.createProject({ name: 'Copy Safety Project' });
    const copySafetyRoots = await projects.getProjectRoots(copySafetyProject);
    const datasetsRoot = path.join(tempRoot, 'scoped-datasets');
    const linkedDataset = path.join(datasetsRoot, 'linked-source');
    const outsideDataset = path.join(tempRoot, 'outside-dataset');
    await fs.mkdir(linkedDataset, { recursive: true });
    await fs.mkdir(outsideDataset, { recursive: true });
    await fs.writeFile(path.join(linkedDataset, 'inside.txt'), 'inside');
    await fs.writeFile(path.join(outsideDataset, 'outside.txt'), 'outside');
    const nestedLink = path.join(linkedDataset, 'redirect');
    await createDirectoryLink(outsideDataset, nestedLink);
    await db.settings.upsert('DATASETS_FOLDER', datasetsRoot);
    settings.flushCache();
    const linkedConfig = {
      config: {
        name: 'linked-copy',
        process: [{ datasets: [{ folder_path: linkedDataset }] }],
      },
    };
    await assert.rejects(
      () => projects.prepareJobConfigForProject(linkedConfig, copySafetyProject),
      error => error?.code === 'PROJECT_ROOT_INVALID' && error.status === 409,
    );
    await assert.rejects(() => fs.access(path.join(copySafetyRoots.datasets, 'linked-source')));
    await fs.unlink(nestedLink);
    const copiedConfig = await projects.prepareJobConfigForProject(linkedConfig, copySafetyProject);
    const copiedDataset = copiedConfig.config.process[0].datasets[0].folder_path;
    assert.equal(copiedDataset, path.join(copySafetyRoots.datasets, 'linked-source'));
    assert.equal(await fs.readFile(path.join(copiedDataset, 'inside.txt'), 'utf8'), 'inside');
    await assert.rejects(() => fs.access(path.join(copiedDataset, 'redirect', 'outside.txt')));
  } finally {
    await disconnectDb();
    if (oldProvider === undefined) delete process.env.AITK_DB_PROVIDER;
    else process.env.AITK_DB_PROVIDER = oldProvider;
    if (oldSqlitePath === undefined) delete process.env.AITK_SQLITE_PATH;
    else process.env.AITK_SQLITE_PATH = oldSqlitePath;
    if (oldAuth === undefined) delete process.env.AI_TOOLKIT_AUTH;
    else process.env.AI_TOOLKIT_AUTH = oldAuth;
    await fs.rm(tempRoot, { recursive: true, force: true });
  }
});
