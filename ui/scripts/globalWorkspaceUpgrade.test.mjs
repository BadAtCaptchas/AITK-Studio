import test from 'node:test';
import assert from 'node:assert/strict';
import sqlite3 from 'sqlite3';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { upgradeSqliteGlobalWorkspace, preflightMongoGlobalUpgrade } from './global-workspace-upgrade.mjs';

const exec = (db, sql) => new Promise((resolve, reject) => db.exec(sql, error => (error ? reject(error) : resolve())));
const all = (db, sql) =>
  new Promise((resolve, reject) => db.all(sql, (error, rows) => (error ? reject(error) : resolve(rows))));
const fixture = async t => {
  const db = new sqlite3.Database(':memory:');
  t.after(() => new Promise(resolve => db.close(resolve)));
  await exec(
    db,
    `CREATE TABLE Job (id TEXT PRIMARY KEY, name TEXT NOT NULL, project_id TEXT, job_config TEXT);
    CREATE TABLE JobReplica (id TEXT PRIMARY KEY, job_id TEXT, remote_project_id TEXT);
    CREATE TABLE Project (id TEXT PRIMARY KEY);
    CREATE TABLE ProjectReplica (id TEXT PRIMARY KEY);
    CREATE TABLE ProjectSyncOperation (id TEXT PRIMARY KEY);
    CREATE TABLE Settings (key TEXT UNIQUE, value TEXT);
    CREATE TABLE user_metrics (value REAL);
    INSERT INTO user_metrics VALUES (0.42);
    INSERT INTO Settings VALUES ('PROJECTS_ENABLED','false'),('DATASETS_FOLDER','D:/my-data');
    CREATE UNIQUE INDEX Job_global_name_key ON Job(name) WHERE project_id IS NULL;
    CREATE UNIQUE INDEX Job_project_id_name_key ON Job(project_id,name) WHERE project_id IS NOT NULL;`,
  );
  return db;
};

test('empty legacy schema upgrades idempotently while preserving global rows, settings and unrelated tables', async t => {
  const db = await fixture(t);
  await exec(
    db,
    `INSERT INTO Job VALUES ('run','example',NULL,'{"config":{}}');
    INSERT INTO JobReplica VALUES ('mapping','run',NULL);`,
  );
  await upgradeSqliteGlobalWorkspace(db);
  const before = await all(db, 'SELECT * FROM sqlite_master ORDER BY name');
  await upgradeSqliteGlobalWorkspace(db);
  assert.deepEqual(await all(db, 'SELECT * FROM sqlite_master ORDER BY name'), before);
  assert.deepEqual(await all(db, 'SELECT * FROM Job'), [{ id: 'run', name: 'example', job_config: '{"config":{}}' }]);
  assert.deepEqual(await all(db, 'SELECT * FROM JobReplica'), [{ id: 'mapping', job_id: 'run' }]);
  assert.deepEqual(await all(db, 'SELECT * FROM Settings'), [{ key: 'DATASETS_FOLDER', value: 'D:/my-data' }]);
  assert.equal((await all(db, 'SELECT * FROM user_metrics'))[0].value, 0.42);
  assert.equal((await all(db, "SELECT name FROM sqlite_master WHERE name LIKE 'Project%'")).length, 0);
  await assert.rejects(exec(db, `INSERT INTO Job VALUES ('other','example','{}')`), /UNIQUE/);
});

for (const [label, seed] of [
  ['project record', "INSERT INTO Project VALUES ('legacy')"],
  ['project job', "INSERT INTO Job VALUES ('run','example','legacy','{}')"],
  ['orphan replica', "INSERT INTO ProjectReplica VALUES ('legacy')"],
  ['pending sync', "INSERT INTO ProjectSyncOperation VALUES ('legacy')"],
  ['remote project mapping', "INSERT INTO JobReplica VALUES ('mapping','run','legacy')"],
  ['scoped watcher', `INSERT INTO Settings VALUES ('DATASET_WATCHERS_V1','{"watchers":[{"projectID":"old"}]}')`],
  [
    'conflicting names',
    "DROP INDEX Job_global_name_key; INSERT INTO Job VALUES ('a','same',NULL,'{}'),('b','same',NULL,'{}')",
  ],
]) {
  test(`refuses ${label} before changing schema or data`, async t => {
    const db = await fixture(t);
    await exec(db, seed);
    const schema = await all(db, 'SELECT * FROM sqlite_master ORDER BY name');
    const rows = await Promise.all(
      ['Job', 'Project', 'ProjectReplica', 'ProjectSyncOperation', 'JobReplica', 'Settings'].map(table =>
        all(db, `SELECT * FROM ${table}`),
      ),
    );
    await assert.rejects(upgradeSqliteGlobalWorkspace(db), /manual migration/);
    assert.deepEqual(await all(db, 'SELECT * FROM sqlite_master ORDER BY name'), schema);
    assert.deepEqual(
      await Promise.all(
        ['Job', 'Project', 'ProjectReplica', 'ProjectSyncOperation', 'JobReplica', 'Settings'].map(table =>
          all(db, `SELECT * FROM ${table}`),
        ),
      ),
      rows,
    );
  });
}

test('fresh empty database has no legacy structures after repeated upgrade', async t => {
  const db = new sqlite3.Database(':memory:');
  t.after(() => new Promise(resolve => db.close(resolve)));
  await upgradeSqliteGlobalWorkspace(db);
  await upgradeSqliteGlobalWorkspace(db);
  assert.deepEqual(await all(db, 'SELECT name FROM sqlite_master'), []);
});

test('Mongo preflight refuses legacy data using read operations only', async () => {
  await assert.rejects(
    preflightMongoGlobalUpgrade({
      collection(name) {
        assert.equal(name, 'projects');
        return {
          async findOne() {
            return { id: 'legacy' };
          },
        };
      },
    }),
    /manual migration/,
  );
});

const executeFile = promisify(execFile);
const uiRoot = fileURLToPath(new URL('../', import.meta.url));
const close = db => new Promise((resolve, reject) => db.close(error => (error ? reject(error) : resolve())));
async function prepare(filename) {
  return executeFile(process.execPath, ['scripts/prepare-db.mjs'], {
    cwd: uiRoot,
    env: {
      ...process.env,
      AITK_DB_PROVIDER: 'sqlite',
      AITK_SQLITE_PATH: filename,
      AITK_SKIP_PRISMA_GENERATE: '1',
      AITK_SQLITE_BACKUP_RETENTION: '3',
    },
    timeout: 60000,
  });
}

test('database preparation creates fresh schema, backs up legacy upgrades and preserves global rows on repeat', async t => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-global-upgrade-'));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  const filename = path.join(directory, 'test.db');
  await prepare(filename);
  let db = new sqlite3.Database(filename);
  assert.equal((await all(db, "SELECT name FROM sqlite_master WHERE name = 'Job'")).length, 1);
  await exec(
    db,
    `ALTER TABLE Job ADD COLUMN project_id TEXT;
    ALTER TABLE JobReplica ADD COLUMN remote_project_id TEXT;
    CREATE TABLE Project (id TEXT PRIMARY KEY);
    CREATE TABLE ProjectReplica (id TEXT PRIMARY KEY);
    CREATE TABLE ProjectSyncOperation (id TEXT PRIMARY KEY);
    CREATE TABLE private_metric (value REAL);
    INSERT INTO private_metric VALUES (0.75);
    INSERT INTO Settings (key,value) VALUES ('PROJECTS_FOLDER','do-not-touch');
    INSERT INTO Job (id,name,gpu_ids,job_config,updated_at) VALUES ('global','retained','0','{}',CURRENT_TIMESTAMP);`,
  );
  await close(db);
  await prepare(filename);
  const backups = await fs.readdir(path.join(directory, '.aitk-backups'));
  assert.equal(backups.length, 1);
  const backup = new sqlite3.Database(path.join(directory, '.aitk-backups', backups[0]), sqlite3.OPEN_READONLY);
  assert.equal((await all(backup, "SELECT name FROM sqlite_master WHERE name='Project'")).length, 1);
  await close(backup);
  await prepare(filename);
  db = new sqlite3.Database(filename);
  try {
    assert.equal((await all(db, "SELECT name FROM sqlite_master WHERE name LIKE 'Project%'")).length, 0);
    assert.equal((await all(db, 'SELECT * FROM private_metric'))[0].value, 0.75);
    assert.equal((await all(db, 'SELECT name FROM Job'))[0].name, 'retained');
    assert.equal(
      (await all(db, 'PRAGMA table_info(Job)')).some(column => column.name === 'project_id'),
      false,
    );
    assert.equal((await all(db, "SELECT * FROM Settings WHERE key='PROJECTS_FOLDER'")).length, 0);
    await assert.rejects(
      exec(
        db,
        `INSERT INTO Job (id,name,gpu_ids,job_config,updated_at) VALUES ('duplicate','retained','0','{}',CURRENT_TIMESTAMP)`,
      ),
      /UNIQUE/,
    );
  } finally {
    await close(db);
  }
});

test('database preparation refuses nonempty legacy data before backup or any database write', async t => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-global-refusal-'));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  const filename = path.join(directory, 'test.db');
  const db = new sqlite3.Database(filename);
  await exec(db, "CREATE TABLE Project (id TEXT); INSERT INTO Project VALUES ('keep-me');");
  await close(db);
  const before = await fs.readFile(filename);
  await assert.rejects(prepare(filename), /manual migration/);
  assert.deepEqual(await fs.readFile(filename), before);
  assert.deepEqual(await fs.readdir(directory), ['test.db']);
});
