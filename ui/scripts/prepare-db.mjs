import { execFileSync } from 'child_process';
import { randomUUID } from 'crypto';
import fs from 'fs';
import { createRequire } from 'module';
import path from 'path';
import { MongoClient } from 'mongodb';
import sqlite3 from 'sqlite3';
import { backupExistingSqliteDatabase } from './sqlite-backup.mjs';
import { resolveSqliteJournalMode } from './sqlite-journal-mode.mjs';

const require = createRequire(import.meta.url);
const provider = (process.env.AITK_DB_PROVIDER || 'sqlite').trim().toLowerCase();
const toolkitRoot = path.resolve(process.cwd(), '..');
const sqlitePath = path.resolve(process.env.AITK_SQLITE_PATH || path.join(toolkitRoot, 'aitk_db.db'));
const mongoUri = process.env.AITK_MONGODB_URI?.trim();
const mongoDbName = process.env.AITK_MONGODB_DB?.trim() || 'ai_toolkit';
const prismaCli = require.resolve('prisma/build/index.js');
const SQLITE_BUSY_TIMEOUT_MS = 30000;

function runPrisma(args, options = {}) {
  execFileSync(process.execPath, [prismaCli, ...args], {
    stdio: options.nonInteractive ? ['ignore', 'inherit', 'inherit'] : 'inherit',
    env: options.nonInteractive ? { ...process.env, CI: '1' } : process.env,
  });
}

function sqliteAll(db, sql, params = []) {
  return new Promise((resolve, reject) => {
    db.all(sql, params, (error, rows) => {
      if (error) reject(error);
      else resolve(rows);
    });
  });
}

function sqliteRun(db, sql, params = []) {
  return new Promise((resolve, reject) => {
    db.run(sql, params, error => {
      if (error) reject(error);
      else resolve();
    });
  });
}

function sqliteIdentifier(value) {
  return `"${String(value).replace(/"/g, '""')}"`;
}

function sqliteColumnDefinition(column, overrides = {}) {
  const name = overrides.name || column.name;
  const type = overrides.type || column.type || 'TEXT';
  const nullable = overrides.nullable ?? !column.notnull;
  const defaultValue = Object.prototype.hasOwnProperty.call(overrides, 'defaultValue')
    ? overrides.defaultValue
    : column.dflt_value;
  const parts = [sqliteIdentifier(name), type];

  if (column.pk) {
    parts.push('PRIMARY KEY');
  }
  if (!nullable) {
    parts.push('NOT NULL');
  }
  if (defaultValue !== null && defaultValue !== undefined) {
    parts.push(`DEFAULT ${defaultValue}`);
  }

  return parts.join(' ');
}

async function configureSqliteConnection(db) {
  const journalMode = resolveSqliteJournalMode(process.env.AI_TOOLKIT_DB_JOURNAL_MODE);
  await sqliteRun(db, `PRAGMA busy_timeout=${SQLITE_BUSY_TIMEOUT_MS};`);
  await sqliteRun(db, `PRAGMA journal_mode=${journalMode};`);
  await sqliteRun(db, 'PRAGMA synchronous=NORMAL;');
}

async function configureSqliteDatabase(filename) {
  const db = new sqlite3.Database(filename);
  try {
    await configureSqliteConnection(db);
  } finally {
    await new Promise(resolve => db.close(resolve));
  }
}

async function ensureColumn(db, table, name, definition) {
  const columns = await sqliteAll(db, `PRAGMA table_info(${table})`);
  if (!columns.some(column => column.name === name)) {
    await sqliteRun(db, `ALTER TABLE ${table} ADD COLUMN ${name} ${definition}`);
  }
}

async function rebuildJobTableWithColumnOverrides(db, columnOverrides = {}) {
  const tempTable = '__aitk_job_rebuild';
  const columns = await sqliteAll(db, 'PRAGMA table_info(Job)');
  if (columns.length === 0) return;
  const columnNames = columns.map(column => column.name);
  const columnList = columnNames.map(sqliteIdentifier).join(', ');
  const columnDefinitions = columns
    .map(column => sqliteColumnDefinition(column, columnOverrides[column.name] || {}))
    .join(',\n      ');

  await sqliteRun(db, `DROP TABLE IF EXISTS ${tempTable};`);
  await sqliteRun(
    db,
    `
    CREATE TABLE ${tempTable} (
      ${columnDefinitions}
    );
    `,
  );
  await sqliteRun(
    db,
    `
    INSERT INTO ${tempTable} (${columnList})
    SELECT ${columnList}
    FROM Job;
    `,
  );
  await sqliteRun(db, 'DROP TABLE Job;');
  await sqliteRun(db, `ALTER TABLE ${tempTable} RENAME TO Job;`);
}

async function ensureJobProjectIdNullable(db) {
  await ensureColumn(db, 'Job', 'project_id', 'TEXT');
  const columns = await sqliteAll(db, 'PRAGMA table_info(Job)');
  const projectColumn = columns.find(column => column.name === 'project_id');
  if (projectColumn && Number(projectColumn.notnull) !== 0) {
    await rebuildJobTableWithColumnOverrides(db, {
      project_id: { type: 'TEXT', nullable: true, defaultValue: null },
    });
  }
}

async function sqliteIndexColumns(db, indexName) {
  const rows = await sqliteAll(db, `PRAGMA index_info(${sqliteIdentifier(indexName)})`);
  return rows.sort((a, b) => Number(a.seqno) - Number(b.seqno)).map(row => row.name);
}

async function rebuildJobTableWithoutNameUnique(db) {
  await rebuildJobTableWithColumnOverrides(db, {
    project_id: { type: 'TEXT', nullable: true, defaultValue: null },
  });
}

async function dropLegacySqliteJobNameUniqueIndexes(db) {
  const indexes = await sqliteAll(db, 'PRAGMA index_list(Job)');
  let needsRebuild = false;
  for (const index of indexes) {
    if (!index.unique) continue;
    const columns = await sqliteIndexColumns(db, index.name);
    if (columns.length !== 1 || columns[0] !== 'name') continue;
    if (String(index.name).startsWith('sqlite_autoindex_')) {
      needsRebuild = true;
      continue;
    }
    try {
      await sqliteRun(db, `DROP INDEX ${sqliteIdentifier(index.name)};`);
    } catch {
      needsRebuild = true;
    }
  }
  if (needsRebuild) {
    await rebuildJobTableWithoutNameUnique(db);
  }
}

async function applySqliteScopedJobNameIndexes(filename) {
  const db = new sqlite3.Database(filename);
  let transactionOpen = false;
  try {
    await configureSqliteConnection(db);
    await sqliteRun(db, 'BEGIN IMMEDIATE;');
    transactionOpen = true;
    await ensureJobProjectIdNullable(db);
    await dropLegacySqliteJobNameUniqueIndexes(db);
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Job_name_idx ON Job(name);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Job_project_id_name_idx ON Job(project_id, name);');
    await sqliteRun(db, 'CREATE UNIQUE INDEX IF NOT EXISTS Job_global_name_key ON Job(name) WHERE project_id IS NULL;');
    await sqliteRun(
      db,
      'CREATE UNIQUE INDEX IF NOT EXISTS Job_project_id_name_key ON Job(project_id, name) WHERE project_id IS NOT NULL;',
    );
    await sqliteRun(db, 'COMMIT;');
    transactionOpen = false;
  } finally {
    if (transactionOpen) await sqliteRun(db, 'ROLLBACK;').catch(() => undefined);
    await new Promise(resolve => db.close(resolve));
  }
}

async function applySqliteCompatibilitySchema(filename) {
  const db = new sqlite3.Database(filename);
  let transactionOpen = false;
  try {
    await configureSqliteConnection(db);

    await sqliteRun(db, 'BEGIN IMMEDIATE;');
    transactionOpen = true;
    await sqliteRun(
      db,
      `
      CREATE TABLE IF NOT EXISTS Settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT NOT NULL UNIQUE,
        value TEXT NOT NULL
      );
      `,
    );
    await sqliteRun(
      db,
      `
      CREATE TABLE IF NOT EXISTS Queue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        worker_id TEXT NOT NULL DEFAULT 'local',
        gpu_ids TEXT NOT NULL,
        is_running BOOLEAN NOT NULL DEFAULT false
      );
      `,
    );
    await ensureColumn(db, 'Queue', 'worker_id', "TEXT NOT NULL DEFAULT 'local'");

    await sqliteRun(
      db,
      `
      CREATE TABLE IF NOT EXISTS Job (
        id TEXT PRIMARY KEY NOT NULL,
        name TEXT NOT NULL,
        worker_id TEXT NOT NULL DEFAULT 'local',
        remote_job_id TEXT,
        remote_sync_at DATETIME,
        remote_error TEXT,
        gpu_ids TEXT NOT NULL,
        job_config TEXT NOT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        status TEXT NOT NULL DEFAULT 'stopped',
        stop BOOLEAN NOT NULL DEFAULT false,
        return_to_queue BOOLEAN NOT NULL DEFAULT false,
        step INTEGER NOT NULL DEFAULT 0,
        info TEXT NOT NULL DEFAULT '',
        speed_string TEXT NOT NULL DEFAULT '',
        queue_position INTEGER NOT NULL DEFAULT 0,
        pid INTEGER,
        job_type TEXT NOT NULL DEFAULT 'train',
        job_ref TEXT,
        save_now BOOLEAN NOT NULL DEFAULT false,
        sample_now BOOLEAN NOT NULL DEFAULT false
      );
      `,
    );
    await ensureColumn(db, 'Job', 'name', "TEXT NOT NULL DEFAULT ''");
    await ensureColumn(db, 'Job', 'worker_id', "TEXT NOT NULL DEFAULT 'local'");
    await ensureColumn(db, 'Job', 'remote_job_id', 'TEXT');
    await ensureColumn(db, 'Job', 'remote_sync_at', 'DATETIME');
    await ensureColumn(db, 'Job', 'remote_error', 'TEXT');
    await ensureColumn(db, 'Job', 'gpu_ids', "TEXT NOT NULL DEFAULT ''");
    await ensureColumn(db, 'Job', 'job_config', "TEXT NOT NULL DEFAULT '{}'");
    await ensureColumn(db, 'Job', 'created_at', "DATETIME NOT NULL DEFAULT '1970-01-01 00:00:00'");
    await ensureColumn(db, 'Job', 'updated_at', "DATETIME NOT NULL DEFAULT '1970-01-01 00:00:00'");
    await ensureColumn(db, 'Job', 'status', "TEXT NOT NULL DEFAULT 'stopped'");
    await ensureColumn(db, 'Job', 'stop', 'BOOLEAN NOT NULL DEFAULT false');
    await ensureColumn(db, 'Job', 'return_to_queue', 'BOOLEAN NOT NULL DEFAULT false');
    await ensureColumn(db, 'Job', 'step', 'INTEGER NOT NULL DEFAULT 0');
    await ensureColumn(db, 'Job', 'info', "TEXT NOT NULL DEFAULT ''");
    await ensureColumn(db, 'Job', 'speed_string', "TEXT NOT NULL DEFAULT ''");
    await ensureColumn(db, 'Job', 'queue_position', 'INTEGER NOT NULL DEFAULT 0');
    await ensureColumn(db, 'Job', 'pid', 'INTEGER');
    await ensureColumn(db, 'Job', 'job_type', "TEXT NOT NULL DEFAULT 'train'");
    await ensureColumn(db, 'Job', 'job_ref', 'TEXT');
    await ensureColumn(db, 'Job', 'save_now', 'BOOLEAN NOT NULL DEFAULT false');
    await ensureColumn(db, 'Job', 'sample_now', 'BOOLEAN NOT NULL DEFAULT false');
    await ensureJobProjectIdNullable(db);

    await sqliteRun(
      db,
      `
      CREATE TABLE IF NOT EXISTS Project (
        id TEXT PRIMARY KEY NOT NULL,
        slug TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        badge_asset TEXT,
        root_path TEXT NOT NULL DEFAULT '',
        storage_root_path TEXT NOT NULL DEFAULT '',
        lifecycle_state TEXT NOT NULL DEFAULT 'active',
        archived_at DATETIME,
        revision INTEGER NOT NULL DEFAULT 1,
        operation_started_at DATETIME,
        operation_error TEXT,
        home_worker_id TEXT NOT NULL DEFAULT 'local',
        home_instance_id TEXT NOT NULL DEFAULT '',
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
      `,
    );
    await ensureColumn(db, 'Project', 'description', "TEXT NOT NULL DEFAULT ''");
    await ensureColumn(db, 'Project', 'badge_asset', 'TEXT');
    await ensureColumn(db, 'Project', 'root_path', "TEXT NOT NULL DEFAULT ''");
    await ensureColumn(db, 'Project', 'storage_root_path', "TEXT NOT NULL DEFAULT ''");
    await ensureColumn(db, 'Project', 'lifecycle_state', "TEXT NOT NULL DEFAULT 'active'");
    await ensureColumn(db, 'Project', 'archived_at', 'DATETIME');
    await ensureColumn(db, 'Project', 'revision', 'INTEGER NOT NULL DEFAULT 1');
    await ensureColumn(db, 'Project', 'operation_started_at', 'DATETIME');
    await ensureColumn(db, 'Project', 'operation_error', 'TEXT');
    await ensureColumn(db, 'Project', 'home_worker_id', "TEXT NOT NULL DEFAULT 'local'");
    await ensureColumn(db, 'Project', 'home_instance_id', "TEXT NOT NULL DEFAULT ''");
    await ensureColumn(db, 'Project', 'created_at', "DATETIME NOT NULL DEFAULT '1970-01-01 00:00:00'");
    await ensureColumn(db, 'Project', 'updated_at', "DATETIME NOT NULL DEFAULT '1970-01-01 00:00:00'");

    await sqliteRun(
      db,
      `
      CREATE TABLE IF NOT EXISTS ProjectReplica (
        id TEXT PRIMARY KEY NOT NULL,
        project_id TEXT NOT NULL,
        worker_id TEXT NOT NULL,
        remote_project_id TEXT,
        remote_instance_id TEXT,
        role TEXT NOT NULL DEFAULT 'execution',
        state TEXT NOT NULL DEFAULT 'creating',
        base_manifest_hash TEXT,
        local_manifest_hash TEXT,
        remote_manifest_hash TEXT,
        last_synced_at DATETIME,
        last_error TEXT,
        auto_pull_results BOOLEAN NOT NULL DEFAULT true,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
      `,
    );
    await sqliteRun(
      db,
      `
      CREATE TABLE IF NOT EXISTS JobReplica (
        id TEXT PRIMARY KEY NOT NULL,
        job_id TEXT NOT NULL,
        worker_id TEXT NOT NULL,
        remote_job_id TEXT NOT NULL,
        remote_project_id TEXT,
        role TEXT NOT NULL DEFAULT 'execution',
        last_synced_at DATETIME,
        last_error TEXT,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
      `,
    );
    await sqliteRun(
      db,
      `
      CREATE TABLE IF NOT EXISTS ProjectSyncOperation (
        id TEXT PRIMARY KEY NOT NULL,
        project_id TEXT NOT NULL,
        worker_id TEXT NOT NULL,
        profile TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'queued',
        phase TEXT NOT NULL DEFAULT 'queued',
        files_total INTEGER NOT NULL DEFAULT 0,
        files_done INTEGER NOT NULL DEFAULT 0,
        bytes_total REAL NOT NULL DEFAULT 0,
        bytes_done REAL NOT NULL DEFAULT 0,
        retry_count INTEGER NOT NULL DEFAULT 0,
        retry_at DATETIME,
        base_manifest_hash TEXT,
        source_manifest_hash TEXT,
        target_manifest_hash TEXT,
        conflicts TEXT NOT NULL DEFAULT '[]',
        error TEXT,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
      `,
    );

    await sqliteRun(
      db,
      `
      CREATE TABLE IF NOT EXISTS WorkerNode (
        id TEXT PRIMARY KEY NOT NULL,
        name TEXT NOT NULL UNIQUE,
        base_url TEXT NOT NULL,
        api_token TEXT NOT NULL,
        enabled BOOLEAN NOT NULL DEFAULT true,
        offline_bypass_enabled BOOLEAN NOT NULL DEFAULT false,
        last_status TEXT NOT NULL DEFAULT 'unknown',
        last_error TEXT,
        last_checked_at DATETIME,
        capabilities TEXT NOT NULL DEFAULT '{}',
        gpus TEXT NOT NULL DEFAULT '[]',
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
      );
      `,
    );
    await ensureColumn(db, 'WorkerNode', 'offline_bypass_enabled', 'BOOLEAN NOT NULL DEFAULT false');

    await sqliteRun(db, 'CREATE UNIQUE INDEX IF NOT EXISTS Queue_worker_id_gpu_ids_key ON Queue(worker_id, gpu_ids);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Queue_worker_id_idx ON Queue(worker_id);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Queue_gpu_ids_idx ON Queue(gpu_ids);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Job_status_idx ON Job(status);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Job_worker_id_idx ON Job(worker_id);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Job_remote_job_id_idx ON Job(remote_job_id);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Job_gpu_ids_idx ON Job(gpu_ids);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Job_job_type_idx ON Job(job_type);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Job_job_ref_idx ON Job(job_ref);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Job_project_id_idx ON Job(project_id);');
    await dropLegacySqliteJobNameUniqueIndexes(db);
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Job_name_idx ON Job(name);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Job_project_id_name_idx ON Job(project_id, name);');
    await sqliteRun(db, 'CREATE UNIQUE INDEX IF NOT EXISTS Job_global_name_key ON Job(name) WHERE project_id IS NULL;');
    await sqliteRun(
      db,
      'CREATE UNIQUE INDEX IF NOT EXISTS Job_project_id_name_key ON Job(project_id, name) WHERE project_id IS NOT NULL;',
    );
    await sqliteRun(db, 'CREATE UNIQUE INDEX IF NOT EXISTS Project_slug_key ON Project(slug);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Project_slug_idx ON Project(slug);');
    await sqliteRun(
      db,
      'CREATE INDEX IF NOT EXISTS Project_lifecycle_state_updated_at_idx ON Project(lifecycle_state, updated_at);',
    );
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS Project_archived_at_idx ON Project(archived_at);');
    await sqliteRun(
      db,
      'CREATE UNIQUE INDEX IF NOT EXISTS ProjectReplica_project_id_worker_id_key ON ProjectReplica(project_id, worker_id);',
    );
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS ProjectReplica_project_id_idx ON ProjectReplica(project_id);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS ProjectReplica_worker_id_idx ON ProjectReplica(worker_id);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS ProjectReplica_state_idx ON ProjectReplica(state);');
    await sqliteRun(
      db,
      'CREATE UNIQUE INDEX IF NOT EXISTS JobReplica_job_id_worker_id_key ON JobReplica(job_id, worker_id);',
    );
    await sqliteRun(
      db,
      'CREATE UNIQUE INDEX IF NOT EXISTS JobReplica_worker_id_remote_job_id_key ON JobReplica(worker_id, remote_job_id);',
    );
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS JobReplica_job_id_idx ON JobReplica(job_id);');
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS JobReplica_worker_id_idx ON JobReplica(worker_id);');
    await sqliteRun(
      db,
      'CREATE INDEX IF NOT EXISTS ProjectSyncOperation_project_id_created_at_idx ON ProjectSyncOperation(project_id, created_at);',
    );
    await sqliteRun(
      db,
      'CREATE INDEX IF NOT EXISTS ProjectSyncOperation_worker_id_idx ON ProjectSyncOperation(worker_id);',
    );
    await sqliteRun(
      db,
      'CREATE INDEX IF NOT EXISTS ProjectSyncOperation_status_retry_at_idx ON ProjectSyncOperation(status, retry_at);',
    );
    await sqliteRun(
      db,
      `INSERT OR IGNORE INTO JobReplica (
         id, job_id, worker_id, remote_job_id, remote_project_id, role,
         last_synced_at, last_error, created_at, updated_at
       )
       SELECT
         'legacy-' || id || '-' || COALESCE(NULLIF(worker_id, ''), 'local'),
         id,
         COALESCE(NULLIF(worker_id, ''), 'local'),
         remote_job_id,
         NULL,
         'execution',
         remote_sync_at,
         remote_error,
         created_at,
         updated_at
       FROM Job
       WHERE remote_job_id IS NOT NULL AND remote_job_id <> '';`,
    );
    await sqliteRun(db, 'CREATE INDEX IF NOT EXISTS WorkerNode_enabled_idx ON WorkerNode(enabled);');

    const instanceRows = await sqliteAll(db, "SELECT value FROM Settings WHERE key = 'AITK_INSTANCE_ID' LIMIT 1");
    const instanceID = String(instanceRows[0]?.value || process.env.AITK_INSTANCE_ID || randomUUID());
    await sqliteRun(db, 'INSERT OR IGNORE INTO Settings (key, value) VALUES (?, ?)', ['AITK_INSTANCE_ID', instanceID]);
    const projectRootRows = await sqliteAll(db, "SELECT value FROM Settings WHERE key = 'PROJECTS_FOLDER' LIMIT 1");
    const configuredProjectsRoot = path.resolve(projectRootRows[0]?.value || path.join(toolkitRoot, 'projects'));
    const projects = await sqliteAll(
      db,
      'SELECT id, slug, root_path, storage_root_path, lifecycle_state, revision, home_instance_id FROM Project',
    );
    for (const project of projects) {
      const rootPath = path.resolve(project.root_path || path.join(configuredProjectsRoot, String(project.slug)));
      const relative = path.relative(configuredProjectsRoot, rootPath);
      const insideConfiguredRoot =
        relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
      let storageRootPath = path.resolve(
        project.storage_root_path || (insideConfiguredRoot ? configuredProjectsRoot : path.dirname(rootPath)),
      );
      if (storageRootPath === path.parse(storageRootPath).root) storageRootPath = configuredProjectsRoot;
      await sqliteRun(
        db,
        `UPDATE Project
         SET root_path = ?,
             storage_root_path = ?,
             lifecycle_state = COALESCE(NULLIF(lifecycle_state, ''), 'active'),
             revision = CASE WHEN revision IS NULL OR revision < 1 THEN 1 ELSE revision END,
             home_worker_id = COALESCE(NULLIF(home_worker_id, ''), 'local'),
             home_instance_id = COALESCE(NULLIF(home_instance_id, ''), ?)
         WHERE id = ?`,
        [rootPath, storageRootPath, instanceID, String(project.id)],
      );
    }
    await sqliteRun(db, 'COMMIT;');
    transactionOpen = false;
  } finally {
    if (transactionOpen) await sqliteRun(db, 'ROLLBACK;').catch(() => undefined);
    await new Promise(resolve => db.close(resolve));
  }
}

async function hasLegacySqliteTables(filename) {
  const currentSchemaTables = new Set([
    'Settings',
    'Queue',
    'WorkerNode',
    'Job',
    'Project',
    'ProjectReplica',
    'JobReplica',
    'ProjectSyncOperation',
    'sqlite_sequence',
  ]);
  const db = new sqlite3.Database(filename);
  try {
    const tables = await sqliteAll(db, "SELECT name FROM sqlite_master WHERE type = 'table'");
    return tables.some(table => !currentSchemaTables.has(table.name));
  } finally {
    await new Promise(resolve => db.close(resolve));
  }
}

function isSingleFieldMongoIndex(index, field) {
  const entries = Object.entries(index?.key || {});
  return entries.length === 1 && entries[0][0] === field && entries[0][1] === 1;
}

async function dropLegacyMongoJobNameUniqueIndex(db) {
  const jobs = db.collection('jobs');
  const indexes = await jobs.indexes().catch(() => []);
  await Promise.all(
    indexes
      .filter(index => index.unique === true && isSingleFieldMongoIndex(index, 'name'))
      .map(index => jobs.dropIndex(index.name).catch(() => undefined)),
  );
}

if (!['sqlite', 'mongodb'].includes(provider)) {
  throw new Error(`Invalid AITK_DB_PROVIDER "${provider}". Expected "sqlite" or "mongodb".`);
}

process.env.DATABASE_URL = `file:${sqlitePath.replace(/\\/g, '/')}`;

const generatedPrismaClient = path.resolve(process.cwd(), 'src', 'generated', 'prisma', 'client.ts');
const skipPrismaGenerate =
  process.env.AITK_SKIP_PRISMA_GENERATE === '1' && fs.existsSync(generatedPrismaClient);

if (skipPrismaGenerate) {
  console.log('Using the existing generated Prisma client.');
} else {
  console.log(`Generating Prisma client for SQLite fallback (${process.env.DATABASE_URL})...`);
  runPrisma(['generate']);
}

if (provider === 'sqlite') {
  console.log('Preparing SQLite database...');
  fs.mkdirSync(path.dirname(sqlitePath), { recursive: true });
  await backupExistingSqliteDatabase(sqlitePath);
  fs.closeSync(fs.openSync(sqlitePath, 'a'));
  await configureSqliteDatabase(sqlitePath);
  if (await hasLegacySqliteTables(sqlitePath)) {
    console.log('Additional SQLite tables detected; preserving them with additive compatibility changes.');
  } else {
    try {
      runPrisma(['db', 'push'], { nonInteractive: true });
    } catch (error) {
      console.warn('Prisma db push could not apply the schema without data loss or a reset.');
      console.warn('Applying additive SQLite compatibility changes instead.');
    }
  }
  await applySqliteCompatibilitySchema(sqlitePath);
  await applySqliteScopedJobNameIndexes(sqlitePath);
  await configureSqliteDatabase(sqlitePath);
  process.exit(0);
}

if (!mongoUri) {
  throw new Error('AITK_MONGODB_URI is required when AITK_DB_PROVIDER=mongodb.');
}

console.log(`Preparing MongoDB database "${mongoDbName}"...`);
const client = new MongoClient(mongoUri);
try {
  await client.connect();
  const db = client.db(mongoDbName);
  await db.collection('jobs').updateMany(
    { sample_now: { $exists: false } },
    { $set: { sample_now: false } },
  );
  const configuredInstanceID = process.env.AITK_INSTANCE_ID?.trim();
  const existingInstance = await db.collection('settings').findOne({ key: 'AITK_INSTANCE_ID' });
  const instanceID = configuredInstanceID || String(existingInstance?.value || randomUUID());
  await db
    .collection('settings')
    .updateOne({ key: 'AITK_INSTANCE_ID' }, { $set: { key: 'AITK_INSTANCE_ID', value: instanceID } }, { upsert: true });
  const projectsSetting = await db.collection('settings').findOne({ key: 'PROJECTS_FOLDER' });
  const configuredProjectsRoot = path.resolve(projectsSetting?.value || path.join(toolkitRoot, 'projects'));
  const projectRows = await db.collection('projects').find({}).toArray();
  if (projectRows.length > 0) {
    await db.collection('projects').bulkWrite(
      projectRows.map(project => {
        const rootPath = path.resolve(project.root_path || path.join(configuredProjectsRoot, String(project.slug || project.id)));
        const relative = path.relative(configuredProjectsRoot, rootPath);
        const insideConfiguredRoot =
          relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
        let storageRootPath = path.resolve(
          project.storage_root_path || (insideConfiguredRoot ? configuredProjectsRoot : path.dirname(rootPath)),
        );
        if (storageRootPath === path.parse(storageRootPath).root) storageRootPath = configuredProjectsRoot;
        return {
          updateOne: {
            filter: { id: String(project.id) },
            update: {
              $set: {
                root_path: rootPath,
                storage_root_path: storageRootPath,
                lifecycle_state: String(project.lifecycle_state || 'active'),
                archived_at: project.archived_at ?? null,
                revision: Math.max(1, Number(project.revision || 1)),
                operation_started_at: project.operation_started_at ?? null,
                operation_error: project.operation_error ?? null,
                home_worker_id: String(project.home_worker_id || 'local'),
                home_instance_id: String(project.home_instance_id || instanceID),
              },
            },
          },
        };
      }),
      { ordered: false },
    );
  }
  const legacyRemoteJobs = await db
    .collection('jobs')
    .find({ remote_job_id: { $type: 'string', $ne: '' } })
    .toArray();
  if (legacyRemoteJobs.length > 0) {
    await db.collection('job_replicas').bulkWrite(
      legacyRemoteJobs.map(job => ({
        updateOne: {
          filter: { job_id: String(job.id), worker_id: String(job.worker_id || 'local') },
          update: {
            $setOnInsert: {
              id: `legacy-${String(job.id)}-${String(job.worker_id || 'local')}`,
              job_id: String(job.id),
              worker_id: String(job.worker_id || 'local'),
              remote_job_id: String(job.remote_job_id),
              remote_project_id: null,
              role: 'execution',
              last_synced_at: job.remote_sync_at ?? null,
              last_error: job.remote_error ?? null,
              created_at: job.created_at ?? new Date(),
              updated_at: job.updated_at ?? new Date(),
            },
          },
          upsert: true,
        },
      })),
      { ordered: false },
    );
  }
  await dropLegacyMongoJobNameUniqueIndex(db);
  await Promise.all([
    db.collection('jobs').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { project_id: 1, name: 1 }, unique: true },
      { key: { name: 1 } },
      { key: { status: 1 } },
      { key: { worker_id: 1 } },
      { key: { remote_job_id: 1 } },
      { key: { gpu_ids: 1 } },
      { key: { job_type: 1 } },
      { key: { job_ref: 1 } },
      { key: { project_id: 1 } },
      { key: { queue_position: 1 } },
    ]),
    db.collection('projects').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { slug: 1 }, unique: true },
      { key: { updated_at: -1 } },
      { key: { lifecycle_state: 1, updated_at: -1 } },
      { key: { archived_at: -1 } },
    ]),
    db.collection('project_replicas').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { project_id: 1, worker_id: 1 }, unique: true },
      { key: { project_id: 1 } },
      { key: { worker_id: 1 } },
      { key: { state: 1 } },
    ]),
    db.collection('job_replicas').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { job_id: 1, worker_id: 1 }, unique: true },
      { key: { worker_id: 1, remote_job_id: 1 }, unique: true },
      { key: { job_id: 1 } },
      { key: { worker_id: 1 } },
    ]),
    db.collection('project_sync_operations').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { project_id: 1, created_at: -1 } },
      { key: { worker_id: 1 } },
      { key: { status: 1, retry_at: 1 } },
    ]),
    db.collection('queues').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { worker_id: 1, gpu_ids: 1 }, unique: true },
      { key: { worker_id: 1 } },
      { key: { gpu_ids: 1 } },
    ]),
    db.collection('worker_nodes').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { name: 1 }, unique: true },
      { key: { enabled: 1 } },
    ]),
    db.collection('settings').createIndexes([{ key: { key: 1 }, unique: true }]),
    db.collection('metrics').createIndexes([
      { key: { job_id: 1, step: 1, key: 1 }, unique: true },
      { key: { job_id: 1, key: 1, step: 1 } },
    ]),
    db.collection('metric_keys').createIndexes([{ key: { job_id: 1, key: 1 }, unique: true }]),
  ]);
  console.log('MongoDB indexes are ready.');
} finally {
  await client.close();
}
