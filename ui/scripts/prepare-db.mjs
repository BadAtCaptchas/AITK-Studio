import { execFileSync } from 'child_process';
import { randomUUID } from 'crypto';
import fs from 'fs';
import { createRequire } from 'module';
import path from 'path';
import { MongoClient } from 'mongodb';
import sqlite3 from 'sqlite3';
import { backupExistingSqliteDatabase } from './sqlite-backup.mjs';
import {
  preflightSqliteGlobalUpgrade,
  upgradeSqliteGlobalWorkspace,
  upgradeMongoGlobalWorkspace,
} from './global-workspace-upgrade.mjs';
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

    await sqliteRun(
      db,
      `
      CREATE TABLE IF NOT EXISTS JobReplica (
        id TEXT PRIMARY KEY NOT NULL,
        job_id TEXT NOT NULL,
        worker_id TEXT NOT NULL,
        remote_job_id TEXT NOT NULL,
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
    await sqliteRun(db, 'CREATE UNIQUE INDEX IF NOT EXISTS Job_name_key ON Job(name);');
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
      `INSERT OR IGNORE INTO JobReplica (
         id, job_id, worker_id, remote_job_id, role,
         last_synced_at, last_error, created_at, updated_at
       )
       SELECT
         'legacy-' || id || '-' || COALESCE(NULLIF(worker_id, ''), 'local'),
         id,
         COALESCE(NULLIF(worker_id, ''), 'local'),
         remote_job_id,
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
    await sqliteRun(db, 'COMMIT;');
    transactionOpen = false;
  } finally {
    if (transactionOpen) await sqliteRun(db, 'ROLLBACK;').catch(() => undefined);
    await new Promise(resolve => db.close(resolve));
  }
}

async function hasLegacySqliteTables(filename) {
  const currentSchemaTables = new Set(['Settings', 'Queue', 'WorkerNode', 'Job', 'JobReplica', 'sqlite_sequence']);
  const db = new sqlite3.Database(filename);
  try {
    const tables = await sqliteAll(db, "SELECT name FROM sqlite_master WHERE type = 'table'");
    return tables.some(table => !currentSchemaTables.has(table.name));
  } finally {
    await new Promise(resolve => db.close(resolve));
  }
}

if (!['sqlite', 'mongodb'].includes(provider)) {
  throw new Error(`Invalid AITK_DB_PROVIDER "${provider}". Expected "sqlite" or "mongodb".`);
}

process.env.DATABASE_URL = `file:${sqlitePath.replace(/\\/g, '/')}`;

const generatedPrismaClient = path.resolve(process.cwd(), 'src', 'generated', 'prisma', 'client.ts');
const skipPrismaGenerate = process.env.AITK_SKIP_PRISMA_GENERATE === '1' && fs.existsSync(generatedPrismaClient);

if (skipPrismaGenerate) {
  console.log('Using the existing generated Prisma client.');
} else {
  console.log(`Generating Prisma client for SQLite fallback (${process.env.DATABASE_URL})...`);
  runPrisma(['generate']);
}

if (provider === 'sqlite') {
  console.log('Preparing SQLite database...');
  fs.mkdirSync(path.dirname(sqlitePath), { recursive: true });
  if (fs.existsSync(sqlitePath)) {
    const preflight = new sqlite3.Database(sqlitePath, sqlite3.OPEN_READONLY);
    try {
      await preflightSqliteGlobalUpgrade(preflight);
    } finally {
      await new Promise(resolve => preflight.close(resolve));
    }
  }
  await backupExistingSqliteDatabase(sqlitePath);
  fs.closeSync(fs.openSync(sqlitePath, 'a'));
  const upgrade = new sqlite3.Database(sqlitePath);
  try {
    await upgradeSqliteGlobalWorkspace(upgrade);
  } finally {
    await new Promise(resolve => upgrade.close(resolve));
  }
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
  await upgradeMongoGlobalWorkspace(db);
  await db.collection('jobs').updateMany({ sample_now: { $exists: false } }, { $set: { sample_now: false } });
  const configuredInstanceID = process.env.AITK_INSTANCE_ID?.trim();
  const existingInstance = await db.collection('settings').findOne({ key: 'AITK_INSTANCE_ID' });
  const instanceID = configuredInstanceID || String(existingInstance?.value || randomUUID());
  await db
    .collection('settings')
    .updateOne({ key: 'AITK_INSTANCE_ID' }, { $set: { key: 'AITK_INSTANCE_ID', value: instanceID } }, { upsert: true });
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
  await Promise.all([
    db
      .collection('jobs')
      .createIndexes([
        { key: { id: 1 }, unique: true },
        { key: { name: 1 }, unique: true },
        { key: { status: 1 } },
        { key: { worker_id: 1 } },
        { key: { remote_job_id: 1 } },
        { key: { gpu_ids: 1 } },
        { key: { job_type: 1 } },
        { key: { job_ref: 1 } },
        { key: { queue_position: 1 } },
      ]),
    db
      .collection('job_replicas')
      .createIndexes([
        { key: { id: 1 }, unique: true },
        { key: { job_id: 1, worker_id: 1 }, unique: true },
        { key: { worker_id: 1, remote_job_id: 1 }, unique: true },
        { key: { job_id: 1 } },
        { key: { worker_id: 1 } },
      ]),
    db
      .collection('queues')
      .createIndexes([
        { key: { id: 1 }, unique: true },
        { key: { worker_id: 1, gpu_ids: 1 }, unique: true },
        { key: { worker_id: 1 } },
        { key: { gpu_ids: 1 } },
      ]),
    db
      .collection('worker_nodes')
      .createIndexes([{ key: { id: 1 }, unique: true }, { key: { name: 1 }, unique: true }, { key: { enabled: 1 } }]),
    db.collection('settings').createIndexes([{ key: { key: 1 }, unique: true }]),
    db
      .collection('metrics')
      .createIndexes([{ key: { job_id: 1, step: 1, key: 1 }, unique: true }, { key: { job_id: 1, key: 1, step: 1 } }]),
    db.collection('metric_keys').createIndexes([{ key: { job_id: 1, key: 1 }, unique: true }]),
  ]);
  console.log('MongoDB indexes are ready.');
} finally {
  await client.close();
}
