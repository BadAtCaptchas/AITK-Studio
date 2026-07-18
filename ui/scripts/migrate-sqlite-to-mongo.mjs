import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import sqlite3 from 'sqlite3';
import { MongoClient } from 'mongodb';

const toolkitRoot = path.resolve(process.cwd(), '..');
const sqlitePath = path.resolve(process.env.AITK_SQLITE_PATH || path.join(toolkitRoot, 'aitk_db.db'));
const mongoUri = process.env.AITK_MONGODB_URI?.trim();
const mongoDbName = process.env.AITK_MONGODB_DB?.trim() || 'ai_toolkit';
const defaultTrainingRoot = path.join(toolkitRoot, 'output');

if (!mongoUri) {
  throw new Error('AITK_MONGODB_URI is required to migrate SQLite data to MongoDB.');
}
if (!fs.existsSync(sqlitePath)) {
  throw new Error(`SQLite database not found: ${sqlitePath}`);
}

function openDb(filename) {
  const db = new sqlite3.Database(filename);
  db.configure('busyTimeout', 30_000);
  return db;
}

function all(db, sql, params = []) {
  return new Promise((resolve, reject) => {
    db.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows);
    });
  });
}

function closeDb(db) {
  return new Promise((resolve, reject) => {
    db.close(err => (err ? reject(err) : resolve()));
  });
}

function asDate(value, fallback = new Date()) {
  if (value instanceof Date) return value;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? fallback : date;
}

function normalizeJob(row) {
  const now = new Date();
  return {
    id: String(row.id),
    name: String(row.name || ''),
    project_id: row.project_id == null ? null : String(row.project_id),
    worker_id: String(row.worker_id || 'local'),
    remote_job_id: row.remote_job_id == null ? null : String(row.remote_job_id),
    remote_sync_at: row.remote_sync_at == null ? null : asDate(row.remote_sync_at, null),
    remote_error: row.remote_error == null ? null : String(row.remote_error),
    gpu_ids: String(row.gpu_ids || ''),
    job_config: String(row.job_config || ''),
    created_at: asDate(row.created_at, now),
    updated_at: asDate(row.updated_at, now),
    status: String(row.status || 'stopped'),
    stop: Boolean(row.stop),
    return_to_queue: Boolean(row.return_to_queue),
    step: Number(row.step || 0),
    info: String(row.info || ''),
    speed_string: String(row.speed_string || ''),
    queue_position: Number(row.queue_position || 0),
    pid: row.pid == null ? null : Number(row.pid),
    job_type: String(row.job_type || 'train'),
    job_ref: row.job_ref == null ? null : String(row.job_ref),
    save_now: Boolean(row.save_now),
    sample_now: Boolean(row.sample_now),
  };
}

function normalizeProject(row, projectsRoot, instanceID) {
  const now = new Date();
  const rootPath = path.resolve(row.root_path || path.join(projectsRoot, String(row.slug || row.id)));
  const relative = path.relative(projectsRoot, rootPath);
  const isConfiguredDescendant =
    relative !== '' && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
  let storageRootPath = path.resolve(
    row.storage_root_path || (isConfiguredDescendant ? projectsRoot : path.dirname(rootPath)),
  );
  if (storageRootPath === path.parse(storageRootPath).root) storageRootPath = projectsRoot;
  return {
    id: String(row.id),
    slug: String(row.slug || ''),
    name: String(row.name || ''),
    description: String(row.description || ''),
    badge_asset: row.badge_asset == null ? null : String(row.badge_asset),
    root_path: rootPath,
    storage_root_path: storageRootPath,
    lifecycle_state: String(row.lifecycle_state || 'active'),
    archived_at: row.archived_at == null ? null : asDate(row.archived_at, null),
    revision: Math.max(1, Number(row.revision || 1)),
    operation_started_at: row.operation_started_at == null ? null : asDate(row.operation_started_at, null),
    operation_error: row.operation_error == null ? null : String(row.operation_error),
    home_worker_id: String(row.home_worker_id || 'local'),
    home_instance_id: String(row.home_instance_id || instanceID),
    created_at: asDate(row.created_at, now),
    updated_at: asDate(row.updated_at, now),
  };
}

function normalizeReplicaDates(row) {
  const now = new Date();
  return {
    ...row,
    id: String(row.id),
    created_at: asDate(row.created_at, now),
    updated_at: asDate(row.updated_at, now),
    ...(row.last_synced_at == null ? { last_synced_at: null } : { last_synced_at: asDate(row.last_synced_at, null) }),
    ...(row.retry_at == null ? {} : { retry_at: asDate(row.retry_at, null) }),
  };
}

async function tableExists(db, tableName) {
  const rows = await all(db, "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", [tableName]);
  return rows.length > 0;
}

async function rowsIfTableExists(db, tableName) {
  return (await tableExists(db, tableName)) ? all(db, `SELECT * FROM "${tableName}"`) : [];
}

async function ensureIndexes(db) {
  const jobIndexes = await db.collection('jobs').indexes().catch(() => []);
  await Promise.all(
    jobIndexes
      .filter(index => {
        const entries = Object.entries(index?.key || {});
        return index.unique === true && entries.length === 1 && entries[0][0] === 'name' && entries[0][1] === 1;
      })
      .map(index => db.collection('jobs').dropIndex(index.name).catch(() => undefined)),
  );

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
}

async function importLossLog(db, job, trainingRoot) {
  const logPath = path.join(trainingRoot, job.name, 'loss_log.db');
  if (!fs.existsSync(logPath)) {
    return { metricKeys: 0, metrics: 0 };
  }

  const sqlite = openDb(logPath);
  try {
    const keys = await all(sqlite, 'SELECT key, first_seen_step, last_seen_step FROM metric_keys');
    const metrics = await all(
      sqlite,
      `
      SELECT
        m.step AS step,
        s.wall_time AS wall_time,
        m.key AS key,
        m.value_real AS value_real,
        m.value_text AS value_text
      FROM metrics m
      JOIN steps s ON s.step = m.step
      `,
    );

    if (keys.length > 0) {
      await db.collection('metric_keys').bulkWrite(
        keys.map(row => ({
          updateOne: {
            filter: { job_id: job.id, key: String(row.key) },
            update: {
              $setOnInsert: { job_id: job.id, key: String(row.key) },
              $min: { first_seen_step: Number(row.first_seen_step || 0) },
              $max: { last_seen_step: Number(row.last_seen_step || 0) },
            },
            upsert: true,
          },
        })),
        { ordered: false },
      );
    }

    if (metrics.length > 0) {
      const batchSize = 1000;
      for (let i = 0; i < metrics.length; i += batchSize) {
        const batch = metrics.slice(i, i + batchSize);
        await db.collection('metrics').bulkWrite(
          batch.map(row => ({
            updateOne: {
              filter: { job_id: job.id, step: Number(row.step), key: String(row.key) },
              update: {
                $set: {
                  job_id: job.id,
                  step: Number(row.step),
                  key: String(row.key),
                  wall_time: Number(row.wall_time || 0),
                  value_real: row.value_real == null ? null : Number(row.value_real),
                  value_text: row.value_text == null ? null : String(row.value_text),
                },
              },
              upsert: true,
            },
          })),
          { ordered: false },
        );
      }
    }

    return { metricKeys: keys.length, metrics: metrics.length };
  } finally {
    await closeDb(sqlite);
  }
}

const sqlite = openDb(sqlitePath);
const client = new MongoClient(mongoUri);

try {
  const [settingsRows, queueRows, projectRows, jobRows, projectReplicaRows, jobReplicaRows, syncOperationRows] = await Promise.all([
    all(sqlite, 'SELECT key, value FROM Settings'),
    all(sqlite, 'SELECT * FROM Queue'),
    rowsIfTableExists(sqlite, 'Project'),
    all(sqlite, 'SELECT * FROM Job'),
    rowsIfTableExists(sqlite, 'ProjectReplica'),
    rowsIfTableExists(sqlite, 'JobReplica'),
    rowsIfTableExists(sqlite, 'ProjectSyncOperation'),
  ]);

  const trainingSetting = settingsRows.find(row => row.key === 'TRAINING_FOLDER');
  const trainingRoot = trainingSetting?.value || defaultTrainingRoot;
  const projectsSetting = settingsRows.find(row => row.key === 'PROJECTS_FOLDER');
  const projectsRoot = path.resolve(projectsSetting?.value || path.join(toolkitRoot, 'projects'));
  const instanceSetting = settingsRows.find(row => row.key === 'AITK_INSTANCE_ID');
  const instanceID = process.env.AITK_INSTANCE_ID?.trim() || String(instanceSetting?.value || randomUUID());
  const settingsToImport = instanceSetting
    ? settingsRows
    : [...settingsRows, { key: 'AITK_INSTANCE_ID', value: instanceID }];
  const projects = projectRows.map(row => normalizeProject(row, projectsRoot, instanceID));

  await client.connect();
  const mongo = client.db(mongoDbName);
  await ensureIndexes(mongo);

  if (settingsToImport.length > 0) {
    await mongo.collection('settings').bulkWrite(
      settingsToImport.map(row => ({
        updateOne: {
          filter: { key: String(row.key) },
          update: { $set: { key: String(row.key), value: String(row.value || '') } },
          upsert: true,
        },
      })),
      { ordered: false },
    );
  }

  if (projects.length > 0) {
    await mongo.collection('projects').bulkWrite(
      projects.map(project => ({
        updateOne: {
          filter: { id: project.id },
          update: { $set: project },
          upsert: true,
        },
      })),
      { ordered: false },
    );
  }

  if (queueRows.length > 0) {
    await mongo.collection('queues').bulkWrite(
      queueRows.map(row => ({
        updateOne: {
          filter: { worker_id: String(row.worker_id || 'local'), gpu_ids: String(row.gpu_ids) },
          update: {
            $set: {
              id: Number(row.id),
              worker_id: String(row.worker_id || 'local'),
              gpu_ids: String(row.gpu_ids),
              is_running: Boolean(row.is_running),
            },
          },
          upsert: true,
        },
      })),
      { ordered: false },
    );
  }

  const jobs = jobRows.map(normalizeJob);
  if (jobs.length > 0) {
    await mongo.collection('jobs').bulkWrite(
      jobs.map(job => ({
        updateOne: {
          filter: { id: job.id },
          update: { $set: job },
          upsert: true,
        },
      })),
      { ordered: false },
    );
  }

  const projectReplicas = projectReplicaRows.map(normalizeReplicaDates);
  if (projectReplicas.length > 0) {
    await mongo.collection('project_replicas').bulkWrite(
      projectReplicas.map(replica => ({
        updateOne: { filter: { id: replica.id }, update: { $set: replica }, upsert: true },
      })),
      { ordered: false },
    );
  }

  const jobReplicas = jobReplicaRows.map(normalizeReplicaDates);
  if (jobReplicas.length > 0) {
    await mongo.collection('job_replicas').bulkWrite(
      jobReplicas.map(replica => ({
        updateOne: { filter: { id: replica.id }, update: { $set: replica }, upsert: true },
      })),
      { ordered: false },
    );
  }
  const migratedReplicaKeys = new Set(jobReplicas.map(replica => `${replica.job_id}\u0000${replica.worker_id}`));
  const legacyJobReplicas = jobs
    .filter(job => job.remote_job_id && !migratedReplicaKeys.has(`${job.id}\u0000${job.worker_id}`))
    .map(job => ({
      id: `legacy-${job.id}-${job.worker_id}`,
      job_id: job.id,
      worker_id: job.worker_id,
      remote_job_id: job.remote_job_id,
      remote_project_id: null,
      role: 'execution',
      last_synced_at: job.remote_sync_at,
      last_error: job.remote_error,
      created_at: job.created_at,
      updated_at: job.updated_at,
    }));
  if (legacyJobReplicas.length > 0) {
    await mongo.collection('job_replicas').bulkWrite(
      legacyJobReplicas.map(replica => ({
        updateOne: {
          filter: { job_id: replica.job_id, worker_id: replica.worker_id },
          update: { $setOnInsert: replica },
          upsert: true,
        },
      })),
      { ordered: false },
    );
  }

  const syncOperations = syncOperationRows.map(normalizeReplicaDates);
  if (syncOperations.length > 0) {
    await mongo.collection('project_sync_operations').bulkWrite(
      syncOperations.map(operation => ({
        updateOne: { filter: { id: operation.id }, update: { $set: operation }, upsert: true },
      })),
      { ordered: false },
    );
  }

  let metricKeyCount = 0;
  let metricCount = 0;
  const projectsByID = new Map(projects.map(project => [project.id, project]));
  for (const job of jobs) {
    const project = job.project_id ? projectsByID.get(job.project_id) : null;
    const imported = await importLossLog(mongo, job, project ? path.join(project.root_path, 'runs') : trainingRoot);
    metricKeyCount += imported.metricKeys;
    metricCount += imported.metrics;
  }

  console.log(`Migrated settings: ${settingsToImport.length}`);
  console.log(`Migrated queues: ${queueRows.length}`);
  console.log(`Migrated projects: ${projects.length}`);
  console.log(`Migrated jobs: ${jobs.length}`);
  console.log(`Migrated project replicas: ${projectReplicas.length}`);
  console.log(`Migrated job replicas: ${jobReplicas.length + legacyJobReplicas.length}`);
  console.log(`Migrated project sync operations: ${syncOperations.length}`);
  console.log(`Migrated metric keys: ${metricKeyCount}`);
  console.log(`Migrated metric points: ${metricCount}`);
  console.log('SQLite files were left untouched.');
} finally {
  await closeDb(sqlite);
  await client.close();
}
