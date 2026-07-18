import { randomUUID } from 'crypto';
import fs from 'fs';
import path from 'path';
import { MongoClient, type Collection, type Db, type Document } from 'mongodb';
import { PrismaBetterSqlite3 } from '@prisma/adapter-better-sqlite3';
import { Prisma, PrismaClient } from '../generated/prisma/client';
import sqlite3 from 'sqlite3';
import { TOOLKIT_ROOT } from '../paths';
import type {
  Job,
  JobReplica,
  Project,
  ProjectLifecycleState,
  ProjectReplicaState,
  ProjectSyncOperation,
  Queue,
  WorkerNode,
} from '../types';
import { buildMetricSeriesResult, normalizeMetricMaxPoints } from './metricsDownsample';

export type DatabaseProvider = 'sqlite' | 'mongodb';

export type DatabaseConfig = {
  provider: DatabaseProvider;
  sqlitePath: string;
  sqliteUrl: string;
  mongoUri: string | null;
  mongoDb: string;
};

export type SettingRecord = {
  id?: number;
  key: string;
  value: string;
};

export type JobCreateInput = {
  id?: string;
  name: string;
  project_id?: string | null;
  worker_id?: string;
  remote_job_id?: string | null;
  remote_sync_at?: Date | string | null;
  remote_error?: string | null;
  gpu_ids: string;
  job_config: string;
  status?: string;
  stop?: boolean;
  return_to_queue?: boolean;
  step?: number;
  info?: string;
  speed_string?: string;
  queue_position?: number;
  pid?: number | null;
  job_type?: string;
  job_ref?: string | null;
  save_now?: boolean;
  sample_now?: boolean;
};

export type JobUpdateInput = Partial<Omit<JobCreateInput, 'id'>>;

export type QueueCreateInput = {
  id?: number;
  worker_id?: string;
  gpu_ids: string;
  is_running?: boolean;
};

export type QueueUpdateInput = Partial<Pick<Queue, 'worker_id' | 'gpu_ids' | 'is_running'>>;

export type WorkerNodeRecord = WorkerNode & {
  api_token: string;
};

export type WorkerNodeCreateInput = {
  id?: string;
  name: string;
  base_url: string;
  api_token: string;
  enabled?: boolean;
  offline_bypass_enabled?: boolean;
  last_status?: string;
  last_error?: string | null;
  last_checked_at?: Date | string | null;
  capabilities?: string;
  gpus?: string;
};

export type WorkerNodeUpdateInput = Partial<Omit<WorkerNodeCreateInput, 'id'>>;

export type ProjectCreateInput = {
  id?: string;
  slug: string;
  name: string;
  description?: string;
  badge_asset?: string | null;
  root_path?: string;
  storage_root_path?: string;
  lifecycle_state?: ProjectLifecycleState;
  archived_at?: Date | string | null;
  revision?: number;
  operation_started_at?: Date | string | null;
  operation_error?: string | null;
  home_worker_id?: string;
  home_instance_id?: string;
};

export type ProjectUpdateInput = Partial<Omit<ProjectCreateInput, 'id' | 'slug'>> & {
  slug?: string;
};

export type ProjectReplicaCreateInput = {
  id?: string;
  project_id: string;
  worker_id: string;
  remote_project_id?: string | null;
  remote_instance_id?: string | null;
  role?: ProjectReplicaState['role'];
  state?: ProjectReplicaState['state'];
  base_manifest_hash?: string | null;
  local_manifest_hash?: string | null;
  remote_manifest_hash?: string | null;
  last_synced_at?: Date | string | null;
  last_error?: string | null;
  auto_pull_results?: boolean;
};

export type ProjectReplicaUpdateInput = Partial<Omit<ProjectReplicaCreateInput, 'id' | 'project_id' | 'worker_id'>>;

export type JobReplicaCreateInput = {
  id?: string;
  job_id: string;
  worker_id: string;
  remote_job_id: string;
  remote_project_id?: string | null;
  role?: JobReplica['role'];
  last_synced_at?: Date | string | null;
  last_error?: string | null;
};

export type JobReplicaUpdateInput = Partial<Omit<JobReplicaCreateInput, 'id' | 'job_id' | 'worker_id'>>;

export type ProjectSyncOperationCreateInput = {
  id?: string;
  project_id: string;
  worker_id: string;
  profile: ProjectSyncOperation['profile'];
  status?: ProjectSyncOperation['status'];
  phase?: string;
  files_total?: number;
  files_done?: number;
  bytes_total?: number;
  bytes_done?: number;
  retry_count?: number;
  retry_at?: Date | string | null;
  base_manifest_hash?: string | null;
  source_manifest_hash?: string | null;
  target_manifest_hash?: string | null;
  conflicts?: string;
  error?: string | null;
};

export type ProjectSyncOperationUpdateInput = Partial<
  Omit<ProjectSyncOperationCreateInput, 'id' | 'project_id' | 'worker_id' | 'profile'>
>;

export type LossPoint = {
  step: number;
  wall_time: number;
  value: number | null;
  value_text?: string | null;
};

export type LossLogResult = {
  key: string;
  keys: string[];
  points: LossPoint[];
};

export type MetricKeyInfo = {
  key: string;
  first_seen_step: number | null;
  last_seen_step: number | null;
};

export type MetricSeriesResult = {
  key: string;
  totalCount: number;
  firstStep: number | null;
  lastStep: number | null;
  latest: LossPoint | null;
  downsampled: boolean;
  points: LossPoint[];
};

export type MetricsResult = {
  keys: string[];
  keyInfo: MetricKeyInfo[];
  series: Record<string, MetricSeriesResult>;
};

function coerceMetricValue(value: number | null | undefined, valueText: string | null | undefined): number | null {
  if (value != null) return Number(value);
  if (!valueText) return null;
  const numericValue = Number(valueText);
  return Number.isFinite(numericValue) ? numericValue : null;
}

const DEFAULT_MONGODB_DB = 'ai_toolkit';
const SQLITE_BUSY_TIMEOUT_MS = 30_000;

declare global {
  // eslint-disable-next-line no-var
  var __aitkPrismaClient: PrismaClient | undefined;
  // eslint-disable-next-line no-var
  var __aitkMongoClientPromise: Promise<MongoClient> | undefined;
}

export class DatabaseConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'DatabaseConfigError';
  }
}

export class UniqueConstraintError extends Error {
  code = 'P2002';

  constructor(message: string) {
    super(message);
    this.name = 'UniqueConstraintError';
  }
}

function normalizeProvider(rawProvider?: string): DatabaseProvider {
  const provider = (rawProvider || 'sqlite').trim().toLowerCase();
  if (provider === 'sqlite' || provider === 'mongodb') {
    return provider;
  }
  throw new DatabaseConfigError(`Invalid AITK_DB_PROVIDER "${rawProvider}". Expected "sqlite" or "mongodb".`);
}

function normalizeSqlitePath(rawPath?: string) {
  return path.resolve(rawPath && rawPath.trim() ? rawPath : path.join(TOOLKIT_ROOT, 'aitk_db.db'));
}

function sqliteFileUrl(sqlitePath: string) {
  return `file:${sqlitePath.replace(/\\/g, '/')}`;
}

export function getDatabaseConfig(): DatabaseConfig {
  const provider = normalizeProvider(process.env.AITK_DB_PROVIDER);
  const sqlitePath = normalizeSqlitePath(process.env.AITK_SQLITE_PATH);
  const mongoUri = process.env.AITK_MONGODB_URI?.trim() || null;
  const mongoDb = process.env.AITK_MONGODB_DB?.trim() || DEFAULT_MONGODB_DB;

  if (provider === 'mongodb' && !mongoUri) {
    throw new DatabaseConfigError('AITK_MONGODB_URI is required when AITK_DB_PROVIDER=mongodb.');
  }

  return {
    provider,
    sqlitePath,
    sqliteUrl: sqliteFileUrl(sqlitePath),
    mongoUri,
    mongoDb,
  };
}

export function isMongoProvider() {
  return getDatabaseConfig().provider === 'mongodb';
}

function getPrisma() {
  if (!globalThis.__aitkPrismaClient) {
    const config = getDatabaseConfig();
    process.env.DATABASE_URL = process.env.DATABASE_URL || config.sqliteUrl;
    const adapter = new PrismaBetterSqlite3(
      { url: config.sqliteUrl, timeout: SQLITE_BUSY_TIMEOUT_MS },
      { timestampFormat: 'unixepoch-ms' },
    );
    globalThis.__aitkPrismaClient = new PrismaClient({ adapter });
  }
  return globalThis.__aitkPrismaClient;
}

async function getMongoClient() {
  const config = getDatabaseConfig();
  if (!config.mongoUri) {
    throw new DatabaseConfigError('AITK_MONGODB_URI is required when AITK_DB_PROVIDER=mongodb.');
  }
  if (!globalThis.__aitkMongoClientPromise) {
    globalThis.__aitkMongoClientPromise = new MongoClient(config.mongoUri).connect();
  }
  return globalThis.__aitkMongoClientPromise;
}

async function getMongoDb() {
  const config = getDatabaseConfig();
  const client = await getMongoClient();
  return client.db(config.mongoDb);
}

function duplicateKeyToUniqueError(error: unknown): never {
  if (typeof error === 'object' && error !== null && (error as any).code === 11000) {
    throw new UniqueConstraintError('Unique constraint failed');
  }
  throw error;
}

function isSingleFieldIndex(index: Document, field: string) {
  const key = index?.key || {};
  const entries = Object.entries(key);
  return entries.length === 1 && entries[0][0] === field && entries[0][1] === 1;
}

async function dropLegacyMongoJobNameUniqueIndex(mongo: Db) {
  const jobs = mongoCollection(mongo, 'jobs');
  const indexes = await jobs.indexes().catch(() => []);
  const legacyIndexNames = indexes
    .filter(index => index.unique === true && isSingleFieldIndex(index, 'name'))
    .map(index => index.name)
    .filter((name): name is string => typeof name === 'string');
  await Promise.all(legacyIndexNames.map(name => jobs.dropIndex(name).catch(() => undefined)));
}

function parseDate(value: unknown, fallback = new Date()) {
  if (value instanceof Date) return value;
  if (typeof value === 'string' || typeof value === 'number') {
    const date = new Date(value);
    if (!Number.isNaN(date.getTime())) return date;
  }
  return fallback;
}

function normalizeJob(raw: any): Job | null {
  if (!raw) return null;
  return {
    id: String(raw.id),
    name: String(raw.name ?? ''),
    project_id: raw.project_id == null ? null : String(raw.project_id),
    worker_id: String(raw.worker_id ?? 'local'),
    remote_job_id: raw.remote_job_id == null ? null : String(raw.remote_job_id),
    remote_sync_at: raw.remote_sync_at == null ? null : parseDate(raw.remote_sync_at),
    remote_error: raw.remote_error == null ? null : String(raw.remote_error),
    gpu_ids: String(raw.gpu_ids ?? ''),
    job_config: String(raw.job_config ?? ''),
    created_at: parseDate(raw.created_at),
    updated_at: parseDate(raw.updated_at),
    status: String(raw.status ?? 'stopped'),
    stop: Boolean(raw.stop),
    return_to_queue: Boolean(raw.return_to_queue),
    step: Number(raw.step ?? 0),
    info: String(raw.info ?? ''),
    speed_string: String(raw.speed_string ?? ''),
    queue_position: Number(raw.queue_position ?? 0),
    pid: raw.pid == null ? null : Number(raw.pid),
    job_type: String(raw.job_type ?? 'train'),
    job_ref: raw.job_ref == null ? null : String(raw.job_ref),
    save_now: Boolean(raw.save_now),
    sample_now: Boolean(raw.sample_now),
  };
}

function normalizeProject(raw: any): Project | null {
  if (!raw) return null;
  const rootPath = String(raw.root_path ?? '');
  const lifecycleStates = new Set<ProjectLifecycleState>(['creating', 'active', 'archived', 'relocating', 'purging']);
  const rawLifecycleState = String(raw.lifecycle_state ?? 'active') as ProjectLifecycleState;
  return {
    id: String(raw.id),
    slug: String(raw.slug ?? ''),
    name: String(raw.name ?? ''),
    description: String(raw.description ?? ''),
    badge_asset: raw.badge_asset == null ? null : String(raw.badge_asset),
    root_path: rootPath,
    storage_root_path: String(raw.storage_root_path || (rootPath ? path.dirname(rootPath) : '')),
    lifecycle_state: lifecycleStates.has(rawLifecycleState) ? rawLifecycleState : 'active',
    archived_at: raw.archived_at == null ? null : parseDate(raw.archived_at),
    revision: Math.max(0, Number(raw.revision ?? 1)),
    operation_started_at: raw.operation_started_at == null ? null : parseDate(raw.operation_started_at),
    operation_error: raw.operation_error == null ? null : String(raw.operation_error),
    home_worker_id: String(raw.home_worker_id ?? 'local'),
    home_instance_id: String(raw.home_instance_id ?? ''),
    created_at: parseDate(raw.created_at),
    updated_at: parseDate(raw.updated_at),
  };
}

function normalizeProjectReplica(raw: any): ProjectReplicaState | null {
  if (!raw) return null;
  const states = new Set<ProjectReplicaState['state']>([
    'creating',
    'syncing',
    'in_sync',
    'dirty',
    'conflict',
    'waiting_for_job',
    'waiting_for_worker',
    'offline',
    'incompatible',
    'error',
    'detached',
  ]);
  const rawState = String(raw.state ?? 'creating') as ProjectReplicaState['state'];
  return {
    id: String(raw.id),
    project_id: String(raw.project_id),
    worker_id: String(raw.worker_id),
    remote_project_id: raw.remote_project_id == null ? null : String(raw.remote_project_id),
    remote_instance_id: raw.remote_instance_id == null ? null : String(raw.remote_instance_id),
    role: raw.role === 'home' ? 'home' : 'execution',
    state: states.has(rawState) ? rawState : 'error',
    base_manifest_hash: raw.base_manifest_hash == null ? null : String(raw.base_manifest_hash),
    local_manifest_hash: raw.local_manifest_hash == null ? null : String(raw.local_manifest_hash),
    remote_manifest_hash: raw.remote_manifest_hash == null ? null : String(raw.remote_manifest_hash),
    last_synced_at: raw.last_synced_at == null ? null : parseDate(raw.last_synced_at),
    last_error: raw.last_error == null ? null : String(raw.last_error),
    auto_pull_results: raw.auto_pull_results !== false,
    created_at: parseDate(raw.created_at),
    updated_at: parseDate(raw.updated_at),
  };
}

function normalizeJobReplica(raw: any): JobReplica | null {
  if (!raw) return null;
  return {
    id: String(raw.id),
    job_id: String(raw.job_id),
    worker_id: String(raw.worker_id),
    remote_job_id: String(raw.remote_job_id),
    remote_project_id: raw.remote_project_id == null ? null : String(raw.remote_project_id),
    role: raw.role === 'home' ? 'home' : 'execution',
    last_synced_at: raw.last_synced_at == null ? null : parseDate(raw.last_synced_at),
    last_error: raw.last_error == null ? null : String(raw.last_error),
    created_at: parseDate(raw.created_at),
    updated_at: parseDate(raw.updated_at),
  };
}

function normalizeProjectSyncOperation(raw: any): ProjectSyncOperation | null {
  if (!raw) return null;
  const profiles = new Set<ProjectSyncOperation['profile']>(['full', 'launch', 'results']);
  const statuses = new Set<ProjectSyncOperation['status']>([
    'queued',
    'running',
    'waiting_for_job',
    'waiting_for_worker',
    'conflict',
    'completed',
    'failed',
    'cancelled',
  ]);
  const rawProfile = String(raw.profile) as ProjectSyncOperation['profile'];
  const rawStatus = String(raw.status ?? 'queued') as ProjectSyncOperation['status'];
  return {
    id: String(raw.id),
    project_id: String(raw.project_id),
    worker_id: String(raw.worker_id),
    profile: profiles.has(rawProfile) ? rawProfile : 'full',
    status: statuses.has(rawStatus) ? rawStatus : 'failed',
    phase: String(raw.phase ?? 'queued'),
    files_total: Math.max(0, Number(raw.files_total ?? 0)),
    files_done: Math.max(0, Number(raw.files_done ?? 0)),
    bytes_total: Math.max(0, Number(raw.bytes_total ?? 0)),
    bytes_done: Math.max(0, Number(raw.bytes_done ?? 0)),
    retry_count: Math.max(0, Number(raw.retry_count ?? 0)),
    retry_at: raw.retry_at == null ? null : parseDate(raw.retry_at),
    base_manifest_hash: raw.base_manifest_hash == null ? null : String(raw.base_manifest_hash),
    source_manifest_hash: raw.source_manifest_hash == null ? null : String(raw.source_manifest_hash),
    target_manifest_hash: raw.target_manifest_hash == null ? null : String(raw.target_manifest_hash),
    conflicts: String(raw.conflicts ?? '[]'),
    error: raw.error == null ? null : String(raw.error),
    created_at: parseDate(raw.created_at),
    updated_at: parseDate(raw.updated_at),
  };
}

function normalizeQueue(raw: any): Queue | null {
  if (!raw) return null;
  return {
    id: Number(raw.id ?? 0),
    worker_id: String(raw.worker_id ?? 'local'),
    gpu_ids: String(raw.gpu_ids ?? ''),
    is_running: Boolean(raw.is_running),
  };
}

function normalizeWorkerNode(raw: any): WorkerNodeRecord | null {
  if (!raw) return null;
  return {
    id: String(raw.id),
    name: String(raw.name ?? ''),
    base_url: String(raw.base_url ?? ''),
    api_token: String(raw.api_token ?? ''),
    enabled: Boolean(raw.enabled ?? true),
    offline_bypass_enabled: Boolean(raw.offline_bypass_enabled ?? false),
    last_status: String(raw.last_status ?? 'unknown'),
    last_error: raw.last_error == null ? null : String(raw.last_error),
    last_checked_at: raw.last_checked_at == null ? null : parseDate(raw.last_checked_at),
    capabilities: String(raw.capabilities ?? '{}'),
    gpus: String(raw.gpus ?? '[]'),
    created_at: parseDate(raw.created_at),
    updated_at: parseDate(raw.updated_at),
  };
}

function normalizeSetting(raw: any): SettingRecord | null {
  if (!raw) return null;
  return {
    id: raw.id == null ? undefined : Number(raw.id),
    key: String(raw.key ?? ''),
    value: String(raw.value ?? ''),
  };
}

function mongoCollection<T extends Document = Document>(db: Db, name: string): Collection<T> {
  return db.collection<T>(name);
}

function openSqliteDb(filename: string) {
  const sqlite = new sqlite3.Database(filename);
  sqlite.configure('busyTimeout', SQLITE_BUSY_TIMEOUT_MS);
  return sqlite;
}

function sqliteAll<T = any>(sqlite: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<T[]>((resolve, reject) => {
    sqlite.all(sql, params, (err, rows) => {
      if (err) reject(err);
      else resolve(rows as T[]);
    });
  });
}

function sqliteGet<T = any>(sqlite: sqlite3.Database, sql: string, params: any[] = []) {
  return new Promise<T | undefined>((resolve, reject) => {
    sqlite.get(sql, params, (err, row) => {
      if (err) reject(err);
      else resolve(row as T | undefined);
    });
  });
}

function closeSqliteDb(sqlite: sqlite3.Database) {
  return new Promise<void>((resolve, reject) => {
    sqlite.close(err => (err ? reject(err) : resolve()));
  });
}

async function pathExists(filePath: string) {
  try {
    await fs.promises.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readSqliteLossLog(
  logPath: string,
  key: string,
  limit: number,
  sinceStep: number | null,
  stride: number,
): Promise<LossLogResult> {
  if (!(await pathExists(logPath))) {
    return { keys: [], key, points: [] };
  }

  const sqlite = openSqliteDb(logPath);
  try {
    const keysRows = await sqliteAll<{ key: string }>(sqlite, `SELECT key FROM metric_keys ORDER BY key ASC`);
    const keys = keysRows.map(row => row.key);
    const points = await sqliteAll<{
      step: number;
      wall_time: number;
      value: number | null;
      value_text: string | null;
    }>(
      sqlite,
      `
      SELECT
        m.step AS step,
        s.wall_time AS wall_time,
        m.value_real AS value,
        m.value_text AS value_text
      FROM metrics m
      JOIN steps s ON s.step = m.step
      WHERE m.key = ?
        AND (? IS NULL OR m.step > ?)
        AND (m.step % ?) = 0
      ORDER BY m.step ASC
      LIMIT ?
      `,
      [key, sinceStep, sinceStep, stride, limit],
    );

    return {
      key,
      keys,
      points: points.map(point => ({
        step: point.step,
        wall_time: point.wall_time,
        value: coerceMetricValue(point.value, point.value_text),
        value_text: point.value_text,
      })),
    };
  } finally {
    await closeSqliteDb(sqlite);
  }
}

async function readMongoLossLog(
  jobID: string,
  key: string,
  limit: number,
  sinceStep: number | null,
  stride: number,
): Promise<LossLogResult> {
  const mongo = await getMongoDb();
  const metricKeys = mongoCollection(mongo, 'metric_keys');
  const metrics = mongoCollection(mongo, 'metrics');

  const keysRows = await metricKeys
    .find({ job_id: jobID }, { projection: { _id: 0, key: 1 } })
    .sort({ key: 1 })
    .toArray();
  const keys = keysRows.map(row => String(row.key));

  const filter: Document = { job_id: jobID, key };
  if (sinceStep !== null) {
    filter.step = { $gt: sinceStep };
  }
  if (stride > 1) {
    filter.$expr = { $eq: [{ $mod: ['$step', stride] }, 0] };
  }

  const rows = await metrics
    .find(filter, { projection: { _id: 0, step: 1, wall_time: 1, value_real: 1, value_text: 1 } })
    .sort({ step: 1 })
    .limit(limit)
    .toArray();

  return {
    key,
    keys,
    points: rows.map(row => ({
      step: Number(row.step ?? 0),
      wall_time: Number(row.wall_time ?? 0),
      value: coerceMetricValue(row.value_real == null ? null : Number(row.value_real), row.value_text),
      value_text: row.value_text == null ? null : String(row.value_text),
    })),
  };
}

function expandMetricKeys(allKeys: string[], requestedKeys: string[]): string[] {
  const out = new Set<string>();
  const sortedKeys = [...allKeys].sort();

  for (const raw of requestedKeys) {
    const token = raw.trim();
    if (!token) continue;

    if (token === '*') {
      for (const key of sortedKeys) out.add(key);
      continue;
    }

    if (token.endsWith('*')) {
      const prefix = token.slice(0, -1);
      for (const key of sortedKeys) {
        if (key.startsWith(prefix)) out.add(key);
      }
      continue;
    }

    out.add(token);
  }

  return Array.from(out).sort();
}

function buildMetricSeries(
  key: string,
  points: LossPoint[],
  totalCount: number,
  firstStep: number | null,
  lastStep: number | null,
  latest: LossPoint | null,
  maxPoints: number,
): MetricSeriesResult {
  return buildMetricSeriesResult(key, points, totalCount, firstStep, lastStep, latest, maxPoints);
}

function parseStepMapValue(value: Record<string, number | null> | undefined, key: string, fallback: number | null) {
  if (!value) return fallback;
  const step = value[key];
  return typeof step === 'number' && Number.isFinite(step) ? step : fallback;
}

async function readSqliteMetrics(
  logPath: string,
  options: {
    keys: string[];
    maxPoints: number;
    sinceStep: number | null;
    sinceSteps?: Record<string, number | null>;
  },
): Promise<MetricsResult> {
  if (!(await pathExists(logPath))) {
    return { keys: [], keyInfo: [], series: {} };
  }

  const sqlite = openSqliteDb(logPath);
  try {
    const keyRows = await sqliteAll<MetricKeyInfo>(
      sqlite,
      `SELECT key, first_seen_step, last_seen_step FROM metric_keys ORDER BY key ASC`,
    );
    const allKeys = keyRows.map(row => row.key);
    const requestedKeys = expandMetricKeys(allKeys, options.keys);
    const maxPoints = normalizeMetricMaxPoints(options.maxPoints);
    const fetchLimit = maxPoints;
    const series: Record<string, MetricSeriesResult> = {};

    for (const key of requestedKeys) {
      const sinceStep = parseStepMapValue(options.sinceSteps, key, options.sinceStep);
      const info = keyRows.find(row => row.key === key);

      if (!info) {
        series[key] = buildMetricSeries(key, [], 0, null, null, null, maxPoints);
        continue;
      }

      const total = await sqliteGet<{ count: number }>(sqlite, `SELECT COUNT(*) AS count FROM metrics WHERE key = ?`, [
        key,
      ]);
      const latestRow = await sqliteGet<{
        step: number;
        wall_time: number;
        value: number | null;
        value_text: string | null;
      }>(
        sqlite,
        `
        SELECT
          m.step AS step,
          s.wall_time AS wall_time,
          m.value_real AS value,
          m.value_text AS value_text
        FROM metrics m
        JOIN steps s ON s.step = m.step
        WHERE m.key = ?
        ORDER BY m.step DESC
        LIMIT 1
        `,
        [key],
      );
      const rows = await sqliteAll<{
        step: number;
        wall_time: number;
        value: number | null;
        value_text: string | null;
      }>(
        sqlite,
        `
        SELECT *
        FROM (
          SELECT
            m.step AS step,
            s.wall_time AS wall_time,
            m.value_real AS value,
            m.value_text AS value_text
          FROM metrics m
          JOIN steps s ON s.step = m.step
          WHERE m.key = ?
            AND (? IS NULL OR m.step > ?)
          ORDER BY m.step DESC
          LIMIT ?
        ) AS bounded
        ORDER BY bounded.step ASC
        `,
        [key, sinceStep, sinceStep, fetchLimit],
      );
      const points = rows.map(point => ({
        step: point.step,
        wall_time: point.wall_time,
        value: coerceMetricValue(point.value, point.value_text),
        value_text: point.value_text,
      }));
      const latest = latestRow
        ? {
            step: latestRow.step,
            wall_time: latestRow.wall_time,
            value: coerceMetricValue(latestRow.value, latestRow.value_text),
            value_text: latestRow.value_text,
          }
        : null;

      series[key] = buildMetricSeries(
        key,
        points,
        Number(total?.count ?? 0),
        info.first_seen_step,
        info.last_seen_step,
        latest,
        maxPoints,
      );
    }

    return { keys: allKeys, keyInfo: keyRows, series };
  } finally {
    await closeSqliteDb(sqlite);
  }
}

async function readMongoMetrics(
  jobID: string,
  options: {
    keys: string[];
    maxPoints: number;
    sinceStep: number | null;
    sinceSteps?: Record<string, number | null>;
  },
): Promise<MetricsResult> {
  const mongo = await getMongoDb();
  const metricKeys = mongoCollection(mongo, 'metric_keys');
  const metrics = mongoCollection(mongo, 'metrics');
  const keyRowsRaw = await metricKeys
    .find({ job_id: jobID }, { projection: { _id: 0, key: 1, first_seen_step: 1, last_seen_step: 1 } })
    .sort({ key: 1 })
    .toArray();
  const keyRows: MetricKeyInfo[] = keyRowsRaw.map(row => ({
    key: String(row.key),
    first_seen_step: row.first_seen_step == null ? null : Number(row.first_seen_step),
    last_seen_step: row.last_seen_step == null ? null : Number(row.last_seen_step),
  }));
  const allKeys = keyRows.map(row => row.key);
  const requestedKeys = expandMetricKeys(allKeys, options.keys);
  const maxPoints = normalizeMetricMaxPoints(options.maxPoints);
  const fetchLimit = maxPoints;
  const series: Record<string, MetricSeriesResult> = {};

  for (const key of requestedKeys) {
    const sinceStep = parseStepMapValue(options.sinceSteps, key, options.sinceStep);
    const info = keyRows.find(row => row.key === key);

    if (!info) {
      series[key] = buildMetricSeries(key, [], 0, null, null, null, maxPoints);
      continue;
    }

    const filter: Document = { job_id: jobID, key };
    const pointFilter: Document = { ...filter };
    if (sinceStep !== null) {
      pointFilter.step = { $gt: sinceStep };
    }

    const [totalCount, latestRow, rows] = await Promise.all([
      metrics.countDocuments(filter),
      metrics
        .find(filter, { projection: { _id: 0, step: 1, wall_time: 1, value_real: 1, value_text: 1 } })
        .sort({ step: -1 })
        .limit(1)
        .next(),
      metrics
        .find(pointFilter, { projection: { _id: 0, step: 1, wall_time: 1, value_real: 1, value_text: 1 } })
        .sort({ step: -1 })
        .limit(fetchLimit)
        .toArray(),
    ]);
    const points = rows.reverse().map(row => ({
      step: Number(row.step ?? 0),
      wall_time: Number(row.wall_time ?? 0),
      value: coerceMetricValue(row.value_real == null ? null : Number(row.value_real), row.value_text),
      value_text: row.value_text == null ? null : String(row.value_text),
    }));
    const latest = latestRow
      ? {
          step: Number(latestRow.step ?? 0),
          wall_time: Number(latestRow.wall_time ?? 0),
          value: coerceMetricValue(
            latestRow.value_real == null ? null : Number(latestRow.value_real),
            latestRow.value_text,
          ),
          value_text: latestRow.value_text == null ? null : String(latestRow.value_text),
        }
      : null;

    series[key] = buildMetricSeries(
      key,
      points,
      totalCount,
      info.first_seen_step,
      info.last_seen_step,
      latest,
      maxPoints,
    );
  }

  return { keys: allKeys, keyInfo: keyRows, series };
}

async function ensureMongoIndexes() {
  const mongo = await getMongoDb();
  await dropLegacyMongoJobNameUniqueIndex(mongo);
  const legacyProjects = await mongoCollection(mongo, 'projects')
    .find(
      {
        $or: [
          { lifecycle_state: { $exists: false } },
          { storage_root_path: { $exists: false } },
          { revision: { $exists: false } },
        ],
      },
      { projection: { _id: 0 } },
    )
    .toArray();
  if (legacyProjects.length > 0) {
    await mongoCollection(mongo, 'projects').bulkWrite(
      legacyProjects.map(project => {
        const rootPath = String(project.root_path ?? '');
        return {
          updateOne: {
            filter: { id: String(project.id) },
            update: {
              $set: {
                root_path: rootPath,
                storage_root_path: String(project.storage_root_path || (rootPath ? path.dirname(rootPath) : '')),
                lifecycle_state: String(project.lifecycle_state || 'active'),
                archived_at: project.archived_at ?? null,
                revision: Number(project.revision ?? 1),
                operation_started_at: project.operation_started_at ?? null,
                operation_error: project.operation_error ?? null,
                home_worker_id: String(project.home_worker_id || 'local'),
                home_instance_id: String(project.home_instance_id || ''),
              },
            },
          },
        };
      }),
      { ordered: false },
    );
  }
  await Promise.all([
    mongoCollection(mongo, 'jobs').createIndexes([
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
    mongoCollection(mongo, 'projects').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { slug: 1 }, unique: true },
      { key: { updated_at: -1 } },
      { key: { lifecycle_state: 1, updated_at: -1 } },
      { key: { archived_at: -1 } },
    ]),
    mongoCollection(mongo, 'project_replicas').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { project_id: 1, worker_id: 1 }, unique: true },
      { key: { project_id: 1 } },
      { key: { worker_id: 1 } },
      { key: { state: 1 } },
    ]),
    mongoCollection(mongo, 'job_replicas').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { job_id: 1, worker_id: 1 }, unique: true },
      { key: { worker_id: 1, remote_job_id: 1 }, unique: true },
      { key: { job_id: 1 } },
      { key: { worker_id: 1 } },
    ]),
    mongoCollection(mongo, 'project_sync_operations').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { project_id: 1, created_at: -1 } },
      { key: { worker_id: 1 } },
      { key: { status: 1, retry_at: 1 } },
    ]),
    mongoCollection(mongo, 'queues').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { worker_id: 1, gpu_ids: 1 }, unique: true },
      { key: { worker_id: 1 } },
      { key: { gpu_ids: 1 } },
    ]),
    mongoCollection(mongo, 'worker_nodes').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { name: 1 }, unique: true },
      { key: { enabled: 1 } },
    ]),
    mongoCollection(mongo, 'settings').createIndexes([{ key: { key: 1 }, unique: true }]),
    mongoCollection(mongo, 'metrics').createIndexes([
      { key: { job_id: 1, step: 1, key: 1 }, unique: true },
      { key: { job_id: 1, key: 1, step: 1 } },
    ]),
    mongoCollection(mongo, 'metric_keys').createIndexes([{ key: { job_id: 1, key: 1 }, unique: true }]),
  ]);

  const legacyRemoteJobs = await mongoCollection(mongo, 'jobs')
    .find({ remote_job_id: { $type: 'string', $ne: '' } }, { projection: { _id: 0 } })
    .toArray();
  if (legacyRemoteJobs.length > 0) {
    await mongoCollection(mongo, 'job_replicas').bulkWrite(
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
}

async function nextMongoQueueId(queues: Collection<Document>) {
  const latest = await queues
    .find({}, { projection: { _id: 0, id: 1 } })
    .sort({ id: -1 })
    .limit(1)
    .next();
  return Number(latest?.id ?? 0) + 1;
}

export const db = {
  settings: {
    async list(): Promise<SettingRecord[]> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const rows = await mongoCollection(mongo, 'settings')
          .find({}, { projection: { _id: 0 } })
          .sort({ key: 1 })
          .toArray();
        return rows.map(normalizeSetting).filter(Boolean) as SettingRecord[];
      }
      return getPrisma().settings.findMany();
    },

    async get(key: string): Promise<SettingRecord | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'settings').findOne({ key }, { projection: { _id: 0 } });
        return normalizeSetting(row);
      }
      return getPrisma().settings.findFirst({ where: { key } });
    },

    async upsert(key: string, value: string): Promise<SettingRecord> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        await mongoCollection(mongo, 'settings').updateOne({ key }, { $set: { key, value } }, { upsert: true });
        return { key, value };
      }
      return getPrisma().settings.upsert({
        where: { key },
        update: { value },
        create: { key, value },
      });
    },

    async upsertMany(settings: Record<string, string>) {
      await Promise.all(Object.entries(settings).map(([key, value]) => db.settings.upsert(key, value ?? '')));
    },

    async delete(key: string): Promise<void> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        await mongoCollection(mongo, 'settings').deleteOne({ key });
        return;
      }

      try {
        await getPrisma().settings.delete({ where: { key } });
      } catch (error: any) {
        if (error?.code !== 'P2025') throw error;
      }
    },
  },

  projects: {
    async list(options: { lifecycle_state?: ProjectLifecycleState | ProjectLifecycleState[] } = {}): Promise<Project[]> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const filter: Document = {};
        if (Array.isArray(options.lifecycle_state)) filter.lifecycle_state = { $in: options.lifecycle_state };
        else if (options.lifecycle_state) filter.lifecycle_state = options.lifecycle_state;
        const rows = await mongoCollection(mongo, 'projects')
          .find(filter, { projection: { _id: 0 } })
          .sort({ updated_at: -1, name: 1 })
          .toArray();
        return rows.map(normalizeProject).filter(Boolean) as Project[];
      }
      const rows = await getPrisma().project.findMany({
        where: Array.isArray(options.lifecycle_state)
          ? { lifecycle_state: { in: options.lifecycle_state } }
          : options.lifecycle_state
            ? { lifecycle_state: options.lifecycle_state }
            : undefined,
        orderBy: [{ updated_at: 'desc' }, { name: 'asc' }],
      });
      return rows.map(normalizeProject).filter(Boolean) as Project[];
    },

    async findById(id: string): Promise<Project | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'projects').findOne({ id }, { projection: { _id: 0 } });
        return normalizeProject(row);
      }
      return normalizeProject(await getPrisma().project.findUnique({ where: { id } }));
    },

    async findBySlug(slug: string): Promise<Project | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'projects').findOne({ slug }, { projection: { _id: 0 } });
        return normalizeProject(row);
      }
      return normalizeProject(await getPrisma().project.findUnique({ where: { slug } }));
    },

    async create(input: ProjectCreateInput): Promise<Project> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const now = new Date();
        const project = normalizeProject({
          id: input.id || randomUUID(),
          slug: input.slug,
          name: input.name,
          description: input.description ?? '',
          badge_asset: input.badge_asset ?? null,
          root_path: input.root_path ?? '',
          storage_root_path: input.storage_root_path ?? '',
          lifecycle_state: input.lifecycle_state ?? 'active',
          archived_at: input.archived_at ?? null,
          revision: input.revision ?? 1,
          operation_started_at: input.operation_started_at ?? null,
          operation_error: input.operation_error ?? null,
          home_worker_id: input.home_worker_id ?? 'local',
          home_instance_id: input.home_instance_id ?? '',
          created_at: now,
          updated_at: now,
        }) as Project;
        try {
          await mongoCollection(mongo, 'projects').insertOne(project);
        } catch (error) {
          duplicateKeyToUniqueError(error);
        }
        return project;
      }

      const project = await getPrisma().project.create({
        data: {
          id: input.id,
          slug: input.slug,
          name: input.name,
          description: input.description ?? '',
          badge_asset: input.badge_asset ?? null,
          root_path: input.root_path ?? '',
          storage_root_path: input.storage_root_path ?? '',
          lifecycle_state: input.lifecycle_state ?? 'active',
          archived_at: input.archived_at == null ? null : new Date(input.archived_at),
          revision: input.revision ?? 1,
          operation_started_at: input.operation_started_at == null ? null : new Date(input.operation_started_at),
          operation_error: input.operation_error ?? null,
          home_worker_id: input.home_worker_id ?? 'local',
          home_instance_id: input.home_instance_id ?? '',
        },
      });
      return normalizeProject(project) as Project;
    },

    async update(id: string, data: ProjectUpdateInput): Promise<Project> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        try {
          const result = await mongoCollection(mongo, 'projects').findOneAndUpdate(
            { id },
            { $set: { ...data, updated_at: new Date() } },
            { returnDocument: 'after', projection: { _id: 0 } },
          );
          const project = normalizeProject(result);
          if (!project) throw new Error(`Project not found: ${id}`);
          return project;
        } catch (error) {
          duplicateKeyToUniqueError(error);
        }
      }

      const project = await getPrisma().project.update({ where: { id }, data });
      return normalizeProject(project) as Project;
    },

    async compareAndSet(
      id: string,
      expected: { revision: number; lifecycle_state?: ProjectLifecycleState | ProjectLifecycleState[] },
      data: ProjectUpdateInput,
    ): Promise<Project | null> {
      const states = Array.isArray(expected.lifecycle_state)
        ? expected.lifecycle_state
        : expected.lifecycle_state
          ? [expected.lifecycle_state]
          : null;
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const filter: Document = { id, revision: expected.revision };
        if (states) filter.lifecycle_state = { $in: states };
        const result = await mongoCollection(mongo, 'projects').findOneAndUpdate(
          filter,
          { $set: { ...data, updated_at: new Date() } },
          { returnDocument: 'after', projection: { _id: 0 } },
        );
        return normalizeProject(result);
      }

      const result = await getPrisma().project.updateMany({
        where: {
          id,
          revision: expected.revision,
          ...(states ? { lifecycle_state: { in: states } } : {}),
        },
        data,
      });
      if (result.count !== 1) return null;
      return normalizeProject(await getPrisma().project.findUnique({ where: { id } }));
    },

    async delete(id: string): Promise<Project | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const result = await mongoCollection(mongo, 'projects').findOneAndDelete({ id }, { projection: { _id: 0 } });
        return normalizeProject(result);
      }
      try {
        return normalizeProject(await getPrisma().project.delete({ where: { id } }));
      } catch (error: any) {
        if (error?.code === 'P2025') return null;
        throw error;
      }
    },
  },

  projectReplicas: {
    async listByProject(projectID: string): Promise<ProjectReplicaState[]> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const rows = await mongoCollection(mongo, 'project_replicas')
          .find({ project_id: projectID }, { projection: { _id: 0 } })
          .sort({ role: 1, worker_id: 1 })
          .toArray();
        return rows.map(normalizeProjectReplica).filter(Boolean) as ProjectReplicaState[];
      }
      const rows = await getPrisma().projectReplica.findMany({
        where: { project_id: projectID },
        orderBy: [{ role: 'asc' }, { worker_id: 'asc' }],
      });
      return rows.map(normalizeProjectReplica).filter(Boolean) as ProjectReplicaState[];
    },

    async findByProjectAndWorker(projectID: string, workerID: string): Promise<ProjectReplicaState | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        return normalizeProjectReplica(
          await mongoCollection(mongo, 'project_replicas').findOne(
            { project_id: projectID, worker_id: workerID },
            { projection: { _id: 0 } },
          ),
        );
      }
      return normalizeProjectReplica(
        await getPrisma().projectReplica.findUnique({
          where: { project_id_worker_id: { project_id: projectID, worker_id: workerID } },
        }),
      );
    },

    async upsert(input: ProjectReplicaCreateInput): Promise<ProjectReplicaState> {
      const now = new Date();
      const create = {
        id: input.id || randomUUID(),
        project_id: input.project_id,
        worker_id: input.worker_id,
        remote_project_id: input.remote_project_id ?? null,
        remote_instance_id: input.remote_instance_id ?? null,
        role: input.role ?? 'execution',
        state: input.state ?? 'creating',
        base_manifest_hash: input.base_manifest_hash ?? null,
        local_manifest_hash: input.local_manifest_hash ?? null,
        remote_manifest_hash: input.remote_manifest_hash ?? null,
        last_synced_at: input.last_synced_at == null ? null : parseDate(input.last_synced_at),
        last_error: input.last_error ?? null,
        auto_pull_results: input.auto_pull_results ?? true,
        created_at: now,
        updated_at: now,
      };
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const update = { ...create };
        delete (update as Partial<typeof update>).id;
        delete (update as Partial<typeof update>).created_at;
        const row = await mongoCollection(mongo, 'project_replicas').findOneAndUpdate(
          { project_id: input.project_id, worker_id: input.worker_id },
          { $set: update, $setOnInsert: { id: create.id, created_at: now } },
          { upsert: true, returnDocument: 'after', projection: { _id: 0 } },
        );
        return normalizeProjectReplica(row) as ProjectReplicaState;
      }
      const row = await getPrisma().projectReplica.upsert({
        where: { project_id_worker_id: { project_id: input.project_id, worker_id: input.worker_id } },
        create,
        update: {
          remote_project_id: create.remote_project_id,
          remote_instance_id: create.remote_instance_id,
          role: create.role,
          state: create.state,
          base_manifest_hash: create.base_manifest_hash,
          local_manifest_hash: create.local_manifest_hash,
          remote_manifest_hash: create.remote_manifest_hash,
          last_synced_at: create.last_synced_at,
          last_error: create.last_error,
          auto_pull_results: create.auto_pull_results,
        },
      });
      return normalizeProjectReplica(row) as ProjectReplicaState;
    },

    async update(id: string, data: ProjectReplicaUpdateInput): Promise<ProjectReplicaState> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'project_replicas').findOneAndUpdate(
          { id },
          { $set: { ...data, updated_at: new Date() } },
          { returnDocument: 'after', projection: { _id: 0 } },
        );
        const replica = normalizeProjectReplica(row);
        if (!replica) throw new Error(`Project replica not found: ${id}`);
        return replica;
      }
      return normalizeProjectReplica(await getPrisma().projectReplica.update({ where: { id }, data })) as ProjectReplicaState;
    },

    async delete(id: string): Promise<ProjectReplicaState | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        return normalizeProjectReplica(
          await mongoCollection(mongo, 'project_replicas').findOneAndDelete({ id }, { projection: { _id: 0 } }),
        );
      }
      try {
        return normalizeProjectReplica(await getPrisma().projectReplica.delete({ where: { id } }));
      } catch (error: any) {
        if (error?.code === 'P2025') return null;
        throw error;
      }
    },

    async deleteByProject(projectID: string): Promise<number> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        return (await mongoCollection(mongo, 'project_replicas').deleteMany({ project_id: projectID })).deletedCount;
      }
      return (await getPrisma().projectReplica.deleteMany({ where: { project_id: projectID } })).count;
    },
  },

  jobReplicas: {
    async listByJob(jobID: string): Promise<JobReplica[]> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const rows = await mongoCollection(mongo, 'job_replicas')
          .find({ job_id: jobID }, { projection: { _id: 0 } })
          .sort({ worker_id: 1 })
          .toArray();
        return rows.map(normalizeJobReplica).filter(Boolean) as JobReplica[];
      }
      const rows = await getPrisma().jobReplica.findMany({ where: { job_id: jobID }, orderBy: { worker_id: 'asc' } });
      return rows.map(normalizeJobReplica).filter(Boolean) as JobReplica[];
    },

    async findByJobAndWorker(jobID: string, workerID: string): Promise<JobReplica | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        return normalizeJobReplica(
          await mongoCollection(mongo, 'job_replicas').findOne(
            { job_id: jobID, worker_id: workerID },
            { projection: { _id: 0 } },
          ),
        );
      }
      return normalizeJobReplica(
        await getPrisma().jobReplica.findUnique({
          where: { job_id_worker_id: { job_id: jobID, worker_id: workerID } },
        }),
      );
    },

    async findByRemote(workerID: string, remoteJobID: string): Promise<JobReplica | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        return normalizeJobReplica(
          await mongoCollection(mongo, 'job_replicas').findOne(
            { worker_id: workerID, remote_job_id: remoteJobID },
            { projection: { _id: 0 } },
          ),
        );
      }
      return normalizeJobReplica(
        await getPrisma().jobReplica.findUnique({
          where: { worker_id_remote_job_id: { worker_id: workerID, remote_job_id: remoteJobID } },
        }),
      );
    },

    async upsert(input: JobReplicaCreateInput): Promise<JobReplica> {
      const now = new Date();
      const create = {
        id: input.id || randomUUID(),
        job_id: input.job_id,
        worker_id: input.worker_id,
        remote_job_id: input.remote_job_id,
        remote_project_id: input.remote_project_id ?? null,
        role: input.role ?? 'execution',
        last_synced_at: input.last_synced_at == null ? null : parseDate(input.last_synced_at),
        last_error: input.last_error ?? null,
        created_at: now,
        updated_at: now,
      };
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const update = { ...create };
        delete (update as Partial<typeof update>).id;
        delete (update as Partial<typeof update>).created_at;
        const row = await mongoCollection(mongo, 'job_replicas').findOneAndUpdate(
          { job_id: input.job_id, worker_id: input.worker_id },
          { $set: update, $setOnInsert: { id: create.id, created_at: now } },
          { upsert: true, returnDocument: 'after', projection: { _id: 0 } },
        );
        return normalizeJobReplica(row) as JobReplica;
      }
      const row = await getPrisma().jobReplica.upsert({
        where: { job_id_worker_id: { job_id: input.job_id, worker_id: input.worker_id } },
        create,
        update: {
          remote_job_id: create.remote_job_id,
          remote_project_id: create.remote_project_id,
          role: create.role,
          last_synced_at: create.last_synced_at,
          last_error: create.last_error,
        },
      });
      return normalizeJobReplica(row) as JobReplica;
    },

    async update(id: string, data: JobReplicaUpdateInput): Promise<JobReplica> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'job_replicas').findOneAndUpdate(
          { id },
          { $set: { ...data, updated_at: new Date() } },
          { returnDocument: 'after', projection: { _id: 0 } },
        );
        const replica = normalizeJobReplica(row);
        if (!replica) throw new Error(`Job replica not found: ${id}`);
        return replica;
      }
      return normalizeJobReplica(await getPrisma().jobReplica.update({ where: { id }, data })) as JobReplica;
    },

    async deleteByJob(jobID: string): Promise<number> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        return (await mongoCollection(mongo, 'job_replicas').deleteMany({ job_id: jobID })).deletedCount;
      }
      return (await getPrisma().jobReplica.deleteMany({ where: { job_id: jobID } })).count;
    },
  },

  projectSyncOperations: {
    async findById(id: string): Promise<ProjectSyncOperation | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        return normalizeProjectSyncOperation(
          await mongoCollection(mongo, 'project_sync_operations').findOne({ id }, { projection: { _id: 0 } }),
        );
      }
      return normalizeProjectSyncOperation(await getPrisma().projectSyncOperation.findUnique({ where: { id } }));
    },

    async list(
      options: { project_id?: string; worker_id?: string; status?: ProjectSyncOperation['status'] | ProjectSyncOperation['status'][] } = {},
    ): Promise<ProjectSyncOperation[]> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const filter: Document = {};
        if (options.project_id) filter.project_id = options.project_id;
        if (options.worker_id) filter.worker_id = options.worker_id;
        if (Array.isArray(options.status)) filter.status = { $in: options.status };
        else if (options.status) filter.status = options.status;
        const rows = await mongoCollection(mongo, 'project_sync_operations')
          .find(filter, { projection: { _id: 0 } })
          .sort({ created_at: -1 })
          .toArray();
        return rows.map(normalizeProjectSyncOperation).filter(Boolean) as ProjectSyncOperation[];
      }
      const rows = await getPrisma().projectSyncOperation.findMany({
        where: {
          project_id: options.project_id,
          worker_id: options.worker_id,
          status: Array.isArray(options.status) ? { in: options.status } : options.status,
        },
        orderBy: { created_at: 'desc' },
      });
      return rows.map(normalizeProjectSyncOperation).filter(Boolean) as ProjectSyncOperation[];
    },

    async create(input: ProjectSyncOperationCreateInput): Promise<ProjectSyncOperation> {
      const now = new Date();
      const operation = {
        id: input.id || randomUUID(),
        project_id: input.project_id,
        worker_id: input.worker_id,
        profile: input.profile,
        status: input.status ?? 'queued',
        phase: input.phase ?? 'queued',
        files_total: input.files_total ?? 0,
        files_done: input.files_done ?? 0,
        bytes_total: input.bytes_total ?? 0,
        bytes_done: input.bytes_done ?? 0,
        retry_count: input.retry_count ?? 0,
        retry_at: input.retry_at == null ? null : parseDate(input.retry_at),
        base_manifest_hash: input.base_manifest_hash ?? null,
        source_manifest_hash: input.source_manifest_hash ?? null,
        target_manifest_hash: input.target_manifest_hash ?? null,
        conflicts: input.conflicts ?? '[]',
        error: input.error ?? null,
        created_at: now,
        updated_at: now,
      };
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        await mongoCollection(mongo, 'project_sync_operations').insertOne(operation);
        return normalizeProjectSyncOperation(operation) as ProjectSyncOperation;
      }
      return normalizeProjectSyncOperation(await getPrisma().projectSyncOperation.create({ data: operation })) as ProjectSyncOperation;
    },

    async update(id: string, data: ProjectSyncOperationUpdateInput): Promise<ProjectSyncOperation> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'project_sync_operations').findOneAndUpdate(
          { id },
          { $set: { ...data, updated_at: new Date() } },
          { returnDocument: 'after', projection: { _id: 0 } },
        );
        const operation = normalizeProjectSyncOperation(row);
        if (!operation) throw new Error(`Project sync operation not found: ${id}`);
        return operation;
      }
      return normalizeProjectSyncOperation(
        await getPrisma().projectSyncOperation.update({ where: { id }, data }),
      ) as ProjectSyncOperation;
    },

    async delete(id: string): Promise<ProjectSyncOperation | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        return normalizeProjectSyncOperation(
          await mongoCollection(mongo, 'project_sync_operations').findOneAndDelete({ id }, { projection: { _id: 0 } }),
        );
      }
      try {
        return normalizeProjectSyncOperation(await getPrisma().projectSyncOperation.delete({ where: { id } }));
      } catch (error: any) {
        if (error?.code === 'P2025') return null;
        throw error;
      }
    },

    async deleteByProject(projectID: string): Promise<number> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        return (await mongoCollection(mongo, 'project_sync_operations').deleteMany({ project_id: projectID })).deletedCount;
      }
      return (await getPrisma().projectSyncOperation.deleteMany({ where: { project_id: projectID } })).count;
    },
  },

  jobs: {
    async findById(id: string): Promise<Job | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'jobs').findOne({ id }, { projection: { _id: 0 } });
        return normalizeJob(row);
      }
      return getPrisma().job.findUnique({ where: { id } });
    },

    async findByName(name: string): Promise<Job | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'jobs').findOne({ name }, { projection: { _id: 0 } });
        return normalizeJob(row);
      }
      return getPrisma().job.findFirst({ where: { name } });
    },

    async findByNameInScope(name: string, project_id: string | null): Promise<Job | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'jobs').findOne(
          { name, project_id: project_id ?? null },
          { projection: { _id: 0 } },
        );
        return normalizeJob(row);
      }
      return getPrisma().job.findFirst({ where: { name, project_id: project_id ?? null } });
    },

    async findLatestByRef(jobRef: string, jobType?: string | null, project_id?: string | null): Promise<Job | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const filter: Document = { job_ref: jobRef };
        if (jobType) filter.job_type = jobType;
        if (project_id !== undefined) filter.project_id = project_id ?? null;
        const row = await mongoCollection(mongo, 'jobs')
          .find(filter, { projection: { _id: 0 } })
          .sort({ updated_at: -1 })
          .limit(1)
          .next();
        return normalizeJob(row);
      }
      const where: Prisma.JobWhereInput = { job_ref: jobRef, ...(jobType ? { job_type: jobType } : {}) };
      if (project_id !== undefined) where.project_id = project_id ?? null;
      return getPrisma().job.findFirst({
        where,
        orderBy: { updated_at: 'desc' },
      });
    },

    async list(
      options: {
        job_type?: string | null;
        status?: string | string[];
        gpu_ids?: string;
        worker_id?: string;
        project_id?: string | null;
        order?: 'created_desc' | 'queue_asc';
      } = {},
    ) {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const filter: Document = {};
        if (options.job_type) filter.job_type = options.job_type;
        if (options.gpu_ids) filter.gpu_ids = options.gpu_ids;
        if (options.worker_id) filter.worker_id = options.worker_id;
        if ('project_id' in options) filter.project_id = options.project_id ?? null;
        if (Array.isArray(options.status)) filter.status = { $in: options.status };
        else if (options.status) filter.status = options.status;
        const sort: Record<string, 1 | -1> = options.order === 'queue_asc' ? { queue_position: 1 } : { created_at: -1 };
        const rows = await mongoCollection(mongo, 'jobs')
          .find(filter, { projection: { _id: 0 } })
          .sort(sort)
          .toArray();
        return rows.map(normalizeJob).filter(Boolean) as Job[];
      }

      const where: any = {};
      if (options.job_type) where.job_type = options.job_type;
      if (options.gpu_ids) where.gpu_ids = options.gpu_ids;
      if (options.worker_id) where.worker_id = options.worker_id;
      if ('project_id' in options) where.project_id = options.project_id ?? null;
      if (Array.isArray(options.status)) where.status = { in: options.status };
      else if (options.status) where.status = options.status;
      return getPrisma().job.findMany({
        where: Object.keys(where).length > 0 ? where : undefined,
        orderBy: options.order === 'queue_asc' ? { queue_position: 'asc' } : { created_at: 'desc' },
      });
    },

    async findFirst(
      options: { status?: string | string[]; gpu_ids?: string; worker_id?: string; order?: 'queue_asc' } = {},
    ) {
      const rows = await db.jobs.list(options);
      return rows[0] ?? null;
    },

    async findByRemoteId(workerId: string, remoteJobId: string): Promise<Job | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'jobs').findOne(
          { worker_id: workerId, remote_job_id: remoteJobId },
          { projection: { _id: 0 } },
        );
        return normalizeJob(row);
      }
      return getPrisma().job.findFirst({
        where: { worker_id: workerId, remote_job_id: remoteJobId },
      });
    },

    async maxQueuePosition() {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'jobs')
          .find({}, { projection: { _id: 0, queue_position: 1 } })
          .sort({ queue_position: -1 })
          .limit(1)
          .next();
        return Number(row?.queue_position ?? 0);
      }

      const highestQueuePosition = await getPrisma().job.aggregate({
        _max: { queue_position: true },
      });
      return highestQueuePosition._max.queue_position || 0;
    },

    async create(input: JobCreateInput): Promise<Job> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const now = new Date();
        const job = normalizeJob({
          id: input.id || randomUUID(),
          name: input.name,
          project_id: input.project_id ?? null,
          worker_id: input.worker_id ?? 'local',
          remote_job_id: input.remote_job_id ?? null,
          remote_sync_at: input.remote_sync_at ?? null,
          remote_error: input.remote_error ?? null,
          gpu_ids: input.gpu_ids,
          job_config: input.job_config,
          created_at: now,
          updated_at: now,
          status: input.status ?? 'stopped',
          stop: input.stop ?? false,
          return_to_queue: input.return_to_queue ?? false,
          step: input.step ?? 0,
          info: input.info ?? '',
          speed_string: input.speed_string ?? '',
          queue_position: input.queue_position ?? 0,
          pid: input.pid ?? null,
          job_type: input.job_type ?? 'train',
          job_ref: input.job_ref ?? null,
          save_now: input.save_now ?? false,
          sample_now: input.sample_now ?? false,
        }) as Job;

        try {
          await mongoCollection(mongo, 'jobs').insertOne(job);
        } catch (error) {
          duplicateKeyToUniqueError(error);
        }
        return job;
      }

      return getPrisma().job.create({ data: input });
    },

    async update(id: string, data: JobUpdateInput): Promise<Job> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        try {
          const result = await mongoCollection(mongo, 'jobs').findOneAndUpdate(
            { id },
            {
              $set: {
                ...data,
                updated_at: new Date(),
              },
            },
            { returnDocument: 'after', projection: { _id: 0 } },
          );
          const job = normalizeJob(result);
          if (!job) throw new Error(`Job not found: ${id}`);
          return job;
        } catch (error) {
          duplicateKeyToUniqueError(error);
        }
      }

      return getPrisma().job.update({ where: { id }, data });
    },

    async delete(id: string): Promise<Job | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const result = await mongoCollection(mongo, 'jobs').findOneAndDelete({ id }, { projection: { _id: 0 } });
        return normalizeJob(result);
      }
      return getPrisma().job.delete({ where: { id } });
    },
  },

  queues: {
    async list(order: 'id' | 'gpu_ids' = 'id', options: { worker_id?: string } = {}): Promise<Queue[]> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const filter: Document = {};
        if (options.worker_id) filter.worker_id = options.worker_id;
        const rows = await mongoCollection(mongo, 'queues')
          .find(filter, { projection: { _id: 0 } })
          .sort({ [order]: 1 })
          .toArray();
        return rows.map(normalizeQueue).filter(Boolean) as Queue[];
      }
      return getPrisma().queue.findMany({
        where: options.worker_id ? { worker_id: options.worker_id } : undefined,
        orderBy: { [order]: 'asc' },
      });
    },

    async findByGpuIds(gpuIds: string, workerId = 'local'): Promise<Queue | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'queues').findOne(
          { worker_id: workerId, gpu_ids: gpuIds },
          { projection: { _id: 0 } },
        );
        return normalizeQueue(row);
      }
      return getPrisma().queue.findUnique({ where: { worker_id_gpu_ids: { worker_id: workerId, gpu_ids: gpuIds } } });
    },

    async create(input: QueueCreateInput): Promise<Queue> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const queues = mongoCollection(mongo, 'queues');
        const queue = normalizeQueue({
          id: input.id ?? (await nextMongoQueueId(queues)),
          worker_id: input.worker_id ?? 'local',
          gpu_ids: input.gpu_ids,
          is_running: input.is_running ?? false,
        }) as Queue;
        try {
          await queues.insertOne(queue);
        } catch (error) {
          duplicateKeyToUniqueError(error);
        }
        return queue;
      }
      return getPrisma().queue.create({ data: input });
    },

    async update(id: number, data: QueueUpdateInput): Promise<Queue> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const result = await mongoCollection(mongo, 'queues').findOneAndUpdate(
          { id },
          { $set: data },
          { returnDocument: 'after', projection: { _id: 0 } },
        );
        const queue = normalizeQueue(result);
        if (!queue) throw new Error(`Queue not found: ${id}`);
        return queue;
      }
      return getPrisma().queue.update({ where: { id }, data });
    },

    async deleteMany(options: { worker_id?: string } = {}): Promise<number> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const filter: Document = {};
        if (options.worker_id) filter.worker_id = options.worker_id;
        const result = await mongoCollection(mongo, 'queues').deleteMany(filter);
        return result.deletedCount ?? 0;
      }
      const result = await getPrisma().queue.deleteMany({
        where: options.worker_id ? { worker_id: options.worker_id } : undefined,
      });
      return result.count;
    },
  },

  workerNodes: {
    async list(options: { enabled?: boolean } = {}): Promise<WorkerNodeRecord[]> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const filter: Document = {};
        if (typeof options.enabled === 'boolean') filter.enabled = options.enabled;
        const rows = await mongoCollection(mongo, 'worker_nodes')
          .find(filter, { projection: { _id: 0 } })
          .sort({ name: 1 })
          .toArray();
        return rows.map(normalizeWorkerNode).filter(Boolean) as WorkerNodeRecord[];
      }
      return getPrisma().workerNode.findMany({
        where: typeof options.enabled === 'boolean' ? { enabled: options.enabled } : undefined,
        orderBy: { name: 'asc' },
      }) as Promise<WorkerNodeRecord[]>;
    },

    async findById(id: string): Promise<WorkerNodeRecord | null> {
      if (id === 'local') return null;
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'worker_nodes').findOne({ id }, { projection: { _id: 0 } });
        return normalizeWorkerNode(row);
      }
      return getPrisma().workerNode.findUnique({ where: { id } }) as Promise<WorkerNodeRecord | null>;
    },

    async create(input: WorkerNodeCreateInput): Promise<WorkerNodeRecord> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const now = new Date();
        const worker = normalizeWorkerNode({
          id: input.id || randomUUID(),
          name: input.name,
          base_url: input.base_url,
          api_token: input.api_token,
          enabled: input.enabled ?? true,
          offline_bypass_enabled: input.offline_bypass_enabled ?? false,
          last_status: input.last_status ?? 'unknown',
          last_error: input.last_error ?? null,
          last_checked_at: input.last_checked_at ?? null,
          capabilities: input.capabilities ?? '{}',
          gpus: input.gpus ?? '[]',
          created_at: now,
          updated_at: now,
        }) as WorkerNodeRecord;
        try {
          await mongoCollection(mongo, 'worker_nodes').insertOne(worker);
        } catch (error) {
          duplicateKeyToUniqueError(error);
        }
        return worker;
      }
      return getPrisma().workerNode.create({ data: input }) as Promise<WorkerNodeRecord>;
    },

    async update(id: string, data: WorkerNodeUpdateInput): Promise<WorkerNodeRecord> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        try {
          const result = await mongoCollection(mongo, 'worker_nodes').findOneAndUpdate(
            { id },
            { $set: { ...data, updated_at: new Date() } },
            { returnDocument: 'after', projection: { _id: 0 } },
          );
          const worker = normalizeWorkerNode(result);
          if (!worker) throw new Error(`Worker node not found: ${id}`);
          return worker;
        } catch (error) {
          duplicateKeyToUniqueError(error);
        }
      }
      return getPrisma().workerNode.update({ where: { id }, data }) as Promise<WorkerNodeRecord>;
    },

    async delete(id: string): Promise<WorkerNodeRecord | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const result = await mongoCollection(mongo, 'worker_nodes').findOneAndDelete(
          { id },
          { projection: { _id: 0 } },
        );
        return normalizeWorkerNode(result);
      }
      try {
        return (await getPrisma().workerNode.delete({ where: { id } })) as WorkerNodeRecord;
      } catch (error: any) {
        if (error?.code === 'P2025') return null;
        throw error;
      }
    },
  },

  metrics: {
    async getLossLog(
      jobID: string,
      logPath: string,
      options: { key: string; limit: number; sinceStep: number | null; stride: number },
    ): Promise<LossLogResult> {
      if (isMongoProvider()) {
        return readMongoLossLog(jobID, options.key, options.limit, options.sinceStep, options.stride);
      }
      return readSqliteLossLog(logPath, options.key, options.limit, options.sinceStep, options.stride);
    },

    async getMetrics(
      jobID: string,
      logPath: string,
      options: {
        keys: string[];
        maxPoints: number;
        sinceStep: number | null;
        sinceSteps?: Record<string, number | null>;
      },
    ): Promise<MetricsResult> {
      if (isMongoProvider()) {
        return readMongoMetrics(jobID, options);
      }
      return readSqliteMetrics(logPath, options);
    },

    async deleteForJob(jobID: string): Promise<void> {
      if (!isMongoProvider()) {
        return;
      }

      const mongo = await getMongoDb();
      await Promise.all([
        mongoCollection(mongo, 'metrics').deleteMany({ job_id: jobID }),
        mongoCollection(mongo, 'metric_keys').deleteMany({ job_id: jobID }),
      ]);
    },
  },

  async prepare() {
    if (isMongoProvider()) {
      await ensureMongoIndexes();
    }
  },
};

export async function disconnectDb() {
  if (globalThis.__aitkPrismaClient) {
    await globalThis.__aitkPrismaClient.$disconnect();
    globalThis.__aitkPrismaClient = undefined;
  }

  if (globalThis.__aitkMongoClientPromise) {
    const client = await globalThis.__aitkMongoClientPromise;
    await client.close();
    globalThis.__aitkMongoClientPromise = undefined;
  }
}
