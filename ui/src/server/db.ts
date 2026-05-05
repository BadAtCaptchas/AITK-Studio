import { randomUUID } from 'crypto';
import fs from 'fs';
import path from 'path';
import { MongoClient, type Collection, type Db, type Document } from 'mongodb';
import { PrismaClient } from '@prisma/client';
import sqlite3 from 'sqlite3';
import { TOOLKIT_ROOT } from '../paths';
import type { Job, Queue } from '../types';

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
};

export type JobUpdateInput = Partial<Omit<JobCreateInput, 'id'>>;

export type QueueCreateInput = {
  id?: number;
  gpu_ids: string;
  is_running?: boolean;
};

export type QueueUpdateInput = Partial<Pick<Queue, 'gpu_ids' | 'is_running'>>;

export type LossPoint = {
  step: number;
  wall_time: number;
  value: number | null;
};

export type LossLogResult = {
  key: string;
  keys: string[];
  points: LossPoint[];
};

const DEFAULT_MONGODB_DB = 'ai_toolkit';

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
    globalThis.__aitkPrismaClient = new PrismaClient({
      datasources: {
        db: {
          url: config.sqliteUrl,
        },
      },
    });
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
  };
}

function normalizeQueue(raw: any): Queue | null {
  if (!raw) return null;
  return {
    id: Number(raw.id ?? 0),
    gpu_ids: String(raw.gpu_ids ?? ''),
    is_running: Boolean(raw.is_running),
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
  sqlite.configure('busyTimeout', 30_000);
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

function closeSqliteDb(sqlite: sqlite3.Database) {
  return new Promise<void>((resolve, reject) => {
    sqlite.close(err => (err ? reject(err) : resolve()));
  });
}

async function readSqliteLossLog(
  logPath: string,
  key: string,
  limit: number,
  sinceStep: number | null,
  stride: number,
): Promise<LossLogResult> {
  if (!fs.existsSync(logPath)) {
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
        value: point.value ?? (point.value_text ? Number(point.value_text) : null),
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
      value: row.value_real == null ? (row.value_text ? Number(row.value_text) : null) : Number(row.value_real),
    })),
  };
}

async function ensureMongoIndexes() {
  const mongo = await getMongoDb();
  await Promise.all([
    mongoCollection(mongo, 'jobs').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { name: 1 }, unique: true },
      { key: { status: 1 } },
      { key: { gpu_ids: 1 } },
      { key: { job_type: 1 } },
      { key: { job_ref: 1 } },
      { key: { queue_position: 1 } },
    ]),
    mongoCollection(mongo, 'queues').createIndexes([
      { key: { id: 1 }, unique: true },
      { key: { gpu_ids: 1 }, unique: true },
    ]),
    mongoCollection(mongo, 'settings').createIndexes([{ key: { key: 1 }, unique: true }]),
    mongoCollection(mongo, 'metrics').createIndexes([
      { key: { job_id: 1, step: 1, key: 1 }, unique: true },
      { key: { job_id: 1, key: 1, step: 1 } },
    ]),
    mongoCollection(mongo, 'metric_keys').createIndexes([{ key: { job_id: 1, key: 1 }, unique: true }]),
  ]);
}

async function nextMongoQueueId(queues: Collection<Document>) {
  const latest = await queues.find({}, { projection: { _id: 0, id: 1 } }).sort({ id: -1 }).limit(1).next();
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
      return getPrisma().job.findUnique({ where: { name } });
    },

    async findLatestByRef(jobRef: string): Promise<Job | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'jobs')
          .find({ job_ref: jobRef }, { projection: { _id: 0 } })
          .sort({ updated_at: -1 })
          .limit(1)
          .next();
        return normalizeJob(row);
      }
      return getPrisma().job.findFirst({
        where: { job_ref: jobRef },
        orderBy: { updated_at: 'desc' },
      });
    },

    async list(options: { job_type?: string | null; status?: string | string[]; gpu_ids?: string; order?: 'created_desc' | 'queue_asc' } = {}) {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const filter: Document = {};
        if (options.job_type) filter.job_type = options.job_type;
        if (options.gpu_ids) filter.gpu_ids = options.gpu_ids;
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
      if (Array.isArray(options.status)) where.status = { in: options.status };
      else if (options.status) where.status = options.status;
      return getPrisma().job.findMany({
        where: Object.keys(where).length > 0 ? where : undefined,
        orderBy: options.order === 'queue_asc' ? { queue_position: 'asc' } : { created_at: 'desc' },
      });
    },

    async findFirst(options: { status?: string | string[]; gpu_ids?: string; order?: 'queue_asc' } = {}) {
      const rows = await db.jobs.list(options);
      return rows[0] ?? null;
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
    async list(order: 'id' | 'gpu_ids' = 'id'): Promise<Queue[]> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const rows = await mongoCollection(mongo, 'queues')
          .find({}, { projection: { _id: 0 } })
          .sort({ [order]: 1 })
          .toArray();
        return rows.map(normalizeQueue).filter(Boolean) as Queue[];
      }
      return getPrisma().queue.findMany({
        orderBy: { [order]: 'asc' },
      });
    },

    async findByGpuIds(gpuIds: string): Promise<Queue | null> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const row = await mongoCollection(mongo, 'queues').findOne({ gpu_ids: gpuIds }, { projection: { _id: 0 } });
        return normalizeQueue(row);
      }
      return getPrisma().queue.findUnique({ where: { gpu_ids: gpuIds } });
    },

    async create(input: QueueCreateInput): Promise<Queue> {
      if (isMongoProvider()) {
        const mongo = await getMongoDb();
        const queues = mongoCollection(mongo, 'queues');
        const queue = normalizeQueue({
          id: input.id ?? (await nextMongoQueueId(queues)),
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
