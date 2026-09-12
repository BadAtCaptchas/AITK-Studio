import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createHash, randomUUID } from 'node:crypto';
import sqlite3 from 'sqlite3';

export const SCHEMA_VERSION = '2026-09-execution-v1';
export function schemaChecksum(uiRoot) {
  const hash = createHash('sha256');
  for (const file of ['prisma/schema.prisma', 'scripts/prepare-db.mjs', 'scripts/global-workspace-upgrade.mjs']) {
    hash.update(fs.readFileSync(path.join(uiRoot, file), 'utf8').replace(/\r\n/g, '\n'));
  }
  return hash.digest('hex');
}
export function sqliteStatement(db, sql, values = []) {
  return new Promise((resolve, reject) => db.run(sql, values, error => (error ? reject(error) : resolve())));
}
function sqliteRow(db, sql, values = []) {
  return new Promise((resolve, reject) => db.get(sql, values, (error, row) => (error ? reject(error) : resolve(row))));
}
export async function beginSqlitePreparation(filename, checksum) {
  fs.mkdirSync(path.dirname(filename), { recursive: true });
  const lock = `${filename}.prepare.lock`;
  const owner = { pid: process.pid, host: os.hostname(), id: randomUUID() };
  const acquire = () => fs.writeFileSync(lock, JSON.stringify(owner), { flag: 'wx', mode: 0o600 });
  try {
    acquire();
  } catch (error) {
    if (error.code !== 'EEXIST') throw error;
    // A stale-owner probe followed by rename can steal a replacement owner's lock.
    // Do not automatically reclaim: migrations are infrequent and require exclusive ownership.
    throw new Error(
      `Database preparation is already owned. If interrupted, inspect ${lock} and stop all preparation processes before removing that lock and retrying.`,
    );
  }
  const release = () => {
    try {
      if (JSON.parse(fs.readFileSync(lock, 'utf8')).id === owner.id) fs.unlinkSync(lock);
    } catch {
      /* Preserve unknown owners. */
    }
  };
  let db;
  try {
    db = new sqlite3.Database(filename);
    await sqliteStatement(db, 'PRAGMA busy_timeout=30000');
    const table = await sqliteRow(db, "SELECT name FROM sqlite_master WHERE type='table' AND name='SchemaMigration'");
    const row = table
      ? await sqliteRow(db, 'SELECT checksum FROM SchemaMigration WHERE version=?', [SCHEMA_VERSION])
      : null;
    if (row && row.checksum !== checksum)
      throw new Error(
        'Applied schema migration checksum changed. Add a new migration version instead of changing an applied migration.',
      );
    return {
      completed: Boolean(row),
      async finish() {
        await sqliteStatement(
          db,
          'CREATE TABLE IF NOT EXISTS SchemaMigration (version TEXT PRIMARY KEY, checksum TEXT NOT NULL, completed_at TEXT NOT NULL)',
        );
        await sqliteStatement(db, 'INSERT OR IGNORE INTO SchemaMigration VALUES (?, ?, ?)', [
          SCHEMA_VERSION,
          checksum,
          new Date().toISOString(),
        ]);
      },
      async close() {
        await new Promise((resolve, reject) => db.close(error => (error ? reject(error) : resolve())));
        release();
      },
    };
  } catch (error) {
    if (db) await new Promise(resolve => db.close(resolve));
    release();
    throw error;
  }
}

export async function beginMongoPreparation(db, checksum) {
  const migrations = db.collection('schema_migrations');
  const previous = await migrations.findOne({ _id: SCHEMA_VERSION });
  if (previous?.completed_at) {
    if (previous.checksum !== checksum)
      throw new Error('Applied MongoDB migration checksum changed. Add a new migration version.');
    return { completed: true, finish: async () => {}, close: async () => {} };
  }
  const owner = randomUUID();
  try {
    await db
      .collection('schema_locks')
      .insertOne({ _id: 'prepare', owner, host: os.hostname(), pid: process.pid, started_at: new Date() });
  } catch (error) {
    if (error.code === 11000)
      throw new Error(
        'MongoDB preparation is already owned. An interrupted migration requires operator inspection before removing its schema_locks entry.',
      );
    throw error;
  }
  return {
    completed: false,
    async finish() {
      await migrations.updateOne(
        { _id: SCHEMA_VERSION },
        { $set: { checksum, completed_at: new Date() } },
        { upsert: true },
      );
    },
    async close() {
      await db.collection('schema_locks').deleteOne({ _id: 'prepare', owner });
    },
  };
}
