import { randomUUID } from 'crypto';
import fs from 'fs';
import path from 'path';
import sqlite3 from 'sqlite3';

const SQLITE_BUSY_TIMEOUT_MS = 30000;
const DEFAULT_SQLITE_BACKUP_RETENTION = 3;
const MAX_SQLITE_BACKUP_RETENTION = 50;

function sqliteRun(db, sql) {
  return new Promise((resolve, reject) => {
    db.run(sql, error => {
      if (error) reject(error);
      else resolve();
    });
  });
}

function sqliteBackup(db, destination) {
  return new Promise((resolve, reject) => {
    const backup = db.backup(destination, initializationError => {
      if (initializationError) {
        reject(initializationError);
        return;
      }

      backup.step(-1, (stepError, completed) => {
        const error = stepError || (completed ? null : new Error('SQLite backup did not reach a complete state.'));
        backup.finish(() => {
          if (error) reject(error);
          else resolve();
        });
      });
    });
  });
}

function closeSqlite(db) {
  return new Promise((resolve, reject) => {
    db.close(error => {
      if (error) reject(error);
      else resolve();
    });
  });
}

export function getSqliteBackupRetention(configured = process.env.AITK_SQLITE_BACKUP_RETENTION?.trim()) {
  if (configured === undefined || configured === '') return DEFAULT_SQLITE_BACKUP_RETENTION;
  if (!/^\d+$/.test(configured)) {
    console.warn(
      `Ignoring invalid AITK_SQLITE_BACKUP_RETENTION="${configured}"; using ${DEFAULT_SQLITE_BACKUP_RETENTION}.`,
    );
    return DEFAULT_SQLITE_BACKUP_RETENTION;
  }
  return Math.min(Number(configured), MAX_SQLITE_BACKUP_RETENTION);
}

export async function backupExistingSqliteDatabase(filename, retention = getSqliteBackupRetention()) {
  if (retention === 0 || !fs.existsSync(filename) || fs.statSync(filename).size === 0) return null;

  const backupDirectory = path.join(path.dirname(filename), '.aitk-backups');
  const backupPrefix = `${path.basename(filename)}.`;
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const backupPath = path.join(backupDirectory, `${backupPrefix}${timestamp}.${randomUUID()}.sqlite3`);
  fs.mkdirSync(backupDirectory, { recursive: true });

  const db = new sqlite3.Database(filename);
  try {
    await sqliteRun(db, `PRAGMA busy_timeout=${SQLITE_BUSY_TIMEOUT_MS};`);
    await sqliteBackup(db, backupPath);
  } catch (error) {
    fs.rmSync(backupPath, { force: true });
    throw new Error(`Could not create the pre-migration SQLite backup at ${backupPath}`, { cause: error });
  } finally {
    await closeSqlite(db);
  }

  const backups = fs
    .readdirSync(backupDirectory, { withFileTypes: true })
    .filter(entry => entry.isFile() && entry.name.startsWith(backupPrefix) && entry.name.endsWith('.sqlite3'))
    .map(entry => entry.name)
    .sort()
    .reverse();
  for (const expiredBackup of backups.slice(retention)) {
    fs.rmSync(path.join(backupDirectory, expiredBackup), { force: true });
  }

  console.log(`Created pre-migration SQLite backup: ${backupPath}`);
  return backupPath;
}
