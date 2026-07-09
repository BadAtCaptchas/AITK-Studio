import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import sqlite3 from 'sqlite3';
import { backupExistingSqliteDatabase, getSqliteBackupRetention } from './sqlite-backup.mjs';

function sqliteRun(db, sql, params = []) {
  return new Promise((resolve, reject) => {
    db.run(sql, params, error => {
      if (error) reject(error);
      else resolve();
    });
  });
}

function sqliteGet(db, sql) {
  return new Promise((resolve, reject) => {
    db.get(sql, (error, row) => {
      if (error) reject(error);
      else resolve(row);
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

function removeTestTree(directory) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) removeTestTree(entryPath);
    else fs.unlinkSync(entryPath);
  }
  fs.rmdirSync(directory);
}

test('SQLite backup retention is bounded and validates configuration', () => {
  assert.equal(getSqliteBackupRetention(undefined), 3);
  assert.equal(getSqliteBackupRetention('0'), 0);
  assert.equal(getSqliteBackupRetention('2'), 2);
  assert.equal(getSqliteBackupRetention('999'), 50);
});

test('pre-migration backups are consistent and prune expired snapshots', async () => {
  const testTempRoot = path.join(process.cwd(), '.test-tmp');
  fs.mkdirSync(testTempRoot, { recursive: true });
  const temporaryDirectory = fs.mkdtempSync(path.join(testTempRoot, 'aitk-sqlite-backup-'));
  const databasePath = path.join(temporaryDirectory, 'fixture.db');
  const db = new sqlite3.Database(databasePath);

  try {
    await sqliteRun(db, 'PRAGMA journal_mode=WAL;');
    await sqliteRun(db, 'CREATE TABLE important_data (value TEXT NOT NULL);');
    await sqliteRun(db, 'INSERT INTO important_data (value) VALUES (?)', ['preserve-me']);
  } finally {
    await closeSqlite(db);
  }

  try {
    await backupExistingSqliteDatabase(databasePath, 2);
    await backupExistingSqliteDatabase(databasePath, 2);
    await backupExistingSqliteDatabase(databasePath, 2);

    const backupDirectory = path.join(temporaryDirectory, '.aitk-backups');
    const backups = fs.readdirSync(backupDirectory).filter(filename => filename.endsWith('.sqlite3'));
    assert.equal(backups.length, 2);

    for (const backupFilename of backups) {
      const backup = new sqlite3.Database(path.join(backupDirectory, backupFilename));
      try {
        assert.deepEqual(await sqliteGet(backup, 'PRAGMA integrity_check;'), { integrity_check: 'ok' });
        assert.deepEqual(await sqliteGet(backup, 'SELECT value FROM important_data;'), { value: 'preserve-me' });
      } finally {
        await closeSqlite(backup);
      }
    }
  } finally {
    removeTestTree(temporaryDirectory);
    if (fs.readdirSync(testTempRoot).length === 0) fs.rmdirSync(testTempRoot);
  }
});
