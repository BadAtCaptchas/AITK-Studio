import assert from 'node:assert/strict';
import { test } from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { beginSqlitePreparation } from './schema-migrations.mjs';

test('SQLite preparations exclude concurrent owners, skip completed versions and reject changed checksums', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-migration-test-'));
  const filename = path.join(directory, 'fixture.sqlite');
  try {
    const first = await beginSqlitePreparation(filename, 'checksum-a');
    assert.equal(first.completed, false);
    await assert.rejects(beginSqlitePreparation(filename, 'checksum-a'), /already owned/);
    await first.finish();
    await first.close();
    const second = await beginSqlitePreparation(filename, 'checksum-a');
    assert.equal(second.completed, true);
    await second.close();
    await assert.rejects(beginSqlitePreparation(filename, 'checksum-b'), /checksum changed/);
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
});
