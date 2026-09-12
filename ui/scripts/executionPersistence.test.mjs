import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

test('real SQLite adapter fences concurrent starts and stale exits and persists operations across reconnect', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-adapter-test-'));
  const old = process.env.AITK_SQLITE_PATH;
  process.env.AITK_SQLITE_PATH = path.join(root, 'test.sqlite');
  process.env.AITK_DB_PROVIDER = 'sqlite';
  let disconnect;
  try {
    await promisify(execFile)(process.execPath, ['scripts/prepare-db.mjs'], { env: process.env, timeout: 60000 });
    const { db, disconnectDb } = await import('../dist/src/server/db.js');
    disconnect = disconnectDb;
    const { claimJobAttempt } = await import('../dist/src/server/jobAttempts.js');
    const { finishObservedJobProcess } = await import('../dist/src/server/jobProcess.js');
    const { enqueueOperation, getOperation } = await import('../dist/src/server/operations.js');
    const job = await db.jobs.create({ name: 'isolated fixture', gpu_ids: '0', job_config: '{}', status: 'queued' });
    const claimed = await Promise.all([claimJobAttempt(job), claimJobAttempt(job)]);
    assert.equal(claimed.filter(Boolean).length, 1);
    const owner = claimed.find(Boolean);
    assert.equal((await db.jobs.findById(job.id)).attempt_id, owner.attempt_id);
    assert.equal(await db.jobs.updateIf(job.id, { attempt_id: 'stale' }, { status: 'completed' }), null);
    await finishObservedJobProcess(owner, 'completed', 'fixture exited');
    const completed = await db.jobs.findById(job.id);
    assert.equal(completed.status, 'completed');
    const operation = await enqueueOperation({
      kind: 'remote-start',
      jobID: job.id,
      durableKeys: false,
      needsEphemeralKeys: false,
      configHash: 'fixture',
    });
    await disconnectDb();
    assert.equal((await getOperation(operation.id)).state, 'queued');
    assert.equal((await db.jobs.findFirst({ status: 'completed' })).id, job.id);
  } finally {
    await disconnect?.();
    if (old === undefined) delete process.env.AITK_SQLITE_PATH;
    else process.env.AITK_SQLITE_PATH = old;
    await fs.rm(root, { recursive: true, force: true });
  }
});
