import assert from 'node:assert/strict';
import test from 'node:test';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { randomUUID } from 'node:crypto';

test(
  'Mongo adapter uses atomic queue IDs, bounded cursors and attempt CAS across independent clients',
  { skip: !process.env.AITK_TEST_MONGODB_URI },
  async () => {
    const { MongoClient } = await import('mongodb');
    const databaseName = 'aitk_test_' + randomUUID().replaceAll('-', '');
    const env = { ...process.env };
    process.env.AITK_DB_PROVIDER = 'mongodb';
    process.env.AITK_MONGODB_URI = process.env.AITK_TEST_MONGODB_URI;
    process.env.AITK_MONGODB_DB = databaseName;
    const { db, disconnectDb } = await import('../dist/src/server/db.js');
    const observer = new MongoClient(process.env.AITK_MONGODB_URI);
    try {
      await promisify(execFile)(process.execPath, ['scripts/prepare-db.mjs'], { env: process.env, timeout: 60000 });
      await promisify(execFile)(process.execPath, ['scripts/prepare-db.mjs'], { env: process.env, timeout: 60000 });
      const cas = await Promise.all([1, 2].map(owner => db.runtime.compareAndSwap('race', null, { owner })));
      assert.equal(cas.filter(Boolean).length, 1);
      const indexes = await observer.db(databaseName).collection('runtime_records').indexes();
      assert.ok(indexes.some(index => index.key.key === 1 && index.unique));
      const queues = await Promise.all(
        Array.from({ length: 40 }, (_, i) => db.queues.create({ gpu_ids: String(i), worker_id: 'local' })),
      );
      assert.equal(new Set(queues.map(queue => queue.id)).size, 40);
      const jobs = await Promise.all(
        Array.from({ length: 20 }, (_, i) =>
          db.jobs.create({ name: 'fixture-' + i, gpu_ids: '0', job_config: '{}', status: 'queued' }),
        ),
      );
      const job = jobs[0];
      const claims = await Promise.all(
        ['one', 'two'].map(attempt_id =>
          db.jobs.updateIf(
            job.id,
            { attempt_id: null, status: 'queued', updated_at: job.updated_at },
            { attempt_id, status: 'starting' },
          ),
        ),
      );
      assert.equal(claims.filter(Boolean).length, 1);
      const winner = claims.find(Boolean);
      const raw = await observer.db(databaseName).collection('jobs').findOne({ id: job.id });
      assert.equal(raw.attempt_id, winner.attempt_id);
      assert.equal(await db.jobs.updateIf(job.id, { attempt_id: 'stale' }, { status: 'completed' }), null);
      const first = await db.jobs.list({ limit: 3 });
      assert.equal(first.length, 3);
      const second = await db.jobs.list({
        limit: 3,
        before: { created_at: new Date(first[2].created_at), id: first[2].id },
      });
      assert.equal(second.length, 3);
      assert.equal(new Set([...first, ...second].map(row => row.id)).size, 6);
      assert.ok(await db.runtime.compareAndSwap('fixture', null, { phase: 'accepted' }));
      await disconnectDb();
      assert.deepEqual((await db.runtime.get('fixture')).value, { phase: 'accepted' });
    } finally {
      await disconnectDb();
      await observer.db(databaseName).dropDatabase();
      await observer.close();
      for (const key of ['AITK_DB_PROVIDER', 'AITK_MONGODB_URI', 'AITK_MONGODB_DB']) {
        if (env[key] === undefined) delete process.env[key];
        else process.env[key] = env[key];
      }
    }
  },
);
