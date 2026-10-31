import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import test from 'node:test';
import Database from 'better-sqlite3';
import { MongoClient } from 'mongodb';
import { db, disconnectDb } from '../dist/src/server/db.js';
import { metricJsonResponse } from '../dist/src/server/metricJsonResponse.js';

for (const provider of ['sqlite', 'mongodb']) {
  test(`${provider} bounded metrics retain tail, late series, text and same-step values beyond 20k history`,
    { skip: provider === 'mongodb' && !process.env.AITK_TEST_MONGODB_URI }, async () => {
      const saved = { ...process.env };
      const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-metrics-'));
      const logPath = path.join(directory, 'loss_log.db');
      const name = 'aitk_test_' + randomUUID().replaceAll('-', '');
      let mongo;
      try {
        process.env.AITK_DB_PROVIDER = provider;
        const rows = Array.from({ length: 21001 }, (_, step) => ({ job_id: 'fixture', key: 'loss', step, wall_time: step, value_real: step / 7, value_text: null }));
        rows.push({ job_id: 'fixture', key: 'phase/name', step: 3, wall_time: 3, value_real: null, value_text: 'Warmup' });
        rows.push({ job_id: 'fixture', key: 'late', step: 2, wall_time: 2, value_real: 0.25, value_text: null });
        const keys = ['loss', 'phase/name', 'late'].map(key => ({ job_id: 'fixture', key, first_seen_step: 0, last_seen_step: key === 'loss' ? 21000 : 3 }));
        if (provider === 'sqlite') {
          const sqlite = new Database(logPath);
          try {
            sqlite.exec('CREATE TABLE steps(step INTEGER PRIMARY KEY, wall_time REAL); CREATE TABLE metrics(step INTEGER, key TEXT, value_real REAL, value_text TEXT, PRIMARY KEY(key, step)); CREATE TABLE metric_keys(key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER)');
            const insertStep = sqlite.prepare('INSERT INTO steps VALUES (?, ?)');
            const insert = sqlite.prepare('INSERT INTO metrics VALUES (?, ?, ?, ?)');
            const insertKey = sqlite.prepare('INSERT INTO metric_keys VALUES (?, ?, ?)');
            sqlite.transaction(() => {
              for (let step = 0; step <= 21000; step++) insertStep.run(step, step);
              for (const row of rows) insert.run(row.step, row.key, row.value_real, row.value_text);
              for (const key of keys) insertKey.run(key.key, key.first_seen_step, key.last_seen_step);
            })();
          } finally { sqlite.close(); }
        } else {
          process.env.AITK_MONGODB_URI = process.env.AITK_TEST_MONGODB_URI;
          process.env.AITK_MONGODB_DB = name;
          mongo = new MongoClient(process.env.AITK_MONGODB_URI);
          await mongo.db(name).collection('metrics').insertMany(rows);
          await mongo.db(name).collection('metric_keys').insertMany(keys);
        }
        const snapshot = await db.metrics.getMetrics('fixture', logPath, { keys: ['*'], maxPoints: 4000, sinceStep: null });
        assert.equal(snapshot.series.loss.totalCount, 21001);
        assert.ok(snapshot.series.loss.points.length <= 4000);
        assert.equal(snapshot.series.loss.points.at(-1).step, 21000);
        assert.equal(snapshot.series.loss.latest.value, 21000 / 7);
        assert.equal(snapshot.series['phase/name'].points[0].value_text, 'Warmup');
        const incremental = await db.metrics.getMetrics('fixture', logPath, { keys: ['*'], maxPoints: 4000, sinceStep: null, sinceSteps: { loss: 21000 } });
        assert.deepEqual(incremental.series.loss.points, []);
        assert.equal(incremental.series.loss.latest.step, 21000);
        assert.equal(incremental.series.late.points[0].step, 2);
        const response = await metricJsonResponse(new Request('http://localhost'), snapshot, 8 * 1024 ** 2);
        assert.equal(response.status, 200);
        assert.deepEqual(await response.json(), snapshot);
      } finally {
        await disconnectDb();
        if (mongo) { await mongo.db(name).dropDatabase(); await mongo.close(); }
        await fs.rm(directory, { recursive: true, force: true });
        for (const key of ['AITK_DB_PROVIDER', 'AITK_MONGODB_URI', 'AITK_MONGODB_DB']) {
          if (saved[key] === undefined) delete process.env[key]; else process.env[key] = saved[key];
        }
      }
    });
}
