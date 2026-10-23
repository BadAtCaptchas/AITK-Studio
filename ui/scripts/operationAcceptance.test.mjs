import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import crypto from 'node:crypto';
import * as timers from 'node:timers/promises';
import test from 'node:test';
import ts from 'typescript';

function load(relative, dependencies = {}) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const loaded = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
    return dependencies[name];
  }, loaded, loaded.exports);
  return loaded.exports;
}

function fixture(race = false) {
  const records = new Map();
  let reads = 0, release;
  const bothRead = new Promise(resolve => { release = resolve; });
  const db = { runtime: {
    get: async key => {
      const snapshot = structuredClone(records.get(key) ?? null);
      if (race && key.startsWith('operation:') && reads < 2) {
        reads++;
        if (reads === 2) release();
        await bothRead;
      }
      return snapshot;
    },
    compareAndSwap: async (key, version, value) => {
      if ((records.get(key)?.version ?? null) !== version) return false;
      records.set(key, { key, version: (version ?? 0) + 1, value: structuredClone(value) });
      return true;
    },
    delete: async (key, version) => records.get(key)?.version === version && records.delete(key),
  } };
  const lease = load('../src/server/processLease.ts', {
    os, crypto, 'timers/promises': timers, './db': { db },
    './processBirth': { getProcessBirth: async () => 1000, processBirthMatches: (a, b) => a === b },
  });
  const commandInput = load('../src/server/commandInput.ts', {
    '../utils/obsoleteWorkspaceGuard': load('../src/utils/obsoleteWorkspaceGuard.ts'),
  });
  const operations = load('../src/server/operations.ts', { crypto, './db': { db }, './processLease': lease, './commandInput': commandInput });
  return { records, operations };
}
const input = jobID => ({ kind: 'remote-start', jobID, configHash: 'fixture', durableKeys: false, needsEphemeralKeys: false });

test('same-key requests on distinct resources cannot publish contradictory operation identities', { timeout: 5000 }, async () => {
  const f = fixture(true);
  const results = await Promise.allSettled([
    f.operations.enqueueOperation(input('first'), 'shared-request-key'),
    f.operations.enqueueOperation(input('second'), 'shared-request-key'),
  ]);
  const accepted = results.filter(result => result.status === 'fulfilled');
  const rejected = results.filter(result => result.status === 'rejected');
  assert.equal(accepted.length, 1);
  assert.equal(rejected.length, 1);
  assert.equal(rejected[0].reason.status, 409);
  const winner = accepted[0].value;
  assert.deepEqual(await f.operations.getOperation(winner.id), winner);
  const resourceRows = [...f.records.values()].filter(row => row.key.startsWith('operation-resource:'));
  assert.equal(resourceRows.length, 1);
  assert.equal(resourceRows[0].key, `operation-resource:${winner.resource}`);
  assert.equal(resourceRows[0].value, winner.id);
});

test('an existing active resource cannot conceal a key owned by another operation', async () => {
  const f = fixture();
  const first = await f.operations.enqueueOperation(input('first'), 'first-request-key');
  const second = await f.operations.enqueueOperation(input('second'), 'second-request-key');
  await assert.rejects(f.operations.enqueueOperation(input('second'), 'first-request-key'), error => error.status === 409);
  assert.equal((await f.operations.activeOperationForResource('job:first')).id, first.id);
  assert.equal((await f.operations.activeOperationForResource('job:second')).id, second.id);
});

test('a prior completed key returns its receipt while a newer command owns the same resource', async () => {
  const f = fixture();
  const first = await f.operations.enqueueOperation(input('first'), 'original-command');
  await f.operations.updateOperation(first.id, { state: 'completed' });
  const newer = await f.operations.enqueueOperation(input('first'), 'newer-command');
  const replay = await f.operations.enqueueOperation(input('first'), 'original-command');
  assert.equal(replay.id, first.id);
  assert.equal(replay.state, 'completed');
  assert.equal((await f.operations.activeOperationForResource('job:first')).id, newer.id);
});

test('a new command identity still attaches to matching active work', async () => {
  const f = fixture();
  const first = await f.operations.enqueueOperation(input('first'), 'original-command');
  const attached = await f.operations.enqueueOperation(input('first'), 'attachment-command');
  assert.equal(attached.id, first.id);
});

test('an interrupted acceptance repairs its resource and pending index on retry', async () => {
  const f = fixture();
  const accepted = await f.operations.enqueueOperation(input('first'), 'interrupted-command');
  f.records.delete('operation-resource:job:first');
  f.records.delete(`operation-pending:${accepted.id}`);
  const replay = await f.operations.enqueueOperation(input('first'), 'interrupted-command');
  assert.deepEqual(replay, accepted);
  assert.equal((await f.operations.activeOperationForResource('job:first')).id, accepted.id);
  assert.equal(f.records.get(`operation-pending:${accepted.id}`).value, accepted.id);
});

test('an interrupted prior command cannot reclaim a resource from a newer active owner', async () => {
  const f = fixture();
  const prior = await f.operations.enqueueOperation(input('first'), 'interrupted-command');
  f.records.delete('operation-resource:job:first');
  const newer = await f.operations.enqueueOperation(input('first'), 'newer-command');
  await assert.rejects(f.operations.enqueueOperation(input('first'), 'interrupted-command'), error => error.status === 409);
  assert.equal((await f.operations.activeOperationForResource('job:first')).id, newer.id);
  assert.equal((await f.operations.getOperation(prior.id)).state, 'queued');
});
