import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import crypto from 'node:crypto';
import * as timers from 'node:timers/promises';
import test from 'node:test';
import ts from 'typescript';

function loadSource(relative, dependencies) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const evaluatedModule = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency: ${name}`);
    return dependencies[name];
  }, evaluatedModule, evaluatedModule.exports);
  return evaluatedModule.exports;
}

const commandInput = loadSource('../src/server/commandInput.ts', {
  '../utils/obsoleteWorkspaceGuard': loadSource('../src/utils/obsoleteWorkspaceGuard.ts', {}),
});
class JobStartError extends Error {
  constructor(payload, status) { super(payload.error); this.payload = payload; this.status = status; }
}

function fixture() {
  const records = new Map();
  const job = { id: 'job', status: 'stopped', attempt_id: null, worker_id: 'local' };
  const db = {
    jobs: { findById: async () => structuredClone(job) },
    runtime: {
      get: async key => structuredClone(records.get(key) ?? null),
      compareAndSwap: async (key, version, value) => {
        if ((records.get(key)?.version ?? null) !== version) return false;
        records.set(key, { version: (version ?? 0) + 1, value: structuredClone(value) });
        return true;
      },
      delete: async (key, version) => records.get(key)?.version === version && records.delete(key),
    },
  };
  const lease = loadSource('../src/server/processLease.ts', {
    os, crypto, 'timers/promises': timers, './db': { db },
    './processBirth': { getProcessBirth: async () => 1000, processBirthMatches: (a, b) => a === b },
  });
  // Distinct module instances represent requests handled by different UI processes.
  const client = (leases = lease) => loadSource('../src/server/startCommand.ts', {
    './db': { db }, './processLease': leases, './jobStart': { JobStartError }, './commandInput': commandInput,
  });
  return { db, job, records, lease, client };
}

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

test('overlapping remote retries wait for the original start and never enqueue twice', { timeout: 5000 }, async () => {
  const f = fixture();
  const entered = deferred(), finish = deferred();
  let starts = 0;
  const start = async () => {
    starts++;
    entered.resolve();
    await finish.promise;
    f.job.status = 'queued';
    return structuredClone(f.job);
  };
  const first = f.client().idempotentStart('job', 'remote-operation', start);
  await entered.promise;
  const second = f.client().idempotentStart('job', 'remote-operation', start);
  const third = f.client().idempotentStart('job', 'remote-operation', start);
  // Register rejection handlers before allowing timers to run.
  const results = Promise.all([first, second, third]);
  try {
    await timers.setTimeout(40);
    assert.equal(starts, 1);
  } finally { finish.resolve(); }
  assert.deepEqual((await results).map(job => job.status), ['queued', 'queued', 'queued']);
  assert.equal(starts, 1);
  assert.equal([...f.records.keys()].some(key => key.startsWith('lease:')), false);

  f.job.status = 'completed';
  assert.equal((await f.client().idempotentStart('job', 'remote-operation', start)).status, 'completed');
  assert.equal(starts, 1, 'replaying an accepted command must not restart a completed run');
});

test('a failed start releases ownership so the same command can be retried', async () => {
  const f = fixture();
  await assert.rejects(f.client().idempotentStart('job', 'retry', async () => { throw new Error('preflight failed'); }), /preflight failed/);
  let starts = 0;
  await f.client().idempotentStart('job', 'retry', async () => { starts++; return f.job; });
  assert.equal(starts, 1);
});

test('waiting is bounded and cannot steal a live operation lease', async () => {
  const f = fixture();
  const owner = await f.lease.acquireProcessLease('busy');
  try {
    await assert.rejects(f.lease.withProcessLease('busy', async () => assert.fail('stole owner'), { waitMs: 30 }), f.lease.LeaseBusyError);
    await owner.assertOwned();
  } finally { await owner.release(); }
});

test('start lease timeout is returned as a retryable JSON conflict instead of an unhandled route error', async () => {
  const f = fixture();
  const startCommand = f.client({
    ...f.lease,
    withProcessLease: async (_key, _work, options) => {
      assert.equal(options.waitMs, 5000);
      throw new f.lease.LeaseBusyError();
    },
  });
  const { POST } = loadSource('../src/app/api/jobs/[jobID]/start/route.ts', {
    '@/server/commandInput': commandInput,
    '@/server/jobStart': { JobStartError, startJobFromRequest: async () => assert.fail('busy command launched') },
    '@/server/db': { db: f.db }, '@/server/remoteClient': { isLocalWorker: () => true },
    '@/server/remoteStartOperation': {}, '@/server/remoteStartProgress': {},
    '@/utils/authSession': { isRequestAuthenticated: async () => true }, '@/server/startCommand': startCommand,
  });
  const response = await POST(new Request('http://localhost/api/jobs/job/start', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ idempotencyKey: 'retry-command' }),
  }), { params: Promise.resolve({ jobID: 'job' }) });
  assert.equal(response.status, 409);
  assert.equal((await response.json()).code, 'START_BUSY');
});

test('a start receipt CAS conflict preserves the retryable command identity', async () => {
  const f = fixture();
  const compareAndSwap = f.db.runtime.compareAndSwap;
  f.db.runtime.compareAndSwap = async (key, version, value) => {
    if (key === 'start-command:job:receipt-conflict' && version === null) {
      // Another writer publishes the receipt after the initial read. Losing the
      // CAS cannot establish that the caller's original command was rejected.
      f.records.set(key, { version: 0, value: { jobID: 'job', accepted: true } });
      f.job.status = 'queued';
      return false;
    }
    return compareAndSwap(key, version, value);
  };
  const { POST } = loadSource('../src/app/api/jobs/[jobID]/start/route.ts', {
    '@/server/commandInput': commandInput,
    '@/server/jobStart': { JobStartError, startJobFromRequest: async () => assert.fail('conflicting command launched again') },
    '@/server/db': { db: f.db }, '@/server/remoteClient': { isLocalWorker: () => true },
    '@/server/remoteStartOperation': {}, '@/server/remoteStartProgress': {},
    '@/utils/authSession': { isRequestAuthenticated: async () => true }, '@/server/startCommand': f.client(),
  });
  const request = () => POST(new Request('http://localhost/api/jobs/job/start', {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ idempotencyKey: 'receipt-conflict' }),
  }), { params: Promise.resolve({ jobID: 'job' }) });
  const busy = await request();
  const response = await busy.json();
  const replay = await request();
  assert.equal(replay.status, 200);
  assert.equal((await replay.json()).status, 'queued');
  assert.equal(busy.status, 409);
  assert.equal(response.code, 'START_BUSY');
});
