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
  const loadedModule = { exports: {} };
  new Function('require', 'module', 'exports', code)(
    name => {
      assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency: ${name}`);
      return dependencies[name];
    },
    loadedModule,
    loadedModule.exports,
  );
  return loadedModule.exports;
}

function deferred() {
  let resolve;
  const promise = new Promise(done => {
    resolve = done;
  });
  return { promise, resolve };
}

class RemoteClientError extends Error {
  constructor(status, body = '') {
    super('Remote command failed');
    this.status = status;
    this.body = body;
  }
}

function fixture(workerID = 'local') {
  const records = new Map();
  const jobs = new Map(
    ['a', 'b', 'c'].map((id, index) => [
      id,
      {
        id,
        worker_id: workerID,
        gpu_ids: '0',
        status: 'queued',
        queue_position: (index + 1) * 1000,
        remote_job_id: workerID === 'local' ? null : `remote-${id}`,
      },
    ]),
  );
  const writes = [],
    remoteCalls = [];
  let beforeUpdate = async () => {},
    remoteCommand = async () => ({});
  const db = {
    runtime: {
      get: async key => structuredClone(records.get(key) ?? null),
      compareAndSwap: async (key, version, value) => {
        if ((records.get(key)?.version ?? null) !== version) return false;
        records.set(key, { version: (version ?? 0) + 1, value: structuredClone(value) });
        return true;
      },
      delete: async (key, version) => records.get(key)?.version === version && records.delete(key),
    },
    jobs: {
      findById: async id => structuredClone(jobs.get(id) ?? null),
      list: async ({ worker_id, gpu_ids, status }) =>
        [...jobs.values()]
          .filter(job => job.worker_id === worker_id && job.gpu_ids === gpu_ids && job.status === status)
          .sort((a, b) => a.queue_position - b.queue_position || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
          .map(job => structuredClone(job)),
      update: async (id, data) => {
        writes.push({ id, data });
        await beforeUpdate(id, data);
        const updated = { ...jobs.get(id), ...data };
        jobs.set(id, updated);
        return structuredClone(updated);
      },
    },
  };
  const lease = loadSource('../src/server/processLease.ts', {
    os,
    crypto,
    'timers/promises': timers,
    './db': { db },
    './processBirth': { getProcessBirth: async () => 1000, processBirthMatches: (a, b) => a === b },
  });
  const coordination = loadSource('../src/server/queueCoordination.ts', {
    './db': { db },
    './processLease': lease,
    'timers/promises': timers,
  });
  const service = loadSource('../src/server/queueReorder.ts', {
    './db': { db },
    './queueCoordination': coordination,
    './processLease': lease,
    './remoteClient': {
      RemoteClientError,
      isLocalWorker: id => id === 'local',
      getRemoteWorker: async id => ({ id, name: 'Fixture worker' }),
      remoteJson: async (_worker, route, init) => {
        remoteCalls.push({ route, body: JSON.parse(init.body) });
        return remoteCommand(JSON.parse(init.body));
      },
    },
  });
  return {
    ...service,
    coordination,
    lease,
    jobs,
    writes,
    remoteCalls,
    pauseWrites: callback => {
      beforeUpdate = callback;
    },
    remoteCommand: callback => {
      remoteCommand = callback;
    },
    reorder: (ids, expected) => service.reorderQueueJobs('0', ids, workerID, undefined, { expectedJobIDs: expected }),
  };
}

test('two concurrent clients with one baseline cannot silently overwrite the winning reorder', async () => {
  const f = fixture(),
    entered = deferred(),
    release = deferred();
  f.pauseWrites(async id => {
    if (id === 'b') {
      entered.resolve();
      await release.promise;
    }
  });
  const first = f.reorder(['b', 'a', 'c'], ['a', 'b', 'c']);
  await entered.promise;
  const second = f.reorder(['c', 'a', 'b'], ['a', 'b', 'c']);
  const results = Promise.allSettled([first, second]);
  release.resolve();
  const [winner, stale] = await results;
  assert.equal(winner.status, 'fulfilled');
  assert.equal(stale.status, 'rejected', 'stale request must not be treated as a successful overwrite');
  assert.equal(stale.reason.status, 409);
  assert.equal(stale.reason.code, 'QUEUE_CHANGED');
  assert.deepEqual(
    f.writes.map(write => write.id),
    ['b', 'a', 'c'],
  );
});

test('reorder owns the exact local transition lease used by cron until its writes settle', async () => {
  const f = fixture(),
    entered = deferred(),
    release = deferred();
  f.pauseWrites(async () => {
    entered.resolve();
    await release.promise;
  });
  const operation = f.reorder(['b', 'a', 'c'], ['a', 'b', 'c']);
  await entered.promise;
  try {
    await assert.rejects(
      f.coordination.withLocalQueueTransition('0', async () => 'cron inspected partial positions', { wait: false }),
      error => error instanceof f.lease.LeaseBusyError,
    );
    assert.equal(await f.coordination.withLocalQueueTransition('1', async () => 'other lane'), 'other lane');
  } finally {
    release.resolve();
    await operation;
  }
  assert.equal(await f.coordination.withLocalQueueTransition('0', async () => 'released'), 'released');
});

test('a precondition covers the entire lane and rejects newly queued jobs before writing', async () => {
  const f = fixture();
  await assert.rejects(f.reorder(['b', 'a'], ['a', 'b']), error => error.status === 409);
  f.jobs.get('c').status = 'running';
  await assert.rejects(f.reorder(['b', 'a', 'c'], ['a', 'b', 'c']), error => error.status === 409);
  assert.equal(f.writes.length, 0);
});

test('malformed preconditions cannot opt out of concurrency protection or introduce different IDs', async () => {
  for (const expected of [null, 'a,b,c', [], ['a', 'a', 'c'], ['a', 'b', 2], ['a', 'b', 'other']]) {
    const f = fixture();
    await assert.rejects(f.reorder(['b', 'a', 'c'], expected), error => error.status === 400);
    assert.equal(f.writes.length, 0);
  }
});

test('legacy callers without a precondition retain partial reorder behavior', async () => {
  const f = fixture();
  await f.reorder(['b', 'a']);
  assert.deepEqual(
    f.writes.map(write => write.id),
    ['b', 'a'],
  );
});

test('remote reorders map desired and expected IDs and leave mirrors unchanged on worker conflict', async () => {
  const f = fixture('worker-1');
  f.remoteCommand(async () => {
    throw new RemoteClientError(409, '{"error":"private worker details"}');
  });
  await assert.rejects(f.reorder(['b', 'a', 'c'], ['a', 'b', 'c']), error => {
    assert.equal(error.status, 409);
    assert.equal(error.code, 'QUEUE_CHANGED');
    assert.doesNotMatch(error.message, /private worker details/);
    return true;
  });
  assert.deepEqual(f.remoteCalls[0].body, {
    worker_id: 'local',
    job_ids: ['remote-b', 'remote-a', 'remote-c'],
    expected_job_ids: ['remote-a', 'remote-b', 'remote-c'],
  });
  assert.equal(f.writes.length, 0);
});

test('remote preconditions are checked at the controller even for a legacy worker', async () => {
  const f = fixture('worker-1');
  await f.reorder(['b', 'a', 'c'], ['a', 'b', 'c']);
  await assert.rejects(f.reorder(['c', 'a', 'b'], ['a', 'b', 'c']), error => error.status === 409);
  assert.equal(f.remoteCalls.length, 1);
});

test('legacy remote payloads omit the optional precondition', async () => {
  const f = fixture('worker-1');
  await f.reorder(['b', 'a']);
  assert.deepEqual(f.remoteCalls[0].body, { worker_id: 'local', job_ids: ['remote-b', 'remote-a'] });
});

test('remote equal positions compare the authoritative worker ID order rather than mirror IDs', async () => {
  const f = fixture('worker-1');
  for (const [id, remote] of Object.entries({ a: 'remote-z', b: 'remote-a', c: 'remote-m' })) {
    Object.assign(f.jobs.get(id), { queue_position: 1000, remote_job_id: remote });
  }
  await f.reorder(['a', 'b', 'c'], ['b', 'c', 'a']);
  assert.deepEqual(f.remoteCalls[0].body.expected_job_ids, ['remote-a', 'remote-m', 'remote-z']);
});

test('a failed write cannot release the lease while another position update is outstanding', async () => {
  const f = fixture(),
    entered = deferred(),
    release = deferred();
  f.pauseWrites(async id => {
    if (id === 'a') throw new Error('Fixture write failed');
    if (id === 'b') {
      entered.resolve();
      await release.promise;
    }
  });
  const operation = assert.rejects(f.reorder(['b', 'a', 'c'], ['a', 'b', 'c']), /Fixture write failed/);
  await entered.promise;
  await timers.setImmediate();
  try {
    await assert.rejects(
      f.coordination.withLocalQueueTransition('0', async () => 'too early', { wait: false }),
      error => error instanceof f.lease.LeaseBusyError,
    );
  } finally {
    release.resolve();
    await operation;
  }
  assert.equal(await f.coordination.withLocalQueueTransition('0', async () => 'released'), 'released');
});

test('concurrent remote requests serialize through acknowledgement and mirror updates', async () => {
  const f = fixture('worker-1'),
    entered = deferred(),
    release = deferred();
  f.remoteCommand(async () => {
    entered.resolve();
    await release.promise;
  });
  const first = f.reorder(['b', 'a', 'c'], ['a', 'b', 'c']);
  await entered.promise;
  const second = f.reorder(['c', 'a', 'b'], ['a', 'b', 'c']);
  const results = Promise.allSettled([first, second]);
  release.resolve();
  const [winner, stale] = await results;
  assert.equal(winner.status, 'fulfilled');
  assert.equal(stale.status, 'rejected');
  assert.equal(stale.reason.status, 409);
  assert.equal(f.remoteCalls.length, 1);
  assert.deepEqual(
    f.writes.map(write => write.id),
    ['b', 'a', 'c'],
  );
});

test('an upgraded worker rejects a stale controller snapshot against its authoritative lane', async () => {
  const controller = fixture('worker-1'),
    worker = fixture();
  for (const [id, job] of [...worker.jobs]) {
    worker.jobs.delete(id);
    worker.jobs.set(`remote-${id}`, { ...job, id: `remote-${id}` });
  }
  await worker.reorder(['remote-c', 'remote-a', 'remote-b']);
  controller.remoteCommand(async body => {
    try {
      return await worker.reorder(body.job_ids, body.expected_job_ids);
    } catch (error) {
      throw new RemoteClientError(error.status);
    }
  });
  await assert.rejects(controller.reorder(['b', 'a', 'c'], ['a', 'b', 'c']), error => error.status === 409);
  assert.equal(controller.writes.length, 0);
  assert.equal(worker.writes.length, 3, 'worker must not apply the stale second order');
});

test('route passes the precondition and returns the conflict code and status', async () => {
  const f = fixture();
  const route = loadSource('../src/app/api/queue/[queueID]/reorder/route.ts', {
    '@/server/commandInput': { readJsonCommand: request => request.json(), withCommandBoundary: handler => handler },
    '@/utils/obsoleteWorkspaceGuard': { assertGlobalPayload: value => value },
    'next/server': { NextResponse: Response },
    '@/server/queueReorder': f,
  });
  const request = body =>
    new Request('http://localhost/api/queue/0/reorder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  const first = await route.POST(request({ job_ids: ['b', 'a', 'c'], expected_job_ids: ['a', 'b', 'c'] }), {
    params: Promise.resolve({ queueID: '0' }),
  });
  assert.equal(first.status, 200);
  const stale = await route.POST(request({ job_ids: ['c', 'a', 'b'], expected_job_ids: ['a', 'b', 'c'] }), {
    params: Promise.resolve({ queueID: '0' }),
  });
  assert.equal(stale.status, 409);
  assert.equal((await stale.json()).code, 'QUEUE_CHANGED');
  const invalid = await route.POST(request({ job_ids: ['a', 'b', 'c'], expected_job_ids: null }), {
    params: Promise.resolve({ queueID: '0' }),
  });
  assert.equal(invalid.status, 400);
});
