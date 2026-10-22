import assert from 'node:assert/strict';
import test from 'node:test';
import { installMemoryRuntime } from './memoryRuntimeFixture.mjs';
import {
  enqueueOperation,
  getOperation,
  updateOperation,
  runPendingOperations,
  checkpointOperation,
  beginOperationCommit,
} from '../dist/src/server/operations.js';
import { withProcessLease } from '../dist/src/server/processLease.js';

test('free operation slots take shared work and admit new requests while another operation is blocked', async () => {
  const fixture = installMemoryRuntime();
  let release;
  const held = new Promise(resolve => { release = resolve; });
  let firstStarted;
  const started = new Promise(resolve => { firstStarted = resolve; });
  const executed = [];
  const input = jobID => ({ kind: 'remote-start', jobID, configHash: 'same', durableKeys: false, needsEphemeralKeys: false });
  let running;
  try {
    const first = await enqueueOperation(input('held'));
    running = runPendingOperations(async operation => {
      executed.push(operation.input.jobID);
      if (operation.id === first.id) { firstStarted(); await held; }
      return {};
    });
    await started;
    for (const name of ['second', 'third', 'fourth']) await enqueueOperation(input(name));
    const deadline = Date.now() + 5000;
    while (executed.length < 4 && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 20));
    assert.equal(executed.length, 4, 'new work must use the available slot before the held job completes');
  } finally { release(); await running; fixture.restore(); }
});

test('a held operation does not starve retry deadlines or cancellation in the free slot', async () => {
  const fixture = installMemoryRuntime();
  let release, signalStarted;
  const held = new Promise(resolve => { release = resolve; });
  const started = new Promise(resolve => { signalStarted = resolve; });
  const input = jobID => ({ kind: 'remote-start', jobID, configHash: 'same', durableKeys: false, needsEphemeralKeys: false });
  let running, attempts = 0, active = 0, maxActive = 0;
  try {
    const first = await enqueueOperation(input('held'));
    running = runPendingOperations(async operation => {
      active++; maxActive = Math.max(maxActive, active);
      try {
        if (operation.id === first.id) { signalStarted(); await held; }
        else if (operation.input.jobID === 'retry' && attempts++ === 0) throw new Error('transient');
        return {};
      } finally { active--; }
    });
    await started;
    const retry = await enqueueOperation(input('retry'));
    const canceled = await enqueueOperation(input('cancel'));
    await updateOperation(canceled.id, { nextAttemptAt: Date.now() + 60_000 });
    await new Promise(resolve => setTimeout(resolve, 1100));
    await updateOperation(canceled.id, { cancelRequested: true });
    const deadline = Date.now() + 5000;
    while (Date.now() < deadline && ((await getOperation(retry.id)).state !== 'completed' || (await getOperation(canceled.id)).state !== 'canceled'))
      await new Promise(resolve => setTimeout(resolve, 20));
    assert.equal((await getOperation(retry.id)).state, 'completed');
    assert.equal((await getOperation(canceled.id)).state, 'canceled');
    assert.equal(maxActive, 2);
  } finally { release(); await running; fixture.restore(); }
});

test('acceptance attaches retries, repairs pending links, and does not repeat completed work', async () => {
  const fixture = installMemoryRuntime();
  let effects = 0;
  try {
    const input = {
      kind: 'remote-start',
      jobID: 'one',
      configHash: 'hash',
      durableKeys: false,
      needsEphemeralKeys: false,
    };
    const first = await enqueueOperation(input, 'repeatable-key');
    assert.equal((await enqueueOperation(input, 'another-key')).id, first.id);
    fixture.records.delete('operation-pending:' + first.id);
    await runPendingOperations(async () => {
      effects++;
      return { remoteJobID: 'remote-one' };
    });
    assert.equal((await getOperation(first.id)).state, 'completed');
    await runPendingOperations(async () => {
      effects++;
    });
    assert.equal(effects, 1);
    assert.equal((await enqueueOperation(input, 'repeatable-key')).id, first.id);
  } finally {
    fixture.restore();
  }
});


test('decryption credentials remain in memory and restart requires unlocking again', async () => {
  const { rememberOperationKeys, readOperationKeys } = await import('../dist/src/server/operationKeys.js');
  const fixture = installMemoryRuntime();
  try {
    const operation = await enqueueOperation({ kind: 'dataset-decrypt', root: '/test', sourceName: 'old', outputName: 'plain' });
    const secret = Buffer.alloc(32, 17).toString('base64');
    rememberOperationKeys(operation.id, [{ datasetPath: 'old', keyB64: secret }]);
    assert.equal((await readOperationKeys(operation.id))[0].keyB64, secret);
    assert.equal(JSON.stringify([...fixture.records.values()]).includes(secret), false);
    globalThis.__aitkOperationKeys.clear();
    assert.equal(await readOperationKeys(operation.id), null);
    await runPendingOperations(async () => { throw Object.assign(new Error('Unlock again'), { code: 'OPERATION_KEYS_REQUIRED' }); });
    assert.equal((await getOperation(operation.id)).state, 'needs-keys');
    rememberOperationKeys(operation.id, [{ datasetPath: 'old', keyB64: secret }]);
    await updateOperation(operation.id, { state: 'queued', nextAttemptAt: 0 });
    await runPendingOperations(async () => ({ datasetName: 'plain' }));
    assert.equal((await getOperation(operation.id)).state, 'completed');
    assert.equal(await readOperationKeys(operation.id), null);
  } finally { fixture.restore(); }
});
test('cancellation and missing keys stop before the next side effect', async () => {
  const fixture = installMemoryRuntime();
  try {
    const one = await enqueueOperation({
      kind: 'remote-start',
      jobID: 'cancel',
      configHash: 'hash',
      durableKeys: false,
      needsEphemeralKeys: false,
    });
    await updateOperation(one.id, { cancelRequested: true });
    await runPendingOperations(async () => assert.fail('canceled operation executed'));
    assert.equal((await getOperation(one.id)).state, 'canceled');
    const two = await enqueueOperation({
      kind: 'remote-start',
      jobID: 'keys',
      configHash: 'hash',
      durableKeys: false,
      needsEphemeralKeys: true,
    });
    await runPendingOperations(async () => {
      throw Object.assign(new Error('missing'), { code: 'OPERATION_KEYS_REQUIRED' });
    });
    assert.equal((await getOperation(two.id)).state, 'needs-keys');
    await updateOperation(two.id, { cancelRequested: true });
    await assert.rejects(checkpointOperation(two.id, 'upload'), /canceled/);
  } finally {
    fixture.restore();
  }
});
test('heartbeats never allow another claimant to steal a live process lease', async () => {
  const fixture = installMemoryRuntime();
  try {
    await withProcessLease('owned', async () => {
      const row = fixture.records.get('lease:owned');
      row.value.heartbeat = 0;
      await assert.rejects(
        withProcessLease('owned', async () => assert.fail('stolen')),
        /already owned/,
      );
    });
    await withProcessLease('owned', async lease => lease.assertOwned());
  } finally {
    fixture.restore();
  }
});

test('cancellation races with commit atomically and input conflicts cannot replace accepted work', async () => {
  const fixture = installMemoryRuntime();
  try {
    const input = {
      kind: 'remote-start',
      jobID: 'commit',
      configHash: 'same',
      durableKeys: false,
      needsEphemeralKeys: false,
    };
    const operation = await enqueueOperation(input);
    await assert.rejects(enqueueOperation({ ...input, configHash: 'changed' }), /different input/);
    await beginOperationCommit(operation.id, 'publishing');
    await assert.rejects(updateOperation(operation.id, { cancelRequested: true }), error => error.status === 409);
    await checkpointOperation(operation.id, 'published');
    const cancelFirst = await enqueueOperation({ ...input, jobID: 'cancel-first' });
    await updateOperation(cancelFirst.id, { cancelRequested: true });
    await assert.rejects(
      beginOperationCommit(cancelFirst.id, 'publishing'),
      error => error.code === 'OPERATION_CANCELED',
    );
    assert.equal((await getOperation(cancelFirst.id)).checkpoint.commitStarted, undefined);
  } finally {
    fixture.restore();
  }
});
