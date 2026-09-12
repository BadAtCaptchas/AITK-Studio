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
