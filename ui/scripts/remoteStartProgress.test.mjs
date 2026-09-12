import assert from 'node:assert/strict';
import test from 'node:test';
import { installMemoryRuntime } from './memoryRuntimeFixture.mjs';
import { enqueueOperation, updateOperation } from '../dist/src/server/operations.js';
import { getRemoteStartProgress, hasActiveRemoteStartForJob } from '../dist/src/server/remoteStartProgress.js';
test('remote progress is persisted, terminal, and exposes missing-key recovery', async () => {
 const fixture = installMemoryRuntime();
 try {
  const operation = await enqueueOperation({ kind: 'remote-start', jobID: 'job-1', configHash: 'hash', durableKeys: false, needsEphemeralKeys: true });
  await updateOperation(operation.id, { progress: { status: 'uploading-dataset', percent: 42, datasetName: 'cats' } });
  assert.equal((await getRemoteStartProgress(operation.id)).percent, 42);
  assert.equal(await hasActiveRemoteStartForJob('job-1'), true);
  await updateOperation(operation.id, { state: 'needs-keys' });
  assert.equal((await getRemoteStartProgress(operation.id)).operationState, 'needs-keys');
  await updateOperation(operation.id, { state: 'failed', error: 'network failed' });
  await updateOperation(operation.id, { state: 'running' });
  assert.equal((await getRemoteStartProgress(operation.id)).error, 'network failed');
  assert.equal(await hasActiveRemoteStartForJob('job-1'), false);
 } finally { fixture.restore(); }
});
