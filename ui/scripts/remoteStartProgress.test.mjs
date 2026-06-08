import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createRemoteStartProgress,
  getRemoteStartProgress,
  hasActiveRemoteStartForJob,
  updateRemoteStartProgress,
} from '../dist/src/server/remoteStartProgress.js';

test('remote start progress stores updates and terminal failures', () => {
  const created = createRemoteStartProgress('job-1');

  assert.equal(created.status, 'queued');
  assert.equal(hasActiveRemoteStartForJob('job-1'), true);

  const uploading = updateRemoteStartProgress(created.startID, {
    status: 'uploading-dataset',
    message: 'Uploading dataset cats',
    percent: 42,
    datasetName: 'cats',
    bytesProcessed: 12,
    bytesTotal: 24,
  });

  assert.equal(uploading?.status, 'uploading-dataset');
  assert.equal(uploading?.datasetName, 'cats');
  assert.equal(uploading?.percent, 42);
  assert.equal(uploading?.bytesProcessed, 12);
  assert.equal(uploading?.bytesTotal, 24);

  const failed = updateRemoteStartProgress(created.startID, {
    status: 'failed',
    message: 'Remote start failed',
    percent: 150,
    error: 'network failed',
  });

  assert.equal(failed?.status, 'failed');
  assert.equal(failed?.percent, 100);
  assert.equal(failed?.error, 'network failed');
  assert.equal(hasActiveRemoteStartForJob('job-1'), false);
  assert.equal(getRemoteStartProgress(created.startID)?.message, 'Remote start failed');
});
