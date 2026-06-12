import assert from 'node:assert/strict';
import test from 'node:test';

import {
  REMOTE_JOB_MISSING_MESSAGE,
  RemoteClientError,
  isRemoteJobMissingError,
  remoteJobMissingUpdate,
} from '../dist/src/server/remoteClient.js';

test('detects remote worker job-not-found responses', () => {
  const missing = new RemoteClientError(
    'Remote worker L40 returned 404 for /api/jobs/job-1/files',
    404,
    '{"error":"Job not found"}',
  );

  assert.equal(isRemoteJobMissingError(missing), true);
  assert.equal(isRemoteJobMissingError(new RemoteClientError('Missing route', 404, 'Not found')), false);
  assert.equal(
    isRemoteJobMissingError(new RemoteClientError('Remote worker L40 returned 500', 500, '{"error":"Job not found"}')),
    false,
  );
});

test('remote missing update clears stale running mirror state', () => {
  const update = remoteJobMissingUpdate();

  assert.equal(update.remote_job_id, null);
  assert.equal(update.status, 'error');
  assert.equal(update.stop, false);
  assert.equal(update.return_to_queue, false);
  assert.equal(update.pid, null);
  assert.equal(update.speed_string, '');
  assert.equal(update.info, 'Remote job was deleted on the worker.');
  assert.equal(update.remote_error, REMOTE_JOB_MISSING_MESSAGE);
  assert.ok(update.remote_sync_at instanceof Date);
});
