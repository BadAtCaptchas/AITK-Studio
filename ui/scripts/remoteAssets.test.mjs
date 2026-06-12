import assert from 'node:assert/strict';
import test from 'node:test';

import { remoteAssetProxyPath, remoteSampleAssetPath } from '../dist/src/server/remoteAssets.js';

test('remote sample asset paths are constrained to the mirrored remote job samples route', () => {
  assert.equal(
    remoteSampleAssetPath('/api/jobs/remote-1/samples/sample%201.png', 'remote-1'),
    '/api/jobs/remote-1/samples/sample%201.png',
  );
  assert.equal(
    remoteSampleAssetPath('/api/jobs/other-job/samples/sample.png', 'remote-1'),
    null,
  );
  assert.equal(
    remoteSampleAssetPath('/api/jobs/remote-1/files/model.safetensors', 'remote-1'),
    null,
  );
});

test('remote asset proxy paths do not pass through traversals or protected worker APIs', () => {
  assert.equal(
    remoteAssetProxyPath('img', '/api/jobs/../settings', 'remote-1'),
    '/api/img/%2Fapi%2Fjobs%2F..%2Fsettings',
  );
  assert.equal(
    remoteAssetProxyPath('img', '/api/jobs/remote-1/samples/..', 'remote-1'),
    '/api/img/%2Fapi%2Fjobs%2Fremote-1%2Fsamples%2F..',
  );
  assert.equal(
    remoteAssetProxyPath('img', '/api/jobs/remote-1/samples/%2E%2E%2Fsettings', 'remote-1'),
    '/api/img/%2Fapi%2Fjobs%2Fremote-1%2Fsamples%2F%252E%252E%252Fsettings',
  );
});
