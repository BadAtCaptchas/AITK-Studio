import assert from 'node:assert/strict';
import test from 'node:test';

import {
  RemoteClientError,
  remoteJson,
  withoutRemoteRedirects,
} from '../dist/src/server/remoteClient.js';

function makeWorker(baseUrl = 'https://worker.example') {
  return {
    id: 'worker-1',
    name: 'Remote One',
    base_url: baseUrl,
    api_token: 'token',
    enabled: true,
    last_status: 'ready',
    last_error: null,
    last_checked_at: null,
    capabilities: '{}',
    gpus: '[]',
    created_at: new Date(),
    updated_at: new Date(),
  };
}

test('remoteJson can reject a redirect before a secret-bearing POST is replayed', async () => {
  const originalFetch = globalThis.fetch;
  const calls = [];

  globalThis.fetch = async (url, init = {}) => {
    calls.push({
      url: String(url),
      redirect: init.redirect,
      authorization: new Headers(init.headers).get('authorization'),
    });
    return new Response('', {
      status: 307,
      headers: {
        Location: 'http://worker.example/api/datasets/combine',
      },
    });
  };

  try {
    await assert.rejects(
      () =>
        remoteJson(
          makeWorker(),
          '/api/datasets/combine',
          withoutRemoteRedirects({
            method: 'POST',
            body: JSON.stringify({
              encryptedDatasetKeys: [{ datasetName: 'locked', keyB64: 'secret-key' }],
              outputKeyB64: 'secret-output-key',
            }),
          }),
        ),
      error => {
        assert.ok(error instanceof RemoteClientError);
        assert.equal(error.status, 307);
        return true;
      },
    );

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, 'https://worker.example/api/datasets/combine');
    assert.equal(calls[0].redirect, 'manual');
    assert.equal(calls[0].authorization, 'Bearer token');
  } finally {
    globalThis.fetch = originalFetch;
  }
});
