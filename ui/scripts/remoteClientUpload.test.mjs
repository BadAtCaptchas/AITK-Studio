import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import test from 'node:test';

import {
  uploadBundleToWorker,
  uploadDatasetArchiveToWorker,
} from '../dist/src/server/remoteClient.js';

function makeWorker() {
  return {
    id: 'worker-1',
    name: 'Remote One',
    base_url: 'https://worker.example',
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

async function readRequestBody(body) {
  if (!body) return Buffer.alloc(0);
  const chunks = [];
  const stream = typeof body.getReader === 'function' ? Readable.fromWeb(body) : body;
  for await (const chunk of stream) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

test('dataset archive upload sends a raw zip body to the worker', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-remote-client-upload-'));
  const zipPath = path.join(tempRoot, 'dataset.zip');
  const bytes = Buffer.alloc(300 * 1024);
  for (let index = 0; index < bytes.length; index += 1) {
    bytes[index] = index % 251;
  }
  fs.writeFileSync(zipPath, bytes);

  const originalFetch = globalThis.fetch;
  const originalChunkMB = process.env.AITK_REMOTE_UPLOAD_CHUNK_MB;
  const progress = [];
  const requests = [];

  globalThis.fetch = async (url, init = {}) => {
    const requestUrl = new URL(String(url));
    const headers = new Headers(init.headers);
    const body = await readRequestBody(init.body);
    requests.push({ url: requestUrl, headers, body });
    if (requestUrl.searchParams.get('aitk_upload') === 'chunk') {
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    return new Response(
      JSON.stringify({
        dataset: { name: 'locked', encrypted: true, path: '/remote/datasets/locked' },
        path: '/remote/datasets/locked',
        renamed: false,
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    );
  };

  try {
    process.env.AITK_REMOTE_UPLOAD_CHUNK_MB = '0.25';
    const result = await uploadDatasetArchiveToWorker(makeWorker(), zipPath, 'locked', item => progress.push(item));

    assert.equal(result.path, '/remote/datasets/locked');
    assert.equal(requests.length, 3);
    assert.equal(requests[0].url.pathname, '/api/datasets/import-archive');
    assert.equal(requests[0].url.searchParams.get('preferredName'), 'locked');
    assert.equal(requests[0].url.searchParams.get('aitk_upload'), 'chunk');
    assert.equal(requests[1].url.searchParams.get('aitk_upload'), 'chunk');
    assert.equal(requests[2].url.searchParams.get('aitk_upload'), 'complete');
    assert.equal(requests[0].headers.get('content-type'), 'application/octet-stream');
    assert.deepEqual(Buffer.concat([requests[0].body, requests[1].body]), bytes);
    assert.equal(progress[0].loaded, 0);
    assert.equal(progress.at(-1).loaded, bytes.length);
    assert.equal(progress.at(-1).total, bytes.length);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalChunkMB == null) {
      delete process.env.AITK_REMOTE_UPLOAD_CHUNK_MB;
    } else {
      process.env.AITK_REMOTE_UPLOAD_CHUNK_MB = originalChunkMB;
    }
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('job bundle upload sends gpu ids in the raw upload request', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-remote-client-upload-'));
  const zipPath = path.join(tempRoot, 'job.zip');
  fs.writeFileSync(zipPath, 'job archive bytes');

  const originalFetch = globalThis.fetch;
  const requests = [];

  globalThis.fetch = async (url, init = {}) => {
    const requestUrl = new URL(String(url));
    requests.push(requestUrl);
    await readRequestBody(init.body);
    if (requestUrl.searchParams.get('aitk_upload') === 'chunk') {
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    return new Response(
      JSON.stringify({
        job: { id: 'remote-job-1' },
        warnings: [],
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    );
  };

  try {
    await uploadBundleToWorker(makeWorker(), zipPath, '0,1');
    assert.equal(requests.at(-1).pathname, '/api/jobs/import');
    assert.equal(requests.at(-1).searchParams.get('gpu_ids'), '0,1');
    assert.equal(requests.at(-1).searchParams.get('aitk_upload'), 'complete');
  } finally {
    globalThis.fetch = originalFetch;
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});
