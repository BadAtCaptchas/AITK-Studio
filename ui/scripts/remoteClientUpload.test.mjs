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
  const bytes = Buffer.from('raw archive bytes');
  fs.writeFileSync(zipPath, bytes);

  const originalFetch = globalThis.fetch;
  const progress = [];
  let capturedUrl = null;
  let capturedHeaders = null;
  let capturedBody = null;

  globalThis.fetch = async (url, init = {}) => {
    capturedUrl = String(url);
    capturedHeaders = new Headers(init.headers);
    capturedBody = await readRequestBody(init.body);
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
    const result = await uploadDatasetArchiveToWorker(makeWorker(), zipPath, 'locked', item => progress.push(item));

    assert.equal(result.path, '/remote/datasets/locked');
    assert.equal(new URL(capturedUrl).pathname, '/api/datasets/import-archive');
    assert.equal(new URL(capturedUrl).searchParams.get('preferredName'), 'locked');
    assert.equal(capturedHeaders.get('content-type'), 'application/zip');
    assert.equal(capturedHeaders.get('content-length'), String(bytes.length));
    assert.deepEqual(capturedBody, bytes);
    assert.equal(progress[0].loaded, 0);
    assert.equal(progress.at(-1).loaded, bytes.length);
    assert.equal(progress.at(-1).total, bytes.length);
  } finally {
    globalThis.fetch = originalFetch;
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});

test('job bundle upload sends gpu ids in the raw upload request', async () => {
  const tempRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-remote-client-upload-'));
  const zipPath = path.join(tempRoot, 'job.zip');
  fs.writeFileSync(zipPath, 'job archive bytes');

  const originalFetch = globalThis.fetch;
  let capturedUrl = null;

  globalThis.fetch = async (url, init = {}) => {
    capturedUrl = String(url);
    await readRequestBody(init.body);
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
    assert.equal(new URL(capturedUrl).pathname, '/api/jobs/import');
    assert.equal(new URL(capturedUrl).searchParams.get('gpu_ids'), '0,1');
  } finally {
    globalThis.fetch = originalFetch;
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});
