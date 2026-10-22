import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import test from 'node:test';
import { createHash } from 'node:crypto';

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

test('interrupted uploads verify acknowledged chunks and send only missing chunks', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-resume-upload-'));
  const zipPath = path.join(root, 'dataset.zip');
  const bytes = Buffer.alloc(600 * 1024, 42);
  fs.writeFileSync(zipPath, bytes);
  const originalFetch = globalThis.fetch, originalChunkMB = process.env.AITK_REMOTE_UPLOAD_CHUNK_MB;
  const requests = [], progress = [];
  let mismatched = false;
  globalThis.fetch = async (url, init = {}) => {
    const parsed = new URL(url), mode = parsed.searchParams.get('aitk_upload');
    if (mode === 'status') return Response.json({ uploadID: 'resume-upload', state: 'receiving', chunksTotal: 3,
      expectedBytes: bytes.length, receivedBytes: 256 * 1024, chunks: { 0: { bytes: 256 * 1024, sha256: mismatched ? '0'.repeat(64) : createHash('sha256').update(bytes.subarray(0, 256 * 1024)).digest('hex') } } });
    requests.push({ mode, index: parsed.searchParams.get('chunkIndex') });
    await readRequestBody(init.body);
    return Response.json(mode === 'complete' ? { path: '/worker/dataset', dataset: { name: 'data', encrypted: false }, renamed: false } : { ok: true });
  };
  try {
    process.env.AITK_REMOTE_UPLOAD_CHUNK_MB = '0.25';
    await uploadDatasetArchiveToWorker(makeWorker(), zipPath, 'data', item => progress.push(item), 'resume-upload');
    assert.deepEqual(requests.map(item => item.index), ['1', '2', null]);
    assert.equal(progress[0].loaded, 256 * 1024);
    assert.equal(progress.at(-1).loaded, bytes.length);
    mismatched = true;
    requests.length = 0;
    await assert.rejects(uploadDatasetArchiveToWorker(makeWorker(), zipPath, 'data', undefined, 'resume-upload'), /source changed/);
    assert.equal(requests.length, 0, 'conflicting source is rejected before writing more chunks');
  } finally {
    globalThis.fetch = originalFetch;
    if (originalChunkMB === undefined) delete process.env.AITK_REMOTE_UPLOAD_CHUNK_MB; else process.env.AITK_REMOTE_UPLOAD_CHUNK_MB = originalChunkMB;
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('resume rejects mismatched upload identities and malformed completed results', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-upload-status-'));
  const file = path.join(root, 'dataset.zip');
  fs.writeFileSync(file, 'zip');
  const originalFetch = globalThis.fetch;
  let payload = { uploadID: 'different-upload', status: 'completed', error: null, result: { path: '/other' } };
  let requests = 0;
  globalThis.fetch = async (_url, init) => { assert.equal(init.method || 'GET', 'GET'); requests++; return Response.json(payload); };
  try {
    await assert.rejects(uploadDatasetArchiveToWorker(makeWorker(), file, 'data', undefined, 'expected-upload'), /different archive upload ID/);
    payload = { ...payload, uploadID: 'expected-upload', result: null };
    await assert.rejects(uploadDatasetArchiveToWorker(makeWorker(), file, 'data', undefined, 'expected-upload'), /invalid imported dataset/);
    payload = { uploadID: 'expected-upload', status: 'importing', error: null, result: null };
    globalThis.fetch = async () => Response.json(requests++ === 2 ? payload : { ...payload, status: 'surprise' });
    await assert.rejects(uploadDatasetArchiveToWorker(makeWorker(), file, 'data', undefined, 'expected-upload'), /invalid archive import status/);
  } finally { globalThis.fetch = originalFetch; fs.rmSync(root, { recursive: true, force: true }); }
});

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
    if (requestUrl.searchParams.get('aitk_upload') === 'complete') {
      return new Response(
        JSON.stringify({
          uploadID: requestUrl.searchParams.get('uploadID'),
          status: 'importing',
          result: null,
          error: null,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      );
    }
    return new Response(
      JSON.stringify({
        uploadID: requestUrl.searchParams.get('uploadID'),
        status: 'completed',
        result: {
          dataset: { name: 'locked', encrypted: true, path: '/remote/datasets/locked' },
          path: '/remote/datasets/locked',
          renamed: false,
        },
        error: null,
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    );
  };

  try {
    process.env.AITK_REMOTE_UPLOAD_CHUNK_MB = '0.25';
    const result = await uploadDatasetArchiveToWorker(makeWorker(), zipPath, 'locked', item => progress.push(item));

    assert.equal(result.path, '/remote/datasets/locked');
    assert.equal(requests.length, 4);
    assert.equal(requests[0].url.pathname, '/api/datasets/import-archive');
    assert.equal(requests[0].url.searchParams.get('preferredName'), 'locked');
    assert.equal(requests[0].url.searchParams.get('aitk_upload'), 'chunk');
    assert.equal(requests[1].url.searchParams.get('aitk_upload'), 'chunk');
    assert.equal(requests[0].url.searchParams.get('fileBytes'), String(bytes.length));
    assert.equal(requests[1].url.searchParams.get('fileBytes'), String(bytes.length));
    assert.equal(requests[2].url.searchParams.get('aitk_upload'), 'complete');
    assert.equal(requests[2].url.searchParams.get('fileBytes'), String(bytes.length));
    assert.equal(requests[2].url.searchParams.get('background'), '1');
    assert.equal(requests[3].url.searchParams.get('aitk_upload'), 'status');
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
        job: { id: 'remote-job-1', name: 'Imported job', gpu_ids: '0,1' },
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
    assert.equal(requests[0].searchParams.get('fileBytes'), String(fs.statSync(zipPath).size));
    assert.equal(requests.at(-1).searchParams.get('fileBytes'), String(fs.statSync(zipPath).size));
  } finally {
    globalThis.fetch = originalFetch;
    fs.rmSync(tempRoot, { recursive: true, force: true });
  }
});
