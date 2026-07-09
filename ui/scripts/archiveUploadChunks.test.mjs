import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  assembleArchiveUploadChunks,
  saveArchiveUploadChunk,
} from '../dist/src/server/archiveUploadChunks.js';

function chunkRequest(uploadID, chunkIndex, chunksTotal, fileBytes, body) {
  const url = new URL('http://localhost/api/archive');
  url.searchParams.set('uploadID', uploadID);
  url.searchParams.set('chunkIndex', String(chunkIndex));
  url.searchParams.set('chunksTotal', String(chunksTotal));
  url.searchParams.set('fileBytes', String(fileBytes));
  const request = new Request(url, {
    method: 'POST',
    headers: {
      'Content-Length': String(body.length),
      'Content-Type': 'application/octet-stream',
    },
    body,
  });
  Object.defineProperty(request, 'nextUrl', { value: url });
  return request;
}

test('archive chunks are assembled in order after bounded streaming writes', async () => {
  const uploadRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-archive-chunks-'));
  const uploadID = 'upload-test-1234';
  const outputPath = path.join(uploadRoot, 'assembled.zip');

  try {
    await saveArchiveUploadChunk(chunkRequest(uploadID, 0, 2, 6, Buffer.from('abc')), uploadRoot, {
      maxArchiveBytes: 100,
    });
    await saveArchiveUploadChunk(chunkRequest(uploadID, 1, 2, 6, Buffer.from('def')), uploadRoot, {
      maxArchiveBytes: 100,
    });
    await assembleArchiveUploadChunks(uploadRoot, uploadID, 2, outputPath, {
      maxBytes: 100,
      expectedBytes: 6,
    });

    assert.equal(await fs.readFile(outputPath, 'utf8'), 'abcdef');
  } finally {
    await fs.rm(uploadRoot, { recursive: true, force: true });
  }
});

test('archive chunk upload rejects a declared archive larger than its route limit', async () => {
  const uploadRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-archive-chunks-'));

  try {
    await assert.rejects(
      saveArchiveUploadChunk(
        chunkRequest('upload-test-5678', 0, 1, 101, Buffer.from('x')),
        uploadRoot,
        { maxArchiveBytes: 100 },
      ),
      error => error?.name === 'UploadTooLargeError' && error?.status === 413,
    );
  } finally {
    await fs.rm(uploadRoot, { recursive: true, force: true });
  }
});

test('archive assembly rejects a size declaration that does not match stored chunks', async () => {
  const uploadRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-archive-chunks-'));
  const uploadID = 'upload-test-9012';
  const outputPath = path.join(uploadRoot, 'assembled.zip');

  try {
    await saveArchiveUploadChunk(chunkRequest(uploadID, 0, 1, 3, Buffer.from('abc')), uploadRoot, {
      maxArchiveBytes: 100,
    });
    await assert.rejects(
      assembleArchiveUploadChunks(uploadRoot, uploadID, 1, outputPath, {
        maxBytes: 100,
        expectedBytes: 4,
      }),
      /Invalid archive upload fileBytes/,
    );
    await assert.rejects(fs.stat(outputPath), error => error?.code === 'ENOENT');
  } finally {
    await fs.rm(uploadRoot, { recursive: true, force: true });
  }
});
