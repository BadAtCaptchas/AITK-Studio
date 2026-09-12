import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
import { createWriteStream } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import archiver from 'archiver';
import { installMemoryRuntime } from './memoryRuntimeFixture.mjs';
import {
  receiveManifestChunk,
  assembleManifestChunks,
  getUploadManifest,
  ensureUploadDirectory,
} from '../dist/src/server/archiveUploadManifest.js';
import { extractZipSafely } from '../dist/src/server/safeArchive.js';
const request = value => new Request('http://localhost/upload', { method: 'POST', body: value });

test('upload staging rejects symlinks at the root, ancestors, upload, and chunks directories', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-staging-paths-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const outside = path.join(root, 'outside');
  await fs.mkdir(outside);
  for (const location of ['root', 'ancestor', 'upload', 'chunks']) {
    const staging = path.join(root, location);
    const id = 'path-test-upload';
    const link = location === 'upload' ? path.join(staging, id)
      : location === 'chunks' ? path.join(staging, id, 'chunks') : staging;
    await fs.mkdir(path.dirname(link), { recursive: true });
    await fs.symlink(outside, link, process.platform === 'win32' ? 'junction' : 'dir');
    await assert.rejects(
      ensureUploadDirectory(location === 'ancestor' ? path.join(staging, 'nested') : staging, id, true),
      /canonical private directories/,
      location,
    );
  }
});
test('chunk retries are identical, aggregate bytes are capped, and finalization freezes input', async () => {
  const fixture = installMemoryRuntime(),
    root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-quota-test-'));
  try {
    const input = {
      uploadID: 'quota-test-upload',
      chunkIndex: 0,
      chunksTotal: 2,
      expectedBytes: 6,
      limit: 6,
      purpose: 'test',
    };
    await receiveManifestChunk(request('abc'), root, input);
    await receiveManifestChunk(request('abc'), root, input);
    await assert.rejects(receiveManifestChunk(request('xyz'), root, input), /different chunk/);
    await assert.rejects(
      receiveManifestChunk(request('defg'), root, { ...input, chunkIndex: 1 }),
      /large|limit|exceed/i,
    );
    await receiveManifestChunk(request('def'), root, { ...input, chunkIndex: 1 });
    const output = path.join(root, input.uploadID, 'upload.zip');
    await assembleManifestChunks(root, input.uploadID, 2, output, { expectedBytes: 6 });
    assert.equal(await fs.readFile(output, 'utf8'), 'abcdef');
    assert.equal((await getUploadManifest(root, input.uploadID)).reservedBytes, 6);
    await assert.rejects(receiveManifestChunk(request('abc'), root, input), /frozen/);
  } finally {
    fixture.restore();
    await fs.rm(root, { recursive: true, force: true });
  }
});
async function zip(filename, entries) {
  const archive = archiver('zip');
  const output = createWriteStream(filename);
  archive.pipe(output);
  const done = new Promise((resolve, reject) => {
    output.on('close', resolve);
    output.on('error', reject);
    archive.on('error', reject);
  });
  for (const [name, data] of entries) archive.append(data, { name });
  await archive.finalize();
  await done;
}
test('archive extraction limits expanded bytes and refuses Windows-special names and duplicate files', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-extract-test-'));
  try {
    for (const [index, entries] of [
      [['CON.txt', 'bad']],
      [['ok', 'a'.repeat(2048)]],
      [
        ['same.txt', 'a'],
        ['same.txt', 'b'],
      ],
    ].entries()) {
      const source = path.join(root, index + '.zip');
      await zip(source, entries);
      await assert.rejects(extractZipSafely(source, path.join(root, 'extract-' + index), { maxExpandedBytes: 1024 }));
    }
    const source = path.join(root, 'valid.zip');
    await zip(source, [['nested/valid.txt', 'valid']]);
    await extractZipSafely(source, path.join(root, 'valid'));
    assert.equal(await fs.readFile(path.join(root, 'valid/nested/valid.txt'), 'utf8'), 'valid');
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
