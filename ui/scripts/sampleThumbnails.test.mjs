import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { resolveSampleThumbnail, removeSampleThumbnails } from '../dist/src/server/sampleThumbnails.js';
import { getSampleThumbnailUrl } from '../dist/src/utils/media.js';

async function makeRoot() {
  return fs.mkdtemp(path.join(os.tmpdir(), 'aitk-sample-thumbnails-'));
}

test('sample thumbnails resolve inside the authorized root and use jpeg content type', async t => {
  const root = await makeRoot();
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const thumbs = path.join(root, '.thumbs');
  await fs.mkdir(thumbs);
  const thumbnail = path.join(thumbs, 'sample.mp4.jpg');
  await fs.writeFile(thumbnail, 'jpeg');

  const resolved = await resolveSampleThumbnail(root, 'sample.mp4');
  assert.equal(resolved?.path, await fs.realpath(thumbnail));
  assert.equal(resolved?.contentType, 'image/jpeg');
  assert.equal(resolved?.stat.isFile(), true);
  assert.equal(await resolveSampleThumbnail(root, 'older-sample.mp4'), null);
});

test('sample thumbnail resolution rejects traversal and symlink escapes', async t => {
  const root = await makeRoot();
  const outside = await makeRoot();
  t.after(() =>
    Promise.all([fs.rm(root, { recursive: true, force: true }), fs.rm(outside, { recursive: true, force: true })]),
  );
  await fs.writeFile(path.join(outside, 'escape.jpg'), 'jpeg');
  await fs.writeFile(path.join(outside, 'escape.png'), 'png');
  await fs.symlink(outside, path.join(root, '.thumbs'), process.platform === 'win32' ? 'junction' : 'dir');

  assert.equal(await resolveSampleThumbnail(root, 'escape'), null);
  assert.equal(await resolveSampleThumbnail(root, '../escape'), null);
  assert.equal(await resolveSampleThumbnail(root, '..\\escape'), null);
  await removeSampleThumbnails(root, 'escape');
  await removeSampleThumbnails(root, '../escape');
  assert.equal(await fs.readFile(path.join(outside, 'escape.png'), 'utf8'), 'png');
  assert.equal(await fs.readFile(path.join(outside, 'escape.jpg'), 'utf8'), 'jpeg');
});

test('PNG thumbnails take precedence with the correct content type and fall back to JPEG', async t => {
  const root = await makeRoot();
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const thumbs = path.join(root, '.thumbs');
  await fs.mkdir(thumbs);
  const png = path.join(thumbs, 'sample.png.png');
  const jpg = path.join(thumbs, 'sample.png.jpg');
  await fs.writeFile(png, 'png');
  await fs.writeFile(jpg, 'jpeg');

  const resolved = await resolveSampleThumbnail(root, 'sample.png');
  assert.equal(resolved?.path, await fs.realpath(png));
  assert.equal(resolved?.contentType, 'image/png');
  assert.equal(resolved?.stat.size, 3);
  await fs.unlink(png);
  assert.equal((await resolveSampleThumbnail(root, 'sample.png'))?.contentType, 'image/jpeg');
  await fs.mkdir(png);
  assert.equal((await resolveSampleThumbnail(root, 'sample.png'))?.path, await fs.realpath(jpg));
});

test('sample thumbnail deletion removes both formats and preserves other samples', async t => {
  const root = await makeRoot();
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const thumbs = path.join(root, '.thumbs');
  await fs.mkdir(thumbs);
  await fs.writeFile(path.join(thumbs, 'sample.png.png'), 'png');
  await fs.writeFile(path.join(thumbs, 'sample.png.jpg'), 'jpeg');
  await fs.writeFile(path.join(thumbs, 'other.png.png'), 'other');

  await removeSampleThumbnails(root, 'sample.png');
  assert.deepEqual(await fs.readdir(thumbs), ['other.png.png']);
  assert.equal(await resolveSampleThumbnail(root, 'sample.png'), null);
  await removeSampleThumbnails(root, 'sample.png');
  await removeSampleThumbnails(root, '../other.png');
  assert.equal(await fs.readFile(path.join(thumbs, 'other.png.png'), 'utf8'), 'other');
});

test('unsafe PNG candidates are skipped without serving or deleting outside the thumbnail root', async t => {
  const root = await makeRoot();
  const outside = await makeRoot();
  t.after(() => Promise.all([root, outside].map(folder => fs.rm(folder, { recursive: true, force: true }))));
  const thumbs = path.join(root, '.thumbs');
  await fs.mkdir(thumbs);
  await fs.writeFile(path.join(outside, 'keep.png'), 'outside');
  await fs.symlink(outside, path.join(thumbs, 'sample.png.png'), process.platform === 'win32' ? 'junction' : 'dir');
  await fs.writeFile(path.join(thumbs, 'sample.png.jpg'), 'jpeg');

  assert.equal((await resolveSampleThumbnail(root, 'sample.png'))?.contentType, 'image/jpeg');
  await removeSampleThumbnails(root, 'sample.png');
  assert.equal(await resolveSampleThumbnail(root, 'sample.png'), null);
  assert.equal(await fs.readFile(path.join(outside, 'keep.png'), 'utf8'), 'outside');
});

test('sample thumbnail URLs preserve existing remote proxy parameters', () => {
  assert.equal(
    getSampleThumbnailUrl('/api/jobs/job-1/samples/sample.png'),
    '/api/jobs/job-1/samples/sample.png?thumb=1',
  );
  assert.equal(
    getSampleThumbnailUrl('remote://job-1/img/%2Fapi%2Fjobs%2Fremote-1%2Fsamples%2Fsample.mp4/sample.mp4'),
    '/api/remote-assets?job_id=job-1&type=img&path=%2Fapi%2Fjobs%2Fremote-1%2Fsamples%2Fsample.mp4&thumb=1',
  );
});
