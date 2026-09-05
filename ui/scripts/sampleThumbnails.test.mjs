import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { resolveSampleThumbnail } from '../dist/src/server/sampleThumbnails.js';
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
  assert.equal(resolved?.path, thumbnail);
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
  await fs.symlink(outside, path.join(root, '.thumbs'), process.platform === 'win32' ? 'junction' : 'dir');

  assert.equal(await resolveSampleThumbnail(root, 'escape'), null);
  assert.equal(await resolveSampleThumbnail(root, '../escape'), null);
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
