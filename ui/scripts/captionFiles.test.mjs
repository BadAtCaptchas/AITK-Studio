import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { test } from 'node:test';

const require = createRequire(import.meta.url);
const {
  captionSidecarPath,
  deleteCaptionSidecars,
  readCaptionSidecar,
  resolveCaptionWritePath,
} = require('../dist/src/server/captionFiles.js');

async function makeMediaFile() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-caption-sidecars-'));
  const mediaPath = path.join(root, '0001.jpg');
  await fs.writeFile(mediaPath, 'image bytes');
  return { root, mediaPath };
}

test('readCaptionSidecar prefers converted JSON over legacy text captions', async () => {
  const { mediaPath } = await makeMediaFile();
  await fs.writeFile(captionSidecarPath(mediaPath, '.txt'), 'legacy text caption');
  await fs.writeFile(captionSidecarPath(mediaPath, '.json'), '{"caption":"converted json caption"}');

  assert.equal(readCaptionSidecar(mediaPath), '{"caption":"converted json caption"}');
});

test('resolveCaptionWritePath keeps edits on the active JSON sidecar', async () => {
  const { mediaPath } = await makeMediaFile();
  await fs.writeFile(captionSidecarPath(mediaPath, '.txt'), 'legacy text caption');
  await fs.writeFile(captionSidecarPath(mediaPath, '.json'), '{"caption":"converted json caption"}');

  assert.equal(resolveCaptionWritePath(mediaPath, '{"caption":"edited"}'), captionSidecarPath(mediaPath, '.json'));
});

test('resolveCaptionWritePath chooses a sidecar extension for new captions', async () => {
  const { mediaPath } = await makeMediaFile();

  assert.equal(resolveCaptionWritePath(mediaPath, 'plain caption'), captionSidecarPath(mediaPath, '.txt'));
  assert.equal(resolveCaptionWritePath(mediaPath, '{"caption":"json"}'), captionSidecarPath(mediaPath, '.json'));
});

test('deleteCaptionSidecars removes JSON and text sidecars together', async () => {
  const { mediaPath } = await makeMediaFile();
  const txtPath = captionSidecarPath(mediaPath, '.txt');
  const jsonPath = captionSidecarPath(mediaPath, '.json');
  await fs.writeFile(txtPath, 'legacy text caption');
  await fs.writeFile(jsonPath, '{"caption":"converted json caption"}');

  deleteCaptionSidecars(mediaPath);

  assert.equal(fsSync.existsSync(txtPath), false);
  assert.equal(fsSync.existsSync(jsonPath), false);
});
