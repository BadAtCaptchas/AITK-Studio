import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const {
  decodeRemoteCaptionBulkPaths,
  mapRemoteCaptionBulkResult,
  performPlainDatasetCaptionBulkAction,
} = require('../dist/src/server/datasetCaptionBulk.js');
const { captionSidecarPath } = require('../dist/src/server/captionFiles.js');
const { makeRemoteDatasetAssetRef } = require('../dist/src/utils/remoteDatasetRefs.js');

async function makeDataset(files) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-caption-bulk-'));
  const dataset = path.join(root, 'source');
  await fs.mkdir(dataset);
  for (const [name, content] of Object.entries(files)) {
    const filePath = path.join(dataset, name);
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, content);
  }
  return { root, dataset };
}

test('remove words updates matching captions only and reports counts', async () => {
  const { root, dataset } = await makeDataset({
    'a.jpg': 'image',
    'a.txt': 'cat dog catfish cat',
    'b.jpg': 'image',
    'b.txt': 'dog only',
  });

  const result = await performPlainDatasetCaptionBulkAction(root, {
    datasetName: 'source',
    action: 'remove_words',
    imgPaths: [path.join(dataset, 'a.jpg'), path.join(dataset, 'b.jpg')],
    query: 'cat',
    matchMode: 'whole-word',
  });

  assert.equal(result.found, 1);
  assert.equal(result.affected, 1);
  assert.equal(result.removedWords, 2);
  assert.equal(await fs.readFile(path.join(dataset, 'a.txt'), 'utf8'), 'dog catfish');
  assert.equal(await fs.readFile(path.join(dataset, 'b.txt'), 'utf8'), 'dog only');
});

test('delete removes media and all caption sidecars for matching items', async () => {
  const { root, dataset } = await makeDataset({
    'a.jpg': 'image',
    'a.txt': 'cat',
    'a.json': '{"caption":"cat"}',
    'b.jpg': 'image',
    'b.txt': 'dog',
  });

  const result = await performPlainDatasetCaptionBulkAction(root, {
    datasetName: 'source',
    action: 'delete',
    imgPaths: [path.join(dataset, 'a.jpg'), path.join(dataset, 'b.jpg')],
    query: 'cat',
  });

  assert.equal(result.found, 1);
  assert.equal(result.affected, 1);
  assert.equal(fsSync.existsSync(path.join(dataset, 'a.jpg')), false);
  assert.equal(fsSync.existsSync(path.join(dataset, 'a.txt')), false);
  assert.equal(fsSync.existsSync(path.join(dataset, 'a.json')), false);
  assert.equal(fsSync.existsSync(path.join(dataset, 'b.jpg')), true);
});

test('move creates a new dataset, moves captions, and removes source files', async () => {
  const { root, dataset } = await makeDataset({
    'nested/a.jpg': 'image',
    'nested/a.txt': 'cat',
    'b.jpg': 'image',
    'b.txt': 'dog',
  });

  const result = await performPlainDatasetCaptionBulkAction(root, {
    datasetName: 'source',
    action: 'move',
    imgPaths: [path.join(dataset, 'nested', 'a.jpg'), path.join(dataset, 'b.jpg')],
    query: 'cat',
    destinationName: 'moved_cats',
  });

  assert.equal(result.found, 1);
  assert.equal(result.affected, 1);
  assert.equal(result.destinationName, 'moved_cats');
  const movedImage = path.join(root, 'moved_cats', 'a.jpg');
  assert.equal(fsSync.existsSync(movedImage), true);
  assert.equal(await fs.readFile(captionSidecarPath(movedImage, '.txt'), 'utf8'), 'cat');
  assert.equal(fsSync.existsSync(path.join(dataset, 'nested', 'a.jpg')), false);
  assert.equal(fsSync.existsSync(path.join(dataset, 'b.jpg')), true);
});

test('bulk action rejects paths outside selected dataset', async () => {
  const { root, dataset } = await makeDataset({ 'a.jpg': 'image', 'a.txt': 'cat' });
  const outside = path.join(root, 'outside.jpg');
  await fs.writeFile(outside, 'image');

  await assert.rejects(
    () =>
      performPlainDatasetCaptionBulkAction(root, {
        datasetName: 'source',
        action: 'delete',
        imgPaths: [path.join(dataset, 'a.jpg'), outside],
        query: 'cat',
      }),
    /Invalid image path/,
  );
});

test('remote bulk helpers decode signed refs and map result paths back to client refs', () => {
  const remotePath = '/remote/datasets/source/a.jpg';
  const ref = makeRemoteDatasetAssetRef('worker-1', 'img', remotePath);
  const decoded = decodeRemoteCaptionBulkPaths([ref], 'worker-1');

  assert.deepEqual(decoded.remotePaths, [remotePath]);
  assert.equal(decoded.refByRemotePath[remotePath], ref);

  const mapped = mapRemoteCaptionBulkResult(
    {
      action: 'remove_words',
      found: 1,
      affected: 1,
      updatedCaptions: { [remotePath]: 'updated caption' },
      removedPaths: [remotePath],
    },
    decoded.refByRemotePath,
  );

  assert.equal(mapped.updatedCaptions[ref], 'updated caption');
  assert.deepEqual(mapped.removedPaths, [ref]);
});
