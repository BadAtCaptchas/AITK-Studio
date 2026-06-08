import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const { deletePlainImagePaths } = require('../dist/src/server/imageDelete.js');

async function makeWorkspace(files) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-image-delete-'));
  const datasetsRoot = path.join(root, 'datasets');
  const trainingRoot = path.join(root, 'training');
  const dataset = path.join(datasetsRoot, 'source');
  await fs.mkdir(dataset, { recursive: true });
  await fs.mkdir(trainingRoot, { recursive: true });
  for (const [name, content] of Object.entries(files)) {
    const filePath = path.join(dataset, name);
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, content);
  }
  return { root, datasetsRoot, trainingRoot, dataset };
}

test('deletePlainImagePaths removes media and caption sidecars', async () => {
  const { datasetsRoot, trainingRoot, dataset } = await makeWorkspace({
    'a.jpg': 'image',
    'a.txt': 'caption',
    'a.json': '{"caption":"caption"}',
    'b.jpg': 'image',
  });

  const result = await deletePlainImagePaths([path.join(dataset, 'a.jpg')], datasetsRoot, trainingRoot);

  assert.equal(result.requested, 1);
  assert.equal(result.deleted, 1);
  assert.equal(result.failed, 0);
  assert.deepEqual(result.removedPaths, [path.join(dataset, 'a.jpg')]);
  assert.equal(fsSync.existsSync(path.join(dataset, 'a.jpg')), false);
  assert.equal(fsSync.existsSync(path.join(dataset, 'a.txt')), false);
  assert.equal(fsSync.existsSync(path.join(dataset, 'a.json')), false);
  assert.equal(fsSync.existsSync(path.join(dataset, 'b.jpg')), true);
});

test('deletePlainImagePaths reports missing files as skipped', async () => {
  const { datasetsRoot, trainingRoot, dataset } = await makeWorkspace({});

  const result = await deletePlainImagePaths([path.join(dataset, 'missing.jpg')], datasetsRoot, trainingRoot);

  assert.equal(result.requested, 1);
  assert.equal(result.deleted, 0);
  assert.equal(result.skipped, 1);
  assert.equal(result.failed, 0);
});

test('deletePlainImagePaths rejects outside paths before deleting valid paths', async () => {
  const { root, datasetsRoot, trainingRoot, dataset } = await makeWorkspace({
    'a.jpg': 'image',
    'a.txt': 'caption',
  });
  const outside = path.join(root, 'outside.jpg');
  await fs.writeFile(outside, 'image');

  await assert.rejects(
    () => deletePlainImagePaths([path.join(dataset, 'a.jpg'), outside], datasetsRoot, trainingRoot),
    /Invalid image path/,
  );
  assert.equal(fsSync.existsSync(path.join(dataset, 'a.jpg')), true);
  assert.equal(fsSync.existsSync(path.join(dataset, 'a.txt')), true);
});
