import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { renameDatasetFolder, DatasetRenameError } = require('../dist/src/server/datasetRename.js');

async function makeRoot() {
  return fs.mkdtemp(path.join(os.tmpdir(), 'aitk-rename-test-'));
}

async function writeDataset(root, name, files = {}) {
  const folder = path.join(root, name);
  await fs.mkdir(folder, { recursive: true });
  await Promise.all(
    Object.entries(files).map(async ([relativePath, data]) => {
      const filePath = path.join(folder, ...relativePath.split('/'));
      await fs.mkdir(path.dirname(filePath), { recursive: true });
      await fs.writeFile(filePath, data);
    }),
  );
  return folder;
}

test('renameDatasetFolder renames a dataset and normalizes the target name', async () => {
  const root = await makeRoot();
  await writeDataset(root, 'source_dataset', {
    'image.png': 'image bytes',
    'nested/caption.txt': 'caption',
  });

  const result = await renameDatasetFolder(root, 'source_dataset', 'Better Dataset');

  assert.equal(result.success, true);
  assert.equal(result.oldName, 'source_dataset');
  assert.equal(result.name, 'better_dataset');
  assert.equal(result.dataset.name, 'better_dataset');
  assert.equal(fsSync.existsSync(path.join(root, 'source_dataset')), false);
  assert.equal(await fs.readFile(path.join(root, 'better_dataset', 'image.png'), 'utf8'), 'image bytes');
  assert.equal(await fs.readFile(path.join(root, 'better_dataset', 'nested', 'caption.txt'), 'utf8'), 'caption');
});

test('renameDatasetFolder rejects missing and conflicting datasets', async () => {
  const root = await makeRoot();
  await writeDataset(root, 'alpha');
  await writeDataset(root, 'beta');

  await assert.rejects(
    () => renameDatasetFolder(root, 'missing', 'renamed'),
    error => error instanceof DatasetRenameError && error.status === 404,
  );

  await assert.rejects(
    () => renameDatasetFolder(root, 'alpha', 'beta'),
    error => error instanceof DatasetRenameError && error.status === 409,
  );
});

test('renameDatasetFolder rejects invalid names', async () => {
  const root = await makeRoot();
  await writeDataset(root, 'alpha');

  await assert.rejects(
    () => renameDatasetFolder(root, '../alpha', 'renamed'),
    error => error instanceof DatasetRenameError && error.status === 400,
  );

  await assert.rejects(
    () => renameDatasetFolder(root, 'alpha', '!!!'),
    error => error instanceof DatasetRenameError && error.status === 400,
  );
});
