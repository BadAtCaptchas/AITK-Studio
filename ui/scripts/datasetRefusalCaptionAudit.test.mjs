import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import test from 'node:test';

const require = createRequire(import.meta.url);
const { auditDatasetRefusalCaptions } = require('../dist/src/server/datasetRefusalCaptionAudit.js');

async function makeDataset(files) {
  const dataset = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-refusal-audit-'));
  for (const [name, content] of Object.entries(files)) {
    const filePath = path.join(dataset, name);
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, content);
  }
  return dataset;
}

test('refusal caption audit caches unchanged datasets and invalidates when sidecars change', async () => {
  const dataset = await makeDataset({
    'good.png': 'image',
    'good.txt': 'A person in a red jacket.',
    'bad.jpg': 'image',
    'bad.txt': 'I cannot fulfill this request.',
    'nested/orphan.caption': "Sorry, but I can't help with that.",
  });

  const first = await auditDatasetRefusalCaptions(dataset);
  assert.equal(first.cached, false);
  assert.equal(first.scanned, 3);
  assert.equal(first.refusalCount, 2);
  assert.equal(first.refusals[path.join(dataset, 'bad.jpg')], 'I cannot fulfill this request.');
  assert.equal(first.refusals[path.join(dataset, 'nested', 'orphan.caption')], "Sorry, but I can't help with that.");

  const second = await auditDatasetRefusalCaptions(dataset);
  assert.equal(second.cached, true);
  assert.deepEqual(second.refusals, first.refusals);

  await fs.writeFile(path.join(dataset, 'good.txt'), 'I cannot access the image.');

  const third = await auditDatasetRefusalCaptions(dataset);
  assert.equal(third.cached, false);
  assert.equal(third.refusalCount, 3);
  assert.equal(third.refusals[path.join(dataset, 'good.png')], 'I cannot access the image.');
});

