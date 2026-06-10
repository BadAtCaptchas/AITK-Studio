import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import { applySelectedDatasetDefaults, normalizeDetectedCaptionExt } from '../dist/src/utils/jobDatasetDefaults.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const uiRoot = path.resolve(__dirname, '..');

test('applies selected model dataset defaults to a new dataset row', () => {
  const dataset = {
    folder_path: '/datasets/cats',
    caption_dropout_rate: 0.05,
    shuffle_tokens: true,
  };

  const next = applySelectedDatasetDefaults(dataset, {
    'config.process[0].datasets[x].caption_dropout_rate': [0, 0.05],
    'config.process[0].datasets[x].shuffle_tokens': [false, undefined],
  });

  assert.equal(next.caption_dropout_rate, 0);
  assert.equal(next.shuffle_tokens, false);
  assert.equal(dataset.caption_dropout_rate, 0.05);
  assert.equal(dataset.shuffle_tokens, true);
});

test('normalizes detected dataset caption extensions for job config use', () => {
  assert.equal(normalizeDetectedCaptionExt('.json'), 'json');
  assert.equal(normalizeDetectedCaptionExt('TXT'), 'txt');
  assert.equal(normalizeDetectedCaptionExt('unknown'), null);
  assert.equal(normalizeDetectedCaptionExt(null), null);
});

test('Ideogram model options declare JSON-safe dataset defaults', async () => {
  const source = await fs.readFile(path.join(uiRoot, 'src/app/jobs/new/options.ts'), 'utf8');

  for (const modelName of ["name: 'ideogram4'", "name: 'ideogram4:fp8'"]) {
    const start = source.indexOf(modelName);
    assert.notEqual(start, -1);
    const end = source.indexOf('\n  },', start);
    const block = source.slice(start, end);

    assert.match(block, /'config\.process\[0\]\.datasets\[x\]\.caption_dropout_rate': \[0, 0\.05\]/);
    assert.match(block, /'config\.process\[0\]\.datasets\[x\]\.shuffle_tokens': \[false, undefined\]/);
  }
});
