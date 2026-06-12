import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  decorateRemoteHfDatasetImportResult,
  normalizeHfDatasetID,
  normalizeHfDatasetImportRequest,
  rankHfCaptionColumns,
} = require('../dist/src/server/hfDatasetImport.js');

test('normalizeHfDatasetID accepts dataset IDs and Hugging Face dataset URLs', () => {
  assert.equal(normalizeHfDatasetID('merve/vlm_test_images'), 'merve/vlm_test_images');
  assert.equal(
    normalizeHfDatasetID('https://huggingface.co/datasets/merve/vlm_test_images?row=1'),
    'merve/vlm_test_images',
  );
  assert.equal(normalizeHfDatasetID('huggingface.co/datasets/mnist'), 'mnist');
});

test('normalizeHfDatasetID rejects path traversal and non-dataset URLs', () => {
  assert.throws(() => normalizeHfDatasetID('../secret'), /valid Hugging Face dataset/i);
  assert.throws(() => normalizeHfDatasetID('https://example.com/datasets/merve/vlm_test_images'), /valid/i);
});

test('normalizeHfDatasetImportRequest validates action and caption column requirements', () => {
  assert.deepEqual(
    normalizeHfDatasetImportRequest({
      action: 'preview',
      dataset: 'merve/vlm_test_images',
      captionMode: 'auto',
      maxRows: '12',
    }),
    {
      action: 'preview',
      worker_id: undefined,
      dataset: 'merve/vlm_test_images',
      config: undefined,
      split: undefined,
      imageColumn: undefined,
      captionMode: 'auto',
      captionColumn: undefined,
      outputName: undefined,
      maxRows: 12,
    },
  );

  assert.throws(
    () =>
      normalizeHfDatasetImportRequest({
        action: 'import',
        dataset: 'merve/vlm_test_images',
        captionMode: 'column',
      }),
    /caption column/i,
  );
});

test('rankHfCaptionColumns prefers common caption names while preserving remaining columns', () => {
  assert.deepEqual(
    rankHfCaptionColumns(['notes', 'Text', 'caption', 'title', 'prompt']),
    ['caption', 'prompt', 'Text', 'title', 'notes'],
  );
});

test('decorateRemoteHfDatasetImportResult marks returned dataset as remote', () => {
  const result = decorateRemoteHfDatasetImportResult(
    { id: 'worker-1', name: 'Worker 1' },
    {
      dataset: {
        name: 'imported',
        encrypted: false,
        source: 'local',
        worker_id: 'local',
        worker_name: 'Local',
        ref: 'aitk-dataset://local/imported',
        path: 'E:/datasets/imported',
      },
      path: 'E:/datasets/imported',
      renamed: false,
      imported: {
        datasetID: 'merve/vlm_test_images',
        config: 'default',
        split: 'train',
        imageColumn: 'image',
        captionColumn: null,
        imagesWritten: 31,
        captionsWritten: 0,
        rowsScanned: 31,
        rowsSkipped: 0,
        warnings: [],
      },
    },
  );

  assert.equal(result.dataset.source, 'remote');
  assert.equal(result.dataset.worker_id, 'worker-1');
  assert.equal(result.dataset.worker_name, 'Worker 1');
  assert.equal(result.dataset.path, undefined);
  assert.match(result.dataset.ref || '', /^aitk-dataset:\/\/remote\/worker-1\//);
  assert.equal(result.path, 'E:/datasets/imported');
});
