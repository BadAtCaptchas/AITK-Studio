import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
import path from 'node:path';
import { getDatasetImageMediaUrl, normalizeDatasetImageListItem } from '../dist/src/utils/datasetMedia.js';

test('remote dataset entries keep operational paths separate from signed media URLs', () => {
  const root = 'E:\\datasets\\cats\\';
  const signed = '/api/remote-datasets/assets?worker_id=worker-1&path=cats/image.png&sig=test';
  const item = normalizeDatasetImageListItem({ img_path: 'image.png', media_url: signed, size_bytes: 123 }, root);

  assert.equal(item?.img_path, `${root}image.png`);
  assert.equal(item?.media_url, signed);
  assert.equal(getDatasetImageMediaUrl(item), signed);
  assert.equal(signed.includes('E%3A') || signed.includes('%5C'), false);
});

test('global dataset entries retain the existing image route fallback', () => {
  const item = normalizeDatasetImageListItem({ img_path: 'C:\\datasets\\cats\\image.png' }, null);
  assert.equal(item?.media_url, null);
  assert.equal(getDatasetImageMediaUrl(item).startsWith('/api/img/'), true);
});

test('dataset studio renders media URLs but keeps paths as item identity', async () => {
  const root = process.cwd();
  const [studioMedia, navigator, editor] = await Promise.all([
    fs.readFile(path.join(root, 'src/components/dataset-image-studio/StudioMedia.tsx'), 'utf8'),
    fs.readFile(path.join(root, 'src/components/dataset-image-studio/ImageNavigator.tsx'), 'utf8'),
    fs.readFile(path.join(root, 'src/components/DatasetEditorPage.tsx'), 'utf8'),
  ]);

  assert.match(studioMedia, /getMediaUrl\(item\.mediaUrl \|\| item\.path\)/);
  assert.match(navigator, /mediaUrl=\{item\.mediaUrl\}/);
  assert.match(editor, /path: img\.img_path/);
  assert.match(editor, /mediaUrl: img\.media_url \?\? null/);
});
