import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createWriteStream } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import sharp from 'sharp';
import archiver from 'archiver';

const require = createRequire(import.meta.url);
const { importLayeredDocument } = require('../dist/src/server/layeredImport.js');
const { listLayeredImages, validateLayeredImageAssets } = require('../dist/src/server/layeredImages.js');
const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-layered-runtime-'));
try {
  const dataset = path.join(root, 'dataset');
  await fs.mkdir(dataset);
  const input = path.join(root, 'interop.ora');
  const png = await sharp({
    create: { width: 64, height: 64, channels: 4, background: { r: 40, g: 90, b: 180, alpha: 0.5 } },
  })
    .png()
    .toBuffer();
  await new Promise((resolve, reject) => {
    const output = createWriteStream(input),
      zip = archiver('zip');
    output.once('close', resolve);
    output.once('error', reject);
    zip.once('error', reject);
    zip.pipe(output);
    zip.append('image/openraster', { name: 'mimetype', store: true });
    zip.append(
      '<image w="64" h="64"><stack><layer name="Foreground" src="data/1.png" x="0" y="0" opacity="1.0" visibility="visible" composite-op="svg:src-over"/><layer name="Background" src="data/0.png" x="0" y="0" opacity="1.0" visibility="visible" composite-op="svg:src-over"/></stack></image>',
      { name: 'stack.xml' },
    );
    zip.append(png, { name: 'data/0.png' });
    zip.append(png, { name: 'data/1.png' });
    void zip.finalize().catch(reject);
  });
  const result = await importLayeredDocument(input, dataset, 'interop.ora');
  assert.equal(result.layer_count, 2);
  const [document] = await listLayeredImages(dataset);
  assert.equal(document.id, result.id);
  assert.deepEqual(
    document.layers.map(layer => layer.name),
    ['Background', 'Foreground'],
  );
  assert.equal(document.source.filename, 'interop.ora');
  await validateLayeredImageAssets(dataset);
  console.log('Real Python ORA import, validated RGBA layers, and atomic dataset publication passed.');
} finally {
  if (path.dirname(root) !== path.resolve(os.tmpdir()) || !path.basename(root).startsWith('aitk-layered-runtime-'))
    throw new Error('Unexpected smoke test cleanup path');
  await fs.rm(root, { recursive: true, force: true });
}
