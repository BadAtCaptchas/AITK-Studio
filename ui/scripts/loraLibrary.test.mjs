import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { findDuplicateUploadedLoraPath } from '../dist/src/server/loraLibrary.js';

function makeTempRoot() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-lora-library-'));
}

test('duplicate LoRA lookup reuses an identical existing upload', async () => {
  const root = makeTempRoot();
  const content = Buffer.from('same lora bytes');
  const existingPath = path.join(root, 'base.safetensors');

  try {
    fs.writeFileSync(existingPath, content);

    const duplicate = await findDuplicateUploadedLoraPath(root, 'base.safetensors', content);

    assert.equal(duplicate, existingPath);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('duplicate LoRA lookup skips same-size files with different content', async () => {
  const root = makeTempRoot();

  try {
    fs.writeFileSync(path.join(root, 'base.safetensors'), Buffer.from('aaaa'));

    const duplicate = await findDuplicateUploadedLoraPath(root, 'base.safetensors', Buffer.from('bbbb'));

    assert.equal(duplicate, null);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('duplicate LoRA lookup can find an identical numbered upload', async () => {
  const root = makeTempRoot();
  const content = Buffer.from('same lora bytes');
  const numberedPath = path.join(root, 'base_1.safetensors');

  try {
    fs.writeFileSync(path.join(root, 'base.safetensors'), Buffer.from('different bytes'));
    fs.writeFileSync(numberedPath, content);

    const duplicate = await findDuplicateUploadedLoraPath(root, 'base.safetensors', content);

    assert.equal(duplicate, numberedPath);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
