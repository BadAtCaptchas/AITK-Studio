import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { moveStagedUploadNoReplace } from '../dist/src/server/streamedUpload.js';

test('concurrent staged uploads cannot replace the same destination', async () => {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-streamed-upload-'));
  try {
    const first = path.join(directory, '.first.tmp');
    const second = path.join(directory, '.second.tmp');
    const destination = path.join(directory, 'asset.bin');
    await Promise.all([fs.writeFile(first, 'first'), fs.writeFile(second, 'second')]);

    const results = await Promise.allSettled([
      moveStagedUploadNoReplace(first, destination),
      moveStagedUploadNoReplace(second, destination),
    ]);
    assert.equal(results.filter(result => result.status === 'fulfilled').length, 1);
    assert.equal(results.filter(result => result.status === 'rejected').length, 1);
    assert.match(await fs.readFile(destination, 'utf8'), /^(first|second)$/);
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
});
