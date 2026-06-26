import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { afterEach, test } from 'node:test';
import { isPathWithinRoot, normalizeStoragePathSetting } from '../dist/src/server/pathContainment.js';

const tempRoots = [];

afterEach(async () => {
  await Promise.all(tempRoots.splice(0).map(root => fs.rm(root, { recursive: true, force: true })));
});

async function makeWorkspace() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-path-containment-'));
  tempRoots.push(root);
  const fallbackRoot = path.join(root, 'datasets');
  const externalRoot = path.join(root, 'external');
  await fs.mkdir(fallbackRoot, { recursive: true });
  await fs.mkdir(externalRoot, { recursive: true });
  return { root, fallbackRoot, externalRoot };
}

test('isPathWithinRoot accepts descendants and rejects sibling prefixes', () => {
  const root = path.resolve('/tmp/aitk/root');
  assert.equal(isPathWithinRoot(root, path.join(root, 'child')), true);
  assert.equal(isPathWithinRoot(root, path.resolve('/tmp/aitk/root-sibling/file.jpg')), false);
});

test('normalizeStoragePathSetting rejects filesystem roots', async () => {
  const { fallbackRoot } = await makeWorkspace();
  assert.equal(await normalizeStoragePathSetting(path.parse(fallbackRoot).root, fallbackRoot), null);
});

test('normalizeStoragePathSetting rejects roots that escape through symlinked ancestors', async t => {
  const { fallbackRoot, externalRoot } = await makeWorkspace();
  const symlinkRoot = path.join(fallbackRoot, 'linked');
  try {
    await fs.symlink(externalRoot, symlinkRoot, process.platform === 'win32' ? 'junction' : 'dir');
  } catch (error) {
    if (error?.code === 'EPERM' || error?.code === 'EINVAL') {
      t.skip('directory symlinks are not available in this environment');
      return;
    }
    throw error;
  }

  assert.equal(await normalizeStoragePathSetting(symlinkRoot, fallbackRoot), null);
  assert.equal(await normalizeStoragePathSetting(path.join(symlinkRoot, 'future'), fallbackRoot), null);
  assert.equal(
    await normalizeStoragePathSetting(symlinkRoot, fallbackRoot, { allowExternal: true }),
    path.resolve(symlinkRoot),
  );
});

test('normalizeStoragePathSetting accepts in-root storage paths', async () => {
  const { fallbackRoot } = await makeWorkspace();
  const nestedRoot = path.join(fallbackRoot, 'nested');
  assert.equal(await normalizeStoragePathSetting(nestedRoot, fallbackRoot), path.resolve(nestedRoot));
});
