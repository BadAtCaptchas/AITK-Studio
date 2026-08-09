import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { ensureUiDependencies } from './install-deps.mjs';

async function fixture() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-ui-deps-'));
  await fs.mkdir(path.join(root, 'node_modules'));
  await fs.writeFile(path.join(root, 'package.json'), '{"name":"fixture","dependencies":{"x":"1"}}\n');
  await fs.writeFile(path.join(root, 'package-lock.json'), '{"lockfileVersion":3,"packages":{}}\n');
  return { root, statePath: path.join(root, 'state', 'deps.json') };
}

test('dependency install restores package-lock bytes and skips an unchanged manifest', async t => {
  const { root, statePath } = await fixture();
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const lockPath = path.join(root, 'package-lock.json');
  const before = await fs.readFile(lockPath);
  let installs = 0;
  const runInstall = async () => {
    installs += 1;
    await fs.writeFile(lockPath, '{"rewritten":true}\n');
    return 0;
  };

  assert.equal((await ensureUiDependencies({ uiRoot: root, statePath, runInstall })).installed, true);
  assert.deepEqual(await fs.readFile(lockPath), before);
  assert.equal((await ensureUiDependencies({ uiRoot: root, statePath, runInstall })).installed, false);
  assert.equal(installs, 1);

  await fs.writeFile(path.join(root, 'package.json'), '{"name":"fixture","dependencies":{"x":"2"}}\n');
  assert.equal((await ensureUiDependencies({ uiRoot: root, statePath, runInstall })).installed, true);
  assert.equal(installs, 2);
  assert.deepEqual(await fs.readFile(lockPath), before);
});

test('failed dependency installs still restore package-lock bytes and do not cache success', async t => {
  const { root, statePath } = await fixture();
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const lockPath = path.join(root, 'package-lock.json');
  const before = await fs.readFile(lockPath);
  await assert.rejects(
    ensureUiDependencies({
      uiRoot: root,
      statePath,
      runInstall: async () => {
        await fs.writeFile(lockPath, 'damaged');
        return 7;
      },
    }),
    /exit code 7/,
  );
  assert.deepEqual(await fs.readFile(lockPath), before);
  await assert.rejects(fs.readFile(statePath), { code: 'ENOENT' });
});
