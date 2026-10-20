import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createInterface } from 'node:readline';
import test from 'node:test';
import { ensureUiDependencies, removeUnsupportedMacstats } from './install-deps.mjs';

async function fixture() {
  // Match production cleanup paths, including Windows temp aliases and casing.
  const root = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-ui-deps-')));
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

async function macstatsLeftover(root) {
  const chalk = path.join(root, 'node_modules', 'macstats', 'node_modules', 'chalk');
  await fs.mkdir(chalk, { recursive: true });
  const file = path.join(chalk, 'package.json');
  await fs.writeFile(file, '{"name":"chalk"}');
  return file;
}

test('non-macOS installs clean only macstats before and after npm, including cached installs', async t => {
  const { root, statePath } = await fixture();
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const target = path.join(root, 'node_modules', 'macstats');
  const preserved = path.join(root, 'node_modules', 'other-dependency.txt');
  await fs.writeFile(preserved, 'keep');
  await macstatsLeftover(root);
  let installs = 0;
  const runInstall = async () => {
    installs += 1;
    await assert.rejects(fs.stat(target), { code: 'ENOENT' });
    await macstatsLeftover(root);
    return 0;
  };
  const options = { uiRoot: root, statePath, runInstall, platform: 'win32' };
  assert.equal((await ensureUiDependencies(options)).installed, true);
  await assert.rejects(fs.stat(target), { code: 'ENOENT' });
  await macstatsLeftover(root);
  assert.equal((await ensureUiDependencies(options)).installed, false);
  await assert.rejects(fs.stat(target), { code: 'ENOENT' });
  assert.equal(installs, 1);
  assert.equal(await fs.readFile(preserved, 'utf8'), 'keep');
});

test('macOS installs keep their native statistics dependency', async t => {
  const { root, statePath } = await fixture();
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const file = await macstatsLeftover(root);
  await ensureUiDependencies({ uiRoot: root, statePath, platform: 'darwin', runInstall: async () => 0 });
  assert.equal(await fs.readFile(file, 'utf8'), '{"name":"chalk"}');
});

test('cleanup never follows node_modules or macstats junctions outside the installation', async t => {
  for (const linkedName of ['node_modules', 'macstats']) {
    const { root } = await fixture();
    t.after(() => fs.rm(root, { recursive: true, force: true }));
    const outside = path.join(root, 'shared-dependencies');
    await fs.mkdir(path.join(outside, 'macstats'), { recursive: true });
    const marker = path.join(outside, 'macstats', 'keep.txt');
    await fs.writeFile(marker, 'keep');
    let link;
    if (linkedName === 'node_modules') {
      link = path.join(root, 'node_modules');
      await fs.rmdir(link);
    } else {
      link = path.join(root, 'node_modules', 'macstats');
    }
    await fs.symlink(outside, link, process.platform === 'win32' ? 'junction' : 'dir');
    await assert.rejects(removeUnsupportedMacstats(root, 'win32'), /symbolic link or junction/);
    assert.equal(await fs.readFile(marker, 'utf8'), 'keep');
  }
});

test('removing leftovers preserves targets of nested dependency junctions', async t => {
  const { root } = await fixture();
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  await macstatsLeftover(root);
  const outside = path.join(root, 'shared-dependencies');
  await fs.mkdir(outside);
  const marker = path.join(outside, 'keep.txt');
  await fs.writeFile(marker, 'keep');
  await fs.symlink(
    outside,
    path.join(root, 'node_modules', 'macstats', 'node_modules', 'shared'),
    process.platform === 'win32' ? 'junction' : 'dir',
  );
  await removeUnsupportedMacstats(root, 'win32');
  await assert.rejects(fs.stat(path.join(root, 'node_modules', 'macstats')), { code: 'ENOENT' });
  assert.equal(await fs.readFile(marker, 'utf8'), 'keep');
});

test('a cleanup warning does not fail a successful install and the next cached run retries cleanup', async t => {
  const { root, statePath } = await fixture();
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  await macstatsLeftover(root);
  const target = path.join(root, 'node_modules', 'macstats');
  const remove = fs.rm;
  let locked = true;
  t.mock.method(fs, 'rm', async (candidate, options) => {
    if (candidate === target && locked) throw Object.assign(new Error('fixture lock'), { code: 'EPERM' });
    return remove(candidate, options);
  });
  const warnings = t.mock.method(console, 'warn', () => {});
  let installs = 0;
  const options = {
    uiRoot: root, statePath, platform: 'win32',
    runInstall: async () => { installs += 1; return 0; },
  };
  assert.equal((await ensureUiDependencies(options)).installed, true);
  assert.equal(warnings.mock.callCount(), 1);
  locked = false;
  assert.equal((await ensureUiDependencies(options)).installed, false);
  await assert.rejects(fs.stat(target), { code: 'ENOENT' });
  assert.equal(installs, 1);
});

test('a persistent cleanup lock is reported without masking an npm install failure', async t => {
  const { root, statePath } = await fixture();
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const file = await macstatsLeftover(root);
  const target = path.join(root, 'node_modules', 'macstats');
  const remove = fs.rm;
  t.mock.method(fs, 'rm', async (candidate, options) => {
    if (candidate === target) throw Object.assign(new Error('fixture lock'), { code: 'EPERM' });
    return remove(candidate, options);
  });
  const warnings = t.mock.method(console, 'warn', () => {});
  await assert.rejects(
    ensureUiDependencies({ uiRoot: root, statePath, platform: 'win32', runInstall: async () => 7 }),
    /exit code 7/,
  );
  assert.equal(warnings.mock.callCount(), 1);
  assert.match(warnings.mock.calls[0].arguments[0], /npm run install_deps again/);
  assert.equal(await fs.readFile(file, 'utf8'), '{"name":"chalk"}');
  await assert.rejects(fs.readFile(statePath), { code: 'ENOENT' });
});

test('Windows cleanup recovers when a locked file is released during retries', {
  skip: process.platform !== 'win32',
  timeout: 15_000,
}, async t => {
  const { root } = await fixture();
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const file = await macstatsLeftover(root);
  const child = spawn('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
    '$file = [System.IO.File]::Open($env:AITK_TEST_LOCK_PATH, "Open", "ReadWrite", "None"); ' +
    'try { [Console]::WriteLine("locked"); [Console]::ReadLine() | Out-Null } finally { $file.Dispose() }',
  ], {
    env: { ...process.env, AITK_TEST_LOCK_PATH: file },
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true,
  });
  const exited = once(child, 'exit');
  const lines = createInterface({ input: child.stdout });
  let release;
  try {
    const ready = await Promise.race([
      once(lines, 'line').then(([line]) => line),
      exited.then(([code]) => { throw new Error(`Lock helper exited early: ${code}`); }),
    ]);
    assert.equal(ready, 'locked');
    await assert.rejects(fs.unlink(file), error => ['EPERM', 'EBUSY', 'EACCES'].includes(error.code));
    release = setTimeout(() => child.stdin.end(), 300);
    await removeUnsupportedMacstats(root);
    await assert.rejects(fs.stat(path.join(root, 'node_modules', 'macstats')), { code: 'ENOENT' });
  } finally {
    clearTimeout(release);
    child.stdin.end();
    await exited;
    lines.close();
  }
});
