import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { loadEnvironment } from './load-env.mjs';

test('environment files retain precedence, expansion and shell overrides', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-env-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  await fs.writeFile(path.join(root, '.env'), 'CHOICE=base\nBASE=base-value\n');
  await fs.writeFile(path.join(root, '.env.production'), 'CHOICE=production\nONLY_MODE=production\n');
  await fs.writeFile(path.join(root, '.env.local'), 'CHOICE=local\nLOCAL_ONLY=yes\n');
  await fs.writeFile(path.join(root, '.env.production.local'), 'CHOICE=mode-local\nSHELL_VALUE=file\nDERIVED=${SHELL_VALUE}/child\n');
  const env = { SHELL_VALUE: 'shell' };
  loadEnvironment(root, 'production', env);
  assert.deepEqual(env, { SHELL_VALUE: 'shell', CHOICE: 'mode-local', DERIVED: 'shell/child', LOCAL_ONLY: 'yes', ONLY_MODE: 'production', BASE: 'base-value' });
  const isolated = {};
  loadEnvironment(root, 'test', isolated);
  assert.deepEqual(isolated, { CHOICE: 'base', BASE: 'base-value' });
});
