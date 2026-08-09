import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { modelsPathFromEnv, resolveModelsPathState } from '../dist/src/server/modelsPath.js';

test('models path uses a nonblank environment value first and reports the lock', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-models-path-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const defaultRoot = path.join(root, 'models');
  const configured = path.join(root, 'shared-models');
  const state = await resolveModelsPathState({
    defaultRoot,
    settingValue: path.join(defaultRoot, 'setting'),
    envValue: `  ${configured}  `,
  });
  assert.deepEqual(state, { path: path.resolve(configured), lockedByEnv: true });
  assert.equal(modelsPathFromEnv('   '), null);
});

test('models settings enforce containment unless authenticated', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-models-security-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const defaultRoot = path.join(root, 'models');
  const inside = path.join(defaultRoot, 'shared');
  const outside = path.join(root, 'external');

  assert.deepEqual(
    await resolveModelsPathState({ defaultRoot, settingValue: inside, envValue: '', allowExternal: false }),
    { path: path.resolve(inside), lockedByEnv: false },
  );
  assert.deepEqual(
    await resolveModelsPathState({ defaultRoot, settingValue: outside, envValue: '', allowExternal: false }),
    { path: path.resolve(defaultRoot), lockedByEnv: false },
  );
  assert.deepEqual(
    await resolveModelsPathState({ defaultRoot, settingValue: outside, envValue: '', allowExternal: true }),
    { path: path.resolve(outside), lockedByEnv: false },
  );
});

test('models environment path refuses a filesystem root', () => {
  const root = path.parse(process.cwd()).root;
  assert.throws(() => modelsPathFromEnv(root), /filesystem root/);
});

test('settings and launcher expose the resolved models path without replacing sensitive env setup', async () => {
  const route = await fs.readFile(path.join(process.cwd(), 'src/app/api/settings/route.ts'), 'utf8');
  const settingsPage = await fs.readFile(path.join(process.cwd(), 'src/app/settings/page.tsx'), 'utf8');
  const launcher = await fs.readFile(path.join(process.cwd(), 'cron/actions/startJob.ts'), 'utf8');
  assert.match(route, /MODELS_PATH_LOCKED/);
  assert.match(route, /allowExternal: access\.authenticated/);
  assert.match(settingsPage, /Locked by the MODELS_PATH environment variable/);
  assert.match(launcher, /MODELS_PATH: modelsRoot/);
  assert.match(launcher, /prepareHfTokenEnv/);
  assert.match(launcher, /offlineChildProcessEnv/);
  assert.match(launcher, /AITK_ENCRYPTED_DATASET_KEYS_B64/);
});
