import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { prepareHfTokenEnv } = require('../dist/src/server/hfTokenEnv.js');

const tempDirs = [];
const tempRoot = path.resolve('.tmp', 'hf-token-env-tests');

async function makeTempDir() {
  await fs.mkdir(tempRoot, { recursive: true });
  const dir = await fs.mkdtemp(path.join(tempRoot, 'case-'));
  tempDirs.push(dir);
  return dir;
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map(dir => fs.rm(dir, { recursive: true, force: true })));
});

test('prepareHfTokenEnv moves UI token into HF_TOKEN_PATH without exposing token env vars', async () => {
  const tokenDir = await makeTempDir();
  const prepared = await prepareHfTokenEnv({
    env: {
      HF_TOKEN: 'host-token',
      HUGGING_FACE_HUB_TOKEN: 'legacy-token',
      HF_TOKEN_PATH: 'existing-token-path',
      KEEP_ME: '1',
    },
    token: 'ui-token',
    tokenDir,
    tokenFilePrefix: 'ui-test',
  });

  assert.equal(prepared.env.HF_TOKEN, undefined);
  assert.equal(prepared.env.HUGGING_FACE_HUB_TOKEN, undefined);
  assert.equal(prepared.env.KEEP_ME, '1');
  assert.match(prepared.env.HF_TOKEN_PATH, /^.+ui-test-.+\.token$/);
  assert.equal(await fs.readFile(prepared.env.HF_TOKEN_PATH, 'utf8'), 'ui-token');

  await prepared.cleanup();
  await assert.rejects(fs.stat(prepared.env.HF_TOKEN_PATH), { code: 'ENOENT' });
});

test('prepareHfTokenEnv preserves Hugging Face auth from host env without exposing HF_TOKEN', async () => {
  const tokenDir = await makeTempDir();
  const prepared = await prepareHfTokenEnv({
    env: {
      HF_TOKEN: 'host-token',
      KEEP_ME: '1',
    },
    tokenDir,
    tokenFilePrefix: 'host-test',
  });

  assert.equal(prepared.env.HF_TOKEN, undefined);
  assert.equal(prepared.env.KEEP_ME, '1');
  assert.equal(await fs.readFile(prepared.env.HF_TOKEN_PATH, 'utf8'), 'host-token');

  await prepared.cleanup();
  await prepared.cleanup();
  await assert.rejects(fs.stat(prepared.env.HF_TOKEN_PATH), { code: 'ENOENT' });
});

test('prepareHfTokenEnv leaves existing HF_TOKEN_PATH when no raw token is present', async () => {
  const prepared = await prepareHfTokenEnv({
    env: {
      HF_TOKEN_PATH: 'existing-token-path',
      KEEP_ME: '1',
    },
  });

  assert.equal(prepared.tokenPath, null);
  assert.equal(prepared.env.HF_TOKEN_PATH, 'existing-token-path');
  assert.equal(prepared.env.KEEP_ME, '1');
});
