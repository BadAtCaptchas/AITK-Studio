import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';

async function loadRemoteUrlHelpers() {
  const source = await readFile(new URL('./repo-updater.mjs', import.meta.url), 'utf8');
  const start = source.indexOf('function redactRemoteCredentials');
  const end = source.indexOf('function normalizeRepoUrlForCompare');
  assert.notEqual(start, -1);
  assert.notEqual(end, -1);

  const context = {};
  vm.runInNewContext(
    `${source.slice(start, end)}\nthis.redactRemoteCredentials = redactRemoteCredentials;\nthis.normalizeRemoteWebUrl = normalizeRemoteWebUrl;`,
    context,
  );
  return context;
}

test('redactRemoteCredentials removes HTTPS remote userinfo', async () => {
  const { redactRemoteCredentials } = await loadRemoteUrlHelpers();

  assert.equal(
    redactRemoteCredentials('https://user:ghp_secret@github.com/BadAtCaptchas/AITK-Studio.git'),
    'https://github.com/BadAtCaptchas/AITK-Studio.git',
  );
  assert.equal(
    redactRemoteCredentials('https://ghp_secret@github.com/BadAtCaptchas/AITK-Studio.git'),
    'https://github.com/BadAtCaptchas/AITK-Studio.git',
  );
  assert.equal(
    redactRemoteCredentials('https://user:p@ss@github.com/BadAtCaptchas/AITK-Studio.git'),
    'https://github.com/BadAtCaptchas/AITK-Studio.git',
  );
});

test('normalizeRemoteWebUrl strips credentials before returning a display URL', async () => {
  const { normalizeRemoteWebUrl } = await loadRemoteUrlHelpers();

  assert.equal(
    normalizeRemoteWebUrl('https://user:ghp_secret@github.com/BadAtCaptchas/AITK-Studio.git'),
    'https://github.com/BadAtCaptchas/AITK-Studio',
  );
  assert.equal(
    normalizeRemoteWebUrl('git@github.com:BadAtCaptchas/AITK-Studio.git'),
    'https://github.com/BadAtCaptchas/AITK-Studio',
  );
});
