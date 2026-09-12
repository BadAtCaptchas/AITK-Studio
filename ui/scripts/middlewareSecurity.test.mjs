import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';

test('middleware authenticates without reading or cloning command bodies', async () => {
  const source = await readFile(new URL('../src/middleware.ts', import.meta.url), 'utf8');
  const authentication = source.indexOf('await isRequestAuthenticated(request, tokenToUse)');
  const bodyParsing = source.indexOf('.clone()');

  assert.notEqual(authentication, -1);
  assert.equal(bodyParsing, -1);
  assert.doesNotMatch(source, /request\.json\(/);
});
