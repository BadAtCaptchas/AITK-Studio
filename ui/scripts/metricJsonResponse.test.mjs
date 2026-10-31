import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import { gunzipSync } from 'node:zlib';
import test from 'node:test';
import ts from 'typescript';
const loaded = { exports: {} };
new Function('require', 'module', 'exports', ts.transpileModule(
  fs.readFileSync(new URL('../src/server/metricJsonResponse.ts', import.meta.url), 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } },
).outputText)(createRequire(import.meta.url), loaded, loaded.exports);
const { metricJsonResponse } = loaded.exports;
const data = { keys: ['loss', 'phase/name'], points: Array.from({ length: 21001 }, (_, step) => ({ step, value: step / 3, value_text: null })) };
for (const [header, compressed] of [['gzip', true], ['br, gzip;q=0.5', true], ['gzip;q=0', false], ['gzip;q=0, *;q=1', false], ['*;q=0.7', true], ['gzip;q=bogus', false], ['', false]]) {
  test(`large bounded history preserves precision and shape with accept-encoding ${header}`, async () => {
    const response = await metricJsonResponse(new Request('http://localhost', { headers: { 'accept-encoding': header } }), data, 8 * 1024 ** 2);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('content-encoding'), compressed ? 'gzip' : null);
    assert.equal(response.headers.get('vary'), 'Accept-Encoding');
    assert.equal(response.headers.get('cache-control'), 'no-store');
    const bytes = Buffer.from(await response.arrayBuffer());
    assert.deepEqual(JSON.parse((compressed ? gunzipSync(bytes) : bytes).toString()), data);
  });
}
test('compression never bypasses the decoded byte limit, even with highly compressible data', async () => {
  const response = await metricJsonResponse(new Request('http://localhost', { headers: { 'accept-encoding': 'gzip' } }), { text: 'x'.repeat(8 * 1024 ** 2) }, 8 * 1024 ** 2);
  assert.equal(response.status, 413);
  assert.equal(response.headers.get('content-encoding'), null);
});
test('empty and null payloads retain JSON semantics without compression overhead', async () => {
  for (const data of [null, {}, { keys: [], points: [] }, { value: null, value_text: 'phase' }]) {
    const response = await metricJsonResponse(new Request('http://localhost', { headers: { 'accept-encoding': 'gzip' } }), data, 8192);
    assert.deepEqual(await response.json(), data);
  }
});
