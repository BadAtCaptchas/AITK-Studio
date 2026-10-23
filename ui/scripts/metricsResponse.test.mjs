import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { createHookHarness } from './reactHookHarness.mjs';

function load(relative, dependencies) {
  const source = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const loadedModule = { exports: {} };
  new Function('require', 'module', 'exports', 'console', source)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
    return dependencies[name];
  }, loadedModule, loadedModule.exports, { error() {} });
  return loadedModule.exports;
}

function fixture(t) {
  const harness = createHookHarness(), requests = [], polls = [];
  t.after(() => harness.cleanup());
  const hook = load('../src/hooks/useJobMetrics.tsx', {
    react: harness.react,
    './useScopedState': load('../src/hooks/useScopedState.ts', { react: harness.react }),
    './useCommittedRef': load('../src/hooks/useCommittedRef.ts', { react: harness.react }),
    './usePollLoop': { __esModule: true, default: callback => polls.push(callback) },
    '@/utils/api': { apiClient: { get(_url, options) {
      return new Promise(resolve => requests.push({ options, resolve: data => resolve({ data }) }));
    } } },
  }).default;
  const render = () => harness.render(() => hook('metric-fixture', 'running', { attemptId: 'attempt-1' }));
  return {
    requests, render,
    async read(payload, { incremental = false } = {}) {
      const value = render();
      const request = incremental ? polls.at(-1)(new AbortController().signal) : value.refreshMetrics();
      requests.at(-1).resolve(payload);
      await request;
      return render();
    },
    poll: () => { render(); return polls.at(-1)(new AbortController().signal); },
  };
}

const numeric = step => ({ step, value: 0.25 });
const payload = step => ({
  keys: ['loss/loss'],
  series: { 'loss/loss': { latest: numeric(step), points: [numeric(step)] } },
});

test('malformed successful metrics refresh preserves cached data and allows a valid manual retry', async t => {
  const f = fixture(t);
  const cached = await f.read(payload(10));
  const failed = await f.read({ keys: { malformed: true }, keyInfo: [], series: {} });
  assert.equal(failed.status, 'error');
  assert.deepEqual(failed.keys, cached.keys);
  assert.deepEqual(failed.series, cached.series);
  assert.deepEqual(failed.latest, cached.latest);
  assert.equal(failed.version, cached.version);
  const recovered = await f.read(payload(20));
  assert.equal(recovered.status, 'success');
  assert.deepEqual(recovered.series['loss/loss'], [numeric(20)]);
  assert.equal(recovered.version, cached.version + 1);
});

const malformed = [
  ['null response', null],
  ['array response', []],
  ['missing keys and series', {}],
  ['non-string metric key', { ...payload(20), keys: [23] }],
  ['prototype setter metric key', { keys: ['__proto__'], series: {} }],
  ['prototype setter series key', JSON.parse('{"keys":["loss/loss"],"series":{"__proto__":{"points":[{"step":20,"value":0.2}]}}}')],
  ['inherited constructor series key', { keys: [], series: { constructor: {} } }],
  ['inherited toString series key', { keys: [], series: { toString: { points: [] } } }],
  ['null series map', { ...payload(20), series: null }],
  ['array series map', { ...payload(20), series: [] }],
  ['null series payload', { ...payload(20), series: { 'loss/loss': null } }],
  ['non-array points', { ...payload(20), series: { 'loss/loss': { points: { step: 20 } } } }],
  ['null point', { ...payload(20), series: { 'loss/loss': { points: [null] } } }],
  ['string step', { ...payload(20), series: { 'loss/loss': { points: [{ step: '20', value: 0.2 }] } } }],
  ['nonfinite step', { ...payload(20), series: { 'loss/loss': { points: [{ step: Infinity, value: 0.2 }] } } }],
  ['string numeric value', { ...payload(20), series: { 'loss/loss': { points: [{ step: 20, value: '0.2' }] } } }],
  ['nonfinite numeric value', { ...payload(20), series: { 'loss/loss': { points: [{ step: 20, value: NaN }] } } }],
  ['invalid text value', { ...payload(20), series: { 'loss/loss': { points: [{ step: 20, value: null, value_text: {} }] } } }],
  ['invalid timestamp', { ...payload(20), series: { 'loss/loss': { points: [{ step: 20, value: 0.2, wall_time: 'now' }] } } }],
  ['invalid latest point', { ...payload(20), series: { 'loss/loss': { points: [], latest: { step: null, value: 0.2 } } } }],
];

for (const [name, response] of malformed) {
  test(`rejects ${name} before publishing metric state`, async t => {
    const f = fixture(t);
    const cached = await f.read(payload(10));
    const failed = await f.read(response);
    assert.equal(failed.status, 'error');
    assert.deepEqual(failed.keys, cached.keys);
    assert.deepEqual(failed.series, cached.series);
    assert.deepEqual(failed.latest, cached.latest);
    assert.equal(failed.version, cached.version);
  });
}

test('a malformed later series cannot partially advance earlier cursors or latest readings', async t => {
  const f = fixture(t);
  const cached = await f.read(payload(10));
  const failed = await f.read({
    keys: ['loss/loss', 'phase/name'],
    series: {
      'loss/loss': { points: [numeric(999)], latest: numeric(999) },
      'phase/name': { points: [null], latest: null },
    },
  }, { incremental: true });
  assert.equal(failed.status, 'error');
  assert.deepEqual(failed.latest, cached.latest);
  assert.deepEqual(failed.series, cached.series);
  const next = f.poll();
  assert.equal(f.requests.at(-1).options.params.since_steps, 'loss%2Floss:10');
  f.requests.at(-1).resolve(payload(11));
  await next;
  assert.deepEqual(f.render().series['loss/loss'], [numeric(10), numeric(11)]);
});

test('accepts numeric, null and textual metrics without requiring unused metadata', async t => {
  const f = fixture(t);
  const name = { step: 10, value: null, value_text: 'Warmup', wall_time: 1700000000 };
  const missing = { step: 10, value: null, value_text: null };
  const state = await f.read({
    keys: ['loss/loss', 'phase/name', 'event/save', 'train/custom'],
    series: {
      'loss/loss': { points: [numeric(10)], latest: numeric(10) },
      'phase/name': { points: [name], latest: name },
      'event/save': { points: [missing], latest: null },
      'train/custom': {},
    },
  });
  assert.equal(state.status, 'success');
  assert.deepEqual(state.lossKeys, ['loss/loss', 'train/custom']);
  assert.deepEqual(state.eventKeys, ['event/save']);
  assert.deepEqual(state.phaseNamePoints, [name]);
  assert.deepEqual(state.series['event/save'], [missing]);
  assert.deepEqual(state.series['train/custom'], []);
  assert.equal(state.latest['train/custom'], null);
  const empty = await f.read({ keys: [], series: {} });
  assert.equal(empty.status, 'success');
  assert.deepEqual(empty.series, {});
});

test('an invalid initial response stays retryable without publishing partial keys', async t => {
  const f = fixture(t);
  const failed = await f.read({ keys: ['loss/loss'], series: { 'loss/loss': { points: [null] } } });
  assert.equal(failed.status, 'error');
  assert.deepEqual(failed.keys, []);
  assert.deepEqual(failed.series, {});
  const recovered = await f.read(payload(1));
  assert.equal(recovered.status, 'success');
  assert.deepEqual(recovered.series['loss/loss'], [numeric(1)]);
});
