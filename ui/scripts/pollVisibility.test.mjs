import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { createHookHarness } from './reactHookHarness.mjs';

function compile(relative) {
  return ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
}
const source = compile('../src/utils/basic.ts');
const settle = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
function fixture(initial = 'visible') {
  let now = 0, timerID = 0;
  const timers = new Map(), listeners = new Set(), errors = [];
  const visibility = {
    visibilityState: initial,
    addEventListener(type, callback) { assert.equal(type, 'visibilitychange'); listeners.add(callback); },
    removeEventListener(type, callback) { assert.equal(type, 'visibilitychange'); listeners.delete(callback); },
  };
  const module = { exports: {} };
  new Function('module', 'exports', 'setTimeout', 'clearTimeout', source)(module, module.exports,
    (callback, delay) => { const id = ++timerID; timers.set(id, { at: now + delay, callback }); return id; },
    id => timers.delete(id));
  return {
    visibility, errors, timers, listeners, basic: module.exports,
    start(callback, interval, visible = true) {
      return module.exports.startSettledPollLoop(callback, interval, error => errors.push(error), visible ? visibility : undefined);
    },
    change(state) { visibility.visibilityState = state; for (const callback of listeners) callback(); },
    async advance(duration) {
      await settle();
      const end = now + duration;
      for (;;) {
        const next = [...timers.entries()].sort((a, b) => a[1].at - b[1].at)[0];
        if (!next || next[1].at > end) break;
        const [id, timer] = next;
        now = timer.at; timers.delete(id); timer.callback(); await settle();
      }
      now = end;
    },
  };
}

test('hidden views make no scheduled requests and foreground refreshes immediately', async () => {
  const f = fixture();
  let count = 0;
  const stop = f.start(() => { count++; }, 2000);
  await f.advance(6000);
  assert.equal(count, 4);
  f.change('hidden');
  await f.advance(600_000);
  assert.equal(count, 4);
  assert.equal(f.timers.size, 0);
  f.change('visible');
  assert.equal(count, 5);
  f.change('visible');
  assert.equal(count, 5, 'duplicate events do not add requests');
  await f.advance(2000);
  assert.equal(count, 6);
  stop();
  assert.equal(f.listeners.size, 0);
  assert.equal(f.timers.size, 0);
});

test('foreground refresh waits for aborted transport settlement without overlap', async () => {
  const f = fixture();
  const requests = [];
  const stop = f.start(signal => new Promise(resolve => requests.push({ signal, resolve })), 2000);
  f.change('hidden');
  assert.equal(requests[0].signal.aborted, true);
  f.change('visible');
  await f.advance(10_000);
  assert.equal(requests.length, 1);
  requests[0].resolve();
  await settle();
  assert.equal(requests.length, 2);
  assert.equal(requests[1].signal.aborted, false);
  stop();
  assert.equal(requests[1].signal.aborted, true);
  requests[1].resolve();
  await f.advance(10_000);
  assert.equal(requests.length, 2);
});

test('initial hidden and one-shot views defer loading, then refresh once per foreground visit', async () => {
  const f = fixture('hidden');
  let count = 0;
  const stop = f.start(() => { count++; }, null);
  await f.advance(600_000);
  assert.equal(count, 0);
  f.change('visible');
  await f.advance(600_000);
  assert.equal(count, 1);
  f.change('hidden'); f.change('visible');
  await settle();
  assert.equal(count, 2);
  stop();
});

test('errors keep retry behavior, canceled failures remain quiet, and disposed scopes cannot restart', async () => {
  const f = fixture();
  const requests = [];
  const stop = f.start(signal => new Promise((resolve, reject) => requests.push({ signal, resolve, reject })), 1000);
  requests[0].reject(new Error('offline'));
  await f.advance(1000);
  assert.equal(f.errors.length, 1);
  assert.equal(requests.length, 2);
  f.change('hidden');
  requests[1].reject(new Error('aborted'));
  await settle();
  assert.equal(f.errors.length, 1);
  stop();
  f.change('visible');
  await f.advance(5000);
  assert.equal(requests.length, 2);
});

test('visibility policy is frontend opt-in; background loops retain their cadence', async () => {
  const f = fixture('hidden');
  let count = 0;
  const stop = f.start(() => { count++; }, 1000, false);
  await f.advance(3000);
  assert.equal(count, 4);
  assert.equal(f.listeners.size, 0);
  stop();
});

test('usePollLoop preserves callback freshness, scope cancellation and visibility listener cleanup', async t => {
  const f = fixture(), harness = createHookHarness(), calls = [];
  const load = (relative, dependencies) => {
    const module = { exports: {} };
    new Function('require', 'module', 'exports', 'document', compile(relative))(
      name => { assert.ok(Object.hasOwn(dependencies, name)); return dependencies[name]; }, module, module.exports, f.visibility);
    return module.exports;
  };
  const usePollLoop = load('../src/hooks/usePollLoop.tsx', {
    react: harness.react,
    './useCommittedRef': load('../src/hooks/useCommittedRef.ts', { react: harness.react }),
    '@/utils/basic': f.basic,
  }).default;
  const render = (scope, value) => harness.render(() => usePollLoop(signal => {
    calls.push({ scope, value, signal });
  }, 1000, [scope]));
  t.after(() => harness.cleanup());
  render('a', 1); await settle();
  render('a', 2); await f.advance(1000);
  assert.deepEqual(calls.map(call => call.value), [1, 2]);
  f.change('hidden');
  render('b', 3);
  assert.equal(f.listeners.size, 1);
  await f.advance(1000);
  assert.equal(calls.length, 2);
  f.change('visible'); await settle();
  assert.equal(calls.at(-1).scope, 'b');
  harness.cleanup();
  assert.equal(f.listeners.size, 0);
  f.change('hidden'); f.change('visible');
  await f.advance(1000);
  assert.equal(calls.length, 3);
});

test('ten-minute request-count benchmark for dashboard and selected-job polling', async t => {
  const intervals = [5000, 5000, 15000, 5000, 3000, 2000];
  const before = fixture(), after = fixture();
  let baseline = 0, optimized = 0;
  const stops = intervals.flatMap(interval => [
    before.start(() => { baseline++; }, interval, false),
    after.start(() => { optimized++; }, interval),
  ]);
  await settle();
  before.change('hidden'); after.change('hidden');
  await before.advance(600_000); await after.advance(600_000);
  assert.equal(baseline, 906);
  assert.equal(optimized, 6);
  after.change('visible'); await settle();
  assert.equal(optimized, 12, 'each visible consumer receives one immediate refresh');
  stops.forEach(stop => stop());
  t.diagnostic('Deterministic 10-minute hidden window, six mounted consumers: 906 baseline requests vs 6 initial requests (900 hidden requests avoided), then 6 foreground refreshes. Counts, not network latency or training throughput.');
});

test('collapsed output pauses log fetching, preserves offsets, and ignores canceled responses on reopening', async t => {
  const f = fixture(), harness = createHookHarness(), requests = [];
  const load = (relative, dependencies) => {
    const module = { exports: {} };
    new Function('require', 'module', 'exports', 'document', compile(relative))(
      name => { assert.ok(Object.hasOwn(dependencies, name), name); return dependencies[name]; }, module, module.exports, f.visibility);
    return module.exports;
  };
  const committed = load('../src/hooks/useCommittedRef.ts', { react: harness.react });
  const usePollLoop = load('../src/hooks/usePollLoop.tsx', { react: harness.react, './useCommittedRef': committed, '@/utils/basic': f.basic });
  const useJobLog = load('../src/hooks/useJobLog.tsx', {
    react: harness.react,
    './useCommittedRef': committed,
    './useScopedState': load('../src/hooks/useScopedState.ts', { react: harness.react }),
    './usePollLoop': usePollLoop,
    '@/utils/terminalEmulator': load('../src/utils/terminalEmulator.ts', {}),
    '@/utils/api': { apiClient: { get(url, options) {
      return new Promise(resolve => requests.push({ url, options, resolve: data => resolve({ data }) }));
    } } },
  }).default;
  const render = enabled => harness.render(() => useJobLog('job-a', 1000, 'attempt-a', enabled));
  t.after(() => harness.cleanup());
  render(false); await f.advance(10_000);
  assert.equal(requests.length, 0);
  render(true);
  assert.equal(requests.length, 1);
  requests[0].resolve({ log: 'hello', reset: true, offset: 5 });
  await settle();
  assert.equal(render(true).log, 'hello');
  await f.advance(1000);
  assert.equal(requests.length, 2);
  render(false);
  assert.equal(requests[1].options.signal.aborted, true);
  await f.advance(10_000);
  assert.equal(requests.length, 2);
  assert.equal(render(false).log, 'hello');
  render(true);
  assert.equal(requests.length, 3);
  assert.equal(requests[2].options.params.offset, 5);
  requests[1].resolve({ log: ' stale', reset: false, offset: 999 });
  requests[2].resolve({ log: ' world', reset: false, offset: 11 });
  await settle();
  assert.equal(render(true).log, 'hello world');
});
