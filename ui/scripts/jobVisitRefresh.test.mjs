import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { createHookHarness } from './reactHookHarness.mjs';

function load(relative, dependencies) {
  const source = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const loaded = { exports: {} };
  new Function('require', 'module', 'exports', 'console', source)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
    return dependencies[name];
  }, loaded, loaded.exports, { error() {} });
  return loaded.exports;
}

function fixture(t) {
  const harness = createHookHarness(), requests = [], polls = [];
  t.after(() => harness.cleanup());
  const hook = load('../src/hooks/useJob.tsx', {
    react: harness.react,
    './useScopedState': load('../src/hooks/useScopedState.ts', { react: harness.react }),
    './useCommittedRef': load('../src/hooks/useCommittedRef.ts', { react: harness.react }),
    './usePollLoop': { __esModule: true, default: callback => polls.push(callback) },
    '@/utils/api': { apiClient: { get(url, options) {
      return new Promise((resolve, reject) => requests.push({ url, options, reject, resolve: data => resolve({ data }) }));
    } } },
  }).default;
  const render = (jobID = 'job-a') => harness.render(() => hook(jobID, 5000));
  return {
    render, requests, unmount: () => harness.cleanup(),
    poll: signal => { render(); return polls.at(-1)(signal); },
  };
}

for (const outcome of ['success', 'failure']) {
  test(`old job visit callback cannot invalidate current refresh through a stale ${outcome} after A/B/A`, async t => {
    const f = fixture(t);
    const previous = f.render().refreshJob;
    f.render('job-b');
    const current = f.render().refreshJob();
    const old = previous();
    const count = f.requests.length;
    f.requests.slice(1).forEach(request => outcome === 'success'
      ? request.resolve({ id: 'job-a', status: 'stopped', step: 0 })
      : request.reject(new Error('Synthetic old visit failure')));
    await old;
    f.requests[0].resolve({ id: 'job-a', status: 'running', step: 20 });
    await current;
    assert.equal(count, 1);
    assert.deepEqual(f.render().job, { id: 'job-a', status: 'running', step: 20 });
    assert.equal(f.render().status, 'success');
  });
}

test('captured job callback cannot start requests after unmount', async t => {
  const f = fixture(t);
  const previous = f.render().refreshJob;
  f.unmount();
  const pending = previous();
  const count = f.requests.length;
  f.requests.forEach(request => request.resolve({ id: 'job-a' }));
  await pending;
  assert.equal(count, 0);
});

test('already-aborted job poll cannot take ownership from a manual refresh', async t => {
  const f = fixture(t);
  const current = f.render().refreshJob();
  const controller = new AbortController();
  controller.abort();
  const old = f.poll(controller.signal);
  const count = f.requests.length;
  f.requests.slice(1).forEach(request => request.resolve({ id: 'job-a', step: 0 }));
  await old;
  f.requests[0].resolve({ id: 'job-a', step: 20 });
  await current;
  assert.equal(count, 1);
  assert.equal(f.render().job.step, 20);
  assert.equal(f.render().status, 'success');
});

test('current job refresh failures retain cached data and remain retryable within the visit', async t => {
  const f = fixture(t);
  const initial = f.render().refreshJob();
  f.requests[0].resolve({ id: 'job-a', step: 20 });
  await initial;
  const failed = f.render().refreshJob();
  f.requests[1].reject(new Error('Synthetic current failure'));
  await failed;
  assert.equal(f.render().job.step, 20);
  assert.equal(f.render().status, 'error');
  const retry = f.render().refreshJob();
  f.requests[2].resolve({ id: 'job-a', step: 21 });
  await retry;
  assert.equal(f.render().job.step, 21);
  assert.equal(f.render().status, 'success');
});

test('empty job identity remains idle and does not send a request', async t => {
  const f = fixture(t);
  await f.render('').refreshJob();
  assert.equal(f.requests.length, 0);
  assert.equal(f.render('').status, 'idle');
});
