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
  const hook = load('../src/hooks/useJobByRef.tsx', {
    react: harness.react,
    './useScopedState': load('../src/hooks/useScopedState.ts', { react: harness.react }),
    './useCommittedRef': load('../src/hooks/useCommittedRef.ts', { react: harness.react }),
    './usePollLoop': { __esModule: true, default: callback => polls.push(callback) },
    '@/utils/api': { apiClient: { get(url, options) {
      return new Promise((resolve, reject) => requests.push({ url, options, reject, resolve: data => resolve({ data }) }));
    } } },
  }).default;
  const render = (ref = 'fixture/dataset', type = 'caption') => harness.render(() => hook(ref, 5000, type));
  return {
    render, requests, unmount: () => harness.cleanup(),
    poll: (signal = new AbortController().signal) => { render(); return polls.at(-1)(signal); },
  };
}

const job = (status, step = 0) => ({ id: 'caption-fixture', job_type: 'caption', status, step, gpu_ids: '0' });

for (const [stale, current] of [[null, job('running')], [job('running', 4), job('stopped', 6)]]) {
  test(`a pre-mutation reference poll cannot replace refreshed ${current.status} job state`, async t => {
    const f = fixture(t);
    const oldPoll = f.poll();
    const refresh = f.render().refreshJob();
    f.requests[1].resolve(current);
    await refresh;
    assert.deepEqual(f.render().job, current);
    f.requests[0].resolve(stale);
    await oldPoll;
    assert.deepEqual(f.render().job, current);
    assert.equal(f.render().status, 'success');
  });
}

test('an older reference poll failure cannot erase a newly refreshed running job', async t => {
  const f = fixture(t);
  const oldPoll = f.poll();
  const refresh = f.render().refreshJob();
  f.requests[1].resolve(job('running'));
  await refresh;
  f.requests[0].reject(new Error('Synthetic old request failure'));
  await oldPoll;
  assert.equal(f.render().job?.status, 'running');
  assert.equal(f.render().status, 'success');
});

test('current reference refresh failure retains known job state until a successful retry reports absence', async t => {
  const f = fixture(t);
  const initial = f.render().refreshJob();
  f.requests[0].resolve(job('running'));
  await initial;
  const failed = f.render().refreshJob();
  f.requests[1].reject(new Error('Synthetic current request failure'));
  await failed;
  assert.equal(f.render().job?.status, 'running');
  assert.equal(f.render().status, 'error');
  const retry = f.render().refreshJob();
  f.requests[2].resolve(null);
  await retry;
  assert.equal(f.render().job, null);
  assert.equal(f.render().status, 'success');
});

test('old scope callbacks cannot start network requests after switching dataset or job type', async t => {
  const f = fixture(t);
  const fromDataset = f.render().refreshJob;
  const fromType = f.render('fixture/other', 'caption').refreshJob;
  f.render('fixture/other', 'train');
  const oldRequests = [fromDataset(), fromType()];
  const requestCount = f.requests.length;
  f.requests.forEach(request => request.resolve(null));
  await Promise.all(oldRequests);
  assert.equal(requestCount, 0);
});

test('already-aborted reference polls do not invalidate a pending manual refresh', async t => {
  const f = fixture(t);
  const refresh = f.render().refreshJob();
  const controller = new AbortController();
  controller.abort();
  const abortedPoll = f.poll(controller.signal);
  const requestCount = f.requests.length;
  f.requests.slice(1).forEach(request => request.resolve(null));
  await abortedPoll;
  f.requests[0].resolve(job('running'));
  await refresh;
  assert.equal(requestCount, 1);
  assert.equal(f.render().job?.status, 'running');
  assert.equal(f.render().status, 'success');
});

test('returning to a prior reference does not accept a response from its previous visit', async t => {
  const f = fixture(t);
  const previous = f.render('fixture/first').refreshJob();
  f.render('fixture/second');
  assert.equal(f.render('fixture/first').job, null);
  f.requests[0].resolve(job('running'));
  await previous;
  assert.equal(f.render('fixture/first').job, null);
  assert.equal(f.render('fixture/first').status, 'idle');
});

test('a captured prior-visit callback cannot invalidate a new request after returning to the same reference', async t => {
  const f = fixture(t);
  const previousRefresh = f.render('fixture/first').refreshJob;
  f.render('fixture/second');
  const current = f.render('fixture/first').refreshJob();
  const previous = previousRefresh();
  const requestCount = f.requests.length;
  f.requests.slice(1).forEach(request => request.resolve(job('stopped')));
  await previous;
  f.requests[0].resolve(job('running'));
  await current;
  assert.equal(requestCount, 1);
  assert.equal(f.render('fixture/first').job?.status, 'running');
  assert.equal(f.render('fixture/first').status, 'success');
});

test('captured reference refresh callback cannot start a new request after unmount', async t => {
  const f = fixture(t);
  const previous = f.render().refreshJob;
  f.unmount();
  const pending = previous();
  const requestCount = f.requests.length;
  f.requests.forEach(request => request.resolve(job('running')));
  await pending;
  assert.equal(requestCount, 0);
});

test('missing reference is idle and valid requests preserve query encoding and job type', async t => {
  const f = fixture(t);
  await f.render(null).refreshJob();
  assert.equal(f.requests.length, 0);
  assert.equal(f.render(null).status, 'idle');
  const refresh = f.render('fixture/path & one', 'caption').refreshJob();
  const url = new URL(f.requests[0].url, 'http://fixture.invalid');
  assert.equal(url.searchParams.get('job_ref'), 'fixture/path & one');
  assert.equal(url.searchParams.get('job_type'), 'caption');
  f.requests[0].resolve(job('completed'));
  await refresh;
  assert.equal(f.render('fixture/path & one', 'caption').job.status, 'completed');
});
