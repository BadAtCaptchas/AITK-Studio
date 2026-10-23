import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { createHookHarness } from './reactHookHarness.mjs';

function load(relative, dependencies) {
  const source = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const evaluatedModule = { exports: {} };
  new Function('require', 'module', 'exports', source)(
    name => {
      assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
      return dependencies[name];
    },
    evaluatedModule,
    evaluatedModule.exports,
  );
  return evaluatedModule.exports;
}

function fixture(t, name) {
  const harness = createHookHarness();
  t.after(() => harness.cleanup());
  const requests = [],
    polls = [];
  const dependencies = {
    react: harness.react,
    './useScopedState': load('../src/hooks/useScopedState.ts', { react: harness.react }),
    './useCommittedRef': load('../src/hooks/useCommittedRef.ts', { react: harness.react }),
    './usePollLoop': { default: (callback, interval) => polls.push({ callback, interval }), __esModule: true },
    '@/utils/api': {
      apiClient: {
        get(url, options) {
          let resolve, reject;
          const promise = new Promise((a, b) => {
            resolve = a;
            reject = b;
          });
          requests.push({ url, options, resolve, reject });
          return promise;
        },
      },
    },
    '@/utils/terminalEmulator': load('../src/utils/terminalEmulator.ts', {}),
  };
  const hook = load(`../src/hooks/${name}.tsx`, dependencies).default;
  return { requests, polls, render: (...args) => harness.render(() => hook(...args)) };
}

const logResponse = text => ({ data: { log: text, reset: false, offset: text.length } });
const metricsResponse = step => ({
  data: {
    keys: ['loss/loss'],
    keyInfo: [],
    series: {
      'loss/loss': {
        key: 'loss/loss',
        totalCount: 1,
        firstStep: step,
        lastStep: step,
        latest: { step, value: 0.25 },
        points: [{ step, value: 0.25 }],
        downsampled: false,
      },
    },
  },
});

test('manual log refresh from A cannot contaminate A after selecting B and returning', async t => {
  const f = fixture(t, 'useJobLog');
  const older = f.render('a').refresh();
  f.render('b');
  const current = f.render('a').refresh();
  f.requests[0].resolve(logResponse('obsolete A\n'));
  await older;
  f.requests[1].resolve(logResponse('current A\n'));
  await current;
  assert.equal(f.render('a').log, 'current A\n');
});

test('canceling an active poll does not suppress the final one-shot log fetch', async t => {
  const f = fixture(t, 'useJobLog');
  f.render('a', 5000);
  const signal = new AbortController();
  const previous = f.polls.at(-1).callback(signal.signal);
  signal.abort();
  f.render('a', null);
  const final = f.polls.at(-1).callback(new AbortController().signal);
  assert.equal(f.requests.length, 2);
  f.requests[0].resolve(logResponse('outdated\n'));
  await previous;
  f.requests[1].resolve(logResponse('Training complete\n'));
  await final;
  assert.equal(f.render('a', null).log, 'Training complete\n');
});

test('a new job attempt clears log text and byte offset while ignoring its previous response', async t => {
  const f = fixture(t, 'useJobLog');
  const first = f.render('a', 5000, 'old-attempt').refresh();
  f.requests[0].resolve(logResponse('old run\n'));
  await first;
  const pending = f.render('a', 5000, 'old-attempt').refresh();
  assert.ok(f.requests[1].options.params.offset > 0);
  const next = f.render('a', 5000, 'new-attempt');
  assert.equal(next.log, '');
  const fresh = next.refresh();
  assert.equal(f.requests[2].options.params, undefined);
  f.requests[1].resolve(logResponse('late old tail\n'));
  await pending;
  f.requests[2].resolve(logResponse('new run\n'));
  await fresh;
  assert.equal(f.render('a', 5000, 'new-attempt').log, 'new run\n');
});

test('a restarted metric stream drops old points/cursors and ignores old attempt responses', async t => {
  const f = fixture(t, 'useJobMetrics');
  f.render('a', 'running', { attemptId: 'old-attempt' });
  const first = f.polls.at(-1).callback(new AbortController().signal);
  f.requests[0].resolve(metricsResponse(1500));
  await first;
  f.render('a', 'running', { attemptId: 'old-attempt' });
  const pending = f.polls.at(-1).callback(new AbortController().signal);
  assert.equal(f.requests[1].options.params.since_steps, 'loss%2Floss:1500');
  const next = f.render('a', 'running', { attemptId: 'new-attempt' });
  assert.deepEqual(next.series, {});
  const fresh = f.polls.at(-1).callback(new AbortController().signal);
  assert.equal(f.requests[2].options.params.since_steps, undefined);
  f.requests[1].resolve(metricsResponse(1600));
  await pending;
  f.requests[2].resolve(metricsResponse(0));
  await fresh;
  assert.deepEqual(f.render('a', 'running', { attemptId: 'new-attempt' }).series['loss/loss'], [
    { step: 0, value: 0.25 },
  ]);
});

test('refreshing a page keeps its new job data ahead of duplicate rows from older cached pages', async t => {
  const f = fixture(t, 'useJobsList');
  const page = (jobs, nextCursor) => ({ data: { jobs, nextCursor, statusCounts: { queued: 2 } } });
  const olderJob = { id: 'moving', step: 10, status: 'queued', updated_at: '2026-10-23T10:00:00Z' };
  const newerJob = { ...olderJob, step: 50, status: 'running', updated_at: '2026-10-23T10:01:00Z' };
  const first = f.render({ view: 'all' }).refreshJobs();
  f.requests[0].resolve(page([{ id: 'removed', step: 0 }], 'next'));
  await first;
  const more = f.render({ view: 'all' }).loadMore();
  f.requests[1].resolve(page([olderJob], null));
  await more;
  const refresh = f.render({ view: 'all' }).refreshJobs();
  f.requests[2].resolve(page([newerJob], 'next'));
  await refresh;
  assert.equal(f.render({ view: 'all' }).jobs.find(job => job.id === 'moving').step, 50);
});
