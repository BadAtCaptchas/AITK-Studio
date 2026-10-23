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

function fixture(t, kind) {
  const harness = createHookHarness(), requests = [], polls = [];
  t.after(() => harness.cleanup());
  const dependencies = {
    react: harness.react,
    './useScopedState': load('../src/hooks/useScopedState.ts', { react: harness.react }),
    './useCommittedRef': load('../src/hooks/useCommittedRef.ts', { react: harness.react }),
    './usePollLoop': { __esModule: true, default: (callback, interval) => polls.push({ callback, interval }) },
    '@/utils/api': { apiClient: { get(url, options) {
      return new Promise((resolve, reject) => requests.push({ url, options, reject, resolve: progress => resolve({ data: { progress } }) }));
    } } },
  };
  const hook = load(`../src/hooks/useJob${kind}Progress.tsx`, {
    ...dependencies,
    './useJobProgress': load('../src/hooks/useJobProgress.ts', dependencies),
  }).default;
  let initial = null, jobId = 'fixture-job', interval = 2000;
  const render = () => harness.render(() => hook(jobId, initial, interval));
  return {
    requests, render,
    initial: value => { initial = value; return render(); },
    job: value => { jobId = value; return render(); },
    poll: (signal = new AbortController().signal) => { render(); return polls.at(-1).callback(signal); },
    interval: () => { render(); return polls.at(-1).interval; },
    setInterval: value => { interval = value; return render(); },
    unmount: () => harness.cleanup(),
  };
}

function progress(kind, second, status = kind === 'Download' ? 'downloading' : 'installing') {
  const updatedAt = `2026-10-23T12:00:${String(second).padStart(2, '0')}.000Z`;
  const common = { version: 1, status, message: 'Fixture progress', percent: second, error: null, updatedAt };
  return kind === 'Download'
    ? { ...common, fileName: 'fixture.bin', activeCount: 1, bytesDownloaded: second, bytesTotal: 100, downloads: [] }
    : { ...common, step: 'install', root: null, startedAt: '2026-10-23T12:00:00.000Z' };
}

for (const kind of ['Download', 'ComfyInstall']) {
  test(`${kind} incorporates fresh job-provided progress before the next poll`, t => {
    const f = fixture(t, kind);
    f.initial(progress(kind, 1));
    assert.deepEqual(f.initial(progress(kind, 5)).progress, progress(kind, 5));
  });

  test(`${kind} a pending poll cannot regress a newer job-provided snapshot`, async t => {
    const f = fixture(t, kind);
    f.initial(progress(kind, 1));
    const pending = f.poll();
    f.initial(progress(kind, 5));
    f.requests[0].resolve(progress(kind, 3));
    await pending;
    assert.deepEqual(f.render().progress, progress(kind, 5));
  });

  test(`${kind} a fresh job response clears expired progress without resurrecting it from a late poll`, async t => {
    const f = fixture(t, kind);
    f.initial(progress(kind, 1, 'completed'));
    const pending = f.poll();
    const expired = f.initial(null).progress;
    f.requests[0].resolve(progress(kind, 1, 'completed'));
    await pending;
    assert.equal(expired, null);
    assert.equal(f.render().progress, null);
  });

  test(`${kind} a late older poll cannot replace a newer refresh`, async t => {
    const f = fixture(t, kind);
    const old = f.poll();
    const current = f.render().refreshProgress();
    f.requests[1].resolve(progress(kind, 5));
    await current;
    f.requests[0].resolve(progress(kind, 1));
    await old;
    assert.deepEqual(f.render().progress, progress(kind, 5));
  });

  test(`${kind} job list responses older than a successful poll do not regress progress`, async t => {
    const f = fixture(t, kind);
    f.initial(progress(kind, 1));
    const request = f.poll();
    f.requests[0].resolve(progress(kind, 5));
    await request;
    assert.deepEqual(f.initial(progress(kind, 3)).progress, progress(kind, 5));
  });

  test(`${kind} a late empty poll cannot clear a fresh job snapshot`, async t => {
    const f = fixture(t, kind);
    f.initial(progress(kind, 1));
    const pending = f.poll();
    f.initial(progress(kind, 5));
    f.requests[0].resolve(null);
    await pending;
    assert.deepEqual(f.render().progress, progress(kind, 5));
    assert.equal(f.interval(), 2000);
  });

  test(`${kind} expiration blocks cached snapshots but allows a newer download or install`, async t => {
    const f = fixture(t, kind);
    f.initial(progress(kind, 5, 'completed'));
    f.initial(null);
    const stale = f.poll();
    f.requests[0].resolve(progress(kind, 5, 'completed'));
    await stale;
    const expired = f.render().progress;
    const next = { ...progress(kind, 6), percent: 0 };
    const fresh = f.poll();
    f.requests[1].resolve(next);
    await fresh;
    assert.equal(expired, null);
    assert.deepEqual(f.render().progress, next);
    assert.equal(f.interval(), 2000);
  });

  test(`${kind} a new job snapshot restarts stopped progress polling`, async t => {
    const f = fixture(t, kind);
    f.initial(progress(kind, 1, 'completed'));
    const expired = f.poll();
    f.requests[0].resolve(null);
    await expired;
    assert.equal(f.interval(), null);
    const next = { ...progress(kind, 5), percent: 0 };
    assert.deepEqual(f.initial(next).progress, next);
    assert.equal(f.interval(), 2000);
  });

  test(`${kind} keeps the ten-empty-poll limit until genuinely fresh progress arrives`, async t => {
    const f = fixture(t, kind);
    for (let i = 0; i < 10; i++) {
      const pending = f.poll();
      f.requests[i].resolve(null);
      await pending;
      assert.equal(f.interval(), i === 9 ? null : 2000);
    }
    f.initial(progress(kind, 5));
    assert.equal(f.interval(), 2000);
    const expired = f.poll();
    f.requests[10].resolve(null);
    await expired;
    assert.equal(f.interval(), null);
    assert.equal(f.render().progress, null);
    // Repeated job responses carrying the same old snapshot cannot resurrect it.
    f.initial(progress(kind, 5));
    assert.equal(f.render().progress, null);
    assert.equal(f.interval(), null);
  });

  test(`${kind} stale refresh callbacks cannot supersede requests after A to B to A`, async t => {
    const f = fixture(t, kind);
    const previous = f.render().refreshProgress;
    f.job('other-job');
    f.job('fixture-job');
    const current = f.render().refreshProgress();
    const old = previous();
    const requestCount = f.requests.length;
    // Settle even an incorrectly issued stale request before assertions.
    for (const request of f.requests.slice(1)) request.resolve(progress(kind, 1));
    await old;
    f.requests[0].resolve(progress(kind, 5));
    await current;
    assert.equal(requestCount, 1);
    assert.deepEqual(f.render().progress, progress(kind, 5));
  });

  test(`${kind} abort and unmount suppress requests and late responses`, async t => {
    const f = fixture(t, kind);
    const abort = new AbortController();
    abort.abort();
    const aborted = f.poll(abort.signal);
    const abortedCount = f.requests.length;
    for (const request of f.requests) request.resolve(progress(kind, 1));
    await aborted;
    const refresh = f.render().refreshProgress;
    const pending = refresh();
    f.unmount();
    const afterUnmount = refresh();
    const requestCount = f.requests.length;
    for (const request of f.requests) request.resolve(progress(kind, 5));
    await Promise.all([pending, afterUnmount]);
    assert.equal(abortedCount, 0);
    assert.equal(requestCount, 1);
    assert.equal(f.render().progress, null);
  });

  test(`${kind} an earlier failed poll cannot hide a successful newer refresh`, async t => {
    const f = fixture(t, kind);
    const old = f.poll();
    const current = f.render().refreshProgress();
    f.requests[1].resolve(progress(kind, 5));
    await current;
    f.requests[0].reject(new Error('Fixture stale transport failure'));
    await old;
    assert.equal(f.render().status, 'success');
    assert.deepEqual(f.render().progress, progress(kind, 5));
  });

  test(`${kind} same-timestamp updates remain visible without letting an earlier poll regress them`, async t => {
    const f = fixture(t, kind);
    f.initial(progress(kind, 5));
    const earlier = f.poll();
    const fromJob = { ...progress(kind, 5), percent: 20 };
    f.initial(fromJob);
    f.requests[0].resolve(progress(kind, 5));
    await earlier;
    assert.deepEqual(f.render().progress, fromJob);
    const later = f.poll();
    const fromPoll = { ...progress(kind, 5), percent: 30 };
    f.requests[1].resolve(fromPoll);
    await later;
    assert.deepEqual(f.render().progress, fromPoll);
    assert.deepEqual(f.initial(fromJob).progress, fromPoll);
  });

  test(`${kind} a genuinely newer operation returned by an earlier request survives parent expiration`, async t => {
    const f = fixture(t, kind);
    f.initial(progress(kind, 1, 'completed'));
    const pending = f.poll();
    f.initial(null);
    const next = { ...progress(kind, 5), percent: 0 };
    f.requests[0].resolve(next);
    await pending;
    assert.deepEqual(f.render().progress, next);
    assert.equal(f.interval(), 2000);
  });

  test(`${kind} malformed responses retain visible progress and recover on valid data`, async t => {
    const f = fixture(t, kind);
    const current = progress(kind, 5);
    f.initial(current);
    for (const malformed of [undefined, {}, { ...current, updatedAt: 'not-a-time' }, { ...current, percent: {} }]) {
      const pending = f.poll();
      f.requests.at(-1).resolve(malformed);
      await pending;
      assert.deepEqual(f.render().progress, current);
      assert.equal(f.render().status, 'error');
      assert.equal(f.interval(), 2000);
    }
    const pending = f.poll();
    f.requests.at(-1).resolve(progress(kind, 6));
    await pending;
    assert.equal(f.render().status, 'success');
    assert.deepEqual(f.render().progress, progress(kind, 6));
  });

  test(`${kind} a successful older endpoint response settles loading without regressing its initial snapshot`, async t => {
    const f = fixture(t, kind);
    const current = progress(kind, 5);
    f.initial(current);
    const pending = f.poll();
    f.requests[0].resolve(progress(kind, 3));
    await pending;
    assert.equal(f.render().status, 'success');
    assert.deepEqual(f.render().progress, current);
  });
}

test('ComfyInstall terminal snapshots stop polling until a newer active install appears', t => {
  const f = fixture(t, 'ComfyInstall');
  f.initial(progress('ComfyInstall', 1));
  f.initial(progress('ComfyInstall', 5, 'completed'));
  assert.equal(f.interval(), null);
  assert.equal(f.initial(progress('ComfyInstall', 3)).progress.status, 'completed');
  assert.equal(f.interval(), null);
  f.initial({ ...progress('ComfyInstall', 6), startedAt: '2026-10-23T12:00:06.000Z' });
  assert.equal(f.interval(), 2000);
  f.initial(progress('ComfyInstall', 7, 'failed'));
  assert.equal(f.interval(), null);
  f.setInterval(null);
  f.setInterval(2000);
  assert.equal(f.interval(), null);
});

test('Download terminal snapshots continue polling until their server grace period expires', async t => {
  const f = fixture(t, 'Download');
  f.initial(progress('Download', 5, 'completed'));
  assert.equal(f.interval(), 2000);
  const pending = f.poll();
  f.requests[0].resolve(null);
  await pending;
  assert.equal(f.interval(), null);
});
