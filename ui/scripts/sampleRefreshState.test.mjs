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
  const hook = load('../src/hooks/useSampleImages.tsx', {
    react: harness.react,
    './useScopedState': load('../src/hooks/useScopedState.ts', { react: harness.react }),
    './useCommittedRef': load('../src/hooks/useCommittedRef.ts', { react: harness.react }),
    './usePollLoop': { __esModule: true, default: callback => polls.push(callback) },
    '@/utils/sampleMetadata': load('../src/utils/sampleMetadata.ts', { './media': { getDisplayPath: path => path } }),
    '@/utils/api': { apiClient: { get(url, options) {
      return new Promise((resolve, reject) => requests.push({
        url, options, reject, resolve: data => resolve({ data }),
      }));
    } } },
  }).default;
  const render = (jobID = 'sample-fixture', includeMetadata = true) => harness.render(() => hook(jobID, 5000, includeMetadata));
  return {
    requests, render, unmount: () => harness.cleanup(),
    poll: (signal = new AbortController().signal) => { render(); return polls.at(-1)(signal); },
  };
}

const metadata = seed => ({
  version: 1, prompt: 'Synthetic regression sample', negative_prompt: '', seed, width: 512, height: 512,
  guidance_scale: 4, guidance_rescale: 0, sample_steps: 20, network_multiplier: 1, num_frames: 1, fps: 16,
  step: 100, prompt_index: 0, sampler: 'flowmatch', iteration_id: 'synthetic',
});

test('a pre-delete sample poll cannot resurrect images after the post-delete refresh', async t => {
  const f = fixture(t);
  const olderPoll = f.poll();
  const refreshAfterDeletion = f.render().refreshSampleImages();
  const retained = '/synthetic/retained.png', deleted = '/synthetic/deleted.png';
  f.requests[1].resolve({ samples: [retained], metadata: { [retained]: metadata(2) } });
  await refreshAfterDeletion;
  const fresh = f.render();
  assert.deepEqual(fresh.sampleImages, [retained]);

  f.requests[0].resolve({ samples: [deleted, retained], metadata: { [deleted]: metadata(1), [retained]: metadata(0) } });
  await olderPoll;
  const afterStalePoll = f.render();
  assert.deepEqual(afterStalePoll.sampleImages, [retained]);
  assert.deepEqual(afterStalePoll.sampleMetadata, fresh.sampleMetadata);
  assert.equal(afterStalePoll.status, 'success');
});

test('an older sample poll failure cannot replace a successful manual refresh status', async t => {
  const f = fixture(t);
  const olderPoll = f.poll();
  const refresh = f.render().refreshSampleImages();
  f.requests[1].resolve({ samples: ['/synthetic/current.png'] });
  await refresh;
  f.requests[0].reject(new Error('Synthetic stale poll failure'));
  await olderPoll;
  assert.equal(f.render().status, 'success');
  assert.deepEqual(f.render().sampleImages, ['/synthetic/current.png']);
});

test('current sample refresh failures preserve the cache and remain retryable', async t => {
  const f = fixture(t);
  const initial = f.render().refreshSampleImages();
  f.requests[0].resolve({ samples: ['/synthetic/cached.png'], metadata: { '/synthetic/cached.png': metadata(1) } });
  await initial;
  const cached = f.render();
  const failed = cached.refreshSampleImages();
  f.requests[1].reject(new Error('Synthetic current request failure'));
  await failed;
  assert.equal(f.render().status, 'error');
  assert.deepEqual(f.render().sampleImages, cached.sampleImages);
  assert.deepEqual(f.render().sampleMetadata, cached.sampleMetadata);
  const retry = f.render().refreshSampleImages();
  f.requests[2].resolve({ samples: [] });
  await retry;
  assert.equal(f.render().status, 'success');
  assert.deepEqual(f.render().sampleImages, []);
  assert.deepEqual(f.render().sampleMetadata, {});
});

test('aborted sample polls cannot supersede a current manual refresh', async t => {
  const f = fixture(t);
  const aborted = new AbortController();
  aborted.abort();
  await f.poll(aborted.signal);
  assert.equal(f.requests.length, 0);
  const oldController = new AbortController();
  const older = f.poll(oldController.signal);
  oldController.abort();
  const current = f.render().refreshSampleImages();
  f.requests[1].resolve({ samples: ['/synthetic/current.png'] });
  await current;
  f.requests[0].resolve({ samples: ['/synthetic/old.png'] });
  await older;
  assert.equal(f.render().status, 'success');
  assert.deepEqual(f.render().sampleImages, ['/synthetic/current.png']);
});

test('changing metadata options invalidates callbacks from the previous sample scope', async t => {
  const f = fixture(t);
  const previous = f.render('sample-fixture', false).refreshSampleImages;
  f.render('sample-fixture', true);
  await previous();
  assert.equal(f.requests.length, 0);
  const current = f.render().refreshSampleImages();
  assert.match(f.requests[0].url, /\?metadata=1$/);
  f.requests[0].resolve({ samples: ['/synthetic/current.png'], metadata: { '/synthetic/current.png': metadata(2) } });
  await current;
  assert.equal(f.render().sampleMetadata['/synthetic/current.png'].seed, 2);
});

for (const changedScope of ['job', 'metadata']) {
  test(`old sample callback cannot invalidate a returned ${changedScope} visit or its current metadata`, async t => {
    const f = fixture(t);
    const previous = f.render().refreshSampleImages;
    if (changedScope === 'job') f.render('other-sample-job');
    else f.render('sample-fixture', false);
    const current = f.render().refreshSampleImages();
    const old = previous();
    const count = f.requests.length;
    f.requests.slice(1).forEach(request => request.resolve({ samples: ['/synthetic/old.png'] }));
    await old;
    f.requests[0].resolve({ samples: ['/synthetic/current.png'], metadata: { '/synthetic/current.png': metadata(42) } });
    await current;
    assert.equal(count, 1);
    assert.equal(f.render().status, 'success');
    assert.deepEqual(f.render().sampleImages, ['/synthetic/current.png']);
    assert.equal(f.render().sampleMetadata['/synthetic/current.png'].seed, 42);
  });
}

test('captured sample callback cannot start a new request after unmount', async t => {
  const f = fixture(t);
  const previous = f.render().refreshSampleImages;
  f.unmount();
  const pending = previous();
  const count = f.requests.length;
  f.requests.forEach(request => request.resolve({ samples: ['/synthetic/old.png'] }));
  await pending;
  assert.equal(count, 0);
});
