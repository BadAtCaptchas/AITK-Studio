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
  new Function('require', 'module', 'exports', source)(
    name => {
      assert.ok(Object.hasOwn(dependencies, name), `Unexpected import ${name}`);
      return dependencies[name];
    },
    loadedModule,
    loadedModule.exports,
  );
  return loadedModule.exports;
}

function browser() {
  const entries = [{ url: 'http://localhost/jobs', state: { __NA: true, tree: 'queue-tree' } }];
  let index = 0;
  const pending = [],
    listeners = new Map(),
    observed = [];
  const history = {
    get state() {
      return entries[index].state;
    },
    pushState(state, _unused, url) {
      entries.splice(index + 1, entries.length, {
        url: new URL(url ?? entries[index].url, entries[index].url).href,
        state,
      });
      index++;
    },
    replaceState(state, _unused, url) {
      entries[index] = { state, url: new URL(url ?? entries[index].url, entries[index].url).href };
    },
    go(delta) {
      const next = index + delta;
      if (next >= 0 && next < entries.length) pending.push(next);
    },
  };
  const win = {
    history,
    crypto: { randomUUID: () => 'fixture-document' },
    get location() {
      return new URL(entries[index].url);
    },
    addEventListener(name, listener, capture) {
      const list = listeners.get(name) || [];
      list.push({ listener, capture: !!capture });
      listeners.set(name, list);
    },
    removeEventListener(name, listener) {
      listeners.set(
        name,
        (listeners.get(name) || []).filter(item => item.listener !== listener),
      );
    },
    setTimeout,
    clearTimeout,
  };
  const emit = (name, payload) => {
    let stopped = false;
    const event = {
      ...payload,
      stopImmediatePropagation: () => {
        stopped = true;
      },
    };
    for (const { listener } of [...(listeners.get(name) || [])].sort((a, b) => Number(b.capture) - Number(a.capture))) {
      listener(event);
      if (stopped) break;
    }
  };
  const flushOne = () => {
    if (!pending.length) return false;
    index = pending.shift();
    emit('popstate', { state: history.state });
    return true;
  };
  return {
    win,
    entries,
    pending,
    observed,
    get index() {
      return index;
    },
    flushOne,
    flush: () => {
      for (let count = 0; flushOne(); count++) assert.ok(count < 20, 'traversal must settle');
    },
    observe: () => win.addEventListener('popstate', () => observed.push(win.location.pathname + win.location.hash)),
    nativeHash: hash => {
      entries.splice(index + 1, entries.length, { url: new URL(hash, entries[index].url).href, state: null });
      index++;
      emit('popstate', { state: null });
      emit('hashchange', {});
    },
  };
}

function fixture(t, { navigation = false, training = false, insecure = false } = {}) {
  const f = browser(),
    prompts = [],
    cleanups = [],
    harness = createHookHarness();
  if (insecure) f.win.crypto = insecure === 'crypto' ? undefined : {};
  if (navigation) f.win.navigation = new EventTarget();
  const priorWindow = globalThis.window,
    priorDocument = globalThis.document;
  globalThis.window = f.win;
  globalThis.document = { addEventListener() {}, removeEventListener() {} };
  const guard = load('../src/components/HistoryNavigationGuard.tsx', {
    react: {
      useLayoutEffect: effect => {
        cleanups.push(effect());
      },
    },
  });
  guard.default();
  f.observe();
  const committed = load('../src/hooks/useCommittedRef.ts', { react: harness.react });
  const router = { push: url => f.win.history.pushState({ __NA: true, tree: 'pushed-tree' }, '', url) };
  const dependencies = {
    react: harness.react,
    '@/navigation': { usePathname: () => '/jobs/new', useRouter: () => router },
    '@/components/ConfirmModal': { openConfirm: value => prompts.push(value) },
    '@/components/HistoryNavigationGuard': guard,
    './useCommittedRef': committed,
    '@/utils/trainingDraftStorage': { saveStoredTrainingDraft() {}, removeStoredTrainingDraft() {} },
  };
  const hook = training
    ? load('../src/hooks/useTrainingDraft.ts', dependencies).useTrainingDraft
    : load('../src/hooks/useUnsavedChangesGuard.ts', dependencies).default;
  t.after(() => {
    harness.cleanup();
    cleanups.forEach(cleanup => cleanup?.());
    globalThis.window = priorWindow;
    globalThis.document = priorDocument;
  });
  return {
    ...f,
    browser: f,
    prompts,
    mount: () =>
      harness.render(() =>
        training
          ? hook('new', { config: { name: 'secret-draft-marker' }, workerID: 'local', gpuIDs: '0' }, true)
          : hook({ dirty: true, revision: 'draft-1', title: 'Discard?', message: 'Unsaved' }),
      ),
    cleanupHook: () => harness.cleanup(),
    cleanup: () => cleanups.forEach(cleanup => cleanup?.()),
  };
}

for (const training of [false, true])
  test(`${training ? 'Training' : 'Notes'} canceling multi-entry Back preserves every destination and a second confirmed Back uses the original stack`, t => {
    const f = fixture(t, { training });
    f.win.history.pushState({ __NA: true, tree: 'editor-tree' }, '', '/jobs/new');
    f.nativeHash('#job-training');
    f.mount();
    const originalURLs = f.entries.map(entry => entry.url);
    f.win.history.go(-2);
    f.flush();
    assert.equal(f.prompts.length, 1);
    f.prompts[0].onCancel();
    assert.equal(f.browser.index, 2, 'cancel restores the current entry, not its contents into the Back destination');
    assert.deepEqual(
      f.entries.map(entry => entry.url),
      originalURLs,
    );
    f.win.history.go(-2);
    f.flush();
    assert.equal(f.prompts.length, 2);
    f.prompts[1].onConfirm();
    f.flush();
    assert.equal(f.browser.index, 0);
    assert.equal(f.win.location.pathname, '/jobs');
    assert.deepEqual(
      f.entries.map(entry => entry.url),
      originalURLs,
    );
    f.win.history.go(1);
    f.flush();
    assert.equal(f.win.location.pathname + f.win.location.hash, '/jobs/new');
    f.win.history.go(1);
    f.flush();
    assert.equal(f.win.location.hash, '#job-training');
    assert.doesNotMatch(JSON.stringify(f.entries), /secret-draft-marker/);
  });

test('confirmation before the restoring popstate waits before resuming the original traversal', t => {
  const f = fixture(t);
  f.win.history.pushState({ __NA: true, tree: 'editor-tree' }, '', '/jobs/new');
  f.mount();
  f.win.history.go(-1);
  f.flushOne();
  assert.equal(f.prompts.length, 1);
  f.prompts[0].onConfirm();
  f.flush();
  assert.equal(f.browser.index, 0);
  assert.deepEqual(
    f.entries.map(entry => entry.url),
    ['http://localhost/jobs', 'http://localhost/jobs/new'],
  );
  assert.deepEqual(f.observed, ['/jobs'], 'router only receives the approved traversal');
});

test('Forward cancellation and confirmation preserve the preceding entry', t => {
  const f = fixture(t);
  f.win.history.pushState({ __NA: true, tree: 'editor-tree' }, '', '/jobs/new');
  f.win.history.pushState({ __NA: true, tree: 'models-tree' }, '', '/models');
  f.win.history.go(-1);
  f.flush();
  f.mount();
  f.win.history.go(1);
  f.flush();
  f.prompts[0].onCancel();
  assert.equal(f.browser.index, 1);
  assert.equal(f.entries[2].url, 'http://localhost/models');
  f.win.history.go(1);
  f.flush();
  f.prompts[1].onConfirm();
  f.flush();
  assert.equal(f.browser.index, 2);
  assert.equal(f.entries[1].url, 'http://localhost/jobs/new');
});

test('metadata preserves Next state and wrapper teardown leaves later router wrappers functional', t => {
  const f = fixture(t);
  f.win.history.pushState({ __NA: true, tree: 'editor-tree', other: 42 }, '', '/jobs/new');
  f.win.history.replaceState({ __NA: true, tree: 'updated-tree', other: 43 }, '', '/jobs/new?tab=config');
  assert.equal(f.win.history.state.tree, 'updated-tree');
  assert.equal(f.win.history.state.other, 43);
  const captured = f.win.history.pushState;
  const outer = (...args) => captured(...args);
  f.win.history.pushState = outer;
  f.cleanup();
  assert.equal(f.win.history.pushState, outer);
  f.win.history.pushState({ __NA: true, tree: 'after-cleanup' }, '', '/models');
  assert.deepEqual(f.win.history.state, { __NA: true, tree: 'after-cleanup' });
});

test('Navigation API browsers receive no fallback history metadata', t => {
  const f = fixture(t, { navigation: true });
  f.win.history.pushState({ __NA: true, tree: 'editor-tree' }, '', '/jobs/new');
  assert.deepEqual(f.win.history.state, { __NA: true, tree: 'editor-tree' });
});

test('Training confirms a Navigation API traversal by its entry key without replacing the Forward stack', t => {
  const f = fixture(t, { navigation: true, training: true });
  f.win.history.pushState({ __NA: true, tree: 'editor-tree' }, '', '/jobs/new');
  f.nativeHash('#job-training');
  const traversed = [];
  f.win.navigation.traverseTo = key => {
    traversed.push(key);
    assert.equal(key, 'queue-entry');
    f.win.history.go(-2);
  };
  f.mount();
  const originalURLs = f.entries.map(entry => entry.url);
  const event = Object.assign(new Event('navigate', { cancelable: true }), {
    canIntercept: true,
    hashChange: false,
    navigationType: 'traverse',
    destination: { url: 'http://localhost/jobs', key: 'queue-entry' },
  });
  f.win.navigation.dispatchEvent(event);
  assert.equal(event.defaultPrevented, true);
  f.prompts[0].onConfirm();
  f.flush();
  assert.deepEqual(traversed, ['queue-entry']);
  assert.equal(f.browser.index, 0);
  assert.deepEqual(
    f.entries.map(entry => entry.url),
    originalURLs,
  );
  f.win.history.go(1);
  f.flush();
  assert.equal(f.win.location.pathname + f.win.location.hash, '/jobs/new');
  f.win.history.go(1);
  f.flush();
  assert.equal(f.win.location.hash, '#job-training');
});

test('unmounting the blocking editor restores its entry but cannot resume a stale confirmation', t => {
  const f = fixture(t);
  f.win.history.pushState({ __NA: true, tree: 'editor-tree' }, '', '/jobs/new');
  f.mount();
  f.win.history.go(-1);
  f.flushOne();
  f.cleanupHook();
  f.flush();
  assert.equal(f.browser.index, 1);
  f.prompts[0].onConfirm();
  f.flush();
  assert.equal(f.browser.index, 1);
  f.win.history.go(-1);
  f.flush();
  assert.equal(f.browser.index, 0);
});

for (const insecure of ['randomUUID', 'crypto'])
  test(`fallback tracking also works without secure-context ${insecure}`, t => {
    const f = fixture(t, { insecure });
    f.win.history.pushState({ __NA: true }, '', '/jobs/new');
    f.mount();
    f.win.history.go(-1);
    f.flush();
    f.prompts[0].onCancel();
    assert.equal(f.browser.index, 1);
    assert.equal(f.entries[0].url, 'http://localhost/jobs');
  });
