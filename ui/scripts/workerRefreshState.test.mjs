import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { createHookHarness } from './reactHookHarness.mjs';

function load(relative, dependencies, logs) {
  const source = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const loaded = { exports: {} };
  new Function('require', 'module', 'exports', 'console', source)(
    name => {
      assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
      return dependencies[name];
    },
    loaded,
    loaded.exports,
    { error: (...args) => logs.push(args) },
  );
  return loaded.exports;
}

const settle = () => new Promise(resolve => setImmediate(resolve));
const worker = (id, name = id) => ({
  id,
  name,
  base_url: `https://${id}.invalid`,
  enabled: false,
  offline_bypass_enabled: false,
  last_status: 'unknown',
  last_error: null,
  last_checked_at: null,
  capabilities: '{}',
  gpus: '[]',
  model_count: 0,
  created_at: '2026-10-23T00:00:00Z',
  updated_at: '2026-10-23T00:00:00Z',
});

function fixture(t, kind) {
  const harness = createHookHarness(),
    requests = [],
    logs = [];
  let disposed = false,
    writesAfterUnmount = 0;
  const react = {
    ...harness.react,
    useState(initial) {
      const [value, setValue] = harness.react.useState(initial);
      return [
        value,
        update => {
          if (disposed) writesAfterUnmount += 1;
          setValue(update);
        },
      ];
    },
  };
  const hook = load(
    `../src/hooks/${kind === 'training' ? 'useWorkers' : 'useRemoteOllamaWorkers'}.tsx`,
    {
      react,
      './useScopedState': load('../src/hooks/useScopedState.ts', { react }, logs),
      './useCommittedRef': load('../src/hooks/useCommittedRef.ts', { react }, logs),
      '@/utils/api': {
        apiClient: {
          get(url, options) {
            return new Promise((resolve, reject) =>
              requests.push({ url, options, reject, resolve: data => resolve({ data }) }),
            );
          },
        },
      },
    },
    logs,
  ).default;
  const unmount = () => {
    harness.cleanup();
    disposed = true;
  };
  t.after(unmount);
  return {
    requests,
    logs,
    unmount,
    render: (enabled = true) => harness.render(() => (kind === 'training' ? hook() : hook({ enabled }))),
    writesAfterUnmount: () => writesAfterUnmount,
  };
}

for (const kind of ['training', 'Ollama']) {
  test(`${kind} old list response cannot resurrect a deleted worker after refresh`, async t => {
    const f = fixture(t, kind);
    f.render();
    f.requests[0].resolve({ workers: [worker('removed'), worker('retained')] });
    await settle();
    f.render().refreshWorkers(); // A health refresh started before deletion.
    const afterDelete = f.render();
    afterDelete.setWorkers(previous => previous.filter(entry => entry.id !== 'removed'));
    afterDelete.refreshWorkers();
    f.requests[2].resolve({ workers: [worker('retained', 'Fresh name')] });
    await settle();
    f.requests[1].resolve({ workers: [worker('removed'), worker('retained')] });
    await settle();
    assert.deepEqual(f.render().workers, [worker('retained', 'Fresh name')]);
    assert.equal(f.render().status, 'success');
  });

  test(`${kind} stale list failure cannot replace a successful save refresh`, async t => {
    const f = fixture(t, kind);
    f.render().refreshWorkers();
    f.requests[1].resolve({ workers: [worker('saved', 'Saved name')] });
    await settle();
    f.requests[0].reject(new Error('Synthetic older list failure'));
    await settle();
    assert.equal(f.render().status, 'success');
    assert.deepEqual(f.render().workers, [worker('saved', 'Saved name')]);
    assert.deepEqual(f.logs, []);
  });

  test(`${kind} newest failure retains cached workers until a successful retry`, async t => {
    const f = fixture(t, kind);
    f.render();
    f.requests[0].resolve({ workers: [worker('cached')] });
    await settle();
    f.render().refreshWorkers();
    f.render().refreshWorkers();
    f.requests[2].reject(new Error('Synthetic current list failure'));
    await settle();
    f.requests[1].resolve({ workers: [worker('stale')] });
    await settle();
    assert.equal(f.render().status, 'error');
    assert.deepEqual(f.render().workers, [worker('cached')]);
    f.render().refreshWorkers();
    f.requests[3].resolve({ workers: [] });
    await settle();
    assert.equal(f.render().status, 'success');
    assert.deepEqual(f.render().workers, []);
  });

  for (const result of ['success', 'failure']) {
    test(`${kind} ${result} after unmount does not write state or log a request error`, async t => {
      const f = fixture(t, kind);
      f.render();
      f.unmount();
      if (result === 'success') f.requests[0].resolve({ workers: [worker('departed')] });
      else f.requests[0].reject(new Error('Synthetic departed failure'));
      await settle();
      assert.equal(f.writesAfterUnmount(), 0);
      assert.deepEqual(f.logs, []);
    });
  }

  test(`${kind} captured refresh callback does not start requests after unmount`, async t => {
    const f = fixture(t, kind);
    const refresh = f.render().refreshWorkers;
    f.unmount();
    refresh();
    assert.equal(f.requests.length, 1);
    assert.equal(f.writesAfterUnmount(), 0);
  });

  test(`${kind} late delete callback cannot update the worker cache after unmount`, t => {
    const f = fixture(t, kind);
    const setWorkers = f.render().setWorkers;
    f.unmount();
    setWorkers(previous => previous.filter(entry => entry.id !== 'removed'));
    assert.equal(f.writesAfterUnmount(), 0);
  });

  test(`${kind} refresh supersession and unmount abort only their own list requests`, async t => {
    const f = fixture(t, kind);
    f.render().refreshWorkers();
    assert.equal(f.requests[0].options?.signal.aborted, true);
    assert.equal(f.requests[1].options?.signal.aborted, false);
    f.requests[0].resolve({ workers: [worker('superseded')] });
    await settle();
    assert.equal(f.requests[1].options?.signal.aborted, false);
    f.unmount();
    assert.equal(f.requests[1].options?.signal.aborted, true);
  });
}

test('Ollama initially disabled discovery stays idle until the editor opens', t => {
  const f = fixture(t, 'Ollama');
  const closed = f.render(false);
  assert.equal(closed.status, 'idle');
  closed.refreshWorkers();
  assert.equal(f.requests.length, 0);
  assert.equal(f.render(true).status, 'loading');
  assert.equal(f.requests.length, 1);
});

test('Ollama disabling an editor rejects pending list data and captured refresh callbacks', async t => {
  const f = fixture(t, 'Ollama');
  const refresh = f.render().refreshWorkers;
  assert.equal(f.render(false).status, 'idle');
  refresh();
  assert.equal(f.requests.length, 1);
  f.requests[0].resolve({ workers: [worker('closed-editor')] });
  await settle();
  assert.deepEqual(f.render(false).workers, []);
  assert.equal(f.render(false).status, 'idle');
});

test('Ollama reopening a disabled editor does not revive the previous request generation', async t => {
  const f = fixture(t, 'Ollama');
  const oldRefresh = f.render().refreshWorkers;
  f.render(false);
  f.render(true);
  f.requests[1].resolve({ workers: [worker('current-editor')] });
  await settle();
  oldRefresh();
  assert.equal(f.requests.length, 2);
  f.requests[0].resolve({ workers: [worker('previous-editor')] });
  await settle();
  assert.deepEqual(f.render().workers, [worker('current-editor')]);
  assert.equal(f.render().status, 'success');
});

test('Ollama closed editor setters cannot change the reopened worker cache', async t => {
  const f = fixture(t, 'Ollama');
  const previousSetter = f.render().setWorkers;
  f.render(false);
  f.render(true);
  f.requests[1].resolve({ workers: [worker('reopened')] });
  await settle();
  previousSetter([]);
  assert.deepEqual(f.render().workers, [worker('reopened')]);
});
