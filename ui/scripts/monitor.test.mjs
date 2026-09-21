import assert from 'node:assert/strict';
import fs from 'node:fs';
import { EventEmitter } from 'node:events';
import test from 'node:test';
import ts from 'typescript';

function loadSource(relative, dependencies, globals = {}) {
  const code = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', ...Object.keys(globals), code)(
    name => {
      assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency: ${name}`);
      return dependencies[name];
    },
    module,
    module.exports,
    ...Object.values(globals),
  );
  return module.exports;
}

const monitorSample = loadSource('../src/utils/monitorSample.ts', {});
const flush = () => new Promise(resolve => setImmediate(resolve));
const cpu = { name: 'CPU', cores: 8, temperature: 40, totalMemory: 64000, availableMemory: 32000, freeMemory: 10000, currentLoad: 25 };
const gpu = { hasNvidiaSmi: true, isMac: false, gpus: [] };

function streamFixture(t, withChannel = false) {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval', 'Date'], now: 100000 });
  const requests = [];
  const messages = [];
  const cleanups = [];
  const snapshots = [];
  class Channel {
    postMessage(message) { messages.push(message); }
    close() {}
  }
  const hooks = loadSource('../src/hooks/useMonitorStream.tsx', {
    react: {
      useState: initial => [initial, next => snapshots.push(next)],
      useEffect: effect => cleanups.push(effect()),
    },
    '@/utils/api': { isAuthorizedState: { set() {} } },
    '@/utils/monitorSample': monitorSample,
  }, {
    BroadcastChannel: withChannel ? Channel : undefined,
    window: { addEventListener() {}, removeEventListener() {} },
    localStorage: { getItem: () => null },
    fetch: async (url, { signal }) => {
      assert.equal(url, '/api/monitor');
      let streamController;
      const body = new ReadableStream({ start(controller) { streamController = controller; } });
      signal.addEventListener('abort', () => streamController.error(new Error('aborted')), { once: true });
      requests.push({
        signal,
        write: text => streamController.enqueue(new TextEncoder().encode(text)),
        sample: (event = 'sample') => streamController.enqueue(new TextEncoder().encode(
          `event: ${event}\ndata: ${JSON.stringify({ t: Date.now(), cpu, gpu, history: [] })}\n\n`,
        )),
      });
      return { ok: true, status: 200, body };
    },
  });
  hooks.default();
  const unmount = () => cleanups.splice(0).forEach(cleanup => cleanup());
  t.after(async () => {
    unmount();
    await flush();
  });
  return { requests, messages, unmount, latest: () => snapshots.at(-1) };
}

test('a monitor connection with no first event is aborted and retried', async t => {
  const fixture = streamFixture(t);
  await flush();
  t.mock.timers.tick(10000);
  await flush();
  assert.equal(fixture.requests[0].signal.aborted, true);
  t.mock.timers.tick(2000);
  await flush();
  assert.equal(fixture.requests.length, 2);
  fixture.requests[1].sample('init');
  await flush();
  assert.equal(fixture.latest().connected, true);
});

test('a silent leader disconnects followers and recovers fresh samples', async t => {
  const fixture = streamFixture(t, true);
  t.mock.timers.tick(1000);
  await flush();
  fixture.requests[0].sample('init');
  await flush();
  assert.equal(fixture.latest().connected, true);

  t.mock.timers.tick(10000);
  await flush();
  assert.equal(fixture.latest().connected, false);
  assert.equal(fixture.messages.at(-1).type, 'snapshot');
  assert.equal(fixture.messages.at(-1).state.connected, false);

  t.mock.timers.tick(2000);
  await flush();
  assert.equal(fixture.requests.length, 2);
  fixture.requests[1].sample('init');
  await flush();
  assert.equal(fixture.latest().connected, true);
  assert.equal(fixture.latest().lastUpdated.getTime(), Date.now());
});

test('complete samples keep the stream alive but comments and partial frames do not', async t => {
  const fixture = streamFixture(t);
  await flush();
  fixture.requests[0].sample('init');
  await flush();
  t.mock.timers.tick(9000);
  fixture.requests[0].sample();
  await flush();
  t.mock.timers.tick(2000);
  await flush();
  assert.equal(fixture.requests[0].signal.aborted, false);
  fixture.requests[0].write(': heartbeat\n\nevent: sample\ndata: ');
  await flush();
  t.mock.timers.tick(8000);
  await flush();
  assert.equal(fixture.requests[0].signal.aborted, true);
  assert.equal(fixture.latest().connected, false);
});

test('unmounting the monitor aborts the stream without reconnecting', async t => {
  const fixture = streamFixture(t);
  await flush();
  fixture.requests[0].sample('init');
  await flush();
  fixture.unmount();
  await flush();
  t.mock.timers.tick(12000);
  await flush();
  assert.equal(fixture.requests.length, 1);
  assert.equal(fixture.requests[0].signal.aborted, true);
});

for (const resource of ['CPU', 'GPU']) {
  test(`${resource} falls back to fresh local polling, stops on stream recovery, and preserves remote polling`, async () => {
    const monitor = { connected: false, cpu: null, gpu: null };
    const requests = [];
    const polls = [];
    const hooks = loadSource(`../src/hooks/use${resource}Info.tsx`, {
      react: {
        useState: initial => [initial, () => {}],
        useEffect() {},
        useRef: initial => ({ current: initial }),
        useCallback: callback => callback,
      },
      '@/utils/api': { apiClient: { get: async (url, options) => {
        requests.push({ url, options });
        return { data: resource === 'GPU' ? gpu : cpu };
      } } },
      '@/utils/sharedAbortableRequest': loadSource('../src/utils/sharedAbortableRequest.ts', {}),
      './useMonitorStream': { default: () => ({ ...monitor }), __esModule: true },
      './usePollLoop': { default: (callback, interval) => polls.push({ callback, interval }), __esModule: true },
    });
    const render = (interval = null, worker = 'local', enabled = true) => resource === 'GPU'
      ? hooks.default(null, interval, worker, { enabled })
      : hooks.default(interval, worker);
    const poll = () => polls.at(-1).callback(new AbortController().signal);

    render();
    assert.equal(polls.at(-1).interval, 5000);
    await poll();
    await poll();
    assert.equal(requests.length, 2, 'fallback polls must bypass cached GPU readings');
    assert.equal(requests[0].url, `/api/${resource.toLowerCase()}`);
    assert.equal(requests[0].options.params.worker_id, 'local');

    monitor.connected = true;
    const result = render();
    assert.equal(polls.at(-1).interval, null);
    await poll();
    assert.equal(requests.length, 2, 'healthy local streaming must not also poll');
    await result[resource === 'GPU' ? 'refreshGpuInfo' : 'refreshCpuInfo']();
    assert.equal(requests.length, 3, 'explicit refresh must still fetch current readings');

    monitor.connected = false;
    render(10000);
    assert.equal(polls.at(-1).interval, 10000);
    await poll();
    assert.equal(requests.length, 4, 'polling must resume if streaming stops again');

    monitor.connected = true;
    render(10000, 'worker-a');
    assert.equal(polls.at(-1).interval, 10000);
    await poll();
    assert.equal(requests.at(-1).options.params.worker_id, 'worker-a');
    render(null, 'worker-b');
    assert.equal(polls.at(-1).interval, null);
    await poll();
    assert.equal(requests.at(-1).options.params.worker_id, 'worker-b');

    if (resource === 'GPU') {
      monitor.connected = false;
      render(10000, 'local', false);
      assert.equal(polls.at(-1).interval, null);
      const count = requests.length;
      await poll();
      assert.equal(requests.length, count, 'disabled telemetry must not poll');
    }
  });
}

test('a stalled fallback GPU query cannot block CPU samples or overlap another query', async t => {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval', 'Date'], now: 100000 });
  const child = new EventEmitter();
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  child.kill = () => child.emit('exit');
  const queries = [];
  let memorySamples = 0;
  const { startMonitor } = loadSource('../src/server/monitor.ts', {
    child_process: {
      spawn: () => child,
      execFile: (command, args, options) => new Promise((resolve, reject) => {
        queries.push({ command, options, resolve, reject });
      }),
    },
    util: { promisify: fn => fn },
    os: { platform: () => 'win32' },
    systeminformation: { cpu: async () => ({ manufacturer: 'Test', brand: 'CPU', cores: 8 }) },
    '@/server/macstats': { loadMacstats: () => null },
    '@/server/cpuStats': {
      createLoadSampler: () => () => 25,
      readCpuTemperature: async () => 40,
      readMemory: async () => ({ total: 64000, available: 32000 - ++memorySamples, free: 10000 }),
    },
    '@/utils/monitorSample': monitorSample,
  }, {
    process: { platform: 'win32', env: {}, once() {} },
    globalThis: {},
    console: { warn() {}, error() {} },
  });
  const monitor = startMonitor();
  const samples = [];
  monitor.subscribe(sample => samples.push(sample));
  await flush();
  // A loop child that never yields output switches to one-shot sampling.
  t.mock.timers.tick(16000);
  await flush();
  t.mock.timers.tick(500);
  await flush();
  assert.equal(queries.length, 1);
  const before = samples.at(-1);
  t.mock.timers.tick(1000);
  await flush();
  assert.ok(samples.at(-1).t > before.t, 'monitor must keep publishing while nvidia-smi is pending');
  assert.ok(samples.at(-1).cpu.availableMemory < before.cpu.availableMemory);
  assert.equal(queries.length, 1, 'only one GPU query can be in flight');
  assert.equal(queries[0].options.timeout, 5000);
  assert.equal(queries[0].options.killSignal, 'SIGKILL');

  queries[0].reject(new Error('GPU query timed out'));
  await flush();
  t.mock.timers.tick(500);
  await flush();
  assert.equal(queries.length, 2, 'a failed query must not disable future sampling');
  queries[1].resolve({ stdout: '0, NVIDIA L40, 560, 70, 99, 80, 48000, 6000, 42000, 250, 300, 1600, 5000, 0\n' });
  await flush();
  t.mock.timers.tick(500);
  await flush();
  assert.equal(samples.at(-1).gpu.gpus[0].memory.used, 42000);
});
