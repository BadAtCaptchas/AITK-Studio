import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

const source = ts.transpileModule(fs.readFileSync(new URL('../src/utils/queueWorkspace.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const loaded = { exports: {} };
new Function('module', 'exports', source)(loaded, loaded.exports);
const {
  createQueueWorkspaceMemory,
  moveQueuedJob,
  clampInspectorWidth,
  hasCompleteQueuedJobs,
  matchesQueuedLaneOrder,
} = loaded.exports;

const job = (id, overrides = {}) => ({ id, status: 'queued', worker_id: 'local', gpu_ids: '0', ...overrides });

test('keyboard and drag ordering move queued jobs only within their worker/GPU lane', () => {
  const jobs = [
    job('running', { status: 'running' }),
    job('a'),
    job('remote', { worker_id: 'remote' }),
    job('b'),
    job('other-gpu', { gpu_ids: '1' }),
    job('c'),
  ];
  assert.deepEqual(moveQueuedJob(jobs, 'c', 'a', 'local', '0'), ['c', 'a', 'b']);
  assert.deepEqual(moveQueuedJob(jobs, 'a', 'b', 'local', '0'), ['b', 'a', 'c']);
  assert.equal(moveQueuedJob(jobs, 'running', 'a', 'local', '0'), null);
  assert.equal(moveQueuedJob(jobs, 'a', 'remote', 'local', '0'), null);
  assert.equal(moveQueuedJob(jobs, 'a', 'other-gpu', 'local', '0'), null);
  assert.equal(moveQueuedJob(jobs, 'a', 'a', 'local', '0'), null);
  assert.deepEqual(
    jobs.map(item => item.id),
    ['running', 'a', 'remote', 'b', 'other-gpu', 'c'],
  );
});

test('a staged order detects another client reordering the same IDs without reacting to unrelated lanes', () => {
  const jobs = [
    job('a', { queue_position: 1000 }),
    job('remote', { worker_id: 'lab-01', queue_position: 1000 }),
    job('b', { queue_position: 2000 }),
    job('c', { queue_position: 3000 }),
  ];
  const baseline = ['a', 'b', 'c'];
  assert.equal(matchesQueuedLaneOrder(jobs, baseline, 'local', '0'), true);
  assert.equal(matchesQueuedLaneOrder([...jobs].reverse(), baseline, 'local', '0'), true);
  assert.equal(
    matchesQueuedLaneOrder(
      jobs.map(item => (item.id === 'remote' ? { ...item, queue_position: 500 } : item)),
      baseline,
      'local',
      '0',
    ),
    true,
  );
  assert.equal(
    matchesQueuedLaneOrder(
      jobs.map(item => (item.id === 'c' ? { ...item, queue_position: 500 } : item)),
      baseline,
      'local',
      '0',
    ),
    false,
  );
  assert.equal(
    matchesQueuedLaneOrder(
      jobs.map(item => (item.id === 'a' ? { ...item, status: 'running' } : item)),
      baseline,
      'local',
      '0',
    ),
    false,
  );
  assert.equal(matchesQueuedLaneOrder([...jobs, job('new', { queue_position: 4000 })], baseline, 'local', '0'), false);
});

test('reordering requires loaded queued rows to match the server count even after the last cursor', () => {
  const jobs = [job('a'), job('b'), job('running', { status: 'running' }), job('old', { status: 'stopped' })];
  assert.equal(hasCompleteQueuedJobs(jobs, 2), true);
  assert.equal(hasCompleteQueuedJobs(jobs, 3), false);
  assert.equal(hasCompleteQueuedJobs(jobs, 1), false);
  assert.equal(hasCompleteQueuedJobs(jobs, null), false);
  assert.equal(hasCompleteQueuedJobs(jobs, undefined), false);
  assert.equal(hasCompleteQueuedJobs(jobs, NaN), false);
  assert.equal(hasCompleteQueuedJobs(jobs, 2.5), false);
  assert.equal(hasCompleteQueuedJobs([], 0), true);
});

test('legacy equal positions use the same ID tie-break as server queue ordering', () => {
  const jobs = [
    job('c', { queue_position: 1000 }),
    job('a', { queue_position: 1000 }),
    job('b', { queue_position: 1000 }),
  ];
  assert.equal(matchesQueuedLaneOrder(jobs, ['a', 'b', 'c'], 'local', '0'), true);
  assert.equal(matchesQueuedLaneOrder(jobs, ['c', 'a', 'b'], 'local', '0'), false);
  const remote = jobs.map(item => ({
    ...item,
    worker_id: 'worker-1',
    remote_job_id: { a: 'remote-z', b: 'remote-a', c: 'remote-m' }[item.id],
  }));
  assert.equal(matchesQueuedLaneOrder(remote, ['b', 'c', 'a'], 'worker-1', '0'), true);
});

test('route memory restores working context without retaining job/config payloads', () => {
  const memory = createQueueWorkspaceMemory();
  memory.write({
    tab: 'history',
    sort: 'name',
    filter: 'portrait',
    selectedJobID: 'run-80',
    scrollTop: 2300,
    loadedCount: 100,
    job_config: 'must not retain',
    token: 'must not retain',
  });
  assert.deepEqual(memory.read(), {
    tab: 'history',
    sort: 'name',
    filter: 'portrait',
    selectedJobID: 'run-80',
    scrollTop: 2300,
    loadedCount: 100,
  });
  memory.write({ selectedJobID: null, filter: '', scrollTop: 0 });
  assert.equal(memory.read().selectedJobID, null);
  const copy = memory.read();
  copy.filter = 'changed outside memory';
  assert.equal(memory.read().filter, '');
  assert.equal(createQueueWorkspaceMemory().read().tab, 'active');
});

test('saved inspector dimensions are bounded, including corrupt local storage', () => {
  assert.equal(clampInspectorWidth(420), 420);
  assert.equal(clampInspectorWidth(100), 280);
  assert.equal(clampInspectorWidth(99999), 520);
  assert.equal(clampInspectorWidth(NaN), 360);
  assert.equal(clampInspectorWidth(Infinity), 360);
});
