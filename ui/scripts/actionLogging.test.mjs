import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import axios, { AxiosError } from 'axios';

function load(relative, dependencies, output) {
  const source = ts.transpileModule(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const loaded = { exports: {} };
  new Function('require', 'module', 'exports', 'console', source)(name => {
    assert.ok(Object.hasOwn(dependencies, name), `Unexpected dependency ${name}`);
    return dependencies[name];
  }, loaded, loaded.exports, {
    log: (...args) => output.push(args), error: (...args) => output.push(args),
  });
  return loaded.exports;
}

function fixture({ failed = false } = {}) {
  const output = [], requests = [];
  let failure;
  const responseData = {
    id: 'synthetic-job', status: 'stopping',
    job_config: JSON.stringify({ private_prompt: 'synthetic-private-prompt', future_token: 'synthetic-config-token' }),
    info: 'synthetic-private-runtime-details',
  };
  // A real Axios adapter creates the request/error objects; it never makes a network call.
  const apiClient = axios.create({ adapter: async config => {
    requests.push(config);
    if (failed) {
      const response = { status: 503, statusText: 'Unavailable', config, headers: {},
        data: { error: 'Synthetic action failure', ...responseData } };
      failure = new AxiosError('Synthetic transport message with private content', 'ERR_BAD_RESPONSE', config,
        { responseText: 'synthetic-private-response' }, response);
      throw failure;
    }
    return { status: 200, statusText: 'OK', config, headers: {}, data: responseData };
  } });
  // Request headers can be extended by callers/interceptors. Never copy them into logs.
  apiClient.defaults.headers.common.Authorization = 'Bearer synthetic-private-token';
  const dependencies = { '@/utils/api': { apiClient }, '@/utils/media': { getDisplayPath: x => x, getDownloadUrl: x => x } };
  return {
    jobs: load('../src/utils/jobs.ts', dependencies, output),
    queue: load('../src/utils/queue.ts', dependencies, output),
    output, requests, get failure() { return failure; },
  };
}

const actions = [
  ['start job', 'jobs', 'startJob', '/api/jobs/synthetic-job/start', 'Failed to start job.:', true],
  ['restart job', 'jobs', 'restartJobFromScratch', '/api/jobs/synthetic-job/restart-from-scratch', 'Failed to restart job from scratch.:', true],
  ['stop job', 'jobs', 'stopJob', '/api/jobs/synthetic-job/stop', 'Error stopping job:', false],
  ['delete job', 'jobs', 'deleteJob', '/api/jobs/synthetic-job/delete', 'Error deleting job:', false],
  ['mark job stopped', 'jobs', 'markJobAsStopped', '/api/jobs/synthetic-job/mark_stopped', 'Error marking job as stopped:', false],
  ['save job now', 'jobs', 'saveJobNow', '/api/jobs/synthetic-job/save_now', 'Error setting job to save on next step:', false],
  ['start queue', 'queue', 'startQueue', '/api/queue/synthetic-job/start', 'Error starting queue:', false],
  ['stop queue', 'queue', 'stopQueue', '/api/queue/synthetic-job/stop', 'Error stopping queue:', false],
];

for (const [label, module, action, url, failureLabel, wrapsError] of actions) {
  test(`${label} logs only its operation label and preserves its error contract`, async () => {
    const f = fixture({ failed: true });
    let caught;
    try { await f[module][action]('synthetic-job'); } catch (error) { caught = error; }
    assert.equal(f.requests.length, 1);
    assert.equal(f.requests[0].url, url);
    assert.equal(f.requests[0].method, 'post');
    assert.ok(caught instanceof Error);
    if (wrapsError) {
      assert.equal(caught.message, 'Synthetic action failure');
      assert.notEqual(caught, f.failure);
    } else assert.equal(caught, f.failure);
    assert.deepEqual(f.output, [[failureLabel]], 'console must not receive an Axios request or response object');
  });
}

for (const [label, module, action] of [...actions, ['sample job now', 'jobs', 'sampleJobNow']]) {
  test(`${label} success does not log the returned job configuration`, async () => {
    const f = fixture();
    assert.equal(await f[module][action]('synthetic-job'), undefined);
    assert.equal(f.requests.length, 1);
    assert.equal(f.output.length, 1);
    assert.equal(f.output[0].length, 1, 'success output must contain only the fixed operation label');
    assert.equal(typeof f.output[0][0], 'string');
    assert.doesNotMatch(f.output[0][0], /synthetic-private|synthetic-config-token/);
  });
}
