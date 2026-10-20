import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

function fixture(t, post) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const code = ts.transpileModule(
    fs.readFileSync(new URL('../src/utils/trainingJobSave.ts', import.meta.url), 'utf8'),
    {
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
    },
  ).outputText;
  const loaded = { exports: {} };
  const requests = [];
  new Function('require', 'module', 'exports', code)(
    name => {
      assert.equal(name, './api');
      return {
        apiClient: {
          post: (...args) => {
            requests.push(args);
            return post(...args);
          },
        },
      };
    },
    loaded,
    loaded.exports,
  );
  return { ...loaded.exports, requests };
}

const payload = { name: 'remote-job', worker_id: 'worker-1', gpu_ids: '0', job_config: {} };

test('a stalled create request aborts, reports a useful error and is never automatically retried', async t => {
  const state = fixture(t, () => new Promise(() => {}));
  const pending = state.saveTrainingJob(payload);
  const rejected = assert.rejects(pending, { message: state.TRAINING_JOB_SAVE_TIMEOUT_MESSAGE });
  assert.equal(state.requests[0][2].signal.aborted, false);
  t.mock.timers.tick(state.TRAINING_JOB_SAVE_TIMEOUT_MS);
  await rejected;
  assert.equal(state.requests[0][2].signal.aborted, true);
  assert.equal(state.requests.length, 1);
});

test('successful saves return a validated job id and clear their deadline', async t => {
  const state = fixture(t, async () => ({ data: { id: 'new-job' } }));
  assert.deepEqual(await state.saveTrainingJob(payload), { id: 'new-job' });
  t.mock.timers.tick(120_000);
  assert.equal(state.requests[0][2].signal.aborted, false);
});

test('existing job updates allow time for worker synchronization but still have a deadline', async t => {
  const state = fixture(t, () => new Promise(() => {}));
  const pending = state.saveTrainingJob({ ...payload, id: 'existing-job' });
  const rejected = assert.rejects(pending, { message: state.TRAINING_JOB_SAVE_TIMEOUT_MESSAGE });
  t.mock.timers.tick(30_000);
  assert.equal(state.requests[0][2].signal.aborted, false);
  t.mock.timers.tick(60_000);
  await rejected;
  assert.equal(state.requests[0][2].signal.aborted, true);
});

test('a late response cannot turn an expired save into success', async t => {
  let finish;
  const state = fixture(
    t,
    () =>
      new Promise(resolve => {
        finish = resolve;
      }),
  );
  let succeeded = false;
  const pending = state.saveTrainingJob(payload).then(() => {
    succeeded = true;
  });
  const rejected = assert.rejects(pending, /did not confirm the save/);
  t.mock.timers.tick(state.TRAINING_JOB_SAVE_TIMEOUT_MS);
  await rejected;
  finish({ data: { id: 'late-job' } });
  await Promise.resolve();
  assert.equal(succeeded, false);
});

test('invalid server confirmations never navigate to an undefined job', async t => {
  for (const data of [null, {}, { id: null }, { id: '' }, { id: 7 }]) {
    const state = fixture(t, async () => ({ data }));
    await assert.rejects(state.saveTrainingJob(payload), /invalid save confirmation/);
    t.mock.timers.reset();
  }
});

test('save errors preserve server feedback and identify name conflicts without logging the request', async t => {
  const failure = { response: { status: 409, data: { error: 'A run with this name already exists.' } } };
  const state = fixture(t, async () => {
    throw failure;
  });
  await assert.rejects(state.saveTrainingJob(payload), error => error === failure);
  assert.deepEqual(state.trainingJobSaveError(failure), { conflict: true, message: failure.response.data.error });
  assert.deepEqual(state.trainingJobSaveError(new Error('Network Error')), {
    conflict: false,
    message: 'Network Error',
  });
  t.mock.timers.tick(120_000);
  assert.equal(state.requests[0][2].signal.aborted, false);
});
