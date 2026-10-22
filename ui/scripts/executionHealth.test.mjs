import assert from 'node:assert/strict';
import test from 'node:test';
import { executionHealthFromHeartbeat } from '../dist/src/server/executionHealth.js';
import { parseWorkerHealth, workerHealthStatus } from '../dist/src/utils/workerHealth.js';

test('execution readiness distinguishes current, missing, stale, and future heartbeats', () => {
  assert.equal(executionHealthFromHeartbeat({ at: 99_000 }, 100_000).executionReady, true);
  for (const value of [null, {}, { at: 0 }, { at: Infinity }, { at: 110_000 }])
    assert.equal(executionHealthFromHeartbeat(value, 100_000).executionReady, false);
});
test('old workers remain reachable with unknown readiness while stopped workers are degraded', () => {
  assert.equal(workerHealthStatus(parseWorkerHealth({ ok: true })), 'reachable');
  assert.equal(workerHealthStatus(parseWorkerHealth({ ok: true, executionReady: false })), 'degraded');
  assert.equal(workerHealthStatus(parseWorkerHealth({ ok: true, executionReady: true })), 'ready');
  assert.throws(() => parseWorkerHealth({ ok: 'true' }), /invalid health/);
});
