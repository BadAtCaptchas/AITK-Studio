import assert from 'node:assert/strict';
import test from 'node:test';
import {
  buildMetricSeriesResult,
  downsampleMetricPoints,
  filterMetricPointsSince,
  normalizeMetricMaxPoints,
} from '../dist/src/server/metricsDownsample.js';

test('downsampleMetricPoints preserves monotonic ordering and point cap', () => {
  const points = Array.from({ length: 1000 }, (_, step) => ({
    step,
    wall_time: step,
    value: Math.sin(step / 12),
  }));

  const sampled = downsampleMetricPoints(points, 100);

  assert.ok(sampled.length <= 100);
  assert.equal(sampled[0].step, 0);
  assert.equal(sampled[sampled.length - 1].step, 999);
  for (let i = 1; i < sampled.length; i++) {
    assert.ok(sampled[i].step > sampled[i - 1].step);
  }
});

test('downsampleMetricPoints keeps bucket extrema visible', () => {
  const points = Array.from({ length: 200 }, (_, step) => ({
    step,
    wall_time: step,
    value: step === 87 ? 999 : step === 130 ? -999 : 1,
  }));

  const sampled = downsampleMetricPoints(points, 60);
  const values = new Set(sampled.map(point => point.value));

  assert.ok(values.has(999));
  assert.ok(values.has(-999));
});

test('downsampleMetricPoints handles text/null metrics', () => {
  const points = Array.from({ length: 80 }, (_, step) => ({
    step,
    wall_time: step,
    value: null,
    value_text: `phase-${step}`,
  }));

  const sampled = downsampleMetricPoints(points, 20);

  assert.ok(sampled.length <= 20);
  assert.equal(sampled[0].step, 0);
  assert.equal(sampled[sampled.length - 1].step, 79);
});

test('normalizeMetricMaxPoints clamps invalid values', () => {
  assert.equal(normalizeMetricMaxPoints(Number.NaN), 4000);
  assert.equal(normalizeMetricMaxPoints(1), 2);
  assert.equal(normalizeMetricMaxPoints(50000), 20000);
});

test('buildMetricSeriesResult returns latest metadata and capped points', () => {
  const points = Array.from({ length: 250 }, (_, step) => ({
    step,
    wall_time: step + 10,
    value: step / 10,
  }));
  const latest = points[points.length - 1];

  const payload = buildMetricSeriesResult('loss/loss', points, points.length, 0, 249, latest, 64);

  assert.equal(payload.key, 'loss/loss');
  assert.equal(payload.totalCount, 250);
  assert.equal(payload.firstStep, 0);
  assert.equal(payload.lastStep, 249);
  assert.deepEqual(payload.latest, latest);
  assert.equal(payload.downsampled, true);
  assert.ok(payload.points.length <= 64);
  assert.equal(payload.points[0].step, 0);
  assert.equal(payload.points[payload.points.length - 1].step, 249);
});

test('filterMetricPointsSince implements incremental since_step semantics', () => {
  const points = [1, 2, 3, 4, 5].map(step => ({ step, wall_time: step, value: step }));

  assert.deepEqual(
    filterMetricPointsSince(points, 3).map(point => point.step),
    [4, 5],
  );
  assert.deepEqual(
    filterMetricPointsSince(points, null).map(point => point.step),
    [1, 2, 3, 4, 5],
  );
});

test('buildMetricSeriesResult preserves text metrics and empty missing keys', () => {
  const textPoint = { step: 8, wall_time: 100, value: null, value_text: 'warmup' };
  const textPayload = buildMetricSeriesResult('phase/name', [textPoint], 1, 8, 8, textPoint, 10);
  const emptyPayload = buildMetricSeriesResult('event/sample', [], 0, null, null, null, 10);

  assert.equal(textPayload.points[0].value_text, 'warmup');
  assert.equal(textPayload.latest?.value_text, 'warmup');
  assert.deepEqual(emptyPayload.points, []);
  assert.equal(emptyPayload.latest, null);
});
