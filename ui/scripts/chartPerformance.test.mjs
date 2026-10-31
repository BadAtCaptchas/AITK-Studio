import assert from 'node:assert/strict';
import test from 'node:test';
import { snapshot, smoothing, loadChart } from './chartPerformanceFixture.mjs';
const point = (step, value = step) => ({ step, value });

test('one changed metric shares every unaffected series and metadata object', () => {
  const initial = snapshot.mergeMetricSnapshot(snapshot.emptyMetricSnapshot(), { keys: ['a', 'b', 'phase/name'], series: {
    a: { points: [point(1)], latest: point(1) }, b: { points: [point(1)], latest: point(1) },
    'phase/name': { points: [{ step: 1, value: null, value_text: 'warmup' }], latest: null },
  } }, true);
  const next = snapshot.mergeMetricSnapshot(initial, { keys: [...initial.keys], series: {
    a: { points: [point(2)], latest: point(2) }, b: { points: [], latest: point(1) },
  } }, false);
  assert.strictEqual(next.keys, initial.keys);
  assert.strictEqual(next.series.b, initial.series.b);
  assert.strictEqual(next.series['phase/name'], initial.series['phase/name']);
  assert.strictEqual(next.latest.b, initial.latest.b);
  assert.deepEqual(next.series.a, [point(1), point(2)]);
  assert.equal(next.version, initial.version + 1);
  assert.deepEqual(initial.series.a, [point(1)]);
});

test('merges unsorted, duplicated, delayed and corrected points without mutating prior snapshots', () => {
  const initial = snapshot.mergeMetricSnapshot(snapshot.emptyMetricSnapshot(), { keys: ['a'], series: { a: { points: [point(2), point(8)], latest: point(8) } } }, true);
  const next = snapshot.mergeMetricSnapshot(initial, { keys: ['a'], series: { a: { points: [point(5), point(2, 20), point(5, 50)], latest: point(8, 80) } } }, false);
  assert.deepEqual(next.series.a, [point(2, 20), point(5, 50), point(8, 80)]);
  const reset = snapshot.mergeMetricSnapshot(next, { keys: [], series: {} }, true);
  assert.deepEqual(reset.series, {});
  assert.deepEqual(reset.latest, {});
});

const series = {
  a: [point(1, -2), point(3, 2), point(7, 5), point(9, 1)],
  b: [point(2, 4), point(3, 8), point(8, 2)],
};
const options = { chartTab: 'loss', series, lossKeys: ['a', 'b'], colorByLossKey: {}, learningRateKeys: [], useLogScale: false, showRaw: true, showSmoothed: false, showTrend: false, clipOutliers: false, smoothing: 82 };
test('disabled smoothing/trend perform no EMA work; toggles restore exact bidirectional outputs', () => {
  const calls = [];
  const chart = loadChart(undefined, (values, alpha) => { calls.push(alpha); return smoothing.emaWithNulls(values, alpha); });
  const raw = chart(options);
  assert.equal(calls.length, 0);
  const smoothed = chart({ ...options, showRaw: false, showSmoothed: true });
  assert.equal(calls.length, 2);
  for (let i = 1; i < raw.data.length; i++) assert.deepEqual(smoothed.data[i], smoothing.emaWithNulls(raw.data[i], 1 - 0.82 * 0.98));
  const trend = chart({ ...options, showRaw: false, showTrend: true });
  assert.equal(calls.length, 4);
  for (let i = 1; i < raw.data.length; i++) assert.deepEqual(trend.data[i], smoothing.emaWithNulls(raw.data[i], 0.015));
});

test('primary smoothing reuse is exact for sparse unions and falls back when logarithmic filtering changes observations', () => {
  const alpha = 1 - 0.82 * 0.98;
  const values = smoothing.emaWithNulls(series.a.map(p => p.value), alpha);
  const primarySmoothing = { key: 'a', values: new Map(series.a.map((point, i) => [point.step, values[i]])) };
  for (const useLogScale of [false, true]) {
    let calls = 0;
    const chart = loadChart(undefined, (values, alpha) => { calls++; return smoothing.emaWithNulls(values, alpha); });
    const settings = { ...options, showSmoothed: true, showTrend: true, useLogScale };
    const reused = chart({ ...settings, primarySmoothing });
    assert.equal(calls, useLogScale ? 4 : 3);
    assert.deepEqual(reused, loadChart()(settings));
  }
});
