// CPU-only microbenchmark; excludes HTTP, React rendering, uPlot and GPU work.
// Run: node scripts/chartPerformance.bench.mjs [baseline-commit]
import { execFileSync } from 'node:child_process';
import { performance } from 'node:perf_hooks';
import { snapshot, loadChart, smoothing } from './chartPerformanceFixture.mjs';
const baseline = process.argv[2] ?? '26cfa8a3c9eb18442261c8bf330fb5b8bc3e11d6';
const baselineGraph = execFileSync('git', ['show', `${baseline}:ui/src/components/JobLossGraph.tsx`], { encoding: 'utf8' });
const beforeChart = loadChart(baselineGraph), afterChart = loadChart();
const keys = Array.from({ length: 24 }, (_, i) => `loss/${i}`);
const history = Object.fromEntries(keys.map((key, i) => [key, Array.from({ length: 4000 }, (_, step) => ({ step, value: Math.sin(step / 70) + 2 + i / 100 }))]));
const initial = snapshot.mergeMetricSnapshot(snapshot.emptyMetricSnapshot(), { keys, series: Object.fromEntries(keys.map(key => [key, { points: history[key], latest: history[key].at(-1) }])) }, true);
const unchanged = { keys: [...keys], series: Object.fromEntries(keys.map(key => [key, { points: [], latest: { ...history[key].at(-1) } }])) };
const single = structuredClone(unchanged);
single.series[keys[0]] = { points: [{ step: 4000, value: 2.5 }], latest: { step: 4000, value: 2.5 } };
// Exact pre-change incremental merge strategy, without React setter overhead.
function beforeMerge(previous, update) {
  const series = { ...previous.series }, latest = { ...previous.latest };
  for (const [key, payload] of Object.entries(update.series)) {
    const incoming = [...payload.points];
    if (payload.latest) incoming.push(payload.latest);
    if (incoming.length) series[key] = Array.from(new Map([...(series[key] ?? []), ...incoming].map(point => [point.step, point])).values()).sort((a, b) => a.step - b.step);
    latest[key] = payload.latest;
  }
  for (const key of Object.keys(series)) if (!update.keys.includes(key)) delete series[key];
  for (const key of Object.keys(latest)) if (!update.keys.includes(key)) delete latest[key];
  return { keys: update.keys, series, latest, version: previous.version + 1 };
}
function measure(fn, iterations = 40) {
  for (let i = 0; i < 10; i++) fn();
  const runs = [];
  for (let round = 0; round < 5; round++) {
    const start = performance.now();
    for (let i = 0; i < iterations; i++) fn();
    runs.push((performance.now() - start) / iterations);
  }
  runs.sort((a, b) => a - b);
  return { median_ms: runs[2], min_ms: runs[0], max_ms: runs[4] };
}
const results = {};
for (const [label, update] of [['unchanged', unchanged], ['single_point', single]]) {
  results[label] = { before: measure(() => beforeMerge(initial, update)), after: measure(() => snapshot.mergeMetricSnapshot(initial, update, false)) };
  const next = snapshot.mergeMetricSnapshot(initial, update, false);
  results[label].retained_series = keys.filter(key => next.series[key] === initial.series[key]).length;
}
const options = { chartTab: 'loss', series: history, lossKeys: keys, colorByLossKey: {}, learningRateKeys: [], useLogScale: false, showRaw: true, showSmoothed: false, showTrend: false, clipOutliers: false, smoothing: 82 };
results.raw_only_chart = { before: measure(() => beforeChart(options), 10), after: measure(() => afterChart(options), 10) };
const primaryValues = smoothing.emaWithNulls(history[keys[0]].map(point => point.value), 1 - 0.82 * 0.98);
const primarySmoothing = { key: keys[0], values: new Map(history[keys[0]].map((point, i) => [point.step, primaryValues[i]])) };
const defaults = { ...options, showRaw: false, showSmoothed: true, showTrend: true };
results.default_chart = { before: measure(() => beforeChart(defaults), 10), after: measure(() => afterChart({ ...defaults, primarySmoothing }), 10) };
console.log(JSON.stringify({ node: process.version, baseline, series: keys.length, points_per_series: 4000, rounds: 5, results }, null, 2));
