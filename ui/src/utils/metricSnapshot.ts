export interface MetricPoint {
  step: number;
  wall_time?: number;
  value: number | null;
  value_text?: string | null;
}

export type MetricSnapshot = {
  keys: string[];
  series: Record<string, MetricPoint[]>;
  latest: Record<string, MetricPoint | null>;
  version: number;
};
export type MetricUpdate = {
  keys: string[];
  series: Record<string, { points: MetricPoint[]; latest: MetricPoint | null }>;
};

export function emptyMetricSnapshot(): MetricSnapshot {
  return { keys: [], series: {}, latest: {}, version: 0 };
}

function samePoint(left: MetricPoint | null | undefined, right: MetricPoint | null | undefined) {
  return left === right || (!!left && !!right && left.step === right.step && left.value === right.value &&
    left.wall_time === right.wall_time && left.value_text === right.value_text);
}

function mergePoints(previous: MetricPoint[], incoming: MetricPoint[], full: boolean): MetricPoint[] {
  if (full) {
    const ordered = Array.from(new Map(incoming.map(point => [point.step, point])).values()).sort((a, b) => a.step - b.step);
    return previous.length === ordered.length && previous.every((point, i) => samePoint(point, ordered[i])) ? previous : ordered;
  }
  // Binary search the already sorted history; unchanged cursor/latest replies
  // neither copy history nor build a Map/sort over thousands of existing points.
  let next = previous;
  for (const point of incoming) {
    let low = 0, high = next.length;
    while (low < high) {
      const mid = (low + high) >>> 1;
      if (next[mid].step < point.step) low = mid + 1;
      else high = mid;
    }
    if (samePoint(next[low], point)) continue;
    if (next === previous) next = previous.slice();
    if (next[low]?.step === point.step) next[low] = point;
    else next.splice(low, 0, point);
  }
  return next;
}

function shareMap<T>(previous: Record<string, T>, next: Record<string, T>): Record<string, T> {
  const keys = Object.keys(next);
  return keys.length === Object.keys(previous).length && keys.every(key => Object.hasOwn(previous, key) && previous[key] === next[key])
    ? previous : next;
}

/** Input is validated at the HTTP boundary. Revisions represent data changes, not polls. */
export function mergeMetricSnapshot(previous: MetricSnapshot, update: MetricUpdate, full: boolean): MetricSnapshot {
  const keys = previous.keys.length === update.keys.length && previous.keys.every((key, i) => key === update.keys[i])
    ? previous.keys : update.keys;
  const wanted = new Set(keys);
  const series: MetricSnapshot['series'] = full ? {} : { ...previous.series };
  const latest: MetricSnapshot['latest'] = full ? {} : { ...previous.latest };
  for (const [key, payload] of Object.entries(update.series)) {
    if (!wanted.has(key)) continue;
    const incoming = payload.latest ? [...payload.points, payload.latest] : payload.points;
    series[key] = mergePoints(previous.series[key] ?? [], incoming, full);
    latest[key] = samePoint(previous.latest[key], payload.latest) ? previous.latest[key] : payload.latest;
  }
  for (const key of Object.keys(series)) if (!wanted.has(key)) delete series[key];
  for (const key of Object.keys(latest)) if (!wanted.has(key)) delete latest[key];
  const sharedSeries = shareMap(previous.series, series);
  const sharedLatest = shareMap(previous.latest, latest);
  if (previous.version > 0 && keys === previous.keys && sharedSeries === previous.series && sharedLatest === previous.latest) return previous;
  return { keys, series: sharedSeries, latest: sharedLatest, version: previous.version + 1 };
}
