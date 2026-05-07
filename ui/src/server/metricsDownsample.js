/**
 * @typedef {Object} MetricPoint
 * @property {number} step
 * @property {number=} wall_time
 * @property {number|null} value
 * @property {string|null=} value_text
 */

function clampMaxPoints(maxPoints) {
  const n = Number(maxPoints);
  if (!Number.isFinite(n)) return 4000;
  return Math.max(2, Math.min(20000, Math.floor(n)));
}

/**
 * Downsample time-series points while preserving first/last points and local
 * extrema in each bucket. This keeps spikes visible without sending every raw
 * point to the browser.
 *
 * @param {MetricPoint[]} points
 * @param {number} maxPoints
 * @returns {MetricPoint[]}
 */
export function downsampleMetricPoints(points, maxPoints = 4000) {
  const cap = clampMaxPoints(maxPoints);
  if (!Array.isArray(points) || points.length <= cap) return points.slice();

  const lastIndex = points.length - 1;
  const bucketCount = Math.max(1, Math.floor((cap - 2) / 2));
  const bucketSize = Math.ceil((points.length - 2) / bucketCount);
  /** @type {MetricPoint[]} */
  const out = [points[0]];

  for (let start = 1; start < lastIndex && out.length < cap - 1; start += bucketSize) {
    const end = Math.min(lastIndex, start + bucketSize);
    let minIdx = -1;
    let maxIdx = -1;
    let minVal = Infinity;
    let maxVal = -Infinity;

    for (let i = start; i < end; i++) {
      const v = points[i].value;
      if (typeof v !== 'number' || !Number.isFinite(v)) continue;
      if (v < minVal) {
        minVal = v;
        minIdx = i;
      }
      if (v > maxVal) {
        maxVal = v;
        maxIdx = i;
      }
    }

    if (minIdx === -1 || maxIdx === -1) {
      out.push(points[start]);
      if (end - 1 !== start && out.length < cap - 1) out.push(points[end - 1]);
      continue;
    }

    if (minIdx === maxIdx) {
      out.push(points[minIdx]);
      continue;
    }

    if (minIdx < maxIdx) {
      out.push(points[minIdx]);
      if (out.length < cap - 1) out.push(points[maxIdx]);
    } else {
      out.push(points[maxIdx]);
      if (out.length < cap - 1) out.push(points[minIdx]);
    }
  }

  if (out[out.length - 1]?.step !== points[lastIndex].step && out.length < cap) {
    out.push(points[lastIndex]);
  } else if (out[out.length - 1]?.step !== points[lastIndex].step) {
    out[out.length - 1] = points[lastIndex];
  }

  return out;
}

export function normalizeMetricMaxPoints(maxPoints, fallback = 4000) {
  return clampMaxPoints(maxPoints ?? fallback);
}

export function buildMetricSeriesResult(key, points, totalCount, firstStep, lastStep, latest, maxPoints = 4000) {
  const sampled = downsampleMetricPoints(points, maxPoints);
  return {
    key,
    totalCount,
    firstStep,
    lastStep,
    latest,
    downsampled: sampled.length < points.length,
    points: sampled,
  };
}

export function filterMetricPointsSince(points, sinceStep) {
  const step = Number(sinceStep);
  if (!Number.isFinite(step)) return points.slice();
  return points.filter(point => point.step > step);
}
