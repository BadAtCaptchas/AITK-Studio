export type MetricDownsamplePoint = {
  step: number;
  wall_time: number;
  value: number | null;
  value_text?: string | null;
};

export type MetricSeriesPayload<TPoint extends MetricDownsamplePoint = MetricDownsamplePoint> = {
  key: string;
  totalCount: number;
  firstStep: number | null;
  lastStep: number | null;
  latest: TPoint | null;
  downsampled: boolean;
  points: TPoint[];
};

function clampMaxPoints(maxPoints: unknown) {
  const n = Number(maxPoints);
  if (!Number.isFinite(n)) return 4000;
  return Math.max(2, Math.min(20000, Math.floor(n)));
}

export function downsampleMetricPoints<TPoint extends MetricDownsamplePoint>(
  points: TPoint[],
  maxPoints = 4000,
): TPoint[] {
  const cap = clampMaxPoints(maxPoints);
  if (!Array.isArray(points) || points.length <= cap) return points.slice();

  const lastIndex = points.length - 1;
  const bucketCount = Math.max(1, Math.floor((cap - 2) / 2));
  const bucketSize = Math.ceil((points.length - 2) / bucketCount);
  const out: TPoint[] = [points[0]];

  for (let start = 1; start < lastIndex && out.length < cap - 1; start += bucketSize) {
    const end = Math.min(lastIndex, start + bucketSize);
    let minIdx = -1;
    let maxIdx = -1;
    let minVal = Infinity;
    let maxVal = -Infinity;

    for (let i = start; i < end; i++) {
      const value = points[i].value;
      if (typeof value !== 'number' || !Number.isFinite(value)) continue;
      if (value < minVal) {
        minVal = value;
        minIdx = i;
      }
      if (value > maxVal) {
        maxVal = value;
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

export function normalizeMetricMaxPoints(maxPoints: unknown, fallback = 4000) {
  return clampMaxPoints(maxPoints ?? fallback);
}

export function buildMetricSeriesResult<TPoint extends MetricDownsamplePoint>(
  key: string,
  points: TPoint[],
  totalCount: number,
  firstStep: number | null,
  lastStep: number | null,
  latest: TPoint | null,
  maxPoints = 4000,
): MetricSeriesPayload<TPoint> {
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

export function filterMetricPointsSince<TPoint extends MetricDownsamplePoint>(
  points: TPoint[],
  sinceStep: number | null | undefined,
) {
  const step = Number(sinceStep);
  if (!Number.isFinite(step)) return points.slice();
  return points.filter(point => point.step > step);
}
