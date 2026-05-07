export type MetricDownsamplePoint = {
  step: number;
  wall_time: number;
  value: number | null;
  value_text?: string | null;
};

export function downsampleMetricPoints(points: MetricDownsamplePoint[], maxPoints?: number): MetricDownsamplePoint[];
export function normalizeMetricMaxPoints(maxPoints: unknown, fallback?: number): number;
export function buildMetricSeriesResult(
  key: string,
  points: MetricDownsamplePoint[],
  totalCount: number,
  firstStep: number | null,
  lastStep: number | null,
  latest: MetricDownsamplePoint | null,
  maxPoints?: number,
): {
  key: string;
  totalCount: number;
  firstStep: number | null;
  lastStep: number | null;
  latest: MetricDownsamplePoint | null;
  downsampled: boolean;
  points: MetricDownsamplePoint[];
};
export function filterMetricPointsSince(
  points: MetricDownsamplePoint[],
  sinceStep: number | null | undefined,
): MetricDownsamplePoint[];
