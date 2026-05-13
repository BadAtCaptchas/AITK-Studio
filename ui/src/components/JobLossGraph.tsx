'use client';

import type { Job, GpuInfo } from '@/types';
import useJobMetrics, { type MetricPoint } from '@/hooks/useJobMetrics';
import useGPUInfo from '@/hooks/useGPUInfo';
import { getTotalSteps } from '@/utils/jobs';
import {
  Activity,
  Clock,
  Gauge,
  Image as ImageIcon,
  RotateCcw,
  Save,
  TrendingDown,
  Zap,
} from 'lucide-react';
import {
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
  startTransition,
  type ReactNode,
} from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';

interface Props {
  job: Job;
}

type ChartTab = 'loss' | 'learning_rate' | 'throughput' | 'timesteps' | 'gradients' | 'memory';
type EventKind = 'sample' | 'checkpoint' | 'phase_change';

type PhaseTransition = {
  step: number;
  index: number;
  name?: string | null;
};

type EventMarker = {
  step: number;
  key: string;
  kind: EventKind;
  label: string;
};

type HoverItem = {
  label: string;
  value: number | null;
  color: string;
};

type HoverState = {
  step: number;
  items: HoverItem[];
};

const FALLBACK_CANVAS_HEIGHT = 360;
const MIN_CANVAS_HEIGHT = 220;
const CHART_MAX_POINTS = 4000;

const LOSS_COLOR = 'rgba(96,165,250,1)';
const LOSS_TREND_COLOR = 'rgba(37,99,235,0.95)';
const LR_COLOR = 'rgba(251,191,36,1)';
const THROUGHPUT_COLOR = 'rgba(52,211,153,1)';
const DIAGNOSTIC_PALETTE = [
  'rgba(96,165,250,1)',
  'rgba(52,211,153,1)',
  'rgba(251,191,36,1)',
  'rgba(244,114,182,1)',
  'rgba(34,211,238,1)',
];
const EVENT_STYLE: Record<EventKind, { color: string; label: string }> = {
  sample: { color: 'rgba(251,191,36,0.9)', label: 'Sample' },
  checkpoint: { color: 'rgba(52,211,153,0.9)', label: 'Checkpoint' },
  phase_change: { color: 'rgba(244,114,182,0.9)', label: 'Phase' },
};

const LOSS_PALETTE = [
  'rgba(96,165,250,1)',
  'rgba(34,211,238,1)',
  'rgba(129,140,248,1)',
  'rgba(52,211,153,1)',
  'rgba(251,191,36,1)',
  'rgba(244,114,182,1)',
  'rgba(248,113,113,1)',
];

function formatNum(v: number | null | undefined, digits = 4) {
  if (v == null || !Number.isFinite(v)) return '--';
  const abs = Math.abs(v);
  if (abs >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (abs >= 10) return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (abs >= 1) return v.toLocaleString(undefined, { maximumFractionDigits: 3 });
  if (abs === 0) return '0';
  return v.toPrecision(digits);
}

function formatCompact(v: number | null | undefined) {
  if (v == null || !Number.isFinite(v)) return '--';
  return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function formatPercent(v: number | null | undefined, digits = 1) {
  if (v == null || !Number.isFinite(v)) return '--';
  return `${v.toFixed(digits)}%`;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function formatDuration(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return '--';
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  if (minutes < 60) return `${minutes}m ${secs}s`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return `${hours}h ${mins}m`;
}

function formatMemory(mb: number | null | undefined) {
  if (mb == null || !Number.isFinite(mb)) return '--';
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

function cleanLabel(key: string) {
  return key
    .replace(/^loss\//, '')
    .replace(/^train\//, '')
    .replace(/_/g, ' ');
}

function hashToIndex(str: string, mod: number) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) % mod;
}

function lossColorForKey(key: string) {
  if (key === 'loss' || key === 'loss/loss') return LOSS_COLOR;
  return LOSS_PALETTE[hashToIndex(key, LOSS_PALETTE.length)];
}

function colorWithAlpha(rgba: string, alpha: number) {
  const match = rgba.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
  if (!match) return rgba;
  return `rgba(${match[1]},${match[2]},${match[3]},${alpha})`;
}

function computeCanvasSize(host: HTMLElement): { width: number; height: number } | null {
  const { width, height } = host.getBoundingClientRect();
  if (width <= 0 || height <= 0) return null;
  return { width, height: Math.max(MIN_CANVAS_HEIGHT, height) };
}

function emaWithNulls(ys: (number | null)[], alpha: number): (number | null)[] {
  const out: (number | null)[] = new Array(ys.length);
  let prev: number | null = null;
  for (let i = 0; i < ys.length; i++) {
    const v = ys[i];
    if (v === null || !Number.isFinite(v)) {
      out[i] = null;
      continue;
    }
    prev = prev === null ? v : alpha * v + (1 - alpha) * prev;
    out[i] = prev;
  }
  return out;
}

function sortedNumericPoints(points: MetricPoint[] | undefined, transform?: (value: number) => number | null) {
  return [...(points ?? [])]
    .map(point => {
      if (point.value == null || !Number.isFinite(point.value)) return null;
      const value = transform ? transform(point.value) : point.value;
      if (value == null || !Number.isFinite(value)) return null;
      return { ...point, value };
    })
    .filter(Boolean)
    .sort((a, b) => (a as MetricPoint).step - (b as MetricPoint).step) as MetricPoint[];
}

function latestNumericPoint(points: MetricPoint[] | undefined) {
  for (let i = (points?.length ?? 0) - 1; i >= 0; i--) {
    const point = points?.[i];
    if (point?.value != null && Number.isFinite(point.value)) return point;
  }
  return null;
}

function latestMetricValue(
  latest: Record<string, MetricPoint | null>,
  series: Record<string, MetricPoint[]>,
  key: string,
) {
  return latest[key]?.value ?? latestNumericPoint(series[key])?.value ?? null;
}

function chartYAxisLabel(chartTab: ChartTab) {
  if (chartTab === 'learning_rate') return 'LR';
  if (chartTab === 'throughput') return 'steps/sec';
  if (chartTab === 'timesteps') return 'timestep / sigma';
  if (chartTab === 'gradients') return 'grad norm';
  if (chartTab === 'memory') return 'GB / percent';
  return 'Loss';
}

function isLearningRateKey(key: string) {
  return /(^|\/)(learning_rate|lr)(\/|$)/i.test(key);
}

function paddedRange(dataMin: number, dataMax: number, chartTab: ChartTab, useLogScale: boolean) {
  if (!Number.isFinite(dataMin) || !Number.isFinite(dataMax)) return [0, 1] as [number, number];

  if (useLogScale) {
    const safeMin = dataMin > 0 ? dataMin : 1e-12;
    const safeMax = dataMax > 0 ? dataMax : safeMin * 10;
    if (safeMin === safeMax) return [safeMin / 1.5, safeMax * 1.5] as [number, number];
    return [safeMin, safeMax] as [number, number];
  }

  const span = dataMax - dataMin;
  if (span === 0) {
    const fallback = chartTab === 'learning_rate' ? 1e-8 : 1;
    const pad = Math.max(Math.abs(dataMin) * 0.25, fallback);
    const min = chartTab === 'learning_rate' && dataMin >= 0 ? Math.max(0, dataMin - pad) : dataMin - pad;
    return [min, dataMax + pad] as [number, number];
  }

  const pad = span * 0.05;
  const min = chartTab === 'learning_rate' && dataMin >= 0 ? Math.max(0, dataMin - pad) : dataMin - pad;
  return [min, dataMax + pad] as [number, number];
}

function getGpuIds(job: Job) {
  if (job.gpu_ids === 'mps') return [0];
  return job.gpu_ids
    .split(',')
    .map(id => Number.parseInt(id.trim(), 10))
    .filter(id => Number.isFinite(id));
}

function safeTotalSteps(job: Job) {
  try {
    return getTotalSteps(job);
  } catch {
    return 0;
  }
}

function parseStepsPerSecond(speedString: string | null | undefined) {
  if (!speedString) return null;
  const iterPerSec = speedString.match(/([\d.]+)\s*(?:it|iter|steps?)\/s(?:ec)?/i);
  if (iterPerSec) {
    const value = Number(iterPerSec[1]);
    return Number.isFinite(value) && value > 0 ? value : null;
  }
  const secPerIter = speedString.match(/([\d.]+)\s*s(?:ec)?\/(?:it|iter|steps?)/i);
  if (secPerIter) {
    const value = Number(secPerIter[1]);
    return Number.isFinite(value) && value > 0 ? 1 / value : null;
  }
  return null;
}

function buildPhaseTransitions(points: MetricPoint[], phaseNames: MetricPoint[]): PhaseTransition[] {
  const nameByStep = new Map<number, string>();
  for (const point of phaseNames) {
    if (point.value_text) nameByStep.set(point.step, point.value_text);
  }

  const sorted = sortedNumericPoints(points);
  const transitions: PhaseTransition[] = [];
  let previousIndex: number | null = null;
  for (const point of sorted) {
    const index = Math.round(point.value as number);
    if (previousIndex === null || index !== previousIndex) {
      transitions.push({ step: point.step, index, name: nameByStep.get(point.step) ?? null });
      previousIndex = index;
    }
  }
  return transitions;
}

function buildEventMarkers(series: Record<string, MetricPoint[]>) {
  const definitions: Array<{ key: string; kind: EventKind }> = [
    { key: 'event/sample', kind: 'sample' },
    { key: 'event/checkpoint', kind: 'checkpoint' },
    { key: 'event/phase_change', kind: 'phase_change' },
  ];

  const markers: EventMarker[] = [];
  for (const definition of definitions) {
    for (const point of series[definition.key] ?? []) {
      markers.push({
        step: point.step,
        key: definition.key,
        kind: definition.kind,
        label: EVENT_STYLE[definition.kind].label,
      });
    }
  }

  return markers.sort((a, b) => a.step - b.step);
}

function drawPhaseBands(u: uPlot, transitions: PhaseTransition[]) {
  if (!transitions.length) return;
  const xs = u.data[0] as number[];
  if (!xs?.length) return;

  const min = u.scales.x.min ?? xs[0];
  const max = u.scales.x.max ?? xs[xs.length - 1];
  const { ctx, bbox } = u;

  ctx.save();
  for (let i = 0; i < transitions.length; i++) {
    const start = Math.max(min, transitions[i].step);
    const end = Math.min(max, transitions[i + 1]?.step ?? max);
    if (end <= min || start >= max || end <= start) continue;

    const x1 = bbox.left + u.valToPos(start, 'x');
    const x2 = bbox.left + u.valToPos(end, 'x');
    ctx.fillStyle = i % 2 === 0 ? 'rgba(96,165,250,0.035)' : 'rgba(251,191,36,0.035)';
    ctx.fillRect(x1, bbox.top, Math.max(1, x2 - x1), bbox.height);
  }
  ctx.restore();
}

function drawEventMarkers(u: uPlot, markers: EventMarker[], transitions: PhaseTransition[]) {
  const { ctx, bbox } = u;
  const min = u.scales.x.min ?? -Infinity;
  const max = u.scales.x.max ?? Infinity;

  ctx.save();

  ctx.font = '11px sans-serif';
  ctx.textBaseline = 'top';
  ctx.lineWidth = 1;

  for (const transition of transitions) {
    if (transition.step < min || transition.step > max) continue;
    const x = bbox.left + u.valToPos(transition.step, 'x');
    ctx.strokeStyle = 'rgba(148,163,184,0.36)';
    ctx.fillStyle = 'rgba(226,232,240,0.78)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, bbox.top);
    ctx.lineTo(x, bbox.top + bbox.height);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillText(`P${transition.index + 1}`, x + 5, bbox.top + 8);
  }

  const visibleMarkers = markers.filter(marker => marker.step >= min && marker.step <= max);
  const showLabels = visibleMarkers.length <= 35;
  for (const marker of visibleMarkers) {
    const x = bbox.left + u.valToPos(marker.step, 'x');
    const style = EVENT_STYLE[marker.kind];
    ctx.strokeStyle = style.color;
    ctx.fillStyle = style.color;
    ctx.globalAlpha = showLabels ? 0.9 : 0.55;
    ctx.beginPath();
    ctx.moveTo(x, bbox.top + bbox.height * 0.08);
    ctx.lineTo(x, bbox.top + bbox.height);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillRect(x - 2, bbox.top + bbox.height - 8, 4, 8);
    if (showLabels) ctx.fillText(marker.label, x + 5, bbox.top + bbox.height - 20);
  }

  ctx.restore();
}

function getPrimaryGpu(gpus: GpuInfo[]) {
  return gpus.length ? gpus[0] : null;
}

function buildChartData({
  chartTab,
  series,
  lossKeys,
  learningRateKeys,
  useLogScale,
  showRaw,
  showSmoothed,
  showTrend,
  clipOutliers,
  smoothing,
}: {
  chartTab: ChartTab;
  series: Record<string, MetricPoint[]>;
  lossKeys: string[];
  learningRateKeys: string[];
  useLogScale: boolean;
  showRaw: boolean;
  showSmoothed: boolean;
  showTrend: boolean;
  clipOutliers: boolean;
  smoothing: number;
}) {
  const data: (number[] | (number | null)[])[] = [];
  const seriesConfigs: uPlot.Series[] = [{}];
  const t = clamp(smoothing / 100, 0, 1);
  const alpha = 1 - t * 0.98;
  const trendAlpha = 0.015;

  if (chartTab === 'loss') {
    const stepSet = new Set<number>();
    for (const key of lossKeys) {
      for (const point of sortedNumericPoints(series[key])) {
        if (useLogScale && (point.value as number) <= 0) continue;
        stepSet.add(point.step);
      }
    }

    const xs = Array.from(stepSet).sort((a, b) => a - b);
    const xsSet = new Set(xs);
    data.push(xs);

    for (const key of lossKeys) {
      const color = lossColorForKey(key);
      const values = new Map<number, number>();
      for (const point of sortedNumericPoints(series[key])) {
        if (!xsSet.has(point.step)) continue;
        if (useLogScale && (point.value as number) <= 0) continue;
        values.set(point.step, point.value as number);
      }
      const raw = xs.map(step => values.get(step) ?? null);
      const smooth = emaWithNulls(raw, alpha);
      const trend = emaWithNulls(raw, trendAlpha);

      if (showRaw) {
        data.push(raw);
        seriesConfigs.push({
          label: `${cleanLabel(key)} raw`,
          stroke: colorWithAlpha(color, 0.24),
          width: 1,
          spanGaps: false,
          points: { show: false },
        });
      }
      if (showSmoothed) {
        data.push(smooth);
        seriesConfigs.push({
          label: cleanLabel(key),
          stroke: color,
          width: 2,
          spanGaps: false,
          points: { show: false },
        });
      }
      if (showTrend) {
        data.push(trend);
        seriesConfigs.push({
          label: `${cleanLabel(key)} trend`,
          stroke: key === lossKeys[0] ? LOSS_TREND_COLOR : colorWithAlpha(color, 0.7),
          width: 2.75,
          spanGaps: false,
          points: { show: false },
        });
      }
    }
  } else if (chartTab === 'learning_rate') {
    const stepSet = new Set<number>();
    for (const key of learningRateKeys) {
      for (const point of sortedNumericPoints(series[key])) {
        if (useLogScale && (point.value as number) <= 0) continue;
        stepSet.add(point.step);
      }
    }

    const xs = Array.from(stepSet).sort((a, b) => a - b);
    const xsSet = new Set(xs);
    data.push(xs);

    for (const key of learningRateKeys) {
      const points = sortedNumericPoints(series[key]).filter(point => xsSet.has(point.step));
      const values = new Map(points.map(point => [point.step, point.value as number]));
      const color = key === learningRateKeys[0] ? LR_COLOR : lossColorForKey(key);
      data.push(xs.map(step => values.get(step) ?? null));
      seriesConfigs.push({
        label: cleanLabel(key),
        stroke: color,
        width: 2.25,
        spanGaps: false,
        points: { show: false },
      });
    }
  } else if (chartTab === 'throughput') {
    const spsPoints = sortedNumericPoints(series['train/steps_per_sec']);
    const stepSecondPoints = sortedNumericPoints(series['train/step_seconds']);

    if (spsPoints.length) {
      const xs = spsPoints.map(point => point.step);
      data.push(xs);
      data.push(spsPoints.map(point => point.value as number));
      seriesConfigs.push({
        label: 'steps/sec',
        stroke: THROUGHPUT_COLOR,
        width: 2.25,
        spanGaps: false,
        points: { show: false },
      });
    } else if (stepSecondPoints.length) {
      const xs = stepSecondPoints.map(point => point.step);
      data.push(xs);
      data.push(stepSecondPoints.map(point => 1 / (point.value as number)));
      seriesConfigs.push({
        label: 'steps/sec',
        stroke: THROUGHPUT_COLOR,
        width: 2.25,
        spanGaps: false,
        points: { show: false },
      });
    } else {
      data.push([]);
    }
  } else {
    const metricKeys =
      chartTab === 'timesteps'
        ? ['train/timestep_mean', 'train/timestep_min', 'train/timestep_max', 'train/sigma_mean']
        : chartTab === 'gradients'
          ? ['train/grad_norm', 'train/grad_norm_mean', 'train/grad_norm_limit']
          : [
              'train/gpu_mem_allocated_gb',
              'train/gpu_mem_reserved_gb',
              'train/gpu_mem_max_allocated_gb',
              'train/gpu_mem_used_pct',
            ];
    const presentKeys = metricKeys.filter(key => (series[key]?.length ?? 0) > 0);
    const stepSet = new Set<number>();
    for (const key of presentKeys) {
      for (const point of sortedNumericPoints(series[key])) stepSet.add(point.step);
    }
    const xs = Array.from(stepSet).sort((a, b) => a - b);
    const xsSet = new Set(xs);
    data.push(xs);

    presentKeys.forEach((key, index) => {
      const points = sortedNumericPoints(series[key]).filter(point => xsSet.has(point.step));
      const values = new Map(points.map(point => [point.step, point.value as number]));
      data.push(xs.map(step => values.get(step) ?? null));
      seriesConfigs.push({
        label: cleanLabel(key),
        stroke: DIAGNOSTIC_PALETTE[index % DIAGNOSTIC_PALETTE.length],
        width: 2.25,
        spanGaps: false,
        points: { show: false },
      });
    });
  }

  let yClip: { min: number; max: number } | null = null;
  const xs = (data[0] ?? []) as number[];
  if (clipOutliers && xs.length >= 10) {
    const vals: number[] = [];
    for (let s = 1; s < data.length; s++) {
      const arr = data[s] as (number | null)[];
      for (const value of arr) {
        if (value != null && Number.isFinite(value)) vals.push(value);
      }
    }
    if (vals.length >= 10) {
      vals.sort((a, b) => a - b);
      const lo = vals[Math.floor(vals.length * 0.02)];
      const hi = vals[Math.ceil(vals.length * 0.98) - 1];
      if (Number.isFinite(lo) && Number.isFinite(hi) && lo !== hi) yClip = { min: lo, max: hi };
    }
  }

  return {
    data: data as uPlot.AlignedData,
    seriesConfigs,
    yClip,
    hasData: xs.length > 1 && data.length > 1,
    totalPoints: xs.length,
  };
}

export default function JobLossGraph({ job }: Props) {
  const {
    series,
    latest,
    lossKeys,
    phasePoints,
    phaseNamePoints,
    status,
    version,
    refreshMetrics,
  } = useJobMetrics(job.id, job.stop && job.status === 'running' ? 'stopping' : job.status, CHART_MAX_POINTS);

  const gpuIds = useMemo(() => getGpuIds(job), [job.gpu_ids]);
  const { gpuList } = useGPUInfo(gpuIds, 5000);

  const [chartTab, setChartTab] = useState<ChartTab>('loss');
  const [useLogScale, setUseLogScale] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [showSmoothed, setShowSmoothed] = useState(true);
  const [showTrend, setShowTrend] = useState(true);
  const [clipOutliers, setClipOutliers] = useState(false);
  const [smoothing, setSmoothing] = useState(82);
  const deferredSmoothing = useDeferredValue(smoothing);
  const [enabledLoss, setEnabledLoss] = useState<Record<string, boolean>>({});
  const [isZoomed, setIsZoomed] = useState(false);
  const [hover, setHover] = useState<HoverState | null>(null);

  useEffect(() => {
    setEnabledLoss(prev => {
      const next = { ...prev };
      for (const key of lossKeys) {
        if (next[key] === undefined) next[key] = true;
      }
      for (const key of Object.keys(next)) {
        if (!lossKeys.includes(key)) delete next[key];
      }
      return next;
    });
  }, [lossKeys]);

  const activeLossKeys = useMemo(
    () => lossKeys.filter(key => enabledLoss[key] !== false && (series[key]?.length ?? 0) > 0),
    [enabledLoss, lossKeys, series],
  );
  const learningRateKeys = useMemo(
    () => Object.keys(series).filter(key => isLearningRateKey(key) && (series[key]?.length ?? 0) > 0).sort(),
    [series],
  );

  const totalSteps = useMemo(() => safeTotalSteps(job), [job]);
  const progressPercent = totalSteps > 0 ? clamp((job.step / totalSteps) * 100, 0, 100) : null;
  const primaryLossKey = activeLossKeys[0] ?? lossKeys.find(key => (series[key]?.length ?? 0) > 0) ?? lossKeys[0];

  const lossSummary = useMemo(() => {
    const points = sortedNumericPoints(series[primaryLossKey]);
    if (!points.length) return { current: null as number | null, deltaPct: null as number | null };
    const alpha = 1 - clamp(deferredSmoothing / 100, 0, 1) * 0.98;
    const smoothed = emaWithNulls(points.map(point => point.value as number), alpha).filter(
      value => value != null && Number.isFinite(value),
    ) as number[];
    if (!smoothed.length) return { current: null, deltaPct: null };
    const current = smoothed[smoothed.length - 1];
    const compareIndex = Math.max(0, smoothed.length - Math.max(20, Math.floor(smoothed.length * 0.12)));
    const previous = smoothed[compareIndex];
    const deltaPct =
      previous != null && previous !== 0 && Number.isFinite(previous) ? ((current - previous) / Math.abs(previous)) * 100 : null;
    return { current, deltaPct };
  }, [deferredSmoothing, primaryLossKey, series]);

  const latestLearningRateKey = learningRateKeys[0];
  const latestLearningRate =
    (latestLearningRateKey ? latest[latestLearningRateKey]?.value : null) ??
    latestNumericPoint(latestLearningRateKey ? series[latestLearningRateKey] : undefined)?.value ??
    null;
  const learningRatePointCount = learningRateKeys.reduce((sum, key) => sum + (series[key]?.length ?? 0), 0);
  const latestStepsPerSec =
    latest['train/steps_per_sec']?.value ??
    latestNumericPoint(series['train/steps_per_sec'])?.value ??
    (() => {
      const seconds = latest['train/step_seconds']?.value ?? latestNumericPoint(series['train/step_seconds'])?.value;
      if (seconds != null && Number.isFinite(seconds) && seconds > 0) return 1 / seconds;
      return parseStepsPerSecond(job.speed_string);
    })();
  const etaSeconds =
    totalSteps > 0 && latestStepsPerSec != null && latestStepsPerSec > 0
      ? Math.max(0, (totalSteps - job.step) / latestStepsPerSec)
      : null;

  const primaryGpu = getPrimaryGpu(gpuList);
  const gpuMemoryPct =
    primaryGpu && primaryGpu.memory.total > 0 ? (primaryGpu.memory.used / primaryGpu.memory.total) * 100 : null;
  const latestTimestepMean = latestMetricValue(latest, series, 'train/timestep_mean');
  const latestSigmaMean = latestMetricValue(latest, series, 'train/sigma_mean');
  const latestGradNorm = latestMetricValue(latest, series, 'train/grad_norm');
  const latestGradLimit = latestMetricValue(latest, series, 'train/grad_norm_limit');
  const latestTrainVramGb =
    latestMetricValue(latest, series, 'train/gpu_mem_allocated_gb') ?? latestMetricValue(latest, series, 'train/gpu_mem_reserved_gb');
  const latestTrainReservedGb = latestMetricValue(latest, series, 'train/gpu_mem_reserved_gb');
  const latestLossFinal = latestMetricValue(latest, series, 'train/loss_final');
  const latestLossUnclipped = latestMetricValue(latest, series, 'train/loss_unclipped');
  const latestBatchSize = latestMetricValue(latest, series, 'train/batch_size');
  const latestEffectiveBatch = latestMetricValue(latest, series, 'train/effective_batch_size');
  const latestNoisePredStd = latestMetricValue(latest, series, 'train/noise_pred_std');
  const latestTargetStd = latestMetricValue(latest, series, 'train/target_std');

  const phaseTransitions = useMemo(
    () => buildPhaseTransitions(phasePoints, phaseNamePoints),
    [phaseNamePoints, phasePoints],
  );
  const eventMarkers = useMemo(() => buildEventMarkers(series), [series]);
  const currentPhase = phaseTransitions.length ? phaseTransitions[phaseTransitions.length - 1] : null;

  const built = useMemo(
    () =>
      buildChartData({
        chartTab,
        series,
        lossKeys: activeLossKeys,
        learningRateKeys,
        useLogScale,
        showRaw,
        showSmoothed,
        showTrend,
        clipOutliers,
        smoothing: deferredSmoothing,
      }),
    [
      activeLossKeys,
      chartTab,
      clipOutliers,
      deferredSmoothing,
      learningRateKeys,
      series,
      showRaw,
      showSmoothed,
      showTrend,
      useLogScale,
      version,
    ],
  );

  const chartHostRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const uplotRef = useRef<uPlot | null>(null);
  const phaseTransitionsRef = useRef<PhaseTransition[]>([]);
  const eventMarkersRef = useRef<EventMarker[]>([]);
  const yClipRef = useRef<{ min: number; max: number } | null>(null);
  const isZoomedRef = useRef(false);
  const hoverRafRef = useRef<number | null>(null);
  const queuedHoverRef = useRef<HoverState | null>(null);

  useEffect(() => {
    phaseTransitionsRef.current = phaseTransitions;
    eventMarkersRef.current = eventMarkers;
    uplotRef.current?.redraw(true, true);
  }, [eventMarkers, phaseTransitions]);

  useEffect(() => {
    yClipRef.current = built.yClip;
  }, [built.yClip]);

  useEffect(() => {
    isZoomedRef.current = isZoomed;
  }, [isZoomed]);

  useEffect(() => {
    return () => {
      if (hoverRafRef.current != null) cancelAnimationFrame(hoverRafRef.current);
    };
  }, []);

  const queueHover = useCallback((next: HoverState | null) => {
    queuedHoverRef.current = next;
    if (hoverRafRef.current != null) return;
    hoverRafRef.current = requestAnimationFrame(() => {
      hoverRafRef.current = null;
      setHover(queuedHoverRef.current);
    });
  }, []);

  const structuralKey = useMemo(
    () =>
      [
        chartTab,
        activeLossKeys.join('|'),
        `raw=${showRaw}`,
        `smooth=${showSmoothed}`,
        `trend=${showTrend}`,
        `log=${useLogScale}`,
        `has=${built.hasData}`,
        `series=${built.seriesConfigs.map(config => config.label).join('|')}`,
      ].join(';'),
    [activeLossKeys, built.hasData, built.seriesConfigs, chartTab, showRaw, showSmoothed, showTrend, useLogScale],
  );

  useEffect(() => {
    if (uplotRef.current) {
      uplotRef.current.destroy();
      uplotRef.current = null;
    }
    if (!containerRef.current || !chartHostRef.current || !built.hasData) return;

    const host = chartHostRef.current;
    const rect = host.getBoundingClientRect();
    const initialHeight = rect.height > 0 ? Math.max(MIN_CANVAS_HEIGHT, rect.height) : FALLBACK_CANVAS_HEIGHT;

    const opts: uPlot.Options = {
      width: rect.width || 900,
      height: initialHeight,
      padding: [12, 16, 0, 4],
      series: built.seriesConfigs,
      scales: {
        x: { time: false },
        y: {
          distr: useLogScale ? 3 : 1,
          range: (_u, dataMin, dataMax) => {
            const c = yClipRef.current;
            if (c) return paddedRange(c.min, c.max, chartTab, useLogScale);
            return paddedRange(dataMin, dataMax, chartTab, useLogScale);
          },
        },
      },
      axes: [
        {
          label: 'Step',
          stroke: 'rgba(203,213,225,0.62)',
          grid: { stroke: 'rgba(148,163,184,0.08)' },
          ticks: { stroke: 'rgba(148,163,184,0.18)' },
          labelFont: '12px sans-serif',
          font: '11px sans-serif',
          values: (_u, ticks) => ticks.map(tick => formatCompact(tick)),
        },
        {
          label: chartYAxisLabel(chartTab),
          stroke: 'rgba(203,213,225,0.62)',
          grid: { stroke: 'rgba(148,163,184,0.08)' },
          ticks: { stroke: 'rgba(148,163,184,0.18)' },
          size: 66,
          labelFont: '12px sans-serif',
          font: '11px sans-serif',
          values: (_u, ticks) => ticks.map(tick => formatNum(tick)),
        },
      ],
      cursor: {
        drag: { x: true, y: false, setScale: true },
        points: { show: false },
      },
      legend: { show: false },
      hooks: {
        drawClear: [u => drawPhaseBands(u, phaseTransitionsRef.current)],
        draw: [u => drawEventMarkers(u, eventMarkersRef.current, phaseTransitionsRef.current)],
        setCursor: [
          u => {
            const idx = u.cursor.idx;
            if (idx == null || idx < 0) {
              queueHover(null);
              return;
            }
            const xs = u.data[0] as number[];
            const step = xs[idx];
            if (step == null) {
              queueHover(null);
              return;
            }
            const items: HoverItem[] = [];
            for (let i = 1; i < u.data.length; i++) {
              const value = (u.data[i] as (number | null)[])[idx] ?? null;
              if (value == null || !Number.isFinite(value)) continue;
              const config = u.series[i];
              items.push({
                label: String(config.label ?? `Series ${i}`),
                value,
                color: typeof config.stroke === 'string' ? config.stroke : 'rgba(203,213,225,1)',
              });
            }
            queueHover({ step, items });
          },
        ],
        setScale: [
          (u, key) => {
            if (key !== 'x') return;
            const xs = u.data[0] as number[];
            if (!xs?.length) return;
            const sx = u.scales.x;
            setIsZoomed(sx.min !== xs[0] || sx.max !== xs[xs.length - 1]);
          },
        ],
      },
    };

    uplotRef.current = new uPlot(opts, built.data, containerRef.current);
    setIsZoomed(false);
    const fitted = computeCanvasSize(host);
    if (fitted) uplotRef.current.setSize(fitted);

    return () => {
      uplotRef.current?.destroy();
      uplotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structuralKey]);

  useEffect(() => {
    const u = uplotRef.current;
    if (!u) return;
    if (isZoomedRef.current) {
      u.setData(built.data, false);
      u.redraw(true, true);
    } else {
      u.setData(built.data, true);
    }
  }, [built]);

  useEffect(() => {
    const el = chartHostRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const u = uplotRef.current;
      if (!u) return;
      const fitted = computeCanvasSize(el);
      if (fitted) u.setSize(fitted);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [built.hasData]);

  const handleResetZoom = useCallback(() => {
    const u = uplotRef.current;
    if (!u) return;
    const xs = u.data[0] as number[];
    if (!xs?.length) return;
    u.setScale('x', { min: xs[0], max: xs[xs.length - 1] });
  }, []);

  const recentTimeline = useMemo(() => {
    const phaseItems = phaseTransitions.map(phase => ({
      step: phase.step,
      color: 'bg-slate-300',
      label: phase.name || `Phase ${phase.index + 1}`,
      detail: 'phase',
    }));
    const eventItems = eventMarkers.map(marker => ({
      step: marker.step,
      color:
        marker.kind === 'sample' ? 'bg-amber-400' : marker.kind === 'checkpoint' ? 'bg-emerald-400' : 'bg-rose-400',
      label: marker.label,
      detail: marker.kind.replace('_', ' '),
    }));
    return [...phaseItems, ...eventItems].sort((a, b) => b.step - a.step).slice(0, 12);
  }, [eventMarkers, phaseTransitions]);

  const noDataMessage =
    status === 'error'
      ? 'Failed to load training metrics.'
      : chartTab === 'learning_rate'
        ? 'No learning-rate points found yet.'
        : chartTab === 'throughput'
          ? 'No throughput telemetry found yet.'
          : chartTab === 'timesteps'
            ? 'No timestep diagnostics found yet.'
            : chartTab === 'gradients'
              ? 'No gradient diagnostics found yet.'
              : chartTab === 'memory'
                ? 'No memory diagnostics found yet.'
                : 'Waiting for loss points...';

  return (
    <div className="bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col h-full">
      <div className="bg-gray-800 px-4 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <div className="h-2 w-2 rounded-full bg-blue-400" />
          <h2 className="text-gray-100 text-sm font-medium">Training Monitor</h2>
          <span className="text-xs text-gray-400 truncate">
            {built.hasData
              ? `${built.totalPoints.toLocaleString()} rendered points`
              : status === 'error'
                ? 'Metrics unavailable'
                : status === 'loading' || status === 'refreshing'
                  ? 'Loading metrics...'
                  : 'No chart data yet'}
          </span>
        </div>

        <button
          type="button"
          onClick={refreshMetrics}
          className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md text-xs bg-gray-700/60 hover:bg-gray-700 text-gray-200 border border-gray-700"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Refresh
        </button>
      </div>

      <div className="p-4 space-y-3 flex-1 min-h-0 overflow-auto">
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-8 gap-3">
          <KpiCard
            icon={<Activity className="h-4 w-4 text-emerald-400" />}
            label="Progress"
            value={`${formatCompact(job.step)} / ${totalSteps ? formatCompact(totalSteps) : '--'}`}
            detail={progressPercent == null ? 'total steps unknown' : `${formatPercent(progressPercent)} complete`}
            accent="emerald"
            progress={progressPercent}
          />
          <KpiCard
            icon={<TrendingDown className="h-4 w-4 text-blue-400" />}
            label="Smoothed loss"
            value={formatNum(lossSummary.current)}
            detail={
              lossSummary.deltaPct == null
                ? primaryLossKey || 'no loss'
                : `${lossSummary.deltaPct >= 0 ? '+' : ''}${lossSummary.deltaPct.toFixed(2)}% trend`
            }
            accent={lossSummary.deltaPct != null && lossSummary.deltaPct > 0 ? 'rose' : 'blue'}
          />
          <KpiCard
            icon={<Gauge className="h-4 w-4 text-amber-400" />}
            label="Learning rate"
            value={formatNum(latestLearningRate, 3)}
            detail={learningRatePointCount > 0 ? `${learningRatePointCount.toLocaleString()} points` : 'not logged yet'}
            accent="amber"
          />
          <KpiCard
            icon={<Clock className="h-4 w-4 text-emerald-400" />}
            label="Speed / ETA"
            value={latestStepsPerSec == null ? job.speed_string || '--' : `${formatNum(latestStepsPerSec, 3)} steps/s`}
            detail={etaSeconds == null ? 'ETA unavailable' : `ETA ${formatDuration(etaSeconds)}`}
            accent="emerald"
          />
          <KpiCard
            icon={<Zap className="h-4 w-4 text-rose-400" />}
            label="GPU health"
            value={
              primaryGpu
                ? `${formatMemory(primaryGpu.memory.used)} / ${formatMemory(primaryGpu.memory.total)}`
                : job.gpu_ids === 'mps'
                  ? 'MPS'
                  : '--'
            }
            detail={
              primaryGpu
                ? `GPU ${primaryGpu.utilization.gpu}% | mem ${formatPercent(gpuMemoryPct)}`
                : 'live hardware data unavailable'
            }
            accent={primaryGpu && (primaryGpu.utilization.gpu > 92 || (gpuMemoryPct ?? 0) > 92) ? 'rose' : 'blue'}
            progress={gpuMemoryPct}
          />
          <KpiCard
            icon={<Gauge className="h-4 w-4 text-blue-400" />}
            label="Timestep"
            value={formatNum(latestTimestepMean, 3)}
            detail={latestSigmaMean == null ? 'sigma not logged yet' : `sigma ${formatNum(latestSigmaMean, 3)}`}
            accent="blue"
          />
          <KpiCard
            icon={<Activity className="h-4 w-4 text-emerald-400" />}
            label="Grad norm"
            value={formatNum(latestGradNorm, 3)}
            detail={latestGradLimit == null ? 'clip limit unknown' : `clip ${formatNum(latestGradLimit, 3)}`}
            accent={latestGradNorm != null && latestGradLimit != null && latestGradNorm > latestGradLimit ? 'rose' : 'emerald'}
          />
          <KpiCard
            icon={<Zap className="h-4 w-4 text-amber-400" />}
            label="Train VRAM"
            value={latestTrainVramGb == null ? '--' : `${formatNum(latestTrainVramGb, 3)} GB`}
            detail={latestTrainReservedGb == null ? 'backend memory not logged yet' : `reserved ${formatNum(latestTrainReservedGb, 3)} GB`}
            accent="amber"
          />
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_300px] gap-3 min-h-[520px]">
          <div className="bg-gray-950 rounded-lg border border-gray-800 min-h-[420px] flex flex-col overflow-hidden">
            <div className="px-3 py-2 border-b border-gray-800 flex flex-wrap items-center gap-2 justify-between">
              <div className="inline-flex rounded-md border border-gray-800 bg-gray-900 p-0.5">
                <ChartTabButton active={chartTab === 'loss'} onClick={() => setChartTab('loss')} label="Loss" />
                <ChartTabButton
                  active={chartTab === 'learning_rate'}
                  onClick={() => setChartTab('learning_rate')}
                  label="Learning Rate"
                />
                <ChartTabButton
                  active={chartTab === 'throughput'}
                  onClick={() => setChartTab('throughput')}
                  label="Throughput"
                />
                <ChartTabButton
                  active={chartTab === 'timesteps'}
                  onClick={() => setChartTab('timesteps')}
                  label="Timesteps"
                />
                <ChartTabButton
                  active={chartTab === 'gradients'}
                  onClick={() => setChartTab('gradients')}
                  label="Gradients"
                />
                <ChartTabButton
                  active={chartTab === 'memory'}
                  onClick={() => setChartTab('memory')}
                  label="Memory"
                />
              </div>

              <div className="flex items-center gap-2 text-xs text-gray-400">
                <span>server cap {CHART_MAX_POINTS.toLocaleString()}</span>
                {isZoomed && (
                  <button
                    type="button"
                    onClick={handleResetZoom}
                    className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs bg-blue-600/80 hover:bg-blue-600 text-white border border-blue-500/50"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    Reset zoom
                  </button>
                )}
              </div>
            </div>

            <div className="relative flex-1 min-h-0">
              {!built.hasData ? (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-400">
                  {noDataMessage}
                </div>
              ) : (
                <>
                  <div ref={chartHostRef} className="absolute inset-0 overflow-hidden">
                    <div ref={containerRef} />
                  </div>
                  <HoverReadout hover={hover} />
                </>
              )}
            </div>
          </div>

          <div className="space-y-3">
            <div className="bg-gray-950 rounded-lg border border-gray-800 p-3">
              <div className="text-xs text-gray-400 mb-2">Display</div>
              <div className="flex flex-wrap gap-2">
                <ToggleButton checked={showSmoothed} onClick={() => setShowSmoothed(v => !v)} label="Smooth" />
                <ToggleButton checked={showTrend} onClick={() => setShowTrend(v => !v)} label="Trend" />
                <ToggleButton checked={showRaw} onClick={() => setShowRaw(v => !v)} label="Raw" />
                <ToggleButton checked={useLogScale} onClick={() => setUseLogScale(v => !v)} label="Log Y" />
                <ToggleButton checked={clipOutliers} onClick={() => setClipOutliers(v => !v)} label="Clip" />
              </div>

              <div className="mt-4">
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-xs text-gray-400">Smoothing</label>
                  <span className="text-xs text-gray-300">{smoothing}%</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={smoothing}
                  onChange={event => {
                    const value = Number(event.target.value);
                    startTransition(() => setSmoothing(value));
                  }}
                  className="w-full accent-blue-500"
                  disabled={!showSmoothed && !showTrend}
                />
              </div>
            </div>

            <div className="bg-gray-950 rounded-lg border border-gray-800 p-3">
              <div className="text-xs text-gray-400 mb-2">Loss Series</div>
              {lossKeys.length === 0 ? (
                <div className="text-sm text-gray-500">No loss keys found yet.</div>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {lossKeys.map(key => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => setEnabledLoss(prev => ({ ...prev, [key]: !(prev[key] ?? true) }))}
                      className={[
                        'px-2.5 py-1 rounded-md text-xs border transition-colors max-w-full truncate',
                        enabledLoss[key] === false
                          ? 'bg-gray-900 text-gray-500 border-gray-800 hover:bg-gray-800/60'
                          : 'bg-gray-900 text-gray-200 border-gray-700 hover:bg-gray-800/80',
                      ].join(' ')}
                      aria-pressed={enabledLoss[key] !== false}
                      title={key}
                    >
                      <span
                        className="inline-block h-2 w-2 rounded-full mr-1.5"
                        style={{ background: lossColorForKey(key) }}
                      />
                      {cleanLabel(key)}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="bg-gray-950 rounded-lg border border-gray-800 p-3">
              <div className="text-xs text-gray-400 mb-2">Latest training stats</div>
              <div className="space-y-1.5">
                <StatRow label="loss final" value={formatNum(latestLossFinal)} />
                <StatRow label="loss unclipped" value={formatNum(latestLossUnclipped)} />
                <StatRow label="batch" value={`${formatNum(latestBatchSize, 3)} / eff ${formatNum(latestEffectiveBatch, 3)}`} />
                <StatRow label="pred std" value={formatNum(latestNoisePredStd, 3)} />
                <StatRow label="target std" value={formatNum(latestTargetStd, 3)} />
                <StatRow label="sigma" value={formatNum(latestSigmaMean, 3)} />
              </div>
            </div>

            <div className="bg-gray-950 rounded-lg border border-gray-800 p-3">
              <div className="text-xs text-gray-400 mb-2">Current Phase</div>
              <div className="text-sm text-gray-100 truncate">
                {currentPhase ? currentPhase.name || `Phase ${currentPhase.index + 1}` : 'No phase data'}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {currentPhase ? `started at step ${formatCompact(currentPhase.step)}` : 'Old jobs may not include phases.'}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-gray-950 rounded-lg border border-gray-800 p-3">
          <div className="flex items-center justify-between gap-3 mb-3">
            <div className="text-xs text-gray-400">Events & phase timeline</div>
            <div className="flex items-center gap-3 text-[11px] text-gray-500">
              <LegendDot className="bg-amber-400" label="samples" icon={<ImageIcon className="h-3 w-3" />} />
              <LegendDot className="bg-emerald-400" label="checkpoints" icon={<Save className="h-3 w-3" />} />
              <LegendDot className="bg-rose-400" label="phase changes" />
            </div>
          </div>

          {recentTimeline.length === 0 ? (
            <div className="text-sm text-gray-500">No samples, checkpoints, or phase changes logged yet.</div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-2">
              {recentTimeline.map((item, index) => (
                <div key={`${item.detail}-${item.step}-${index}`} className="flex items-center gap-2 rounded-md bg-gray-900/70 border border-gray-800 px-3 py-2 min-w-0">
                  <span className={`h-2 w-2 rounded-full shrink-0 ${item.color}`} />
                  <div className="min-w-0">
                    <div className="text-xs text-gray-200 truncate">{item.label}</div>
                    <div className="text-[11px] text-gray-500">step {formatCompact(item.step)}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <style jsx global>{`
        .uplot,
        .uplot * {
          font-family: inherit;
        }
        .uplot {
          background: transparent;
        }
        .uplot .u-select {
          background: rgba(59, 130, 246, 0.14);
          border: 1px solid rgba(59, 130, 246, 0.38);
        }
      `}</style>
    </div>
  );
}

function KpiCard({
  icon,
  label,
  value,
  detail,
  accent,
  progress,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
  accent: 'blue' | 'amber' | 'emerald' | 'rose';
  progress?: number | null;
}) {
  const barColor =
    accent === 'emerald' ? 'bg-emerald-500' : accent === 'amber' ? 'bg-amber-500' : accent === 'rose' ? 'bg-rose-500' : 'bg-blue-500';

  return (
    <div className="bg-gray-950 border border-gray-800 rounded-lg p-3 min-w-0">
      <div className="flex items-center gap-2 text-xs text-gray-400">
        {icon}
        <span>{label}</span>
      </div>
      <div className="mt-2 text-lg font-semibold text-gray-100 truncate">{value}</div>
      <div className="mt-1 text-xs text-gray-500 truncate">{detail}</div>
      {progress != null && Number.isFinite(progress) && (
        <div className="mt-3 h-1 rounded-full bg-gray-800 overflow-hidden">
          <div className={`h-full rounded-full ${barColor}`} style={{ width: `${clamp(progress, 0, 100)}%` }} />
        </div>
      )}
    </div>
  );
}

function ChartTabButton({ active, onClick, label }: { active: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'px-3 py-1.5 rounded text-xs transition-colors',
        active ? 'bg-blue-500/15 text-blue-200' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800',
      ].join(' ')}
    >
      {label}
    </button>
  );
}

function ToggleButton({ checked, onClick, label }: { checked: boolean; onClick: () => void; label: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        'px-2.5 py-1 rounded-md text-xs border transition-colors',
        checked
          ? 'bg-blue-500/10 text-blue-300 border-blue-500/30 hover:bg-blue-500/15'
          : 'bg-gray-900 text-gray-400 border-gray-800 hover:bg-gray-800/60',
      ].join(' ')}
      aria-pressed={checked}
    >
      {label}
    </button>
  );
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 text-xs">
      <span className="text-gray-500 truncate">{label}</span>
      <span className="text-gray-200 font-medium truncate">{value}</span>
    </div>
  );
}

function HoverReadout({ hover }: { hover: HoverState | null }) {
  if (!hover || hover.items.length === 0) return null;
  return (
    <div className="absolute top-3 left-3 z-10 rounded-md border border-gray-800 bg-gray-950/92 px-3 py-2 shadow-lg max-w-[260px]">
      <div className="text-[11px] text-gray-500 mb-1">step {formatCompact(hover.step)}</div>
      <div className="space-y-1">
        {hover.items.slice(0, 6).map(item => (
          <div key={item.label} className="flex items-center justify-between gap-3 text-xs">
            <span className="min-w-0 truncate text-gray-300">
              <span className="inline-block h-2 w-2 rounded-full mr-1.5" style={{ background: item.color }} />
              {item.label}
            </span>
            <span className="font-medium text-gray-100">{formatNum(item.value)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function LegendDot({ className, label, icon }: { className: string; label: string; icon?: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1">
      {icon}
      <span className={`h-2 w-2 rounded-full ${className}`} />
      {label}
    </span>
  );
}
