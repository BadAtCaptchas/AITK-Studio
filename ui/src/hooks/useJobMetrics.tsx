'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';

export interface MetricPoint {
  step: number;
  wall_time?: number;
  value: number | null;
  value_text?: string | null;
}

export interface MetricSeriesPayload {
  key: string;
  totalCount: number;
  firstStep: number | null;
  lastStep: number | null;
  latest: MetricPoint | null;
  downsampled: boolean;
  points: MetricPoint[];
}

export interface MetricsResponse {
  keys: string[];
  keyInfo: { key: string; first_seen_step: number | null; last_seen_step: number | null }[];
  series: Record<string, MetricSeriesPayload>;
}

type SeriesMap = Record<string, MetricPoint[]>;
type LatestMap = Record<string, MetricPoint | null>;
type Status = 'idle' | 'loading' | 'success' | 'error' | 'refreshing';
type UseJobMetricsOptions =
  | number
  | {
      maxPoints?: number;
      keys?: string[];
      enabled?: boolean;
    };

const DEFAULT_METRIC_KEYS = ['*'];
const DEFAULT_MAX_POINTS = 4000;
const COMPACT_THRESHOLD = 5000;

function isTerminalStatus(status?: string) {
  return status === 'completed' || status === 'error' || status === 'stopped';
}

function isSlowPollingStatus(status?: string) {
  return status === 'queued' || status === 'stopping';
}

function encodeSinceSteps(lastStepByKey: Record<string, number | null>) {
  const parts = Object.entries(lastStepByKey)
    .filter(([, step]) => step != null)
    .map(([key, step]) => `${encodeURIComponent(key)}:${step}`);
  return parts.length ? parts.join(',') : undefined;
}

function normalizeOptions(options: UseJobMetricsOptions | undefined) {
  if (typeof options === 'number') {
    return {
      maxPoints: options,
      keys: DEFAULT_METRIC_KEYS,
      enabled: true,
    };
  }

  const keys = options?.keys?.map(key => key.trim()).filter(Boolean);
  return {
    maxPoints: options?.maxPoints ?? DEFAULT_MAX_POINTS,
    keys: keys?.length ? keys : DEFAULT_METRIC_KEYS,
    enabled: options?.enabled ?? true,
  };
}

function isChartMetricKey(key: string) {
  return !key.startsWith('event/') && !key.startsWith('phase/');
}

export default function useJobMetrics(
  jobID: string,
  jobStatus?: string,
  options: UseJobMetricsOptions = DEFAULT_MAX_POINTS,
) {
  const normalizedOptions = normalizeOptions(options);
  const maxPoints = normalizedOptions.maxPoints;
  const metricKeys = normalizedOptions.keys.join(',');
  const enabled = normalizedOptions.enabled;
  const [series, setSeries] = useState<SeriesMap>({});
  const [latest, setLatest] = useState<LatestMap>({});
  const [keys, setKeys] = useState<string[]>([]);
  const [status, setStatus] = useState<Status>('idle');
  const [version, setVersion] = useState(0);

  const didInitialLoadRef = useRef(false);
  const inFlightRef = useRef(false);
  const needsFullRefreshRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const lastStepByKeyRef = useRef<Record<string, number | null>>({});

  const refreshMetrics = useCallback(
    async (options: { full?: boolean } = {}) => {
      if (!enabled || !jobID || inFlightRef.current) return;

      const full = options.full || !didInitialLoadRef.current || needsFullRefreshRef.current;
      needsFullRefreshRef.current = false;
      inFlightRef.current = true;
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setStatus(didInitialLoadRef.current ? 'refreshing' : 'loading');

      try {
        const params: Record<string, string | number | undefined> = {
          keys: metricKeys,
          max_points: maxPoints,
        };
        if (!full) {
          params.since_steps = encodeSinceSteps(lastStepByKeyRef.current);
        }

        const response = await apiClient
          .get(`/api/jobs/${jobID}/metrics`, { params, signal: controller.signal })
          .then(res => res.data as MetricsResponse);

        setKeys(response.keys ?? []);
        setLatest(prev => {
          const next = full ? {} : { ...prev };
          for (const [key, payload] of Object.entries(response.series ?? {})) {
            next[key] = payload.latest ?? next[key] ?? null;
          }
          return next;
        });
        setSeries(prev => {
          const next: SeriesMap = full ? {} : { ...prev };
          let shouldCompact = false;

          for (const [key, payload] of Object.entries(response.series ?? {})) {
            const incoming = payload.points ?? [];
            if (full || !didInitialLoadRef.current) {
              next[key] = incoming;
            } else if (incoming.length) {
              const existing = next[key] ?? [];
              const prevLast = existing.length ? existing[existing.length - 1].step : null;
              const filtered = prevLast == null ? incoming : incoming.filter(point => point.step > prevLast);
              next[key] = filtered.length ? [...existing, ...filtered] : existing;
            } else {
              next[key] = next[key] ?? [];
            }

            const finalArr = next[key] ?? [];
            lastStepByKeyRef.current[key] = finalArr.length
              ? finalArr[finalArr.length - 1].step
              : (lastStepByKeyRef.current[key] ?? null);

            if (finalArr.length > COMPACT_THRESHOLD) {
              shouldCompact = true;
            }
          }

          if (shouldCompact) needsFullRefreshRef.current = true;
          return next;
        });
        setVersion(v => v + 1);
        setStatus('success');
        didInitialLoadRef.current = true;
      } catch (error: any) {
        if (error?.name !== 'CanceledError' && error?.code !== 'ERR_CANCELED') {
          console.error('Error fetching job metrics:', error);
          setStatus('error');
        }
      } finally {
        inFlightRef.current = false;
      }
    },
    [enabled, jobID, maxPoints, metricKeys],
  );

  useEffect(() => {
    abortRef.current?.abort();
    didInitialLoadRef.current = false;
    needsFullRefreshRef.current = false;
    lastStepByKeyRef.current = {};
    setSeries({});
    setLatest({});
    setKeys([]);
    setVersion(0);
    setStatus('idle');
    if (enabled) {
      refreshMetrics({ full: true });
    }

    return () => abortRef.current?.abort();
  }, [enabled, jobID, refreshMetrics]);

  const pollInterval = useMemo(() => {
    if (!enabled) return null;
    if (isTerminalStatus(jobStatus)) return null;
    if (isSlowPollingStatus(jobStatus)) return 10_000;
    return 2_000;
  }, [enabled, jobStatus]);

  useEffect(() => {
    if (!pollInterval) return;
    const interval = setInterval(() => {
      refreshMetrics({ full: needsFullRefreshRef.current });
    }, pollInterval);
    return () => clearInterval(interval);
  }, [pollInterval, refreshMetrics]);

  const lossKeys = useMemo(() => keys.filter(isChartMetricKey).sort(), [keys]);
  const eventKeys = useMemo(() => keys.filter(key => key.startsWith('event/')).sort(), [keys]);

  return {
    series,
    latest,
    keys,
    lossKeys: lossKeys.length ? lossKeys : ['loss'],
    eventKeys,
    phasePoints: series['phase/index'] ?? [],
    phaseNamePoints: series['phase/name'] ?? [],
    status,
    version,
    refreshMetrics: () => refreshMetrics({ full: true }),
  };
}
