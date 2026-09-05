'use client';

import { useEffect, useState, useRef, useCallback, useMemo } from 'react';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';

export interface LossPoint {
  step: number;
  wall_time?: number;
  value: number | null;
  value_text?: string | null;
}

type SeriesMap = Record<string, LossPoint[]>;

function isMarkerKey(key: string) {
  return key === 'phase/index' || key.startsWith('event/');
}

export default function useJobLossLog(jobID: string, reloadInterval: null | number = null) {
  const [series, setSeries] = useState<SeriesMap>({});
  const [keys, setKeys] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const didInitialLoadRef = useRef(false);
  const inFlightRef = useRef(false);
  const activeJobIDRef = useRef(jobID);

  // track last step per key so polling is incremental per series
  const lastStepByKeyRef = useRef<Record<string, number | null>>({});

  const lossKeys = useMemo(() => {
    const base = (keys ?? []).filter(key => !isMarkerKey(key));
    // if keys table is empty early on, fall back to just "loss"
    if (base.length === 0) return ['loss'];
    return [...base].sort();
  }, [keys]);

  const refreshLoss = useCallback(
    async (signal?: AbortSignal) => {
      if (!jobID) return;

      if (inFlightRef.current) return;
      inFlightRef.current = true;

      const loadStatus: 'loading' | 'refreshing' = didInitialLoadRef.current ? 'refreshing' : 'loading';
      setStatus(loadStatus);

      try {
        // Step 1: get key list (we can do this by calling endpoint once; it returns keys)
        // Keep it cheap: limit=1.
        const first = await apiClient
          .get(`/api/jobs/${jobID}/loss`, { params: { key: 'loss', limit: 1 }, signal })
          .then(res => res.data as { keys?: string[] });

        if (signal?.aborted || activeJobIDRef.current !== jobID) return;
        const newKeys = first.keys ?? [];
        setKeys(newKeys);

        const wantedKeys = (newKeys.length ? [...newKeys] : ['loss']).sort();

        // Step 2: fetch each chart key incrementally (since_step per key if polling)
        const requests = wantedKeys.map(k => {
          const params: Record<string, any> = { key: k };

          if (reloadInterval && lastStepByKeyRef.current[k] != null) {
            params.since_step = lastStepByKeyRef.current[k];
          }

          params.limit = 1000000;

          return apiClient
            .get(`/api/jobs/${jobID}/loss`, { params, signal })
            .then(res => res.data as { key: string; points?: LossPoint[] });
        });

        const results = await Promise.all(requests);
        if (signal?.aborted || activeJobIDRef.current !== jobID) return;

        setSeries(prev => {
          const next: SeriesMap = { ...prev };

          for (const r of results) {
            const k = r.key;
            const newPoints = (r.points ?? []).filter(p => p.value !== null);

            if (!didInitialLoadRef.current) {
              // initial: replace
              next[k] = newPoints;
            } else if (newPoints.length) {
              const existing = next[k] ?? [];
              const prevLast = existing.length ? existing[existing.length - 1].step : null;
              const filtered = prevLast == null ? newPoints : newPoints.filter(p => p.step > prevLast);
              next[k] = filtered.length ? [...existing, ...filtered] : existing;
            } else {
              // no new points: keep existing
              next[k] = next[k] ?? [];
            }

            // update last step per key
            const finalArr = next[k] ?? [];
            lastStepByKeyRef.current[k] = finalArr.length
              ? finalArr[finalArr.length - 1].step
              : (lastStepByKeyRef.current[k] ?? null);
          }

          // remove stale chart keys that no longer exist (rare, but keeps UI clean)
          for (const existingKey of Object.keys(next)) {
            if (!wantedKeys.includes(existingKey)) {
              delete next[existingKey];
              delete lastStepByKeyRef.current[existingKey];
            }
          }

          return next;
        });

        setStatus('success');
        didInitialLoadRef.current = true;
      } catch (err) {
        if (signal?.aborted || activeJobIDRef.current !== jobID) return;
        console.error('Error fetching loss logs:', err);
        setStatus('error');
      } finally {
        if (activeJobIDRef.current === jobID) {
          inFlightRef.current = false;
        }
      }
    },
    [jobID, reloadInterval],
  );

  useEffect(() => {
    // reset when job changes
    activeJobIDRef.current = jobID;
    inFlightRef.current = false;
    didInitialLoadRef.current = false;
    lastStepByKeyRef.current = {};
    setSeries({});
    setKeys([]);
    setStatus('idle');
  }, [jobID]);

  usePollLoop(signal => refreshLoss(signal), reloadInterval, [jobID]);

  return {
    series,
    keys,
    lossKeys,
    phasePoints: series['phase/index'] ?? [],
    status,
    refreshLoss: () => refreshLoss(),
    setSeries,
  };
}
