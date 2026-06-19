'use client';

import { useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';

type DatasetWatcherListItem = {
  id: string;
  enabled?: boolean;
};

type DatasetWatcherStatus = {
  lastScanAt?: string | null;
  lastImportedAt?: string | null;
  lastImportedCount?: number;
  lastCaptionedCount?: number;
};

type DatasetWatcherLiveRefreshOptions = {
  enabled: boolean;
  datasetName?: string | null;
  projectID?: string | null;
  workerID?: string;
  intervalMs?: number;
  onRefresh: () => void;
};

function watcherStatusSignature(statuses: Record<string, DatasetWatcherStatus>) {
  return Object.entries(statuses)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([id, status]) =>
      [id, status.lastImportedAt || '', status.lastImportedCount || 0, status.lastCaptionedCount || 0].join(':'),
    )
    .join('|');
}

function watcherStatusesHaveRecentWork(statuses: Record<string, DatasetWatcherStatus>, sinceMs: number) {
  return Object.values(statuses).some(status => {
    const importedAt = status.lastImportedAt ? Date.parse(status.lastImportedAt) : NaN;
    if (Number.isFinite(importedAt) && importedAt >= sinceMs) return true;

    const scanAt = status.lastScanAt ? Date.parse(status.lastScanAt) : NaN;
    if (!Number.isFinite(scanAt) || scanAt < sinceMs) return false;
    return Boolean((status.lastImportedCount || 0) > 0 || (status.lastCaptionedCount || 0) > 0);
  });
}

export default function useDatasetWatcherLiveRefresh({
  enabled,
  datasetName,
  projectID = null,
  workerID = 'local',
  intervalMs = 5000,
  onRefresh,
}: DatasetWatcherLiveRefreshOptions) {
  const [hasActiveWatchers, setHasActiveWatchers] = useState(false);
  const onRefreshRef = useRef(onRefresh);
  const signatureRef = useRef('');
  const initializedRef = useRef(false);
  const busyRef = useRef(false);
  const startedAtRef = useRef(Date.now());

  useEffect(() => {
    onRefreshRef.current = onRefresh;
  }, [onRefresh]);

  useEffect(() => {
    signatureRef.current = '';
    initializedRef.current = false;
    startedAtRef.current = Date.now();
  }, [datasetName, enabled, projectID, workerID]);

  useEffect(() => {
    if (!enabled) {
      setHasActiveWatchers(false);
      return;
    }

    let cancelled = false;

    const pollWatchers = async () => {
      if (busyRef.current) return;
      busyRef.current = true;
      try {
        const res = await apiClient.get('/api/datasets/watchers', {
          params: {
            ...(datasetName ? { datasetName } : {}),
            worker_id: workerID,
            ...(projectID ? { project_id: projectID } : {}),
          },
        });
        if (cancelled) return;

        const watchers = Array.isArray(res.data?.watchers) ? (res.data.watchers as DatasetWatcherListItem[]) : [];
        const statuses = (res.data?.statuses || {}) as Record<string, DatasetWatcherStatus>;
        setHasActiveWatchers(watchers.some(watcher => watcher.enabled !== false));

        const signature = watcherStatusSignature(statuses);
        const initialized = initializedRef.current;
        const changed = initialized && signature !== signatureRef.current;
        const recentInitialWork = !initialized && watcherStatusesHaveRecentWork(statuses, startedAtRef.current);

        signatureRef.current = signature;
        initializedRef.current = true;

        if (changed || recentInitialWork) onRefreshRef.current();
      } catch (error) {
        if (!cancelled) {
          console.warn('Dataset watcher status refresh failed:', error);
          setHasActiveWatchers(false);
        }
      } finally {
        busyRef.current = false;
      }
    };

    void pollWatchers();
    const interval = window.setInterval(pollWatchers, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [datasetName, enabled, intervalMs, projectID, workerID]);

  return hasActiveWatchers;
}
