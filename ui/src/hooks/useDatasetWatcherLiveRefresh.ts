'use client';
import { useEffect, useRef } from 'react';
import useScopedState from './useScopedState';
import { apiClient } from '@/utils/api';
import type { DatasetWatcherListItem, DatasetWatcherStatus } from '@/utils/datasetWatcherStatus';
type DatasetWatcherLiveRefreshOptions = {
  enabled: boolean;
  datasetName?: string | null;
  workerID?: string;
  intervalMs?: number;
  onRefresh: () => void;
};
function watcherStatusSignature(statuses: Record<string, DatasetWatcherStatus>) {
  return Object.entries(statuses)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([id, status]) =>
      [
        id,
        status.lastImportedAt || '',
        status.lastImportedCount || 0,
        status.lastCaptionedCount || 0,
        status.autoCaptionTotalCount || 0,
        status.autoCaptionPendingCount || 0,
        status.autoCaptionCompletedCount || 0,
        status.autoCaptionActivePath || '',
      ].join(':'),
    )
    .join('|');
}
function datasetRefreshSignature(statuses: Record<string, DatasetWatcherStatus>) {
  return Object.entries(statuses)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([id, status]) =>
      [
        id,
        status.lastImportedAt || '',
        status.lastImportedCount || 0,
        status.lastCaptionedCount || 0,
        status.lastError || '',
      ].join(':'),
    )
    .join('|');
}
function watcherStatusesHaveRecentWork(statuses: Record<string, DatasetWatcherStatus>, sinceMs: number) {
  return Object.values(statuses).some(status => {
    const importedAt = status.lastImportedAt ? Date.parse(status.lastImportedAt) : NaN;
    if (Number.isFinite(importedAt) && importedAt >= sinceMs) return true;
    const scanAt = status.lastScanAt ? Date.parse(status.lastScanAt) : NaN;
    if (!Number.isFinite(scanAt) || scanAt < sinceMs) return false;
    return Boolean(
      (status.lastImportedCount || 0) > 0 ||
        (status.lastCaptionedCount || 0) > 0 ||
        (status.autoCaptionPendingCount || 0) > 0 ||
        (status.autoCaptionCompletedCount || 0) > 0,
    );
  });
}
export default function useDatasetWatcherLiveRefresh({
  enabled,
  datasetName,
  workerID = 'local',
  intervalMs = 5000,
  onRefresh,
}: DatasetWatcherLiveRefreshOptions) {
  const scope = JSON.stringify([enabled, datasetName, workerID]);
  const [hasActiveWatchers, setHasActiveWatchers] = useScopedState(scope, false);
  const [watchers, setWatchers] = useScopedState<DatasetWatcherListItem[]>(scope, []);
  const [statuses, setStatuses] = useScopedState<Record<string, DatasetWatcherStatus>>(scope, {});
  const onRefreshRef = useRef(onRefresh);
  const statusSignatureRef = useRef('');
  const refreshSignatureRef = useRef('');
  const initializedRef = useRef(false);
  const busyRef = useRef(false);
  const startedAtRef = useRef(0);
  useEffect(() => {
    onRefreshRef.current = onRefresh;
  }, [onRefresh]);
  useEffect(() => {
    statusSignatureRef.current = '';
    refreshSignatureRef.current = '';
    initializedRef.current = false;
    startedAtRef.current = Date.now();
  }, [datasetName, enabled, null, workerID]);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const pollWatchers = async () => {
      if (busyRef.current) return;
      busyRef.current = true;
      try {
        const res = await apiClient.get('/api/datasets/watchers', {
          params: {
            ...(datasetName ? { datasetName } : {}),
            worker_id: workerID,
          },
        });
        if (cancelled) return;
        const watchers = Array.isArray(res.data?.watchers) ? (res.data.watchers as DatasetWatcherListItem[]) : [];
        const statuses = (res.data?.statuses || {}) as Record<string, DatasetWatcherStatus>;
        setHasActiveWatchers(watchers.some(watcher => watcher.enabled !== false));
        setWatchers(watchers);
        const statusSignature = watcherStatusSignature(statuses);
        const refreshSignature = datasetRefreshSignature(statuses);
        const initialized = initializedRef.current;
        const statusChanged = !initialized || statusSignature !== statusSignatureRef.current;
        const changed = initialized && refreshSignature !== refreshSignatureRef.current;
        const recentInitialWork = !initialized && watcherStatusesHaveRecentWork(statuses, startedAtRef.current);
        if (statusChanged) setStatuses(statuses);
        statusSignatureRef.current = statusSignature;
        refreshSignatureRef.current = refreshSignature;
        initializedRef.current = true;
        if (changed || recentInitialWork) onRefreshRef.current();
      } catch (error) {
        if (!cancelled) {
          console.warn('Dataset watcher status refresh failed:', error);
          setHasActiveWatchers(false);
          setWatchers([]);
          setStatuses({});
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
  }, [datasetName, enabled, intervalMs, null, workerID, setHasActiveWatchers, setWatchers, setStatuses]);
  return { hasActiveWatchers, watchers, statuses };
}
