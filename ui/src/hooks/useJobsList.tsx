'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { Job } from '@/types';
import { apiClient } from '@/utils/api';
import usePollLoop from './usePollLoop';
type UseJobsListProps = { onlyActive?: boolean; reloadInterval?: number | null; job_type?: string | null; view?: 'active' | 'history' | 'failed' | 'all' };
type Page = { jobs: Job[]; nextCursor: string | null; freshness?: { updatedAt: string; failed: number } | null };
export default function useJobsList({ onlyActive = false, reloadInterval = null, job_type = null, view = 'all' }: UseJobsListProps = {}) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [isRefreshing, setRefreshing] = useState(false);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [freshness, setFreshness] = useState<Page['freshness']>(null);
  const scope = JSON.stringify([job_type, onlyActive, view]);
  const activeScope = useRef(scope); activeScope.current = scope;
  const pages = useRef<Array<Page & { cursor: string | null }>>([]);
  const pollPage = useRef(0);
  const inflight = useRef<AbortController | null>(null);
  const refreshJobs = useCallback(async (signal?: AbortSignal, more = false) => {
    if (inflight.current || activeScope.current !== scope) return;
    const controller = new AbortController(); inflight.current = controller;
    const abort = () => controller.abort(); signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) controller.abort();
    setRefreshing(true);
    setStatus(current => current === 'idle' ? 'loading' : current);
    try {
      // One bounded page per poll; rotate through loaded pages instead of re-fetching
      // the entire history on every tick. Load more fetches only the next page.
      const index = more ? pages.current.length : pollPage.current % Math.max(1, pages.current.length);
      const cursor = more ? pages.current.at(-1)?.nextCursor ?? null : pages.current[index]?.cursor ?? null;
      if (more && !cursor) return;
      const response = await apiClient.get<Page>('/api/jobs', { params: { job_type: job_type || undefined,
        view: onlyActive ? 'active' : view, cursor: cursor || undefined, limit: 50 }, signal: controller.signal });
      const data = response.data;
      if (!Array.isArray(data.jobs) || data.nextCursor !== null && typeof data.nextCursor !== 'string') throw new Error('Invalid job page');
      if (controller.signal.aborted || activeScope.current !== scope) return;
      pages.current[index] = { ...data, cursor };
      if (!data.nextCursor) pages.current = pages.current.slice(0, index + 1);
      else if (pages.current[index + 1]) pages.current[index + 1].cursor = data.nextCursor;
      pollPage.current = index + 1;
      setFreshness(data.freshness);
      setJobs([...new Map(pages.current.flatMap(page => page.jobs).map(job => [job.id, job])).values()]);
      setNextCursor(pages.current.at(-1)?.nextCursor ?? null); setStatus('success');
    } catch {
      if (!controller.signal.aborted && activeScope.current === scope) setStatus('error');
    } finally { signal?.removeEventListener('abort', abort); if (inflight.current === controller) inflight.current = null; if (activeScope.current === scope) setRefreshing(false); }
  }, [job_type, onlyActive, view, scope]);
  useEffect(() => {
    pages.current = []; pollPage.current = 0; setJobs([]); setNextCursor(null); setFreshness(null); setStatus('idle');
    return () => { inflight.current?.abort(); inflight.current = null; };
  }, [scope]);
  usePollLoop(signal => refreshJobs(signal), reloadInterval, [scope]);
  return { jobs, setJobs, status, isRefreshing, freshness, hasMore: !!nextCursor,
    loadMore: () => refreshJobs(undefined, true), refreshJobs: () => refreshJobs() };
}
