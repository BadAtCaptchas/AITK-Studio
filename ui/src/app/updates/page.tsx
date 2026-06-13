'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import classNames from 'classnames';
import {
  AlertCircle,
  CheckCircle2,
  Download,
  ExternalLink,
  GitCommitHorizontal,
  GitCompareArrows,
  GitPullRequestArrow,
  Loader2,
  Power,
  RefreshCw,
} from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import { UPDATER_STATUS_EVENT, broadcastUpdaterStatus } from '@/utils/updaterEvents';

type RepoUpdateState =
  | 'pending'
  | 'checking'
  | 'up_to_date'
  | 'update_available'
  | 'unknown_current'
  | 'updating'
  | 'restarting'
  | 'updated'
  | 'update_failed'
  | 'update_blocked'
  | 'update_conflict'
  | 'error'
  | 'unsupported'
  | 'disabled'
  | 'stopped';

interface RepoUpdateCommit {
  sha: string;
  shortSha: string;
  message: string;
  body?: string | null;
  authorName?: string | null;
  authorDate?: string | null;
  committerName?: string | null;
  committerDate?: string | null;
  url?: string | null;
}

interface RepoUpdateStatus {
  state: RepoUpdateState;
  message: string;
  checkedAt?: string | null;
  updatedAt?: string | null;
  branch?: string | null;
  upstream?: string | null;
  repoFullName?: string | null;
  repoWebUrl?: string | null;
  downloadUrl?: string | null;
  latestVersion?: string | null;
  remote?: string | null;
  remoteCommitDate?: string | null;
  sourceRemoteWebUrl?: string | null;
  sourceRemoteMatchesCanonical?: boolean | null;
  compareUrl?: string | null;
  localVersion?: string | null;
  localShortCommit?: string | null;
  remoteShortCommit?: string | null;
  recentCommits?: RepoUpdateCommit[];
  ahead?: number | null;
  behind?: number | null;
  canApplyUpdate?: boolean | null;
  applyUpdateUnavailableReason?: string | null;
  updateStep?: string | null;
  updateError?: string | null;
  restartStartedAt?: string | null;
  restartStep?: string | null;
  restartPid?: number | null;
  restartChildPid?: number | null;
  restartError?: string | null;
  stashRef?: string | null;
  needsRestart?: boolean | null;
  error?: string | null;
}

const toneByState: Record<string, string> = {
  up_to_date: 'border-emerald-800 bg-emerald-950/20 text-emerald-100',
  updated: 'border-emerald-800 bg-emerald-950/20 text-emerald-100',
  update_available: 'border-amber-800 bg-amber-950/20 text-amber-100',
  updating: 'border-cyan-800 bg-cyan-950/20 text-cyan-100',
  restarting: 'border-cyan-800 bg-cyan-950/20 text-cyan-100',
  checking: 'border-cyan-800 bg-cyan-950/20 text-cyan-100',
  update_blocked: 'border-amber-800 bg-amber-950/20 text-amber-100',
  update_conflict: 'border-amber-800 bg-amber-950/20 text-amber-100',
  update_failed: 'border-rose-800 bg-rose-950/20 text-rose-100',
  error: 'border-rose-800 bg-rose-950/20 text-rose-100',
};

const APPLY_FEEDBACK_TIMEOUT_MS = 2 * 60 * 1000;
const RESTART_WAIT_TIMEOUT_MS = 5 * 60 * 1000;
const RESTART_RELOAD_DELAY_MS = 8000;

function formatDate(value?: string | null) {
  if (!value) return 'Unknown';
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return 'Unknown';
  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function plural(value: number, singular: string, pluralLabel = `${singular}s`) {
  return `${value} ${value === 1 ? singular : pluralLabel}`;
}

function getStatusLabel(status: RepoUpdateStatus | null) {
  if (!status) return 'Loading updates';
  if (status.state === 'up_to_date') return 'Up to date';
  if (status.state === 'update_available') return 'Update available';
  if (status.state === 'updating') return 'Updating';
  if (status.state === 'restarting') return 'Restarting';
  if (status.state === 'updated') return 'Updated';
  if (status.state === 'unknown_current') return 'Latest on GitHub';
  if (status.state === 'update_blocked') return 'Manual update needed';
  if (status.state === 'update_conflict') return 'Update needs attention';
  if (status.state === 'update_failed') return 'Update failed';
  if (status.state === 'checking') return 'Checking';
  return status.message || 'Updates';
}

function getStatusDetail(status: RepoUpdateStatus | null) {
  if (!status) return 'Waiting for status';
  if (status.state === 'update_available') {
    const behind = Number(status.behind || 0);
    if (behind > 0) return `${plural(behind, 'commit')} behind ${status.upstream || 'GitHub'}`;
  }
  if (status.state === 'updating') return status.updateStep?.replaceAll('-', ' ') || 'Applying update';
  if (status.state === 'restarting') return status.restartStep?.replaceAll('-', ' ') || 'Restarting app';
  if (status.state === 'updated' && status.needsRestart) return 'Restart required';
  if (status.state === 'update_blocked') return status.applyUpdateUnavailableReason || status.message;
  if (status.state === 'update_conflict') return status.stashRef ? `Local changes saved in ${status.stashRef}` : status.message;
  if (status.state === 'update_failed') return status.updateError || status.error || status.message;
  return status.message || `Checked ${formatDate(status.checkedAt)}`;
}

function getStatusTime(status: RepoUpdateStatus) {
  const time = new Date(status.updatedAt || status.checkedAt || '').getTime();
  return Number.isFinite(time) ? time : null;
}

function queuedApplyStatus(status: RepoUpdateStatus | null): RepoUpdateStatus | null {
  if (!status) return status;
  return {
    ...status,
    state: 'updating',
    message: 'Update requested',
    updateStep: 'waiting-for-updater',
    canApplyUpdate: false,
    updateError: null,
    error: null,
    updatedAt: new Date().toISOString(),
  };
}

function queuedRestartStatus(status: RepoUpdateStatus | null): RepoUpdateStatus | null {
  if (!status) return status;
  return {
    ...status,
    state: 'restarting',
    message: 'Restart requested',
    restartStep: 'waiting-for-updater',
    canApplyUpdate: false,
    restartError: null,
    error: null,
    updatedAt: new Date().toISOString(),
  };
}

function isRepoUpdateStatus(value: unknown): value is RepoUpdateStatus {
  return Boolean(value && typeof value === 'object' && 'state' in value && 'message' in value);
}

export default function UpdatesPage() {
  const [status, setStatus] = useState<RepoUpdateStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [checking, setChecking] = useState(false);
  const [applying, setApplying] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const applyRequestedAtRef = useRef<number | null>(null);
  const applyFeedbackExpiresAtRef = useRef<number | null>(null);
  const restartRequestedAtRef = useRef<number | null>(null);
  const restartPollRef = useRef<number | null>(null);

  const refresh = async () => {
    setLoading(true);
    try {
      const res = await apiClient.get('/api/updater');
      const nextStatus = res.data as RepoUpdateStatus;
      const applyRequestedAt = applyRequestedAtRef.current;

      if (applyRequestedAt) {
        const statusTime = getStatusTime(nextStatus);
        const expiresAt = applyFeedbackExpiresAtRef.current || 0;
        const staleStatus = statusTime == null || statusTime < applyRequestedAt;

        if (staleStatus && Date.now() < expiresAt) {
          return;
        }

        if (nextStatus.state !== 'checking' && nextStatus.state !== 'updating') {
          applyRequestedAtRef.current = null;
          applyFeedbackExpiresAtRef.current = null;
          setApplying(false);
        }
      }

      setStatus(nextStatus);
      broadcastUpdaterStatus(nextStatus);
    } catch (error) {
      console.error('Error fetching updater status:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    const handleStatusEvent = (event: Event) => {
      const nextStatus = (event as CustomEvent<unknown>).detail;
      if (isRepoUpdateStatus(nextStatus)) {
        setStatus(nextStatus);
      }
    };
    window.addEventListener(UPDATER_STATUS_EVENT, handleStatusEvent);
    return () => window.removeEventListener(UPDATER_STATUS_EVENT, handleStatusEvent);
  }, []);

  useEffect(() => {
    return () => {
      if (restartPollRef.current) {
        window.clearInterval(restartPollRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (status?.state !== 'checking' && status?.state !== 'updating') return;
    const interval = window.setInterval(() => {
      void refresh();
    }, 3000);
    return () => window.clearInterval(interval);
  }, [status?.state]);

  const requestCheck = async () => {
    setChecking(true);
    try {
      await apiClient.post('/api/updater', { action: 'check' });
      window.setTimeout(() => void refresh(), 750);
    } finally {
      setChecking(false);
    }
  };

  const requestApply = async () => {
    setApplying(true);
    try {
      const response = await apiClient.post('/api/updater', { action: 'apply' });
      const requestedAt = new Date(response.data?.request?.requestedAt || Date.now()).getTime();
      applyRequestedAtRef.current = Number.isFinite(requestedAt) ? requestedAt : Date.now();
      applyFeedbackExpiresAtRef.current = Date.now() + APPLY_FEEDBACK_TIMEOUT_MS;
      const optimisticStatus = queuedApplyStatus(status);
      setStatus(optimisticStatus);
      if (optimisticStatus) broadcastUpdaterStatus(optimisticStatus);
      window.setTimeout(() => void refresh(), 1000);
      window.setTimeout(() => void refresh(), 4000);
    } catch (error) {
      console.error('Error requesting updater apply:', error);
      applyRequestedAtRef.current = null;
      applyFeedbackExpiresAtRef.current = null;
      setApplying(false);
    }
  };

  const startRestartPolling = (requestedAt: number) => {
    if (restartPollRef.current) {
      window.clearInterval(restartPollRef.current);
    }

    restartPollRef.current = window.setInterval(async () => {
      const elapsed = Date.now() - requestedAt;
      if (elapsed > RESTART_WAIT_TIMEOUT_MS) {
        if (restartPollRef.current) {
          window.clearInterval(restartPollRef.current);
          restartPollRef.current = null;
        }
        setRestarting(false);
        void refresh();
        return;
      }

      if (elapsed < RESTART_RELOAD_DELAY_MS) {
        return;
      }

      try {
        const res = await apiClient.get('/api/updater');
        const nextStatus = res.data as RepoUpdateStatus;
        const statusTime = getStatusTime(nextStatus);
        if (statusTime != null && restartRequestedAtRef.current && statusTime < restartRequestedAtRef.current) {
          return;
        }

        if (nextStatus.state === 'restarting') {
          setStatus(nextStatus);
          broadcastUpdaterStatus(nextStatus);
          return;
        }

        if (nextStatus.state === 'error' && nextStatus.restartError) {
          setStatus(nextStatus);
          broadcastUpdaterStatus(nextStatus);
          setRestarting(false);
          if (restartPollRef.current) {
            window.clearInterval(restartPollRef.current);
            restartPollRef.current = null;
          }
          return;
        }

        window.location.reload();
      } catch {
        // The server is expected to be offline during the restart window.
      }
    }, 5000);
  };

  const requestRestart = async () => {
    setRestarting(true);
    try {
      const response = await apiClient.post('/api/updater', { action: 'restart' });
      const requestedAt = new Date(response.data?.request?.requestedAt || Date.now()).getTime();
      const normalizedRequestedAt = Number.isFinite(requestedAt) ? requestedAt : Date.now();
      restartRequestedAtRef.current = normalizedRequestedAt;
      const optimisticStatus = queuedRestartStatus(status);
      setStatus(optimisticStatus);
      if (optimisticStatus) broadcastUpdaterStatus(optimisticStatus);
      startRestartPolling(normalizedRequestedAt);
    } catch (error) {
      console.error('Error requesting updater restart:', error);
      restartRequestedAtRef.current = null;
      setRestarting(false);
    }
  };

  const commits = status?.recentCommits || [];
  const repoUrl = status?.repoWebUrl || 'https://github.com/BadAtCaptchas/ai-toolkit-revamped';
  const statusTone = toneByState[status?.state || ''] || 'border-gray-800 bg-gray-900/60 text-gray-100';
  const StatusIcon = status?.state === 'updating' || status?.state === 'checking' || status?.state === 'restarting' ? Loader2 : status?.state?.includes('fail') || status?.state === 'error' ? AlertCircle : CheckCircle2;
  const repoName = status?.repoFullName || 'BadAtCaptchas/ai-toolkit-revamped';
  const isBusy = status?.state === 'checking' || status?.state === 'updating' || status?.state === 'restarting';
  const isRestarting = restarting || status?.state === 'restarting';
  const canRestart = Boolean(status?.needsRestart || status?.state === 'updated' || isRestarting);
  const recommendedRemote = status?.remote || `${repoUrl.replace(/\.git$/, '')}.git`;
  const remoteSetUrlCommand = `git remote set-url origin ${recommendedRemote}`;

  const facts = useMemo(
    () => [
      { label: 'Source', value: repoName },
      { label: 'Branch', value: status?.upstream || status?.branch || 'Unknown' },
      { label: 'Local', value: status?.localShortCommit || status?.localVersion || 'Unknown' },
      { label: 'Latest', value: status?.remoteShortCommit || status?.latestVersion || 'Unknown' },
      { label: 'Checked', value: formatDate(status?.checkedAt || status?.updatedAt) },
    ],
    [repoName, status],
  );

  return (
    <>
      <TopBar>
        <div className="flex shrink-0 items-center gap-2">
          <GitPullRequestArrow className="h-4 w-4 text-cyan-300" />
          <h1 className="text-base font-semibold">Updates</h1>
        </div>
        <div className="flex-1"></div>
        <button type="button" onClick={requestCheck} disabled={checking || isBusy} className="operator-button py-1 text-xs">
          {checking || status?.state === 'checking' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          Check
        </button>
        {(status?.canApplyUpdate || applying || status?.state === 'updating') && (
          <button type="button" onClick={requestApply} disabled={applying || isBusy} className="operator-button border-amber-800 bg-amber-950/40 py-1 text-xs text-amber-100 hover:bg-amber-900/50">
            {applying || status?.state === 'updating' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
            {applying || status?.state === 'updating' ? 'Updating' : 'Update'}
          </button>
        )}
        {canRestart && (
          <button type="button" onClick={requestRestart} disabled={isRestarting} className="operator-button border-cyan-800 bg-cyan-950/30 py-1 text-xs text-cyan-100 hover:bg-cyan-900/40">
            {isRestarting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Power className="h-3.5 w-3.5" />}
            {isRestarting ? 'Restarting' : 'Restart'}
          </button>
        )}
        <a href={repoUrl} target="_blank" rel="noreferrer" className="operator-button py-1 text-xs">
          <ExternalLink className="h-3.5 w-3.5" />
          GitHub
        </a>
      </TopBar>
      <MainContent>
        <section className={classNames('border px-4 py-3', statusTone)}>
          <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <StatusIcon className={classNames('h-4 w-4 flex-none', isBusy ? 'animate-spin' : '')} />
                <h2 className="truncate text-sm font-semibold">{getStatusLabel(status)}</h2>
              </div>
              <p className="mt-1 text-xs opacity-85">{getStatusDetail(status)}</p>
            </div>
            {status?.compareUrl && (
              <a href={status.compareUrl} target="_blank" rel="noreferrer" className="operator-button shrink-0 py-1 text-xs">
                <GitCompareArrows className="h-3.5 w-3.5" />
                Compare
              </a>
            )}
          </div>
        </section>

        <section className="mt-4 grid grid-cols-1 gap-2 md:grid-cols-5">
          {facts.map(fact => (
            <div key={fact.label} className="operator-panel px-3 py-2">
              <div className="text-[10px] font-semibold uppercase text-gray-500">{fact.label}</div>
              <div className="mt-1 truncate text-xs text-gray-200">{fact.value}</div>
            </div>
          ))}
        </section>

        {status?.sourceRemoteMatchesCanonical === false && status.sourceRemoteWebUrl && (
          <section className="mt-4 border border-amber-900 bg-amber-950/20 px-4 py-3 text-sm text-amber-100">
            <div className="font-medium">Local remote differs from update source</div>
            <div className="mt-1 truncate text-xs text-amber-200/80">Current origin: {status.sourceRemoteWebUrl}</div>
            <div className="mt-2 text-xs text-amber-100/90">
              Suggested origin: <span className="text-amber-50">{recommendedRemote}</span>
            </div>
            <div className="mt-2 overflow-x-auto rounded-sm border border-amber-900/70 bg-gray-950/70 px-2 py-1.5">
              <code className="whitespace-nowrap text-[11px] text-amber-50">{remoteSetUrlCommand}</code>
            </div>
            <div className="mt-2 text-xs text-amber-200/70">
              Use this if this checkout should receive updates from {repoName}.
            </div>
          </section>
        )}

        <section className="mt-5">
          <div className="mb-2 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <GitCommitHorizontal className="h-4 w-4 text-gray-500" />
              <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-400">Recent commits</h2>
            </div>
            <span className="text-xs text-gray-500">{commits.length}</span>
          </div>
          <div className="operator-surface divide-y divide-gray-800">
            {commits.map(commit => (
              <article key={commit.sha} className="px-3 py-3">
                <div className="flex min-w-0 flex-col gap-2 md:flex-row md:items-start md:justify-between">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-gray-100">{commit.message}</div>
                    {commit.body && <pre className="mt-1 whitespace-pre-wrap text-xs text-gray-500">{commit.body}</pre>}
                    <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-gray-500">
                      <span>{commit.authorName || 'Unknown author'}</span>
                      <span>{formatDate(commit.authorDate || commit.committerDate)}</span>
                    </div>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    <code className="rounded-sm border border-gray-800 bg-gray-950 px-2 py-1 text-xs text-cyan-200">
                      {commit.shortSha}
                    </code>
                    {commit.url && (
                      <a href={commit.url} target="_blank" rel="noreferrer" className="operator-icon-button" title="Open commit" aria-label="Open commit">
                        <ExternalLink className="h-4 w-4" />
                      </a>
                    )}
                  </div>
                </div>
              </article>
            ))}
            {!loading && commits.length === 0 && (
              <div className="px-3 py-6 text-sm text-gray-500">No commit data has been cached yet.</div>
            )}
            {loading && commits.length === 0 && (
              <div className="flex items-center gap-2 px-3 py-6 text-sm text-gray-500">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading commits
              </div>
            )}
          </div>
        </section>
      </MainContent>
    </>
  );
}
