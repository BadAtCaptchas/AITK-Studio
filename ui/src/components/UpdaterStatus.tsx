'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import classNames from 'classnames';
import {
  AlertCircle,
  CheckCircle2,
  CircleDashed,
  Download,
  ExternalLink,
  GitBranch,
  Loader2,
  Power,
  RefreshCw,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { apiClient } from '@/utils/api';

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

interface RepoUpdateStatus {
  state: RepoUpdateState;
  message: string;
  checkedAt?: string | null;
  updatedAt?: string | null;
  nextCheckAt?: string | null;
  branch?: string | null;
  upstream?: string | null;
  installKind?: string | null;
  repoFullName?: string | null;
  repoWebUrl?: string | null;
  downloadUrl?: string | null;
  latestVersion?: string | null;
  latestReleaseUrl?: string | null;
  remote?: string | null;
  remoteWebUrl?: string | null;
  sourceRemoteWebUrl?: string | null;
  sourceRemoteMatchesCanonical?: boolean | null;
  compareUrl?: string | null;
  localVersion?: string | null;
  localShortCommit?: string | null;
  remoteShortCommit?: string | null;
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
  localChangesRestored?: boolean | null;
  needsRestart?: boolean | null;
  error?: string | null;
}

interface StatusMeta {
  label: string;
  icon: LucideIcon;
  textClass: string;
  subtleClass: string;
  active?: boolean;
}

const stateMeta: Record<RepoUpdateState, StatusMeta> = {
  pending: {
    label: 'Update check pending',
    icon: CircleDashed,
    textClass: 'text-gray-400',
    subtleClass: 'text-gray-500',
  },
  checking: {
    label: 'Checking for updates',
    icon: Loader2,
    textClass: 'text-cyan-300',
    subtleClass: 'text-cyan-500/80',
    active: true,
  },
  up_to_date: {
    label: 'Up to date',
    icon: CheckCircle2,
    textClass: 'text-emerald-300',
    subtleClass: 'text-gray-500',
  },
  update_available: {
    label: 'Update available',
    icon: Download,
    textClass: 'text-amber-300',
    subtleClass: 'text-amber-500/80',
  },
  unknown_current: {
    label: 'Latest on GitHub',
    icon: GitBranch,
    textClass: 'text-cyan-300',
    subtleClass: 'text-gray-500',
  },
  updating: {
    label: 'Updating',
    icon: Loader2,
    textClass: 'text-cyan-300',
    subtleClass: 'text-cyan-500/80',
    active: true,
  },
  restarting: {
    label: 'Restarting',
    icon: Loader2,
    textClass: 'text-cyan-300',
    subtleClass: 'text-cyan-500/80',
    active: true,
  },
  updated: {
    label: 'Updated',
    icon: CheckCircle2,
    textClass: 'text-emerald-300',
    subtleClass: 'text-emerald-500/80',
  },
  update_failed: {
    label: 'Update failed',
    icon: AlertCircle,
    textClass: 'text-rose-300',
    subtleClass: 'text-gray-500',
  },
  update_blocked: {
    label: 'Manual update needed',
    icon: GitBranch,
    textClass: 'text-amber-300',
    subtleClass: 'text-gray-500',
  },
  update_conflict: {
    label: 'Update needs attention',
    icon: AlertCircle,
    textClass: 'text-amber-300',
    subtleClass: 'text-gray-500',
  },
  error: {
    label: 'Update check failed',
    icon: AlertCircle,
    textClass: 'text-rose-300',
    subtleClass: 'text-gray-500',
  },
  unsupported: {
    label: 'Updates unavailable',
    icon: GitBranch,
    textClass: 'text-gray-400',
    subtleClass: 'text-gray-500',
  },
  disabled: {
    label: 'Updates off',
    icon: GitBranch,
    textClass: 'text-gray-400',
    subtleClass: 'text-gray-500',
  },
  stopped: {
    label: 'Updater stopped',
    icon: AlertCircle,
    textClass: 'text-gray-400',
    subtleClass: 'text-gray-500',
  },
};

const APPLY_FEEDBACK_TIMEOUT_MS = 2 * 60 * 1000;
const RESTART_WAIT_TIMEOUT_MS = 5 * 60 * 1000;
const RESTART_RELOAD_DELAY_MS = 8000;

function plural(value: number, singular: string, pluralLabel = `${singular}s`) {
  return `${value} ${value === 1 ? singular : pluralLabel}`;
}

function formatCheckedAt(value?: string | null) {
  if (!value) {
    return 'Not checked yet';
  }

  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) {
    return 'Last check unknown';
  }

  return `Checked ${date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`;
}

function getDetail(status: RepoUpdateStatus | null) {
  if (!status) {
    return 'Waiting for checker';
  }

  if (status.state === 'update_available') {
    const behind = Number(status.behind || 0);
    if (behind > 0) return `${plural(behind, 'commit')} behind`;
    if (status.localVersion && status.latestVersion) return `${status.localVersion} -> ${status.latestVersion}`;
    return status.latestVersion ? `Latest ${status.latestVersion}` : 'Remote has newer commits';
  }

  if (status.state === 'unknown_current') {
    if (status.latestVersion) return `Latest ${status.latestVersion}`;
    if (status.remoteShortCommit) return `Latest ${status.remoteShortCommit}`;
    return status.repoFullName || 'GitHub source found';
  }

  if (status.state === 'checking') {
    return status.repoFullName ? `Checking ${status.repoFullName}` : 'Checking GitHub';
  }

  if (status.state === 'updating') {
    return status.updateStep ? status.updateStep.replaceAll('-', ' ') : 'Applying update';
  }

  if (status.state === 'restarting') {
    return status.restartStep ? status.restartStep.replaceAll('-', ' ') : 'Restarting app';
  }

  if (status.state === 'updated') {
    return status.needsRestart ? 'Restart required' : formatCheckedAt(status.checkedAt);
  }

  if (status.state === 'update_blocked') {
    return status.applyUpdateUnavailableReason || 'Open GitHub';
  }

  if (status.state === 'update_conflict') {
    return status.stashRef ? `Saved in ${status.stashRef}` : 'Local changes need attention';
  }

  if (status.state === 'update_failed') {
    return status.updateError || status.error || formatCheckedAt(status.checkedAt);
  }

  if (status.state === 'error') {
    return status.error || formatCheckedAt(status.checkedAt);
  }

  if (status.state === 'unsupported' || status.state === 'disabled' || status.state === 'stopped') {
    return status.message;
  }

  return formatCheckedAt(status.checkedAt);
}

function getTitle(status: RepoUpdateStatus | null, detail: string) {
  if (!status) {
    return 'Repository update checker';
  }

  const parts = [status.message, detail];
  if (status.branch) parts.push(`Branch: ${status.branch}`);
  if (status.upstream) parts.push(`Upstream: ${status.upstream}`);
  if (status.repoFullName) parts.push(`Source: ${status.repoFullName}`);
  if (status.localVersion) parts.push(`Local version: ${status.localVersion}`);
  if (status.latestVersion) parts.push(`Latest release: ${status.latestVersion}`);
  if (status.localShortCommit) parts.push(`Local: ${status.localShortCommit}`);
  if (status.remoteShortCommit) parts.push(`Remote: ${status.remoteShortCommit}`);
  if (status.applyUpdateUnavailableReason) parts.push(status.applyUpdateUnavailableReason);
  if (status.updateError) parts.push(`Update error: ${status.updateError}`);
  if (status.restartStep) parts.push(`Restart step: ${status.restartStep.replaceAll('-', ' ')}`);
  if (status.restartError) parts.push(`Restart error: ${status.restartError}`);
  if (status.stashRef) parts.push(`Local changes stash: ${status.stashRef}`);
  if (status.needsRestart) parts.push('Restart the app to use the update.');
  if (status.sourceRemoteMatchesCanonical === false && status.sourceRemoteWebUrl) {
    parts.push(`Local git remote: ${status.sourceRemoteWebUrl}`);
    const recommendedRemote = status.remote || `${(status.repoWebUrl || status.remoteWebUrl || 'https://github.com/rmcc3/ai-toolkit-revamped').replace(/\.git$/, '')}.git`;
    parts.push(`Suggested origin: ${recommendedRemote}`);
    parts.push(`git remote set-url origin ${recommendedRemote}`);
  }
  return parts.filter(Boolean).join('\n');
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

export default function UpdaterStatus({ compact = false }: { compact?: boolean }) {
  const [status, setStatus] = useState<RepoUpdateStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [requesting, setRequesting] = useState(false);
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
    } catch (error) {
      console.error('Error fetching updater status:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(
      () => {
        void refresh();
      },
      status?.state === 'checking' || status?.state === 'updating' || status?.state === 'restarting' ? 3000 : 300000,
    );
    return () => window.clearInterval(interval);
  }, [status?.state]);

  useEffect(() => {
    return () => {
      if (restartPollRef.current) {
        window.clearInterval(restartPollRef.current);
      }
    };
  }, []);

  const requestCheck = async () => {
    setRequesting(true);
    try {
      await apiClient.post('/api/updater', { action: 'check' });
      window.setTimeout(() => {
        void refresh();
      }, 750);
    } catch (error) {
      console.error('Error requesting updater check:', error);
    } finally {
      setRequesting(false);
    }
  };

  const requestApplyUpdate = async () => {
    setApplying(true);
    try {
      const response = await apiClient.post('/api/updater', { action: 'apply' });
      const requestedAt = new Date(response.data?.request?.requestedAt || Date.now()).getTime();
      applyRequestedAtRef.current = Number.isFinite(requestedAt) ? requestedAt : Date.now();
      applyFeedbackExpiresAtRef.current = Date.now() + APPLY_FEEDBACK_TIMEOUT_MS;
      setStatus(prev => queuedApplyStatus(prev));
      window.setTimeout(() => {
        void refresh();
      }, 1000);
      window.setTimeout(() => {
        void refresh();
      }, 4000);
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
          return;
        }

        if (nextStatus.state === 'error' && nextStatus.restartError) {
          setStatus(nextStatus);
          setRestarting(false);
          if (restartPollRef.current) {
            window.clearInterval(restartPollRef.current);
            restartPollRef.current = null;
          }
          return;
        }

        window.location.reload();
      } catch {
        // The server is expected to be offline while npm run build_and_start is running.
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
      setStatus(prev => queuedRestartStatus(prev));
      startRestartPolling(normalizedRequestedAt);
    } catch (error) {
      console.error('Error requesting updater restart:', error);
      restartRequestedAtRef.current = null;
      setRestarting(false);
    }
  };

  const meta = useMemo(() => stateMeta[status?.state || 'pending'], [status?.state]);
  const detail = getDetail(status);
  const Icon = requesting || loading ? RefreshCw : meta.icon;
  const spinning = requesting || loading || applying || restarting || meta.active;
  const title = getTitle(status, detail);
  const canApplyUpdate = Boolean(status?.canApplyUpdate && status.state === 'update_available');
  const isRestarting = restarting || status?.state === 'restarting';
  const canRestart = Boolean(status?.needsRestart || status?.state === 'updated' || isRestarting);
  const repoUrl = status?.repoWebUrl || status?.remoteWebUrl || 'https://github.com/rmcc3/ai-toolkit-revamped';
  const shouldLinkRepo = Boolean(
    status &&
      repoUrl &&
      !canApplyUpdate &&
      ['update_available', 'unknown_current', 'update_blocked', 'update_failed', 'update_conflict', 'error'].includes(
        status.state,
      ),
  );

  if (compact) {
    return (
      <span className="inline-flex items-center gap-1">
        <Link
          href="/updates"
          title={title}
          aria-label={meta.label}
          className={classNames(
            'inline-flex h-8 w-8 items-center justify-center rounded-sm border border-transparent text-gray-400 transition-colors hover:border-gray-800 hover:bg-gray-900 hover:text-gray-100',
            meta.textClass,
          )}
        >
          <Icon className={classNames('h-4 w-4', spinning ? 'animate-spin' : '')} />
        </Link>
        <button
          type="button"
          onClick={requestCheck}
          disabled={requesting || loading}
          title="Check for updates"
          aria-label="Check for updates"
          className="inline-flex h-8 w-8 items-center justify-center rounded-sm border border-transparent text-gray-400 transition-colors hover:border-gray-800 hover:bg-gray-900 hover:text-gray-100 disabled:opacity-50"
        >
          <RefreshCw className={classNames('h-4 w-4', requesting || loading ? 'animate-spin' : '')} />
        </button>
        {(canApplyUpdate || applying) && (
          <button
            type="button"
            onClick={requestApplyUpdate}
            disabled={applying}
            title="Apply update"
            aria-label="Apply update"
            className="inline-flex h-8 w-8 items-center justify-center rounded-sm border border-amber-800/60 text-amber-300 transition-colors hover:bg-amber-950/30 disabled:opacity-50"
          >
            {applying ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
          </button>
        )}
        {canRestart && (
          <button
            type="button"
            onClick={requestRestart}
            disabled={isRestarting}
            title="Restart app"
            aria-label="Restart app"
            className="inline-flex h-8 w-8 items-center justify-center rounded-sm border border-cyan-800/60 text-cyan-300 transition-colors hover:bg-cyan-950/30 disabled:opacity-50"
          >
            {isRestarting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Power className="h-4 w-4" />}
          </button>
        )}
        {shouldLinkRepo && (
          <a
            href={repoUrl}
            target="_blank"
            rel="noreferrer"
            title="Open update repository"
            aria-label="Open update repository"
            className="inline-flex h-8 w-8 items-center justify-center rounded-sm border border-transparent text-gray-400 transition-colors hover:border-gray-800 hover:bg-gray-900 hover:text-gray-100"
          >
            <ExternalLink className="h-4 w-4" />
          </a>
        )}
      </span>
    );
  }

  return (
    <div className="flex w-full min-w-0 items-center gap-1 border-t border-gray-800 px-2 py-2 transition-colors hover:bg-gray-900">
      <Link
        href="/updates"
        title={title}
        aria-label={`${meta.label}. ${detail}`}
        className="flex min-w-0 flex-1 items-center gap-2 text-left"
      >
        <Icon className={classNames('h-4 w-4 flex-none', meta.textClass, spinning ? 'animate-spin' : '')} />
        <span className="min-w-0 flex-1">
          <span className={classNames('block truncate text-[11px] font-medium', meta.textClass)}>{meta.label}</span>
          <span className={classNames('block truncate text-[10px]', meta.subtleClass)}>{detail}</span>
        </span>
      </Link>
      <button
        type="button"
        onClick={requestCheck}
        disabled={requesting || loading}
        title="Check for updates"
        aria-label="Check for updates"
        className="inline-flex h-8 w-8 flex-none items-center justify-center rounded-sm border border-transparent text-gray-400 transition-colors hover:border-gray-800 hover:bg-gray-950 hover:text-gray-100 disabled:opacity-50"
      >
        <RefreshCw className={classNames('h-3.5 w-3.5', requesting || loading ? 'animate-spin' : '')} />
      </button>
      {(canApplyUpdate || applying) && (
        <button
          type="button"
          onClick={requestApplyUpdate}
          disabled={applying}
          title="Apply update"
          aria-label="Apply update"
          className="inline-flex h-8 w-8 flex-none items-center justify-center rounded-sm border border-amber-800/60 text-amber-300 transition-colors hover:bg-amber-950/30 disabled:opacity-50"
        >
          {applying ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
        </button>
      )}
      {canRestart && (
        <button
          type="button"
          onClick={requestRestart}
          disabled={isRestarting}
          title="Restart app"
          aria-label="Restart app"
          className="inline-flex h-8 w-8 flex-none items-center justify-center rounded-sm border border-cyan-800/60 text-cyan-300 transition-colors hover:bg-cyan-950/30 disabled:opacity-50"
        >
          {isRestarting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Power className="h-4 w-4" />}
        </button>
      )}
      {shouldLinkRepo && (
        <a
          href={repoUrl}
          target="_blank"
          rel="noreferrer"
          title="Open update repository"
          aria-label="Open update repository"
          className="inline-flex h-8 w-8 flex-none items-center justify-center rounded-sm border border-transparent text-gray-400 transition-colors hover:border-gray-800 hover:bg-gray-950 hover:text-gray-100"
        >
          <ExternalLink className="h-4 w-4" />
        </a>
      )}
    </div>
  );
}
