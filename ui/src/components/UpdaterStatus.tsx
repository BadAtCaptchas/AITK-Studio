'use client';

import { useEffect, useMemo, useState } from 'react';
import classNames from 'classnames';
import { AlertCircle, CheckCircle2, CircleDashed, Download, GitBranch, Loader2, RefreshCw } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { apiClient } from '@/utils/api';

type RepoUpdateState =
  | 'pending'
  | 'checking'
  | 'up_to_date'
  | 'update_available'
  | 'error'
  | 'unsupported'
  | 'disabled'
  | 'stopped';

interface RepoUpdateStatus {
  state: RepoUpdateState;
  message: string;
  checkedAt?: string | null;
  nextCheckAt?: string | null;
  branch?: string | null;
  upstream?: string | null;
  remoteWebUrl?: string | null;
  compareUrl?: string | null;
  localShortCommit?: string | null;
  remoteShortCommit?: string | null;
  ahead?: number | null;
  behind?: number | null;
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
    return behind > 0 ? `${plural(behind, 'commit')} behind` : 'Remote has newer commits';
  }

  if (status.state === 'checking') {
    return status.branch ? `Branch ${status.branch}` : 'Comparing with origin';
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
  if (status.localShortCommit) parts.push(`Local: ${status.localShortCommit}`);
  if (status.remoteShortCommit) parts.push(`Remote: ${status.remoteShortCommit}`);
  return parts.filter(Boolean).join('\n');
}

export default function UpdaterStatus({ compact = false }: { compact?: boolean }) {
  const [status, setStatus] = useState<RepoUpdateStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [requesting, setRequesting] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      const res = await apiClient.get('/api/updater');
      setStatus(res.data);
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
      status?.state === 'checking' ? 10000 : 300000,
    );
    return () => window.clearInterval(interval);
  }, [status?.state]);

  const requestCheck = async () => {
    setRequesting(true);
    try {
      await apiClient.post('/api/updater');
      window.setTimeout(() => {
        void refresh();
      }, 750);
    } catch (error) {
      console.error('Error requesting updater check:', error);
    } finally {
      setRequesting(false);
    }
  };

  const meta = useMemo(() => stateMeta[status?.state || 'pending'], [status?.state]);
  const detail = getDetail(status);
  const Icon = requesting || loading ? RefreshCw : meta.icon;
  const spinning = requesting || loading || meta.active;
  const title = getTitle(status, detail);

  if (compact) {
    return (
      <button
        type="button"
        onClick={requestCheck}
        title={title}
        aria-label={meta.label}
        className={classNames(
          'inline-flex h-8 w-8 items-center justify-center rounded-sm border border-transparent text-gray-400 transition-colors hover:border-gray-800 hover:bg-gray-900 hover:text-gray-100',
          meta.textClass,
        )}
      >
        <Icon className={classNames('h-4 w-4', spinning ? 'animate-spin' : '')} />
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={requestCheck}
      title={title}
      aria-label={`${meta.label}. ${detail}`}
      className="flex w-full min-w-0 items-center gap-2 border-t border-gray-800 px-3 py-2 text-left transition-colors hover:bg-gray-900"
    >
      <Icon className={classNames('h-4 w-4 flex-none', meta.textClass, spinning ? 'animate-spin' : '')} />
      <span className="min-w-0 flex-1">
        <span className={classNames('block truncate text-[11px] font-medium', meta.textClass)}>{meta.label}</span>
        <span className={classNames('block truncate text-[10px]', meta.subtleClass)}>{detail}</span>
      </span>
      <RefreshCw className="h-3.5 w-3.5 flex-none text-gray-600" />
    </button>
  );
}
