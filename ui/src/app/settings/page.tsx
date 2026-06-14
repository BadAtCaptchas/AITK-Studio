'use client';

import { useEffect, useRef, useState } from 'react';
import useSettings from '@/hooks/useSettings';
import useWorkers from '@/hooks/useWorkers';
import useRemoteOllamaWorkers from '@/hooks/useRemoteOllamaWorkers';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import type { ComfyInstallProgress, RemoteOllamaWorker, WorkerNode } from '@/types';
import { ComfyInstallProgressBand } from '@/components/ComfyInstallProgress';
import {
  AlertTriangle,
  Box,
  CheckCircle2,
  ChevronDown,
  Cloud,
  Database,
  Download,
  Eye,
  EyeOff,
  FolderOpen,
  KeyRound,
  Loader2,
  Power,
  RefreshCw,
  Save,
  ServerCog,
  TerminalSquare,
  UsersRound,
} from 'lucide-react';

type ComfyManagedInstallStatus = {
  installed: boolean;
  installing: boolean;
  root: string;
  progressPath: string;
  logPath: string;
  pid: number | null;
  progress: ComfyInstallProgress | null;
  message: string;
  error: string | null;
};

type CloudflaredStatus = {
  configured: boolean;
  enabled: boolean;
  mode: 'named' | 'quick';
  detected: boolean;
  bin: string;
  downloadAvailable: boolean;
  downloadUrl: string | null;
  installPath: string;
  running: boolean;
  pid: number | null;
  publicUrl: string | null;
  targetUrl: string;
  metricsAddr: string;
  message: string;
  error: string | null;
};

type WorkerUpdaterState =
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

type WorkerUpdaterStatus = {
  state: WorkerUpdaterState;
  message: string;
  checkedAt?: string | null;
  updatedAt?: string | null;
  upstream?: string | null;
  localShortCommit?: string | null;
  remoteShortCommit?: string | null;
  ahead?: number | null;
  behind?: number | null;
  canApplyUpdate?: boolean | null;
  applyUpdateUnavailableReason?: string | null;
  updateStep?: string | null;
  updateError?: string | null;
  restartStep?: string | null;
  restartError?: string | null;
  needsRestart?: boolean | null;
  error?: string | null;
};

type WorkerUpdaterAction = 'idle' | 'checking' | 'updating' | 'restarting';

type WorkerUpdaterUiState = {
  status?: WorkerUpdaterStatus | null;
  action?: WorkerUpdaterAction;
  error?: string | null;
};

const CLOUDFLARED_AUTO_DOWNLOAD_KEY = 'AITK_CLOUDFLARED_AUTO_DOWNLOAD';
const WORKER_UPDATER_POLL_MS = 5000;
const WORKER_UPDATER_WAIT_MS = 2 * 60 * 1000;
const WORKER_RESTART_GRACE_MS = 8000;
const WORKER_RESTART_WAIT_MS = 10 * 60 * 1000;

const emptyWorkerForm = {
  id: '',
  name: '',
  base_url: '',
  api_token: '',
  enabled: true,
};

const emptyOllamaWorkerForm = {
  id: '',
  name: '',
  base_url: '',
  auth_token: '',
  enabled: true,
};

function formatUpdaterTime(value?: string | null) {
  if (!value) return 'Not checked yet';
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return 'Last check unknown';
  return `Checked ${date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`;
}

function plural(value: number, singular: string, pluralLabel = `${singular}s`) {
  return `${value} ${value === 1 ? singular : pluralLabel}`;
}

function workerUpdaterLabel(status?: WorkerUpdaterStatus | null, error?: string | null) {
  if (error) return 'Updater unavailable';
  if (!status) return 'Updater not checked';
  if (status.state === 'up_to_date') return 'Up to date';
  if (status.state === 'update_available') return 'Update available';
  if (status.state === 'checking') return 'Checking';
  if (status.state === 'updating') return 'Updating';
  if (status.state === 'restarting') return 'Restarting';
  if (status.state === 'updated') return 'Restart required';
  if (status.state === 'update_blocked') return 'Manual attention needed';
  if (status.state === 'update_conflict') return 'Local changes need attention';
  if (status.state === 'update_failed') return 'Update failed';
  if (status.state === 'unknown_current') return 'Latest on GitHub';
  if (status.state === 'error') return 'Updater error';
  return status.message || 'Updater status';
}

function workerUpdaterDetail(status?: WorkerUpdaterStatus | null, error?: string | null) {
  if (error) return error;
  if (!status) return 'Use the refresh action to check this worker.';
  if (status.state === 'update_available') {
    const behind = Number(status.behind || 0);
    if (behind > 0) return `${plural(behind, 'commit')} behind ${status.upstream || 'GitHub'}`;
  }
  if (status.state === 'updating') return status.updateStep?.replaceAll('-', ' ') || 'Applying update';
  if (status.state === 'restarting') return status.restartStep?.replaceAll('-', ' ') || 'Restarting worker';
  if (status.state === 'updated' || status.needsRestart) return 'Restart the worker to use the update.';
  if (status.state === 'update_blocked') return status.applyUpdateUnavailableReason || status.message;
  if (status.state === 'update_failed') return status.updateError || status.error || status.message;
  if (status.state === 'error') return status.restartError || status.error || status.message;
  if (status.localShortCommit && status.remoteShortCommit) {
    return `${status.localShortCommit} -> ${status.remoteShortCommit}`;
  }
  return status.message || formatUpdaterTime(status.checkedAt || status.updatedAt);
}

function workerUpdaterStatusTime(status: WorkerUpdaterStatus) {
  const time = new Date(status.updatedAt || status.checkedAt || '').getTime();
  return Number.isFinite(time) ? time : null;
}

type SettingsSectionKey = 'essentials' | 'access' | 'storage' | 'workers' | 'comfy' | 'advanced';

const sectionNav: Array<{ id: SettingsSectionKey; label: string }> = [
  { id: 'essentials', label: 'Essentials' },
  { id: 'access', label: 'Access' },
  { id: 'storage', label: 'Storage' },
  { id: 'workers', label: 'Workers' },
  { id: 'comfy', label: 'ComfyUI' },
  { id: 'advanced', label: 'Advanced' },
];

function FieldShell({
  id,
  label,
  detail,
  children,
}: {
  id: string;
  label: string;
  detail: string;
  children: React.ReactNode;
}) {
  return (
    <div className="border-b border-gray-900 pb-4 last:border-b-0 last:pb-0">
      <label htmlFor={id} className="block text-sm font-semibold text-gray-100">
        {label}
      </label>
      <div className="mt-1 text-sm text-gray-500">{detail}</div>
      <div className="mt-2">{children}</div>
    </div>
  );
}

function StatusDot({ tone = 'ok' }: { tone?: 'ok' | 'warn' | 'idle' }) {
  return (
    <span
      className={
        tone === 'ok'
          ? 'h-1.5 w-1.5 rounded-full bg-emerald-400'
          : tone === 'warn'
            ? 'h-1.5 w-1.5 rounded-full bg-amber-400'
            : 'h-1.5 w-1.5 rounded-full bg-gray-600'
      }
    />
  );
}

function SettingSwitch({ checked, onChange }: { checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 flex-none items-center rounded-full border border-gray-700 transition-colors ${
        checked ? 'bg-cyan-500/90' : 'bg-gray-700'
      }`}
    >
      <span className="sr-only">Toggle setting</span>
      <span
        className={`h-5 w-5 rounded-full bg-white transition-transform ${
          checked ? 'translate-x-[1.3rem]' : 'translate-x-0.5'
        }`}
      />
    </button>
  );
}

export default function Settings() {
  const { settings, setSettings } = useSettings();
  const { workers, setWorkers, refreshWorkers } = useWorkers();
  const { workers: ollamaWorkers, refreshWorkers: refreshOllamaWorkers } = useRemoteOllamaWorkers();
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [workerForm, setWorkerForm] = useState(emptyWorkerForm);
  const [workerStatus, setWorkerStatus] = useState<'idle' | 'saving' | 'error'>('idle');
  const [ollamaWorkerForm, setOllamaWorkerForm] = useState(emptyOllamaWorkerForm);
  const [ollamaWorkerStatus, setOllamaWorkerStatus] = useState<'idle' | 'saving' | 'error'>('idle');
  const [cloudflared, setCloudflared] = useState<CloudflaredStatus | null>(null);
  const [cloudflaredAction, setCloudflaredAction] = useState<'idle' | 'starting' | 'downloading' | 'error'>('idle');
  const [cloudflaredActionError, setCloudflaredActionError] = useState('');
  const [cloudflaredAutoDownload, setCloudflaredAutoDownload] = useState(true);
  const [comfyInstall, setComfyInstall] = useState<ComfyManagedInstallStatus | null>(null);
  const [comfyInstallAction, setComfyInstallAction] = useState<'idle' | 'installing' | 'error'>('idle');
  const [comfyInstallActionError, setComfyInstallActionError] = useState('');
  const [workerUpdater, setWorkerUpdater] = useState<Record<string, WorkerUpdaterUiState>>({});
  const [activeSection, setActiveSection] = useState<SettingsSectionKey>('essentials');
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [showSecrets, setShowSecrets] = useState({ hf: false, openRouter: false });
  const workerUpdaterPolls = useRef<Record<string, number>>({});
  const comfyInstallPoll = useRef<number | null>(null);
  const loadedWorkerUpdaterIds = useRef<Set<string>>(new Set());

  const updateWorkerUpdaterState = (
    workerID: string,
    patch: WorkerUpdaterUiState | ((current: WorkerUpdaterUiState) => WorkerUpdaterUiState),
  ) => {
    setWorkerUpdater(prev => {
      const current = prev[workerID] || { action: 'idle', status: null, error: null };
      const next = typeof patch === 'function' ? patch(current) : { ...current, ...patch };
      return { ...prev, [workerID]: next };
    });
  };

  const clearWorkerUpdaterPoll = (workerID: string) => {
    const interval = workerUpdaterPolls.current[workerID];
    if (interval) {
      window.clearInterval(interval);
      delete workerUpdaterPolls.current[workerID];
    }
  };

  const workerUpdaterBusy = (workerID: string) => {
    const action = workerUpdater[workerID]?.action || 'idle';
    return action === 'checking' || action === 'updating' || action === 'restarting';
  };

  const fetchWorkerUpdaterStatus = async (
    workerID: string,
    options: { suppressError?: boolean; keepAction?: boolean; minUpdatedAt?: number } = {},
  ) => {
    try {
      const res = await apiClient.get(`/api/workers/${workerID}/updater`);
      const status = res.data?.status as WorkerUpdaterStatus;
      const statusTime = workerUpdaterStatusTime(status);
      if (options.minUpdatedAt && (statusTime == null || statusTime < options.minUpdatedAt)) {
        updateWorkerUpdaterState(workerID, current => ({
          ...current,
          error: null,
          action: options.keepAction ? current.action || 'idle' : 'idle',
        }));
        return status;
      }
      updateWorkerUpdaterState(workerID, current => ({
        ...current,
        status,
        error: null,
        action: options.keepAction ? current.action || 'idle' : 'idle',
      }));
      return status;
    } catch (error: any) {
      const message = error?.response?.data?.error || 'Remote updater could not be reached.';
      if (!options.suppressError) {
        updateWorkerUpdaterState(workerID, current => ({
          ...current,
          error: message,
          action: options.keepAction ? current.action || 'idle' : 'idle',
        }));
      }
      throw error;
    }
  };

  const startWorkerUpdaterPolling = (workerID: string, action: WorkerUpdaterAction, requestedAt = Date.now()) => {
    clearWorkerUpdaterPoll(workerID);
    workerUpdaterPolls.current[workerID] = window.setInterval(async () => {
      const elapsed = Date.now() - requestedAt;
      const waitMs = action === 'restarting' ? WORKER_RESTART_WAIT_MS : WORKER_UPDATER_WAIT_MS;

      if (action === 'restarting' && elapsed < WORKER_RESTART_GRACE_MS) {
        return;
      }

      if (elapsed > waitMs) {
        clearWorkerUpdaterPoll(workerID);
        updateWorkerUpdaterState(workerID, current => ({ ...current, action: 'idle' }));
        void fetchWorkerUpdaterStatus(workerID).catch(() => undefined);
        return;
      }

      try {
        const status = await fetchWorkerUpdaterStatus(workerID, {
          suppressError: action === 'restarting',
          keepAction: true,
          minUpdatedAt: requestedAt,
        });
        const statusTime = workerUpdaterStatusTime(status);
        if (statusTime == null || statusTime < requestedAt) {
          return;
        }
        const stillBusy = status.state === 'checking' || status.state === 'updating' || status.state === 'restarting';
        if (!stillBusy) {
          clearWorkerUpdaterPoll(workerID);
          updateWorkerUpdaterState(workerID, current => ({ ...current, action: 'idle' }));
          refreshWorkers();
        }
      } catch {
        if (action !== 'restarting') {
          clearWorkerUpdaterPoll(workerID);
          updateWorkerUpdaterState(workerID, current => ({ ...current, action: 'idle' }));
        }
      }
    }, WORKER_UPDATER_POLL_MS);
  };

  const requestWorkerUpdaterAction = async (
    workerID: string,
    action: 'check' | 'apply' | 'restart',
    optimisticStatus: WorkerUpdaterStatus,
    uiAction: WorkerUpdaterAction,
  ) => {
    updateWorkerUpdaterState(workerID, current => ({
      ...current,
      action: uiAction,
      status: { ...(current.status || optimisticStatus), ...optimisticStatus },
      error: null,
    }));

    try {
      await apiClient.post(`/api/workers/${workerID}/updater`, { action });
      startWorkerUpdaterPolling(workerID, uiAction, Date.now());
    } catch (error: any) {
      clearWorkerUpdaterPoll(workerID);
      updateWorkerUpdaterState(workerID, current => ({
        ...current,
        action: 'idle',
        error: error?.response?.data?.error || `Failed to request remote ${action}.`,
      }));
    }
  };

  const refreshCloudflared = () => {
    apiClient
      .get('/api/cloudflared')
      .then(res => setCloudflared(res.data))
      .catch(error => console.error('Error fetching cloudflared status:', error));
  };

  const stopComfyInstallPolling = () => {
    if (comfyInstallPoll.current === null) return;
    window.clearInterval(comfyInstallPoll.current);
    comfyInstallPoll.current = null;
  };

  const startComfyInstallPolling = () => {
    if (comfyInstallPoll.current !== null) return;
    comfyInstallPoll.current = window.setInterval(() => {
      void refreshComfyInstall();
    }, 2000);
  };

  const applyComfyInstallStatus = (nextStatus: ComfyManagedInstallStatus) => {
    setComfyInstall(nextStatus);
    if (nextStatus.installing) {
      startComfyInstallPolling();
    } else {
      stopComfyInstallPolling();
      setComfyInstallAction(current => (current === 'installing' ? 'idle' : current));
    }
  };

  const refreshComfyInstall = () => {
    apiClient
      .get('/api/comfy/install')
      .then(res => applyComfyInstallStatus(res.data))
      .catch(error => console.error('Error fetching ComfyUI install status:', error));
  };

  useEffect(() => {
    refreshCloudflared();
    refreshComfyInstall();
    if (typeof window !== 'undefined') {
      setCloudflaredAutoDownload(window.localStorage.getItem(CLOUDFLARED_AUTO_DOWNLOAD_KEY) !== 'false');
    }
  }, []);

  useEffect(() => {
    return () => {
      Object.values(workerUpdaterPolls.current).forEach(interval => window.clearInterval(interval));
      workerUpdaterPolls.current = {};
      stopComfyInstallPolling();
    };
  }, []);

  useEffect(() => {
    workers.forEach(worker => {
      if (!worker.enabled || loadedWorkerUpdaterIds.current.has(worker.id)) return;
      loadedWorkerUpdaterIds.current.add(worker.id);
      void fetchWorkerUpdaterStatus(worker.id, { suppressError: true }).catch(() => undefined);
    });
  }, [workers]);

  const setAutoDownloadCloudflared = (checked: boolean) => {
    setCloudflaredAutoDownload(checked);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(CLOUDFLARED_AUTO_DOWNLOAD_KEY, checked ? 'true' : 'false');
    }
  };

  const startCloudflared = async () => {
    setCloudflaredAction('starting');
    setCloudflaredActionError('');
    try {
      const res = await apiClient.post('/api/cloudflared', { autoDownload: cloudflaredAutoDownload });
      setCloudflared(res.data);
      setCloudflaredAction('idle');
    } catch (error: any) {
      setCloudflaredAction('error');
      setCloudflaredActionError(error?.response?.data?.error || 'Failed to start cloudflared.');
      refreshCloudflared();
    }
  };

  const downloadCloudflared = async () => {
    setCloudflaredAction('downloading');
    setCloudflaredActionError('');
    try {
      const res = await apiClient.put('/api/cloudflared');
      setCloudflared(res.data?.status || null);
      setCloudflaredAction('idle');
    } catch (error: any) {
      setCloudflaredAction('error');
      setCloudflaredActionError(error?.response?.data?.error || 'Failed to download cloudflared.');
      refreshCloudflared();
    }
  };

  const installComfyNow = async () => {
    setComfyInstallAction('installing');
    setComfyInstallActionError('');
    try {
      const res = await apiClient.post('/api/comfy/install');
      applyComfyInstallStatus(res.data);
    } catch (error: any) {
      setComfyInstallAction('error');
      setComfyInstallActionError(error?.response?.data?.error || 'Failed to start managed ComfyUI install.');
      refreshComfyInstall();
    }
  };

  const saveSettings = async () => {
    setStatus('saving');

    apiClient
      .post('/api/settings', settings)
      .then(() => {
        setStatus('success');
      })
      .catch(error => {
        console.error('Error saving settings:', error);
        setStatus('error');
      })
      .finally(() => {
        setTimeout(() => setStatus('idle'), 2000);
      });
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setSettings(prev => ({ ...prev, [name]: value }));
  };

  const saveWorker = async (e: React.FormEvent) => {
    e.preventDefault();
    setWorkerStatus('saving');
    try {
      await apiClient.post('/api/workers', workerForm);
      setWorkerForm(emptyWorkerForm);
      refreshWorkers();
      setWorkerStatus('idle');
    } catch (error) {
      console.error('Error saving worker:', error);
      setWorkerStatus('error');
    }
  };

  const editWorker = (worker: WorkerNode) => {
    setWorkerForm({
      id: worker.id,
      name: worker.name,
      base_url: worker.base_url,
      api_token: '',
      enabled: worker.enabled,
    });
  };

  const checkWorker = async (workerID: string) => {
    await apiClient.post(`/api/workers/${workerID}/check`).catch(error => {
      console.error('Worker check failed:', error);
    });
    refreshWorkers();
  };

  const checkWorkerUpdates = async (workerID: string) => {
    await requestWorkerUpdaterAction(
      workerID,
      'check',
      {
        state: 'checking',
        message: 'Checking worker for updates',
        updateStep: null,
      },
      'checking',
    );
  };

  const updateRemoteWorker = async (workerID: string) => {
    await requestWorkerUpdaterAction(
      workerID,
      'apply',
      {
        state: 'updating',
        message: 'Update requested',
        updateStep: 'waiting-for-updater',
        canApplyUpdate: false,
      },
      'updating',
    );
  };

  const restartRemoteWorker = async (workerID: string) => {
    await requestWorkerUpdaterAction(
      workerID,
      'restart',
      {
        state: 'restarting',
        message: 'Restart requested',
        restartStep: 'waiting-for-updater',
        canApplyUpdate: false,
      },
      'restarting',
    );
  };

  const deleteWorker = async (workerID: string) => {
    try {
      await apiClient.delete(`/api/workers/${workerID}`);
      clearWorkerUpdaterPoll(workerID);
      loadedWorkerUpdaterIds.current.delete(workerID);
      setWorkerUpdater(prev => {
        const next = { ...prev };
        delete next[workerID];
        return next;
      });
      setWorkers(prev => prev.filter(worker => worker.id !== workerID));
      refreshWorkers();
    } catch (error: any) {
      alert(error.response?.data?.error || 'Failed to delete worker.');
    }
  };

  const saveOllamaWorker = async (e: React.FormEvent) => {
    e.preventDefault();
    setOllamaWorkerStatus('saving');
    try {
      await apiClient.post('/api/ollama-workers', ollamaWorkerForm);
      setOllamaWorkerForm(emptyOllamaWorkerForm);
      refreshOllamaWorkers();
      setOllamaWorkerStatus('idle');
    } catch (error) {
      console.error('Error saving Remote Ollama worker:', error);
      setOllamaWorkerStatus('error');
    }
  };

  const editOllamaWorker = (worker: RemoteOllamaWorker) => {
    setOllamaWorkerForm({
      id: worker.id,
      name: worker.name,
      base_url: worker.base_url,
      auth_token: '',
      enabled: worker.enabled,
    });
  };

  const checkOllamaWorker = async (workerID: string) => {
    await apiClient.post(`/api/ollama-workers/${workerID}/check`).catch(error => {
      console.error('Remote Ollama check failed:', error);
    });
    refreshOllamaWorkers();
  };

  const deleteOllamaWorker = async (workerID: string) => {
    try {
      await apiClient.delete(`/api/ollama-workers/${workerID}`);
      refreshOllamaWorkers();
    } catch (error: any) {
      alert(error.response?.data?.error || 'Failed to delete Remote Ollama endpoint.');
    }
  };

  const jumpToSection = (section: SettingsSectionKey) => {
    setActiveSection(section);
    if (section === 'workers' || section === 'comfy' || section === 'advanced') setAdvancedOpen(true);
    const target = section === 'access' || section === 'storage' ? 'essentials' : section;
    window.requestAnimationFrame(() => {
      document.getElementById(target)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  };

  const activeWorkers = workers.filter(worker => worker.enabled).length;
  const firstOllamaWorker = ollamaWorkers.find(worker => worker.enabled) || ollamaWorkers[0] || null;
  const ollamaModelCount = ollamaWorkers.reduce((sum, worker) => sum + (typeof worker.model_count === 'number' ? worker.model_count : 0), 0);
  const hasHealthyOllama = Boolean(firstOllamaWorker && !firstOllamaWorker.last_error);
  const saveStatusLabel = status === 'saving' ? 'Saving' : status === 'error' ? 'Needs review' : 'Saved';
  const appVersion = process.env.NEXT_PUBLIC_APP_VERSION || 'local';

  return (
    <>
      <TopBar className="h-24 !overflow-hidden border-gray-900 bg-gray-950 px-4 sm:px-7">
        <div className="min-w-0 flex-1">
          <h1 className="text-2xl font-semibold text-gray-100">Settings</h1>
          <p className="mt-0.5 hidden truncate text-sm text-gray-500 sm:block">
            Configure access, storage, workers, and managed runtimes.
          </p>
        </div>
        <div className="ml-auto flex flex-none items-center gap-2">
          <span
            className={`hidden h-9 items-center gap-2 border px-3 text-sm sm:inline-flex ${
              status === 'error'
                ? 'border-rose-500/35 bg-rose-950/20 text-rose-200'
                : status === 'saving'
                  ? 'border-cyan-500/35 bg-cyan-950/20 text-cyan-100'
                  : 'border-cyan-500/25 bg-cyan-950/20 text-gray-300'
            }`}
          >
            {status === 'saving' ? <Loader2 className="h-4 w-4 animate-spin" /> : <CheckCircle2 className="h-4 w-4 text-cyan-300" />}
            {saveStatusLabel}
          </span>
          <button
            type="button"
            onClick={saveSettings}
            disabled={status === 'saving'}
            aria-label="Save settings"
            title="Save settings"
            className="inline-flex h-9 w-10 items-center justify-center gap-2 border border-cyan-500 bg-cyan-500 px-0 text-sm font-semibold text-gray-950 transition-colors hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto sm:px-4"
          >
            {status === 'saving' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
            <span className="hidden sm:inline">Save changes</span>
          </button>
        </div>
      </TopBar>

      <MainContent className="bg-gray-950 px-0 pt-24">
        <div className="min-h-full">
          <div className="grid min-h-[calc(100dvh-4rem)] grid-cols-1 xl:grid-cols-[210px_minmax(0,1fr)_360px]">
            <aside className="hidden border-r border-gray-900 px-5 py-6 xl:block">
              <nav className="sticky top-6 space-y-1">
                {sectionNav.map(section => {
                  const active = activeSection === section.id;
                  return (
                    <button
                      key={section.id}
                      type="button"
                      onClick={() => jumpToSection(section.id)}
                      className={`flex h-11 w-full items-center border-l-2 px-3 text-left text-sm transition-colors ${
                        active
                          ? 'border-cyan-400 bg-gray-900/55 text-cyan-200'
                          : 'border-transparent text-gray-400 hover:bg-gray-900/35 hover:text-gray-200'
                      }`}
                    >
                      {section.label}
                    </button>
                  );
                })}
              </nav>
            </aside>

            <div className="min-w-0 px-4 py-6 sm:px-6 xl:px-7">
              <div className="mx-auto max-w-3xl xl:mx-0">
                <div className="operator-scrollbar-none mb-5 flex gap-1 overflow-x-auto xl:hidden">
                  {sectionNav.map(section => (
                    <button
                      key={section.id}
                      type="button"
                      onClick={() => jumpToSection(section.id)}
                      className={`h-9 flex-none border-b-2 px-3 text-sm ${
                        activeSection === section.id
                          ? 'border-cyan-400 text-cyan-100'
                          : 'border-transparent text-gray-400'
                      }`}
                    >
                      {section.label}
                    </button>
                  ))}
                </div>

                <section id="essentials" className="scroll-mt-20">
                  <div className="mb-6">
                    <h2 className="text-base font-semibold text-gray-100">Essentials</h2>
                    <p className="mt-1 text-sm text-gray-500">Core configuration for tokens and storage paths.</p>
                  </div>

                  <div className="space-y-4">
                    <FieldShell
                      id="HF_TOKEN"
                      label="Hugging Face token"
                      detail="Required to access gated or private models."
                    >
                      <div className="flex h-10 items-center border border-gray-800 bg-gray-950">
                        <KeyRound className="ml-3 h-4 w-4 text-gray-600" />
                        <input
                          type={showSecrets.hf ? 'text' : 'password'}
                          id="HF_TOKEN"
                          name="HF_TOKEN"
                          value={settings.HF_TOKEN}
                          onChange={handleChange}
                          className="min-w-0 flex-1 bg-transparent px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600"
                          placeholder="Enter your Hugging Face token"
                        />
                        <button
                          type="button"
                          onClick={() => setShowSecrets(prev => ({ ...prev, hf: !prev.hf }))}
                          className="flex h-full w-10 items-center justify-center border-l border-gray-800 text-gray-400 hover:text-gray-100"
                          title={showSecrets.hf ? 'Hide token' : 'Show token'}
                        >
                          {showSecrets.hf ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                        </button>
                      </div>
                    </FieldShell>

                    <FieldShell
                      id="OPENROUTER_API_KEY"
                      label="OpenRouter API key"
                      detail="Used for OpenRouter caption jobs."
                    >
                      <div className="flex h-10 items-center border border-gray-800 bg-gray-950">
                        <KeyRound className="ml-3 h-4 w-4 text-gray-600" />
                        <input
                          type={showSecrets.openRouter ? 'text' : 'password'}
                          id="OPENROUTER_API_KEY"
                          name="OPENROUTER_API_KEY"
                          value={settings.OPENROUTER_API_KEY}
                          onChange={handleChange}
                          className="min-w-0 flex-1 bg-transparent px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600"
                          placeholder="Enter your OpenRouter API key"
                        />
                        <button
                          type="button"
                          onClick={() => setShowSecrets(prev => ({ ...prev, openRouter: !prev.openRouter }))}
                          className="flex h-full w-10 items-center justify-center border-l border-gray-800 text-gray-400 hover:text-gray-100"
                          title={showSecrets.openRouter ? 'Hide API key' : 'Show API key'}
                        >
                          {showSecrets.openRouter ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                        </button>
                      </div>
                    </FieldShell>

                    <FieldShell
                      id="TRAINING_FOLDER"
                      label="Training folder"
                      detail="Where training outputs and logs are stored."
                    >
                      <div className="flex h-10 items-center border border-gray-800 bg-gray-950">
                        <input
                          type="text"
                          id="TRAINING_FOLDER"
                          name="TRAINING_FOLDER"
                          value={settings.TRAINING_FOLDER}
                          onChange={handleChange}
                          className="min-w-0 flex-1 bg-transparent px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600"
                          placeholder="Enter training folder path"
                        />
                        <button
                          type="button"
                          className="flex h-full w-12 items-center justify-center border-l border-gray-800 bg-gray-900/60 text-gray-300"
                          title="Training folder"
                        >
                          <FolderOpen className="h-4 w-4" />
                        </button>
                      </div>
                    </FieldShell>

                    <FieldShell
                      id="DATASETS_FOLDER"
                      label="Dataset folder"
                      detail="Where datasets are stored and discovered."
                    >
                      <div className="flex h-10 items-center border border-gray-800 bg-gray-950">
                        <input
                          type="text"
                          id="DATASETS_FOLDER"
                          name="DATASETS_FOLDER"
                          value={settings.DATASETS_FOLDER}
                          onChange={handleChange}
                          className="min-w-0 flex-1 bg-transparent px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600"
                          placeholder="Enter datasets folder path"
                        />
                        <button
                          type="button"
                          className="flex h-full w-12 items-center justify-center border-l border-gray-800 bg-gray-900/60 text-gray-300"
                          title="Dataset folder"
                        >
                          <FolderOpen className="h-4 w-4" />
                        </button>
                      </div>
                      <div className="mt-2 flex items-center gap-2 border border-amber-700/45 bg-amber-950/15 px-3 py-2 text-xs text-amber-200">
                        <AlertTriangle className="h-4 w-4 flex-none" />
                        Keep a backup elsewhere. Changes to datasets by this software are not reversible.
                      </div>
                    </FieldShell>
                  </div>
                </section>

                <section className="mt-7 border-t border-gray-900 pt-6">
                  <div className="mb-3">
                    <h2 className="text-base font-semibold text-gray-100">Automation</h2>
                    <p className="mt-1 text-sm text-gray-500">Optional behaviors that streamline your workflow.</p>
                  </div>
                  <div className="border border-gray-900">
                    <div className="flex items-center gap-3 border-b border-gray-900 px-3 py-3">
                      <TerminalSquare className="h-5 w-5 flex-none text-gray-500" />
                      <div className="min-w-0 flex-1">
                        <div className="text-sm font-semibold text-gray-100">Training Advisor (experimental)</div>
                        <div className="mt-0.5 text-xs text-gray-500">Enable advisor checks on training forms and completed job pages.</div>
                      </div>
                      <SettingSwitch
                        checked={settings.TRAINING_ADVISOR_ENABLED === 'true'}
                        onChange={checked => setSettings(prev => ({ ...prev, TRAINING_ADVISOR_ENABLED: checked ? 'true' : 'false' }))}
                      />
                    </div>
                    <div className="flex items-center gap-3 px-3 py-3">
                      <Cloud className="h-5 w-5 flex-none text-gray-500" />
                      <div className="min-w-0 flex-1">
                        <div className="text-sm font-semibold text-gray-100">Auto-install managed ComfyUI</div>
                        <div className="mt-0.5 text-xs text-gray-500">Automatically install the trainer-managed ComfyUI backend when required.</div>
                      </div>
                      <SettingSwitch
                        checked={settings.COMFY_AUTO_INSTALL === 'true'}
                        onChange={checked => setSettings(prev => ({ ...prev, COMFY_AUTO_INSTALL: checked ? 'true' : 'false' }))}
                      />
                    </div>
                  </div>
                </section>

                <section id="advanced" className="mt-5 scroll-mt-20">
                  <button
                    type="button"
                    onClick={() => {
                      setAdvancedOpen(open => !open);
                      setActiveSection('advanced');
                    }}
                    className="flex w-full items-center justify-between border border-gray-900 bg-gray-900/30 px-4 py-4 text-left transition-colors hover:bg-gray-900/50"
                  >
                    <span>
                      <span className="block text-sm font-semibold text-gray-100">Advanced configuration</span>
                      <span className="mt-1 block text-sm text-gray-500">Low-level settings for power users.</span>
                    </span>
                    <ChevronDown className={`h-5 w-5 text-gray-400 transition-transform ${advancedOpen ? 'rotate-180' : ''}`} />
                  </button>

                  {advancedOpen && (
                    <div className="mt-5 space-y-5">
                      <section id="comfy" className="scroll-mt-20 border-y border-gray-900 py-5">
                        <div className="mb-6 space-y-4 border-b border-gray-900 pb-5">
                          <FieldShell
                            id="COMFY_EXTERNAL_URL"
                            label="External ComfyUI URL"
                            detail="Used by the Ideogram workflow builder for live export, import, preflight, and generation."
                          >
                            <input
                              type="url"
                              id="COMFY_EXTERNAL_URL"
                              name="COMFY_EXTERNAL_URL"
                              value={settings.COMFY_EXTERNAL_URL}
                              onChange={handleChange}
                              className="h-10 w-full border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600 focus:border-cyan-700"
                              placeholder="http://127.0.0.1:8188"
                            />
                          </FieldShell>
                          <FieldShell
                            id="COMFY_EXTERNAL_LORA_DIR"
                            label="External ComfyUI LoRA folder"
                            detail="Absolute path to the external ComfyUI models/loras folder for copying Toolkit LoRAs."
                          >
                            <input
                              type="text"
                              id="COMFY_EXTERNAL_LORA_DIR"
                              name="COMFY_EXTERNAL_LORA_DIR"
                              value={settings.COMFY_EXTERNAL_LORA_DIR}
                              onChange={handleChange}
                              className="h-10 w-full border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600 focus:border-cyan-700"
                              placeholder="E:\\ComfyUI\\models\\loras"
                            />
                          </FieldShell>
                        </div>
                        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                          <div>
                            <h2 className="text-base font-semibold text-gray-100">Managed ComfyUI</h2>
                            <p className="mt-1 text-sm text-gray-500">Download and install the trainer-owned ComfyUI copy.</p>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            <button
                              type="button"
                              onClick={installComfyNow}
                              disabled={comfyInstallAction === 'installing' || comfyInstall?.installing}
                              className="inline-flex h-9 items-center gap-2 border border-cyan-800 bg-cyan-950/40 px-3 text-sm text-cyan-100 hover:bg-cyan-900 disabled:opacity-50"
                            >
                              {comfyInstallAction === 'installing' || comfyInstall?.installing ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <Download className="h-4 w-4" />
                              )}
                              {comfyInstallAction === 'installing' || comfyInstall?.installing
                                ? 'Installing'
                                : comfyInstall?.installed
                                  ? 'Refresh install'
                                  : 'Download install'}
                            </button>
                            <button
                              type="button"
                              onClick={refreshComfyInstall}
                              className="inline-flex h-9 items-center gap-2 border border-gray-800 bg-gray-950 px-3 text-sm text-gray-300 hover:bg-gray-900"
                            >
                              <RefreshCw className="h-4 w-4" />
                              Refresh
                            </button>
                          </div>
                        </div>
                        <div className="mt-4 border border-gray-900 bg-gray-950 px-3 py-3 text-sm text-gray-300">
                          <div>Status: {comfyInstall?.message || 'Not checked'}</div>
                          <div>Installed: {comfyInstall?.installed ? 'Yes' : 'No'}</div>
                          <div className="truncate">Root: {comfyInstall?.root || 'Not set'}</div>
                          <div className="truncate">Log: {comfyInstall?.logPath || 'Not set'}</div>
                          {comfyInstall?.pid && <div>Installer PID: {comfyInstall.pid}</div>}
                          {comfyInstall?.error && <div className="mt-2 text-rose-400">{comfyInstall.error}</div>}
                          {comfyInstallAction === 'error' && <div className="mt-2 text-rose-400">{comfyInstallActionError}</div>}
                        </div>
                        <div className="mt-3">
                          <ComfyInstallProgressBand progress={comfyInstall?.progress || null} />
                        </div>
                      </section>

                      <section id="workers" className="scroll-mt-20 border-y border-gray-900 py-5">
                        <div className="mb-4">
                          <h2 className="text-base font-semibold text-gray-100">Remote Workers</h2>
                          <p className="mt-1 text-sm text-gray-500">Central UI sends bundled jobs to these authenticated workers.</p>
                        </div>
                        <form onSubmit={saveWorker} className="grid gap-3 sm:grid-cols-2">
                          <input
                            type="text"
                            value={workerForm.name}
                            onChange={e => setWorkerForm(prev => ({ ...prev, name: e.target.value }))}
                            className="h-10 border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600"
                            placeholder="Worker name"
                          />
                          <input
                            type="url"
                            value={workerForm.base_url}
                            onChange={e => setWorkerForm(prev => ({ ...prev, base_url: e.target.value }))}
                            className="h-10 border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600"
                            placeholder="https://worker.example.com"
                          />
                          <input
                            type="password"
                            value={workerForm.api_token}
                            onChange={e => setWorkerForm(prev => ({ ...prev, api_token: e.target.value }))}
                            className="h-10 border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600 sm:col-span-2"
                            placeholder={workerForm.id ? 'Leave blank to keep existing API token' : 'Worker AI_TOOLKIT_AUTH token'}
                          />
                          <label className="flex items-center gap-2 text-sm text-gray-300">
                            <input
                              type="checkbox"
                              checked={workerForm.enabled}
                              onChange={e => setWorkerForm(prev => ({ ...prev, enabled: e.target.checked }))}
                              className="h-4 w-4 accent-cyan-500"
                            />
                            Enabled
                          </label>
                          <div className="flex justify-end gap-2">
                            {workerForm.id && (
                              <button
                                type="button"
                                onClick={() => setWorkerForm(emptyWorkerForm)}
                                className="h-9 border border-gray-800 px-3 text-sm text-gray-300 hover:bg-gray-900"
                              >
                                Cancel
                              </button>
                            )}
                            <button
                              type="submit"
                              disabled={workerStatus === 'saving'}
                              className="h-9 border border-gray-700 bg-gray-900 px-3 text-sm text-gray-100 hover:bg-gray-800 disabled:opacity-50"
                            >
                              {workerForm.id ? 'Update worker' : 'Add worker'}
                            </button>
                          </div>
                          {workerStatus === 'error' && <p className="text-sm text-rose-400 sm:col-span-2">Failed to save worker.</p>}
                        </form>

                        <div className="mt-5 divide-y divide-gray-900 border-y border-gray-900">
                          {workers.map(worker => {
                            const updater = workerUpdater[worker.id] || {};
                            const updaterStatus = updater.status;
                            const updaterAction = updater.action || 'idle';
                            const updaterBusy = workerUpdaterBusy(worker.id);
                            const canApplyWorkerUpdate = Boolean(
                              worker.enabled && updaterStatus?.canApplyUpdate && updaterStatus.state === 'update_available',
                            );
                            const restartSuggested = Boolean(updaterStatus?.needsRestart || updaterStatus?.state === 'updated');
                            const updaterLabel =
                              updaterAction === 'checking'
                                ? 'Checking'
                                : updaterAction === 'updating'
                                  ? 'Updating'
                                  : updaterAction === 'restarting'
                                    ? 'Restarting'
                                    : workerUpdaterLabel(updaterStatus, updater.error);
                            const updaterDetail =
                              updaterAction === 'checking'
                                ? 'Waiting for worker updater'
                                : updaterAction === 'updating'
                                  ? 'Waiting for worker update'
                                  : updaterAction === 'restarting'
                                    ? 'Worker may disconnect while it rebuilds'
                                    : workerUpdaterDetail(updaterStatus, updater.error);

                            return (
                              <div key={worker.id} className="py-3">
                                <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                                  <div className="min-w-0">
                                    <div className="truncate text-sm font-medium text-gray-100">{worker.name}</div>
                                    <div className="truncate text-xs text-gray-500">{worker.base_url}</div>
                                    <div className="mt-1 text-xs text-gray-400">
                                      {worker.last_status}
                                      {worker.last_error ? `: ${worker.last_error}` : ''}
                                    </div>
                                    <div className="mt-2 text-xs text-gray-500">
                                      Updater: {updaterLabel} · {updaterDetail}
                                    </div>
                                  </div>
                                  <div className="flex flex-wrap gap-2">
                                    <button type="button" className="h-8 border border-gray-800 px-2 text-xs text-gray-300" onClick={() => checkWorker(worker.id)}>
                                      Health
                                    </button>
                                    <button type="button" className="h-8 border border-gray-800 px-2 text-xs text-gray-300" onClick={() => editWorker(worker)}>
                                      Edit
                                    </button>
                                    <button type="button" className="h-8 border border-gray-800 px-2 text-xs text-gray-300" onClick={() => checkWorkerUpdates(worker.id)} disabled={!worker.enabled || updaterBusy}>
                                      Check
                                    </button>
                                    <button type="button" className="h-8 border border-amber-800 px-2 text-xs text-amber-100 disabled:opacity-45" onClick={() => updateRemoteWorker(worker.id)} disabled={!canApplyWorkerUpdate || updaterBusy}>
                                      Update
                                    </button>
                                    <button type="button" className={`h-8 border px-2 text-xs disabled:opacity-45 ${restartSuggested ? 'border-cyan-700 text-cyan-100' : 'border-gray-800 text-gray-300'}`} onClick={() => restartRemoteWorker(worker.id)} disabled={!worker.enabled || updaterBusy}>
                                      {updaterAction === 'restarting' ? 'Restarting' : 'Restart'}
                                    </button>
                                    <button type="button" className="h-8 border border-rose-900 px-2 text-xs text-rose-200" onClick={() => deleteWorker(worker.id)}>
                                      Delete
                                    </button>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                          {workers.length === 0 && <div className="py-3 text-sm text-gray-500">No remote workers configured.</div>}
                        </div>
                      </section>

                      <section className="grid gap-5 lg:grid-cols-2">
                        <div className="border-y border-gray-900 py-5">
                          <h2 className="text-base font-semibold text-gray-100">Remote Ollama</h2>
                          <p className="mt-1 text-sm text-gray-500">Direct Ollama HTTP endpoints for captioning and image tools.</p>
                          <form onSubmit={saveOllamaWorker} className="mt-4 space-y-3">
                            <input
                              type="text"
                              value={ollamaWorkerForm.name}
                              onChange={e => setOllamaWorkerForm(prev => ({ ...prev, name: e.target.value }))}
                              className="h-10 w-full border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600"
                              placeholder="Endpoint name"
                            />
                            <input
                              type="url"
                              value={ollamaWorkerForm.base_url}
                              onChange={e => setOllamaWorkerForm(prev => ({ ...prev, base_url: e.target.value }))}
                              className="h-10 w-full border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600"
                              placeholder="http://ollama-host:11434"
                            />
                            <input
                              type="password"
                              value={ollamaWorkerForm.auth_token}
                              onChange={e => setOllamaWorkerForm(prev => ({ ...prev, auth_token: e.target.value }))}
                              className="h-10 w-full border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600"
                              placeholder={ollamaWorkerForm.id ? 'Leave blank to keep existing bearer token' : 'Optional bearer token'}
                            />
                            <div className="flex items-center justify-between gap-2">
                              <label className="flex items-center gap-2 text-sm text-gray-300">
                                <input
                                  type="checkbox"
                                  checked={ollamaWorkerForm.enabled}
                                  onChange={e => setOllamaWorkerForm(prev => ({ ...prev, enabled: e.target.checked }))}
                                  className="h-4 w-4 accent-cyan-500"
                                />
                                Enabled
                              </label>
                              <div className="flex gap-2">
                                {ollamaWorkerForm.id && (
                                  <button
                                    type="button"
                                    onClick={() => setOllamaWorkerForm(emptyOllamaWorkerForm)}
                                    className="h-9 border border-gray-800 px-3 text-sm text-gray-300"
                                  >
                                    Cancel
                                  </button>
                                )}
                                <button
                                  type="submit"
                                  disabled={ollamaWorkerStatus === 'saving'}
                                  className="h-9 border border-gray-700 bg-gray-900 px-3 text-sm text-gray-100 disabled:opacity-50"
                                >
                                  {ollamaWorkerForm.id ? 'Update endpoint' : 'Add endpoint'}
                                </button>
                              </div>
                            </div>
                            {ollamaWorkerStatus === 'error' && <p className="text-sm text-rose-400">Failed to save Remote Ollama endpoint.</p>}
                          </form>
                          <div className="mt-4 divide-y divide-gray-900 border-y border-gray-900">
                            {ollamaWorkers.map(worker => (
                              <div key={worker.id} className="flex items-start justify-between gap-3 py-3">
                                <div className="min-w-0">
                                  <div className="truncate text-sm font-medium text-gray-100">{worker.name}</div>
                                  <div className="truncate text-xs text-gray-500">{worker.base_url}</div>
                                  <div className="mt-1 text-xs text-gray-400">
                                    {worker.last_status}
                                    {typeof worker.model_count === 'number' ? `: ${worker.model_count} model${worker.model_count === 1 ? '' : 's'}` : ''}
                                    {worker.last_error ? `: ${worker.last_error}` : ''}
                                  </div>
                                </div>
                                <div className="flex flex-none gap-2">
                                  <button type="button" className="h-8 border border-gray-800 px-2 text-xs" onClick={() => checkOllamaWorker(worker.id)}>Health</button>
                                  <button type="button" className="h-8 border border-gray-800 px-2 text-xs" onClick={() => editOllamaWorker(worker)}>Edit</button>
                                  <button type="button" className="h-8 border border-rose-900 px-2 text-xs text-rose-200" onClick={() => deleteOllamaWorker(worker.id)}>Delete</button>
                                </div>
                              </div>
                            ))}
                            {ollamaWorkers.length === 0 && <div className="py-3 text-sm text-gray-500">No Remote Ollama endpoints configured.</div>}
                          </div>
                        </div>

                        <div className="border-y border-gray-900 py-5">
                          <h2 className="text-base font-semibold text-gray-100">Cloudflared</h2>
                          <p className="mt-1 text-sm text-gray-500">Managed tunnel status from AITK_CLOUDFLARED_* variables.</p>
                          <div className="mt-4 space-y-1 border border-gray-900 bg-gray-950 px-3 py-3 text-sm text-gray-300">
                            <div>Status: {cloudflared?.message || 'Unknown'}</div>
                            <div>Mode: {cloudflared ? (cloudflared.mode === 'named' ? 'Named tunnel' : 'Quick tunnel') : 'Unknown'}</div>
                            <div>Detected: {cloudflared?.detected ? 'Yes' : 'No'}</div>
                            <div className="truncate">Public URL: {cloudflared?.publicUrl || (cloudflared?.running ? 'Waiting for cloudflared' : 'Not set')}</div>
                            <div className="truncate">Target URL: {cloudflared?.targetUrl || 'Not set'}</div>
                            {cloudflared?.error && <div className="mt-2 text-rose-400">{cloudflared.error}</div>}
                            {cloudflaredAction === 'error' && <div className="mt-2 text-rose-400">{cloudflaredActionError}</div>}
                          </div>
                          <div className="mt-3 flex items-center justify-between gap-3 border border-gray-900 px-3 py-3">
                            <div>
                              <div className="text-sm font-medium text-gray-100">Auto-download missing cloudflared</div>
                              <div className="mt-0.5 text-xs text-gray-500">Use the official Cloudflare release for this OS.</div>
                            </div>
                            <SettingSwitch checked={cloudflaredAutoDownload} onChange={setAutoDownloadCloudflared} />
                          </div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            <button type="button" onClick={startCloudflared} disabled={cloudflaredAction === 'starting' || cloudflaredAction === 'downloading'} className="h-9 border border-emerald-800 px-3 text-sm text-emerald-100 disabled:opacity-50">
                              {cloudflaredAction === 'starting' ? 'Starting' : 'Start'}
                            </button>
                            <button type="button" onClick={downloadCloudflared} disabled={!cloudflared?.downloadAvailable || cloudflaredAction === 'starting' || cloudflaredAction === 'downloading'} className="inline-flex h-9 items-center gap-2 border border-cyan-800 px-3 text-sm text-cyan-100 disabled:opacity-50">
                              {cloudflaredAction === 'downloading' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                              Download
                            </button>
                            <button type="button" onClick={() => apiClient.delete('/api/cloudflared').finally(refreshCloudflared)} className="h-9 border border-rose-900 px-3 text-sm text-rose-200">
                              Stop
                            </button>
                            <button type="button" onClick={refreshCloudflared} className="h-9 border border-gray-800 px-3 text-sm text-gray-300">
                              Refresh
                            </button>
                          </div>
                        </div>
                      </section>
                    </div>
                  )}
                </section>
              </div>
            </div>

            <aside className="hidden border-l border-gray-900 px-6 py-6 xl:block">
              <div className="sticky top-6">
                <h2 className="text-base font-semibold text-gray-100">System status</h2>
                <p className="mt-1 text-sm text-gray-500">Overview of key services and runtimes.</p>

                <div className="mt-8 space-y-6">
                  <div className="border-b border-gray-900 pb-5">
                    <div className="flex items-center gap-3">
                      <Box className="h-5 w-5 text-gray-300" />
                      <div className="min-w-0 flex-1 text-sm font-semibold text-gray-100">Managed ComfyUI</div>
                      <StatusDot tone={comfyInstall?.installed ? 'ok' : 'idle'} />
                      <span className="text-xs text-gray-400">{comfyInstall?.installed ? 'Installed' : 'Not installed'}</span>
                    </div>
                    <div className="mt-4 grid grid-cols-[6rem_1fr] gap-y-2 pl-8 text-sm">
                      <div className="text-gray-500">Backend</div>
                      <div className="truncate text-right text-gray-300">aitk_comfy</div>
                      <div className="text-gray-500">Root</div>
                      <div className="truncate text-right text-gray-300">{comfyInstall?.root || 'Not set'}</div>
                    </div>
                  </div>

                  <div className="border-b border-gray-900 pb-5">
                    <div className="flex items-center gap-3">
                      <UsersRound className="h-5 w-5 text-gray-300" />
                      <div className="min-w-0 flex-1 text-sm font-semibold text-gray-100">Remote workers</div>
                      <StatusDot tone={activeWorkers > 0 ? 'ok' : 'idle'} />
                      <span className="text-xs text-gray-400">{activeWorkers > 0 ? 'Connected' : 'Not configured'}</span>
                    </div>
                    <div className="mt-4 grid grid-cols-[7rem_1fr] gap-y-2 pl-8 text-sm">
                      <div className="text-gray-500">Active workers</div>
                      <div className="text-right text-gray-300">{activeWorkers}</div>
                    </div>
                  </div>

                  <div className="border-b border-gray-900 pb-5">
                    <div className="flex items-center gap-3">
                      <ServerCog className="h-5 w-5 text-gray-300" />
                      <div className="min-w-0 flex-1 text-sm font-semibold text-gray-100">Ollama endpoint</div>
                      <StatusDot tone={hasHealthyOllama ? 'ok' : 'idle'} />
                      <span className="text-xs text-gray-400">{hasHealthyOllama ? 'Healthy' : 'Not configured'}</span>
                    </div>
                    <div className="mt-4 grid grid-cols-[5rem_1fr] gap-y-2 pl-8 text-sm">
                      <div className="text-gray-500">Endpoint</div>
                      <div className="truncate text-right text-gray-300">{firstOllamaWorker?.base_url || 'Not set'}</div>
                      <div className="text-gray-500">Models</div>
                      <div className="text-right text-gray-300">{ollamaModelCount || '-'}</div>
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center gap-3">
                      <Database className="h-5 w-5 text-gray-300" />
                      <div className="min-w-0 flex-1 text-sm font-semibold text-gray-100">Updates</div>
                      <StatusDot tone="ok" />
                      <span className="text-xs text-gray-400">Ready</span>
                    </div>
                    <div className="mt-4 grid grid-cols-[5rem_1fr] gap-y-2 pl-8 text-sm">
                      <div className="text-gray-500">AI Toolkit</div>
                      <div className="text-right text-gray-300">v{appVersion}</div>
                      <div className="text-gray-500">Settings</div>
                      <div className="text-right text-gray-300">{saveStatusLabel}</div>
                    </div>
                  </div>
                </div>

                <div className="mt-8 space-y-2">
                  <button
                    type="button"
                    onClick={installComfyNow}
                    disabled={comfyInstallAction === 'installing' || comfyInstall?.installing}
                    className="inline-flex h-10 w-full items-center justify-center gap-2 border border-gray-800 text-sm text-gray-200 hover:bg-gray-900 disabled:opacity-50"
                  >
                    {comfyInstallAction === 'installing' || comfyInstall?.installing ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <RefreshCw className="h-4 w-4" />
                    )}
                    Refresh install
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      refreshComfyInstall();
                      refreshWorkers();
                      refreshOllamaWorkers();
                      refreshCloudflared();
                    }}
                    className="inline-flex h-10 w-full items-center justify-center gap-2 border border-gray-800 text-sm text-gray-200 hover:bg-gray-900"
                  >
                    <RefreshCw className="h-4 w-4" />
                    Refresh status
                  </button>
                </div>
              </div>
            </aside>
          </div>
        </div>
      </MainContent>
    </>
  );
}
