'use client';

import { useEffect, useRef, useState } from 'react';
import useSettings from '@/hooks/useSettings';
import useWorkers from '@/hooks/useWorkers';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import type { WorkerNode } from '@/types';
import { Checkbox } from '@/components/formInputs';
import { Download, Loader2, Power, RefreshCw } from 'lucide-react';

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

export default function Settings() {
  const { settings, setSettings } = useSettings();
  const { workers, refreshWorkers } = useWorkers();
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [workerForm, setWorkerForm] = useState(emptyWorkerForm);
  const [workerStatus, setWorkerStatus] = useState<'idle' | 'saving' | 'error'>('idle');
  const [cloudflared, setCloudflared] = useState<CloudflaredStatus | null>(null);
  const [cloudflaredAction, setCloudflaredAction] = useState<'idle' | 'starting' | 'downloading' | 'error'>('idle');
  const [cloudflaredActionError, setCloudflaredActionError] = useState('');
  const [cloudflaredAutoDownload, setCloudflaredAutoDownload] = useState(true);
  const [workerUpdater, setWorkerUpdater] = useState<Record<string, WorkerUpdaterUiState>>({});
  const workerUpdaterPolls = useRef<Record<string, number>>({});
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

  useEffect(() => {
    refreshCloudflared();
    if (typeof window !== 'undefined') {
      setCloudflaredAutoDownload(window.localStorage.getItem(CLOUDFLARED_AUTO_DOWNLOAD_KEY) !== 'false');
    }
  }, []);

  useEffect(() => {
    return () => {
      Object.values(workerUpdaterPolls.current).forEach(interval => window.clearInterval(interval));
      workerUpdaterPolls.current = {};
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
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
    await apiClient.delete(`/api/workers/${workerID}`).catch(error => {
      alert(error.response?.data?.error || 'Failed to delete worker.');
    });
    refreshWorkers();
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg">Settings</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
            <div>
              <div className="space-y-4">
                <div>
                  <label htmlFor="HF_TOKEN" className="block text-sm font-medium mb-2">
                    Hugging Face Token
                    <div className="text-gray-500 text-sm ml-1">
                      Create a Read token on{' '}
                      <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
                        {' '}
                        Huggingface
                      </a>{' '}
                      if you need to access gated/private models.
                    </div>
                  </label>
                  <input
                    type="password"
                    id="HF_TOKEN"
                    name="HF_TOKEN"
                    value={settings.HF_TOKEN}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter your Hugging Face token"
                  />
                </div>

                <div>
                  <label htmlFor="TRAINING_FOLDER" className="block text-sm font-medium mb-2">
                    Training Folder Path
                    <div className="text-gray-500 text-sm ml-1">
                      We will store your training information here. Must be an absolute path. If blank, it will default
                      to the output folder in the project root.
                    </div>
                  </label>
                  <input
                    type="text"
                    id="TRAINING_FOLDER"
                    name="TRAINING_FOLDER"
                    value={settings.TRAINING_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter training folder path"
                  />
                </div>

                <div>
                  <label htmlFor="DATASETS_FOLDER" className="block text-sm font-medium mb-2">
                    Dataset Folder Path
                    <div className="text-gray-500 text-sm ml-1">
                      Where we store and find your datasets.{' '}
                      <span className="text-orange-800">
                        Warning: This software may modify datasets so it is recommended you keep a backup somewhere else
                        or have a dedicated folder for this software.
                      </span>
                    </div>
                  </label>
                  <input
                    type="text"
                    id="DATASETS_FOLDER"
                    name="DATASETS_FOLDER"
                    value={settings.DATASETS_FOLDER}
                    onChange={handleChange}
                    className="w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-gray-600 focus:border-transparent"
                    placeholder="Enter datasets folder path"
                  />
                </div>

                <div className="rounded-lg border border-gray-800 bg-gray-900/70 p-4">
                  <Checkbox
                    checked={settings.TRAINING_ADVISOR_ENABLED === 'true'}
                    onChange={checked =>
                      setSettings(prev => ({ ...prev, TRAINING_ADVISOR_ENABLED: checked ? 'true' : 'false' }))
                    }
                    label={
                      <span>
                        Training Advisor (experimental)
                        <span className="mt-1 block text-xs font-normal text-gray-500">
                          Enable advisor checks on training forms and completed job pages.
                        </span>
                      </span>
                    }
                  />
                </div>
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={status === 'saving'}
            className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {status === 'saving' ? 'Saving...' : 'Save Settings'}
          </button>

          {status === 'success' && <p className="text-green-500 text-center">Settings saved successfully!</p>}
          {status === 'error' && <p className="text-red-500 text-center">Error saving settings. Please try again.</p>}
        </form>

        <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
          <section className="rounded-xl border border-gray-800 bg-gray-900 p-4">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h2 className="text-base font-semibold text-gray-100">Remote Workers</h2>
                <p className="text-sm text-gray-500">Central UI sends bundled jobs to these authenticated workers.</p>
              </div>
            </div>

            <form onSubmit={saveWorker} className="space-y-3">
              <input
                type="text"
                value={workerForm.name}
                onChange={e => setWorkerForm(prev => ({ ...prev, name: e.target.value }))}
                className="w-full rounded-lg border border-gray-700 bg-gray-800 px-4 py-2"
                placeholder="Worker name"
              />
              <input
                type="url"
                value={workerForm.base_url}
                onChange={e => setWorkerForm(prev => ({ ...prev, base_url: e.target.value }))}
                className="w-full rounded-lg border border-gray-700 bg-gray-800 px-4 py-2"
                placeholder="https://worker.example.com"
              />
              <input
                type="password"
                value={workerForm.api_token}
                onChange={e => setWorkerForm(prev => ({ ...prev, api_token: e.target.value }))}
                className="w-full rounded-lg border border-gray-700 bg-gray-800 px-4 py-2"
                placeholder={workerForm.id ? 'Leave blank to keep existing API token' : 'Worker AI_TOOLKIT_AUTH token'}
              />
              <label className="flex items-center gap-2 text-sm text-gray-300">
                <input
                  type="checkbox"
                  checked={workerForm.enabled}
                  onChange={e => setWorkerForm(prev => ({ ...prev, enabled: e.target.checked }))}
                />
                Enabled
              </label>
              <div className="flex gap-2">
                <button
                  type="submit"
                  disabled={workerStatus === 'saving'}
                  className="rounded-lg bg-gray-700 px-4 py-2 text-sm hover:bg-gray-600 disabled:opacity-50"
                >
                  {workerForm.id ? 'Update Worker' : 'Add Worker'}
                </button>
                {workerForm.id && (
                  <button
                    type="button"
                    onClick={() => setWorkerForm(emptyWorkerForm)}
                    className="rounded-lg bg-gray-800 px-4 py-2 text-sm hover:bg-gray-700"
                  >
                    Cancel
                  </button>
                )}
              </div>
              {workerStatus === 'error' && <p className="text-sm text-red-400">Failed to save worker.</p>}
            </form>

            <div className="mt-5 space-y-2">
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
                  <div key={worker.id} className="rounded-lg border border-gray-800 bg-gray-950 px-3 py-2">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="truncate text-sm font-medium text-gray-100">{worker.name}</div>
                        <div className="truncate text-xs text-gray-500">{worker.base_url}</div>
                        <div className="mt-1 text-xs text-gray-400">
                          {worker.last_status}
                          {worker.last_error ? `: ${worker.last_error}` : ''}
                        </div>
                      </div>
                      <div className="flex shrink-0 gap-2">
                        <button className="rounded bg-gray-800 px-2 py-1 text-xs" onClick={() => checkWorker(worker.id)}>
                          Health
                        </button>
                        <button className="rounded bg-gray-800 px-2 py-1 text-xs" onClick={() => editWorker(worker)}>
                          Edit
                        </button>
                        <button className="rounded bg-red-900 px-2 py-1 text-xs" onClick={() => deleteWorker(worker.id)}>
                          Delete
                        </button>
                      </div>
                    </div>

                    <div className="mt-3 rounded border border-gray-800 bg-gray-900/70 px-3 py-2">
                      <div className="flex flex-col gap-2 xl:flex-row xl:items-center xl:justify-between">
                        <div className="min-w-0">
                          <div className="truncate text-xs font-medium text-gray-200">
                            Updater: {updaterLabel}
                          </div>
                          <div className="mt-0.5 truncate text-xs text-gray-500">{updaterDetail}</div>
                        </div>
                        <div className="flex shrink-0 flex-wrap gap-2">
                          <button
                            type="button"
                            onClick={() => checkWorkerUpdates(worker.id)}
                            disabled={!worker.enabled || updaterBusy}
                            className="inline-flex items-center gap-1 rounded bg-gray-800 px-2 py-1 text-xs hover:bg-gray-700 disabled:opacity-50"
                            title="Check worker updates"
                          >
                            <RefreshCw className={`h-3.5 w-3.5 ${updaterAction === 'checking' ? 'animate-spin' : ''}`} />
                            Check
                          </button>
                          <button
                            type="button"
                            onClick={() => updateRemoteWorker(worker.id)}
                            disabled={!canApplyWorkerUpdate || updaterBusy}
                            className="inline-flex items-center gap-1 rounded bg-amber-900/70 px-2 py-1 text-xs text-amber-100 hover:bg-amber-800 disabled:opacity-50"
                            title={updaterStatus?.applyUpdateUnavailableReason || 'Update worker'}
                          >
                            {updaterAction === 'updating' ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Download className="h-3.5 w-3.5" />
                            )}
                            Update
                          </button>
                          <button
                            type="button"
                            onClick={() => restartRemoteWorker(worker.id)}
                            disabled={!worker.enabled || updaterBusy}
                            className={`inline-flex items-center gap-1 rounded px-2 py-1 text-xs hover:bg-cyan-800 disabled:opacity-50 ${
                              restartSuggested ? 'bg-cyan-900/80 text-cyan-100' : 'bg-gray-800 text-gray-200'
                            }`}
                            title="Restart worker"
                          >
                            {updaterAction === 'restarting' ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Power className="h-3.5 w-3.5" />
                            )}
                            Restart
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
              {workers.length === 0 && <div className="text-sm text-gray-500">No remote workers configured.</div>}
            </div>
          </section>

          <section className="rounded-xl border border-gray-800 bg-gray-900 p-4">
            <h2 className="text-base font-semibold text-gray-100">Cloudflared</h2>
            <p className="mt-1 text-sm text-gray-500">
              Managed tunnel status comes from AITK_CLOUDFLARED_* environment variables.
            </p>
            <div className="mt-4 rounded-lg border border-gray-800 bg-gray-950 p-3 text-sm">
              <div>Status: {cloudflared?.message || 'Unknown'}</div>
              <div>Mode: {cloudflared ? (cloudflared.mode === 'named' ? 'Named tunnel' : 'Quick tunnel') : 'Unknown'}</div>
              <div>Binary: {cloudflared?.bin || 'Not checked'}</div>
              <div>Detected: {cloudflared?.detected ? 'Yes' : 'No'}</div>
              <div>Public URL: {cloudflared?.publicUrl || (cloudflared?.running ? 'Waiting for cloudflared' : 'Not set')}</div>
              <div>Target URL: {cloudflared?.targetUrl || 'Not set'}</div>
              <div>Metrics: {cloudflared?.metricsAddr || 'Not set'}</div>
              {!cloudflared?.detected && cloudflared?.downloadAvailable && (
                <div className="mt-2 text-amber-300">cloudflared can be downloaded to {cloudflared.installPath}.</div>
              )}
              {cloudflared?.error && <div className="mt-2 text-red-400">{cloudflared.error}</div>}
              {cloudflaredAction === 'error' && <div className="mt-2 text-red-400">{cloudflaredActionError}</div>}
            </div>
            <div className="mt-4 rounded-lg border border-gray-800 bg-gray-950 p-3">
              <Checkbox
                checked={cloudflaredAutoDownload}
                onChange={setAutoDownloadCloudflared}
                label={
                  <span>
                    Auto-download missing cloudflared
                    <span className="mt-1 block text-xs font-normal text-gray-500">
                      Uses the official Cloudflare GitHub release for this OS and stores it in the local bin folder.
                    </span>
                  </span>
                }
              />
            </div>
            <div className="mt-4 flex gap-2">
              <button
                type="button"
                onClick={startCloudflared}
                disabled={cloudflaredAction === 'starting' || cloudflaredAction === 'downloading'}
                className="rounded-lg bg-green-700 px-4 py-2 text-sm hover:bg-green-600"
              >
                {cloudflaredAction === 'starting' ? 'Starting...' : 'Start'}
              </button>
              <button
                type="button"
                onClick={downloadCloudflared}
                disabled={!cloudflared?.downloadAvailable || cloudflaredAction === 'starting' || cloudflaredAction === 'downloading'}
                className="inline-flex items-center gap-2 rounded-lg bg-blue-700 px-4 py-2 text-sm hover:bg-blue-600 disabled:opacity-50"
              >
                {cloudflaredAction === 'downloading' ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Download className="h-4 w-4" />
                )}
                Download
              </button>
              <button
                type="button"
                onClick={() => apiClient.delete('/api/cloudflared').finally(refreshCloudflared)}
                className="rounded-lg bg-red-900 px-4 py-2 text-sm hover:bg-red-800"
              >
                Stop
              </button>
              <button
                type="button"
                onClick={refreshCloudflared}
                className="rounded-lg bg-gray-800 px-4 py-2 text-sm hover:bg-gray-700"
              >
                Refresh
              </button>
            </div>
          </section>
        </div>
      </MainContent>
    </>
  );
}
