'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Button } from '@headlessui/react';
import { Eye, FolderSync, Loader2, Play, Plus, Save, Trash2, X } from 'lucide-react';
import { Modal } from '@/components/Modal';
import { Checkbox, CreatableSelectInput, NumberInput, SelectInput, TextAreaInput, TextInput } from '@/components/formInputs';
import { StatusBadge } from '@/components/OperatorPrimitives';
import { apiClient } from '@/utils/api';
import useRemoteOllamaWorkers from '@/hooks/useRemoteOllamaWorkers';
import { defaultImageCaptionPrompt } from '@/helpers/captionOptions';
import {
  AUTO_BOX_PROVIDERS,
  DEFAULT_OLLAMA_VISION_MODEL,
  DEFAULT_OPENROUTER_BOX_MODEL,
  OLLAMA_VISION_MODELS,
  OPENROUTER_BOX_MODELS,
} from '@/components/dataset-image-studio/constants';

type RecaptionProvider = 'openrouter' | 'ollama' | 'remote_ollama';
type RecaptionOutputFormat = 'text' | 'ideogram_json';

type DatasetWatcherAutoCaption = {
  enabled: boolean;
  provider: RecaptionProvider;
  model: string;
  prompt?: string;
  systemPrompt?: string;
  outputFormat?: RecaptionOutputFormat;
  maxNewTokens?: number | null;
  remoteWorkerId?: string;
};

type DatasetWatcher = {
  id: string;
  datasetName: string;
  projectID: string | null;
  enabled: boolean;
  sourcePath: string;
  includeSubfolders: boolean;
  preserveRelativePaths: boolean;
  autoCaption: DatasetWatcherAutoCaption | null;
  createdAt: string;
  updatedAt: string;
};

type DatasetWatcherStatus = {
  state: string;
  lastScanAt: string | null;
  lastImportedAt: string | null;
  lastImportedCount: number;
  lastCaptionedCount: number;
  autoCaptionTotalCount?: number;
  autoCaptionPendingCount?: number;
  autoCaptionCompletedCount?: number;
  autoCaptionActivePath?: string | null;
  lastError: string | null;
  warnings: string[];
};

type DatasetWatcherForm = {
  id: string | null;
  enabled: boolean;
  sourcePath: string;
  includeSubfolders: boolean;
  preserveRelativePaths: boolean;
  autoCaptionEnabled: boolean;
  provider: RecaptionProvider;
  model: string;
  outputFormat: RecaptionOutputFormat;
  prompt: string;
  systemPrompt: string;
  maxNewTokens: number | null;
  remoteWorkerId: string;
};

type Props = {
  datasetName: string;
  projectID?: string | null;
  workerID?: string;
  defaultSourcePath?: string | null;
  label?: string;
  className?: string;
  icon?: 'folderSync' | 'eye';
  iconOnly?: boolean;
  onRefresh?: () => void;
};

const emptyForm = (defaultSourcePath = ''): DatasetWatcherForm => ({
  id: null,
  enabled: true,
  sourcePath: defaultSourcePath,
  includeSubfolders: true,
  preserveRelativePaths: true,
  autoCaptionEnabled: false,
  provider: 'openrouter',
  model: DEFAULT_OPENROUTER_BOX_MODEL,
  outputFormat: 'text',
  prompt: defaultImageCaptionPrompt,
  systemPrompt: '',
  maxNewTokens: 256,
  remoteWorkerId: '',
});

const outputFormatOptions = [
  { value: 'text', label: 'Text captions' },
  { value: 'ideogram_json', label: 'Ideogram JSON' },
];

function providerOptions() {
  return AUTO_BOX_PROVIDERS.map(provider => ({ value: provider.value, label: provider.label }));
}

function stateLabel(status?: DatasetWatcherStatus) {
  if (!status) return 'Idle';
  if (status.state === 'captioning') return 'Captioning';
  if (status.state === 'importing') return 'Importing';
  if (status.state === 'scanning') return 'Scanning';
  if (status.state === 'disabled') return 'Disabled';
  if (status.state === 'error') return 'Error';
  return 'Idle';
}

function stateForBadge(status?: DatasetWatcherStatus) {
  if (!status) return 'stopped';
  if (status.state === 'error') return 'error';
  if (status.state === 'disabled') return 'stopped';
  if (status.state === 'captioning' || status.state === 'importing' || status.state === 'scanning') return 'running';
  return 'completed';
}

function formatDate(value: string | null | undefined) {
  if (!value) return 'Never';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Never' : date.toLocaleString();
}

function watcherAutoCaptionLabel(status?: DatasetWatcherStatus) {
  const pending = Math.max(0, Math.floor(Number(status?.autoCaptionPendingCount || 0)));
  if (pending <= 0) return '';
  const total = Math.max(0, Math.floor(Number(status?.autoCaptionTotalCount || 0)));
  const completed = Math.max(0, Math.floor(Number(status?.autoCaptionCompletedCount || 0)));
  const progress = total > 0 ? `${completed.toLocaleString()}/${total.toLocaleString()}` : completed.toLocaleString();
  return `Captioning ${progress}, ${pending.toLocaleString()} left`;
}

function watcherAutoCaptionTitle(status?: DatasetWatcherStatus) {
  const activePath = status?.autoCaptionActivePath;
  return activePath ? `Auto-captioning ${activePath}` : 'Auto-captioning imported images';
}

function formFromWatcher(watcher: DatasetWatcher): DatasetWatcherForm {
  return {
    id: watcher.id,
    enabled: watcher.enabled,
    sourcePath: watcher.sourcePath,
    includeSubfolders: watcher.includeSubfolders,
    preserveRelativePaths: watcher.preserveRelativePaths,
    autoCaptionEnabled: watcher.autoCaption?.enabled === true,
    provider: watcher.autoCaption?.provider || 'openrouter',
    model: watcher.autoCaption?.model || DEFAULT_OPENROUTER_BOX_MODEL,
    outputFormat: watcher.autoCaption?.outputFormat || 'text',
    prompt: watcher.autoCaption?.prompt || defaultImageCaptionPrompt,
    systemPrompt: watcher.autoCaption?.systemPrompt || '',
    maxNewTokens: watcher.autoCaption?.maxNewTokens || 256,
    remoteWorkerId: watcher.autoCaption?.remoteWorkerId || '',
  };
}

export default function DatasetWatchFoldersButton({
  datasetName,
  projectID = null,
  workerID = 'local',
  defaultSourcePath = '',
  label = 'Watch Folders',
  className = 'operator-button whitespace-nowrap py-1 text-sm',
  icon = 'folderSync',
  iconOnly = false,
  onRefresh,
}: Props) {
  const [open, setOpen] = useState(false);
  const [watchers, setWatchers] = useState<DatasetWatcher[]>([]);
  const [statuses, setStatuses] = useState<Record<string, DatasetWatcherStatus>>({});
  const [form, setForm] = useState<DatasetWatcherForm>(() => emptyForm(defaultSourcePath || ''));
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');
  const statusRefreshSignatureRef = useRef('');
  const systemPromptTouchedRef = useRef(false);
  const autoSystemPromptRef = useRef('');
  const rootCaptionSourceRef = useRef('');
  const { workers } = useRemoteOllamaWorkers({ enabled: open });

  const remoteWorkerOptions = useMemo(
    () => workers.filter(worker => worker.enabled).map(worker => ({ value: worker.id, label: worker.name })),
    [workers],
  );
  const modelOptions = useMemo(() => {
    if (form.provider === 'openrouter') return OPENROUTER_BOX_MODELS.map(option => ({ ...option }));
    return OLLAMA_VISION_MODELS.map(option => ({ ...option }));
  }, [form.provider]);

  useEffect(() => {
    statusRefreshSignatureRef.current = '';
  }, [datasetName, projectID, workerID]);

  useEffect(() => {
    if (form.provider !== 'remote_ollama' || form.remoteWorkerId || remoteWorkerOptions.length === 0) return;
    setForm(current =>
      current.provider === 'remote_ollama' && !current.remoteWorkerId
        ? { ...current, remoteWorkerId: remoteWorkerOptions[0].value }
        : current,
    );
  }, [form.provider, form.remoteWorkerId, remoteWorkerOptions]);

  const loadWatchers = useCallback(async () => {
    if (!open) return;
    setIsLoading(true);
    setError('');
    try {
      const res = await apiClient.get('/api/datasets/watchers', {
        params: {
          datasetName,
          worker_id: workerID,
          ...(projectID ? { project_id: projectID } : {}),
        },
      });
      setWatchers(res.data?.watchers || []);
      const nextStatuses = res.data?.statuses || {};
      setStatuses(nextStatuses);
      const refreshSignature = Object.entries(nextStatuses)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([id, status]) => {
          const watcherStatus = status as DatasetWatcherStatus;
          return [
            id,
            watcherStatus.lastImportedAt || '',
            watcherStatus.lastImportedCount || 0,
            watcherStatus.lastCaptionedCount || 0,
            watcherStatus.autoCaptionTotalCount || 0,
            watcherStatus.autoCaptionPendingCount || 0,
            watcherStatus.autoCaptionCompletedCount || 0,
            watcherStatus.autoCaptionActivePath || '',
          ].join(':');
        })
        .join('|');
      const hasImportedWork = Object.values(nextStatuses).some(status => {
        const watcherStatus = status as DatasetWatcherStatus;
        return Boolean(
          watcherStatus.lastImportedAt ||
            watcherStatus.lastImportedCount ||
            watcherStatus.lastCaptionedCount ||
            watcherStatus.autoCaptionPendingCount ||
            watcherStatus.autoCaptionCompletedCount,
        );
      });
      if (
        (statusRefreshSignatureRef.current && refreshSignature !== statusRefreshSignatureRef.current) ||
        (!statusRefreshSignatureRef.current && hasImportedWork)
      ) {
        onRefresh?.();
      }
      statusRefreshSignatureRef.current = refreshSignature;
    } catch (requestError: any) {
      setError(requestError?.response?.data?.error || requestError?.message || 'Failed to load watch folders.');
    } finally {
      setIsLoading(false);
    }
  }, [datasetName, onRefresh, open, projectID, workerID]);

  useEffect(() => {
    void loadWatchers();
  }, [loadWatchers]);

  useEffect(() => {
    if (!open) return;
    const interval = window.setInterval(() => void loadWatchers(), 5000);
    return () => window.clearInterval(interval);
  }, [loadWatchers, open]);

  useEffect(() => {
    if (!defaultSourcePath) return;
    setForm(current => (current.id || current.sourcePath.trim() ? current : { ...current, sourcePath: defaultSourcePath }));
  }, [defaultSourcePath]);

  useEffect(() => {
    if (!open || !form.autoCaptionEnabled) return;
    const sourcePath = form.sourcePath.trim();
    if (!sourcePath || systemPromptTouchedRef.current) return;
    if (form.systemPrompt.trim() && form.systemPrompt !== autoSystemPromptRef.current) return;

    const sourceSignature = [workerID, projectID || '', sourcePath].join('\n');
    if (rootCaptionSourceRef.current === sourceSignature) return;
    rootCaptionSourceRef.current = sourceSignature;

    let cancelled = false;
    const timer = window.setTimeout(() => {
      apiClient
        .get('/api/datasets/watchers', {
          params: {
            action: 'root-caption',
            sourcePath,
            worker_id: workerID,
            ...(projectID ? { project_id: projectID } : {}),
          },
        })
        .then(res => {
          if (cancelled) return;
          const systemPrompt = typeof res.data?.systemPrompt === 'string' ? res.data.systemPrompt.trim() : '';
          if (!res.data?.found || !systemPrompt) return;
          setForm(current => {
            if (
              current.sourcePath.trim() !== sourcePath ||
              !current.autoCaptionEnabled ||
              systemPromptTouchedRef.current ||
              (current.systemPrompt.trim() && current.systemPrompt !== autoSystemPromptRef.current)
            ) {
              return current;
            }
            autoSystemPromptRef.current = systemPrompt;
            return { ...current, systemPrompt };
          });
        })
        .catch(requestError => {
          if (!cancelled) console.warn('Could not load watch folder ROOT_CAPTION.txt:', requestError);
        });
    }, 300);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [form.autoCaptionEnabled, form.sourcePath, form.systemPrompt, open, projectID, workerID]);

  const updateForm = <K extends keyof DatasetWatcherForm>(key: K, value: DatasetWatcherForm[K]) => {
    if (key === 'systemPrompt') {
      systemPromptTouchedRef.current = true;
      autoSystemPromptRef.current = '';
    }
    if (key === 'sourcePath') {
      rootCaptionSourceRef.current = '';
    }
    setForm(current => {
      if (key === 'enabled' && value === true && !current.id && !current.sourcePath.trim() && defaultSourcePath) {
        return { ...current, enabled: true, sourcePath: defaultSourcePath };
      }
      if (
        key === 'sourcePath' &&
        !systemPromptTouchedRef.current &&
        current.systemPrompt &&
        current.systemPrompt === autoSystemPromptRef.current
      ) {
        autoSystemPromptRef.current = '';
        return { ...current, sourcePath: value as string, systemPrompt: '' };
      }
      return { ...current, [key]: value };
    });
  };

  const handleProviderChange = (value: string) => {
    const provider = value === 'ollama' || value === 'remote_ollama' ? value : 'openrouter';
    setForm(current => ({
      ...current,
      provider,
      model:
        provider === 'openrouter'
          ? current.model && !current.model.startsWith('qwen3.5:') && !current.model.startsWith('gemma4:')
            ? current.model
            : DEFAULT_OPENROUTER_BOX_MODEL
          : current.model && current.model !== DEFAULT_OPENROUTER_BOX_MODEL
            ? current.model
            : DEFAULT_OLLAMA_VISION_MODEL,
      remoteWorkerId:
        provider === 'remote_ollama' ? current.remoteWorkerId || remoteWorkerOptions[0]?.value || '' : current.remoteWorkerId,
    }));
  };

  const resetForm = () => {
    systemPromptTouchedRef.current = false;
    autoSystemPromptRef.current = '';
    rootCaptionSourceRef.current = '';
    setForm(emptyForm(defaultSourcePath || ''));
  };

  const saveWatcher = async () => {
    setIsSaving(true);
    setError('');
    try {
      const payload = {
        id: form.id || undefined,
        datasetName,
        projectID,
        worker_id: workerID,
        enabled: form.enabled,
        sourcePath: form.sourcePath,
        includeSubfolders: form.includeSubfolders,
        preserveRelativePaths: form.preserveRelativePaths,
        autoCaption: form.autoCaptionEnabled
          ? {
              enabled: true,
              provider: form.provider,
              model: form.model,
              outputFormat: form.outputFormat,
              prompt: form.prompt,
              systemPrompt: form.systemPrompt,
              maxNewTokens: form.maxNewTokens,
              remoteWorkerId: form.remoteWorkerId,
            }
          : null,
      };
      const res = form.id
        ? await apiClient.patch('/api/datasets/watchers', payload)
        : await apiClient.post('/api/datasets/watchers', payload);
      const saved = res.data?.watcher;
      await loadWatchers();
      if (saved) {
        const nextForm = formFromWatcher(saved);
        systemPromptTouchedRef.current = Boolean(nextForm.systemPrompt.trim());
        autoSystemPromptRef.current = '';
        rootCaptionSourceRef.current = '';
        setForm(nextForm);
      }
    } catch (requestError: any) {
      setError(requestError?.response?.data?.error || requestError?.message || 'Failed to save watch folder.');
    } finally {
      setIsSaving(false);
    }
  };

  const deleteWatcher = async (watcher: DatasetWatcher) => {
    setIsSaving(true);
    setError('');
    try {
      await apiClient.delete('/api/datasets/watchers', {
        data: {
          id: watcher.id,
          worker_id: workerID,
          ...(projectID ? { project_id: projectID } : {}),
        },
      });
      if (form.id === watcher.id) resetForm();
      await loadWatchers();
    } catch (requestError: any) {
      setError(requestError?.response?.data?.error || requestError?.message || 'Failed to delete watch folder.');
    } finally {
      setIsSaving(false);
    }
  };

  const runWatcher = async (watcher: DatasetWatcher) => {
    setIsSaving(true);
    setError('');
    try {
      await apiClient.post('/api/datasets/watchers', {
        action: 'run',
        id: watcher.id,
        worker_id: workerID,
        ...(projectID ? { project_id: projectID } : {}),
      });
      await loadWatchers();
      onRefresh?.();
    } catch (requestError: any) {
      setError(requestError?.response?.data?.error || requestError?.message || 'Failed to sync watch folder.');
    } finally {
      setIsSaving(false);
    }
  };
  const TriggerIcon = icon === 'eye' ? Eye : FolderSync;

  return (
    <>
      <Button className={className} onClick={() => setOpen(true)} title={label} aria-label={label}>
        <TriggerIcon className="h-3.5 w-3.5" />
        <span className={iconOnly ? 'sr-only' : ''}>{label}</span>
      </Button>
      <Modal isOpen={open} onClose={() => setOpen(false)} title="Watch Folders" size="xl">
        <div className="space-y-4 text-sm text-gray-300">
          {error && <div className="border border-red-900 bg-red-950/50 px-3 py-2 text-red-200">{error}</div>}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_1.15fr]">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Configured</div>
                {isLoading && <Loader2 className="h-4 w-4 animate-spin text-gray-500" />}
              </div>
              <div className="max-h-[54vh] overflow-y-auto border border-gray-800">
                {watchers.length === 0 && (
                  <div className="px-3 py-4 text-sm text-gray-500">No watch folders.</div>
                )}
                {watchers.map(watcher => {
                  const status = statuses[watcher.id];
                  const autoCaptionLabel = watcherAutoCaptionLabel(status);
                  return (
                    <div key={watcher.id} className="border-b border-gray-800 px-3 py-3 last:border-b-0">
                      <div className="flex min-w-0 items-start gap-2">
                        <div className="min-w-0 flex-1">
                          <div className="truncate font-medium text-gray-100">{watcher.sourcePath}</div>
                          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-gray-500">
                            <StatusBadge status={stateForBadge(status)} label={stateLabel(status)} />
                            <span>Imported {status?.lastImportedCount || 0}</span>
                            <span>Captioned {status?.lastCaptionedCount || 0}</span>
                            {autoCaptionLabel && (
                              <span className="text-fuchsia-200" title={watcherAutoCaptionTitle(status)}>
                                {autoCaptionLabel}
                              </span>
                            )}
                          </div>
                          <div className="mt-1 text-xs text-gray-500">Last scan: {formatDate(status?.lastScanAt)}</div>
                          {status?.lastError && <div className="mt-1 truncate text-xs text-red-400">{status.lastError}</div>}
                        </div>
                        <div className="flex flex-none gap-1">
                          <button
                            type="button"
                            className="operator-icon-button"
                            onClick={() => {
                              const nextForm = formFromWatcher(watcher);
                              systemPromptTouchedRef.current = Boolean(nextForm.systemPrompt.trim());
                              autoSystemPromptRef.current = '';
                              rootCaptionSourceRef.current = '';
                              setForm(nextForm);
                            }}
                            title="Edit"
                          >
                            <Save className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            className="operator-icon-button"
                            onClick={() => runWatcher(watcher)}
                            title="Sync now"
                            disabled={isSaving}
                          >
                            <Play className="h-3.5 w-3.5" />
                          </button>
                          <button
                            type="button"
                            className="operator-icon-button text-red-300"
                            onClick={() => deleteWatcher(watcher)}
                            title="Delete"
                            disabled={isSaving}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="space-y-4 border border-gray-800 bg-gray-950/50 p-4">
              <div className="flex items-center justify-between">
                <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                  {form.id ? 'Edit Watch Folder' : 'New Watch Folder'}
                </div>
                {form.id && (
                  <button type="button" className="operator-icon-button" onClick={resetForm} title="New watcher">
                    <Plus className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
              <TextInput label="Folder Path" value={form.sourcePath} onChange={value => updateForm('sourcePath', value)} />
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                <Checkbox label="Enabled" checked={form.enabled} onChange={value => updateForm('enabled', value)} />
                <Checkbox
                  label="Subfolders"
                  checked={form.includeSubfolders}
                  onChange={value => updateForm('includeSubfolders', value)}
                />
                <Checkbox
                  label="Keep paths"
                  checked={form.preserveRelativePaths}
                  onChange={value => updateForm('preserveRelativePaths', value)}
                />
              </div>
              <Checkbox
                label="Auto-caption images"
                checked={form.autoCaptionEnabled}
                onChange={value => updateForm('autoCaptionEnabled', value)}
              />
              {form.autoCaptionEnabled && (
                <div className="space-y-4 border-t border-gray-800 pt-4">
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <SelectInput
                      label="Provider"
                      value={form.provider}
                      onChange={handleProviderChange}
                      options={providerOptions()}
                    />
                    <SelectInput
                      label="Output"
                      value={form.outputFormat}
                      onChange={value => updateForm('outputFormat', value === 'ideogram_json' ? 'ideogram_json' : 'text')}
                      options={outputFormatOptions}
                    />
                  </div>
                  {form.provider === 'remote_ollama' && (
                    <SelectInput
                      label="Remote Ollama"
                      value={form.remoteWorkerId}
                      onChange={value => updateForm('remoteWorkerId', value)}
                      options={remoteWorkerOptions}
                      disabled={remoteWorkerOptions.length === 0}
                    />
                  )}
                  <CreatableSelectInput
                    label="Model"
                    value={form.model}
                    onChange={value => updateForm('model', value)}
                    options={modelOptions}
                  />
                  <NumberInput
                    label="Max New Tokens"
                    value={form.maxNewTokens}
                    min={1}
                    onChange={value => updateForm('maxNewTokens', value)}
                  />
                  <TextAreaInput label="Prompt" value={form.prompt} onChange={value => updateForm('prompt', value)} rows={4} />
                  <TextAreaInput
                    label="System Prompt"
                    value={form.systemPrompt}
                    onChange={value => updateForm('systemPrompt', value)}
                    rows={3}
                  />
                </div>
              )}
              <div className="flex justify-end gap-2 border-t border-gray-800 pt-4">
                <button type="button" className="operator-button" onClick={resetForm} disabled={isSaving}>
                  <X className="h-3.5 w-3.5" />
                  Clear
                </button>
                <button type="button" className="operator-button border-blue-800 bg-blue-950/70 text-blue-100" onClick={saveWatcher} disabled={isSaving}>
                  {isSaving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                  Save
                </button>
              </div>
            </div>
          </div>
        </div>
      </Modal>
    </>
  );
}
