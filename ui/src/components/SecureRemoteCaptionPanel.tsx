'use client';

import type React from 'react';
import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { Loader2, RotateCcw, Save } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';
import type { SelectOption } from '@/types';
import { apiClient } from '@/utils/api';
import { pathJoin } from '@/utils/basic';
import { startJob } from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import { getRememberedEncryptedDatasetKey } from '@/utils/encryptedDatasets';
import useDatasetList from '@/hooks/useDatasetList';
import useSettings from '@/hooks/useSettings';
import useWorkers from '@/hooks/useWorkers';
import {
  Checkbox,
  CreatableSelectInput,
  NumberInput,
  SelectInput,
  TextAreaInput,
} from '@/components/formInputs';

const LAST_MODEL_KEY = 'AITK_SECURE_REMOTE_OLLAMA_MODEL';
const IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp', 'bmp'];
const DEFAULT_PROMPT =
  'Caption this image for training an image generation model. Be specific and decisive. Describe the subject, setting, composition, style, lighting, colors, and notable details. No preamble.';

type OllamaModel = {
  name?: string;
  model?: string;
  details?: {
    parameter_size?: string;
    quantization_level?: string;
  };
};

function modelName(model: OllamaModel) {
  return model.model || model.name || '';
}

export default function SecureRemoteCaptionPanel() {
  const { settings, isSettingsLoaded } = useSettings();
  const { workers, status: workerStatus } = useWorkers();
  const { datasets, status: datasetStatus } = useDatasetList();
  const [workerID, setWorkerID] = useState('');
  const [datasetName, setDatasetName] = useState('');
  const [model, setModel] = useState('');
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [modelStatus, setModelStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [modelError, setModelError] = useState('');
  const [captionPrompt, setCaptionPrompt] = useState(DEFAULT_PROMPT);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [systemPromptStatus, setSystemPromptStatus] = useState<
    'idle' | 'loading' | 'saving' | 'success' | 'error'
  >('idle');
  const [systemPromptError, setSystemPromptError] = useState('');
  const [recaption, setRecaption] = useState(false);
  const [maxRes, setMaxRes] = useState(768);
  const [maxNewTokens, setMaxNewTokens] = useState(180);
  const [submitStatus, setSubmitStatus] = useState<'idle' | 'starting' | 'success' | 'error'>('idle');
  const [submitError, setSubmitError] = useState('');
  const [startedJobId, setStartedJobId] = useState<string | null>(null);

  const enabledWorkers = useMemo(() => workers.filter(worker => worker.enabled), [workers]);
  const workerOptions = useMemo(
    () => enabledWorkers.map(worker => ({ value: worker.id, label: worker.name })),
    [enabledWorkers],
  );
  const datasetOptions = useMemo(
    () =>
      datasets.map(dataset => ({
        value: dataset.name,
        label: dataset.encrypted ? `${dataset.name} (encrypted)` : dataset.name,
      })),
    [datasets],
  );
  const selectedDataset = useMemo(
    () => datasets.find(dataset => dataset.name === datasetName) || null,
    [datasetName, datasets],
  );
  const datasetPath = useMemo(() => {
    if (!isSettingsLoaded || !settings.DATASETS_FOLDER || !datasetName) return '';
    return pathJoin(settings.DATASETS_FOLDER, datasetName);
  }, [datasetName, isSettingsLoaded, settings.DATASETS_FOLDER]);
  const modelOptions: SelectOption[] = useMemo(
    () =>
      models
        .map(item => {
          const value = modelName(item);
          const detail = [item.details?.parameter_size, item.details?.quantization_level].filter(Boolean).join(' ');
          return value ? { value, label: detail ? `${value} (${detail})` : value } : null;
        })
        .filter((item): item is SelectOption => item !== null),
    [models],
  );

  const loadModels = async (selectedWorkerID = workerID) => {
    if (!selectedWorkerID) return;
    setModelStatus('loading');
    setModelError('');
    try {
      const res = await apiClient.get('/api/secure-caption/ollama/models', {
        params: { worker_id: selectedWorkerID },
      });
      setModels(res.data?.models || []);
      setModelStatus('success');
    } catch (error: any) {
      setModels([]);
      setModelStatus('error');
      setModelError(error?.response?.data?.error || 'Could not reach remote Ollama.');
    }
  };

  const saveDatasetSystemPrompt = async (
    selectedDatasetName = datasetName,
    selectedSystemPrompt = systemPrompt,
  ) => {
    if (!selectedDatasetName) return '';
    setSystemPromptStatus('saving');
    setSystemPromptError('');
    const res = await apiClient.post('/api/secure-caption/dataset-system-prompt', {
      datasetName: selectedDatasetName,
      systemPrompt: selectedSystemPrompt,
    });
    const savedSystemPrompt = res.data?.systemPrompt || '';
    setSystemPrompt(savedSystemPrompt);
    setSystemPromptStatus('success');
    return savedSystemPrompt;
  };

  useEffect(() => {
    if (typeof window !== 'undefined') {
      setModel(window.localStorage.getItem(LAST_MODEL_KEY) || '');
    }
  }, []);

  useEffect(() => {
    if (!workerID && enabledWorkers.length > 0) {
      setWorkerID(enabledWorkers[0].id);
    }
  }, [enabledWorkers, workerID]);

  useEffect(() => {
    if (!datasetName && datasets.length > 0) {
      setDatasetName(datasets[0].name);
    }
  }, [datasetName, datasets]);

  useEffect(() => {
    if (workerID) {
      void loadModels(workerID);
    }
  }, [workerID]);

  useEffect(() => {
    let cancelled = false;

    const loadDatasetSystemPrompt = async () => {
      if (!datasetName) {
        setSystemPrompt('');
        setSystemPromptStatus('idle');
        setSystemPromptError('');
        return;
      }
      setSystemPromptStatus('loading');
      setSystemPromptError('');
      try {
        const res = await apiClient.get('/api/secure-caption/dataset-system-prompt', {
          params: { datasetName },
        });
        if (cancelled) return;
        setSystemPrompt(res.data?.systemPrompt || '');
        setSystemPromptStatus('idle');
      } catch (error: any) {
        if (cancelled) return;
        setSystemPrompt('');
        setSystemPromptStatus('error');
        setSystemPromptError(error?.response?.data?.error || 'Could not load the dataset system prompt.');
      }
    };

    void loadDatasetSystemPrompt();
    return () => {
      cancelled = true;
    };
  }, [datasetName]);

  const startSecureCaptionJob = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!workerID || !datasetPath || !model.trim()) return;

    setSubmitStatus('starting');
    setSubmitError('');
    setStartedJobId(null);
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(LAST_MODEL_KEY, model.trim());
      }
      const savedSystemPrompt = await saveDatasetSystemPrompt(datasetName, systemPrompt);
      const jobName = `secure_remote_caption_${uuidv4()}`;
      const jobConfig = {
        job: 'extension',
        config: {
          name: 'Secure Remote Ollama Caption',
          process: [
            {
              type: 'SecureRemoteOllamaCaptioner',
              sqlite_db_path: './aitk_db.db',
              device: 'cpu',
              caption: {
                model_name_or_path: model.trim(),
                device: 'cpu',
                dtype: 'bf16',
                quantize: false,
                qtype: 'float8',
                low_vram: false,
                extensions: IMAGE_EXTENSIONS,
                path_to_caption: datasetPath,
                recaption,
                caption_prompt: captionPrompt,
                max_res: maxRes,
                max_new_tokens: maxNewTokens,
                remote_worker_id: workerID,
                system_prompt: savedSystemPrompt,
              },
            },
          ],
        },
      };
      const saved = await apiClient
        .post('/api/jobs', {
          name: jobName,
          worker_id: 'local',
          gpu_ids: '0',
          job_config: jobConfig,
          job_type: 'caption',
          job_ref: datasetPath,
        })
        .then(res => res.data);
      const rememberedKey =
        selectedDataset?.encrypted && datasetPath
          ? getRememberedEncryptedDatasetKey(datasetPath) || getRememberedEncryptedDatasetKey(datasetName)
          : null;
      await startJob(
        saved.id,
        rememberedKey ? [{ datasetPath, keyB64: rememberedKey }] : undefined,
      );
      await startQueue('0');
      setStartedJobId(saved.id);
      setSubmitStatus('success');
    } catch (error: any) {
      setSubmitStatus('error');
      setSubmitError(error?.response?.data?.error || error?.message || 'Failed to start secure caption job.');
    }
  };

  const isLoading = workerStatus === 'loading' || datasetStatus === 'loading' || !isSettingsLoaded;
  const canStart =
    !!workerID &&
    !!datasetPath &&
    !!model.trim() &&
    submitStatus !== 'starting' &&
    systemPromptStatus !== 'loading' &&
    systemPromptStatus !== 'saving' &&
    !isLoading;

  return (
    <section className="rounded-xl border border-gray-800 bg-gray-900 p-4">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-gray-100">Secure Remote Captioning</h2>
          <p className="text-sm text-gray-500">Image captions through a remote worker's local Ollama.</p>
        </div>
        {startedJobId && (
          <Link href={`/jobs/${startedJobId}`} className="rounded-md bg-gray-800 px-3 py-1 text-sm text-gray-100">
            View Job
          </Link>
        )}
      </div>

      <form onSubmit={startSecureCaptionJob} className={`space-y-3 ${isLoading ? 'pointer-events-none opacity-60' : ''}`}>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <SelectInput
            label="Remote Worker"
            value={workerID}
            onChange={value => setWorkerID(value)}
            options={workerOptions}
            disabled={workerOptions.length === 0}
          />
          <SelectInput
            label="Dataset"
            value={datasetName}
            onChange={value => setDatasetName(value)}
            options={datasetOptions}
            disabled={datasetOptions.length === 0}
          />
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-[1fr_auto] md:items-end">
          <CreatableSelectInput
            label="Ollama Model"
            value={model}
            onChange={value => setModel(value)}
            options={modelOptions}
            placeholder="llava:latest"
            required
          />
          <button
            type="button"
            onClick={() => void loadModels()}
            disabled={!workerID || modelStatus === 'loading'}
            className="mb-[1px] inline-flex h-8 items-center justify-center gap-1.5 rounded-md bg-gray-800 px-3 text-sm text-gray-200 hover:bg-gray-700 disabled:opacity-50"
          >
            <RotateCcw className={`h-4 w-4 ${modelStatus === 'loading' ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {modelStatus === 'error' && <div className="text-sm text-red-400">{modelError}</div>}
        {modelStatus === 'success' && (
          <div className="text-xs text-gray-500">
            {models.length} remote Ollama model{models.length === 1 ? '' : 's'} found.
          </div>
        )}

        <TextAreaInput
          label="Caption Prompt"
          value={captionPrompt}
          onChange={setCaptionPrompt}
          rows={4}
          required
        />

        <div className="grid grid-cols-1 gap-3 md:grid-cols-[1fr_auto] md:items-end">
          <TextAreaInput
            label="Dataset System Prompt"
            value={systemPrompt}
            onChange={value => {
              setSystemPrompt(value);
              if (systemPromptStatus === 'success') setSystemPromptStatus('idle');
            }}
            rows={3}
            disabled={!datasetName || systemPromptStatus === 'loading' || systemPromptStatus === 'saving'}
          />
          <button
            type="button"
            onClick={() => void saveDatasetSystemPrompt()}
            disabled={!datasetName || systemPromptStatus === 'loading' || systemPromptStatus === 'saving'}
            className="mb-[1px] inline-flex h-8 items-center justify-center gap-1.5 rounded-md bg-gray-800 px-3 text-sm text-gray-200 hover:bg-gray-700 disabled:opacity-50"
          >
            {systemPromptStatus === 'saving' ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            Save
          </button>
        </div>
        {systemPromptStatus === 'error' && <div className="text-sm text-red-400">{systemPromptError}</div>}
        {systemPromptStatus === 'success' && <div className="text-sm text-green-400">Dataset system prompt saved.</div>}

        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <NumberInput label="Max Resolution" value={maxRes} onChange={value => setMaxRes(value || 768)} min={128} />
          <NumberInput
            label="Max Tokens"
            value={maxNewTokens}
            onChange={value => setMaxNewTokens(value || 180)}
            min={1}
          />
          <div className="pt-7">
            <Checkbox label="Recaption existing files" checked={recaption} onChange={setRecaption} />
          </div>
        </div>

        {selectedDataset?.encrypted && (
          <div className="rounded-md border border-amber-900 bg-amber-950/30 px-3 py-2 text-sm text-amber-200">
            Encrypted datasets must be unlocked in this browser before starting, unless the password can be entered at start.
          </div>
        )}

        <button
          type="submit"
          disabled={!canStart}
          className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-blue-700 px-4 py-2 text-sm text-white hover:bg-blue-600 disabled:opacity-50"
        >
          {submitStatus === 'starting' && <Loader2 className="h-4 w-4 animate-spin" />}
          Start Secure Caption Job
        </button>
        {submitStatus === 'success' && <div className="text-sm text-green-400">Secure caption job started.</div>}
        {submitStatus === 'error' && <div className="text-sm text-red-400">{submitError}</div>}
      </form>
    </section>
  );
}
