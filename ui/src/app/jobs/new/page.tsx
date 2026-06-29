'use client';

import { useEffect, useRef, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { defaultJobConfig, defaultDatasetConfig, migrateJobConfig } from './jobConfig';
import { jobTypeOptions, modelArchs } from './options';
import type { JobConfig } from '@/types';
import { objectCopy } from '@/utils/basic';
import { useNestedState, setNestedValue } from '@/utils/hooks';
import { SelectInput } from '@/components/formInputs';
import useSettings from '@/hooks/useSettings';
import useGPUInfo from '@/hooks/useGPUInfo';
import useDatasetList from '@/hooks/useDatasetList';
import useWorkers from '@/hooks/useWorkers';
import YAML from 'yaml';
import path from 'path';
import { TopBar, MainContent } from '@/components/layout';
import { Button } from '@headlessui/react';
import { ChevronLeft, SlidersHorizontal, Sparkles, TerminalSquare, X } from 'lucide-react';
import SimpleJob from './SimpleJob';
import AdvancedConfigEditor from '@/components/AdvancedConfigEditor';
import ErrorBoundary from '@/components/ErrorBoundary';
import { apiClient } from '@/utils/api';
import type { SelectOption } from '@/types';
import { PageNotice } from '@/components/OperatorPrimitives';
import {
  getRememberedEncryptedDatasetKey,
  rememberEncryptedDatasetKey,
} from '@/utils/encryptedDatasets';
import { normalizeDetectedCaptionExt } from '@/utils/jobDatasetDefaults';
import {
  makeRemoteDatasetRef,
  parseRemoteDatasetRef,
  remoteDatasetRememberKey,
  shouldImportRemoteDatasetForWorker,
} from '@/utils/remoteDatasetRefs';
import { AUTHENLORA_BUILTIN_CODEC_BITS } from '@/utils/authenloraCodecs';

const isDev = process.env.NODE_ENV === 'development';

type ValidationMessage = {
  level: 'error' | 'warning';
  message: string;
};

export default function TrainingForm({
  projectIDOverride = null,
}: {
  projectIDOverride?: string | null;
} = {}) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const runId = searchParams.get('id');
  const cloneId = searchParams.get('cloneId');
  const projectID = projectIDOverride ?? searchParams.get('project_id');
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const [workerID, setWorkerID] = useState('local');
  const { settings, isSettingsLoaded } = useSettings();
  const { workers, status: workerStatus } = useWorkers();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo(null, null, workerID);
  const { datasets, status: datasetFetchStatus } = useDatasetList({ includeRemote: !projectID, projectID });
  const [datasetOptions, setDatasetOptions] = useState<
    Array<
      SelectOption & {
        encrypted: boolean;
        name: string;
        source: 'local' | 'remote';
        worker_id: string;
        ref: string;
        detectedCaptionExt?: string | null;
      }
    >
  >([]);
  const [showAdvancedView, setShowAdvancedView] = useState(false);
  const [rawConfigOpen, setRawConfigOpen] = useState(false);

  const [jobConfig, setJobConfig] = useNestedState<JobConfig>(objectCopy(migrateJobConfig(defaultJobConfig)));
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [validationMessages, setValidationMessages] = useState<ValidationMessage[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImportConfig = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = reader.result as string;
        let parsed: any;
        if (file.name.endsWith('.json') || file.name.endsWith('.jsonc')) {
          parsed = JSON.parse(text.replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, ''));
        } else {
          parsed = YAML.parse(text);
        }

        // Set required fields (same pattern as AdvancedJob.handleChange)
        try {
          parsed.config.process[0].sqlite_db_path = './aitk_db.db';
          parsed.config.process[0].training_folder = settings.TRAINING_FOLDER;
          parsed.config.process[0].device = 'cuda';
          parsed.config.process[0].performance_log_every = 10;
        } catch (err) {
          console.warn('Could not set required fields on imported config:', err);
        }

        migrateJobConfig(parsed);
        setJobConfig(parsed);
      } catch (err) {
        console.error('Failed to parse config file:', err);
        alert('Failed to parse config file. Please check the file format.');
      }
    };
    reader.readAsText(file);

    // Reset so the same file can be re-imported
    e.target.value = '';
  };

  useEffect(() => {
    if (!isSettingsLoaded) return;
    if (datasetFetchStatus !== 'success') return;

    const datasetOptions = datasets.map(dataset => {
      const source = (dataset.source || 'local') as 'local' | 'remote';
      const workerID = dataset.worker_id || 'local';
      const ref = source === 'remote' ? dataset.ref || makeRemoteDatasetRef(workerID, dataset.name) : dataset.ref || '';
      return {
        value: source === 'remote' ? ref : dataset.path || path.join(settings.DATASETS_FOLDER, dataset.name),
        label:
          source === 'remote'
            ? `${dataset.name}${dataset.encrypted ? ' (encrypted)' : ''} - ${dataset.worker_name || workerID}`
            : dataset.encrypted
              ? `${dataset.name} (encrypted)`
              : dataset.name,
        encrypted: dataset.encrypted,
        name: dataset.name,
        source,
        worker_id: workerID,
        ref,
        detectedCaptionExt: dataset.detectedCaptionExt ?? null,
      };
    });
    setDatasetOptions(datasetOptions);

    if (datasetOptions.length > 0) {
      const defaultDatasetPath = defaultDatasetConfig.folder_path;
      // Use functional updater so we check the *current* state, not a stale closure
      setJobConfig((prev: JobConfig) => {
        let updated = prev;
        for (let i = 0; i < prev.config.process[0].datasets.length; i++) {
          if (prev.config.process[0].datasets[i].folder_path === defaultDatasetPath) {
            updated = setNestedValue(updated, datasetOptions[0].value, `config.process[0].datasets[${i}].folder_path`);
            const detectedCaptionExt = normalizeDetectedCaptionExt(datasetOptions[0].detectedCaptionExt);
            if (detectedCaptionExt) {
              updated = setNestedValue(updated, detectedCaptionExt, `config.process[0].datasets[${i}].caption_ext`);
            }
          }
        }
        return updated;
      });
    }
  }, [datasets, settings, isSettingsLoaded, datasetFetchStatus]);

  // clone existing job
  useEffect(() => {
    if (cloneId) {
      apiClient
        .get(`/api/jobs?id=${cloneId}`)
        .then(res => res.data)
        .then(data => {
          console.log('Clone Training:', data);
          setGpuIDs(data.gpu_ids);
          setWorkerID(data.worker_id || 'local');
          const newJobConfig = migrateJobConfig(JSON.parse(data.job_config));
          newJobConfig.config.name = `${newJobConfig.config.name}_copy`;
          setJobConfig(newJobConfig);
        })
        .catch(error => console.error('Error fetching training:', error));
    }
  }, [cloneId]);

  useEffect(() => {
    if (runId) {
      apiClient
        .get(`/api/jobs?id=${runId}`)
        .then(res => res.data)
        .then(data => {
          console.log('Training:', data);
          setGpuIDs(data.gpu_ids);
          setWorkerID(data.worker_id || 'local');
          setJobConfig(migrateJobConfig(JSON.parse(data.job_config)));
        })
        .catch(error => console.error('Error fetching training:', error));
    }
  }, [runId]);

  useEffect(() => {
    if (isGPUInfoLoaded) {
      if (gpuIDs === null) {
        setGpuIDs(gpuList.length > 0 ? `${gpuList[0].index}` : '0');
      }
    }
  }, [gpuList, isGPUInfoLoaded]);

  useEffect(() => {
    if (isSettingsLoaded) {
      setJobConfig(settings.TRAINING_FOLDER, 'config.process[0].training_folder');
    }
  }, [settings, isSettingsLoaded]);

  useEffect(() => {
    if (!rawConfigOpen) return;

    const previousOverflow = document.body.style.overflow;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setRawConfigOpen(false);
      }
    };

    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [rawConfigOpen]);

  const copyRememberedRemoteDatasetKey = (remoteRef: string, imported: { name: string; path: string }) => {
    const parsed = parseRemoteDatasetRef(remoteRef);
    if (!parsed) return;
    const remembered =
      getRememberedEncryptedDatasetKey(remoteRef) ||
      getRememberedEncryptedDatasetKey(remoteDatasetRememberKey(parsed.workerID, parsed.datasetName)) ||
      getRememberedEncryptedDatasetKey(parsed.datasetName);
    if (!remembered) return;
    rememberEncryptedDatasetKey(imported.path, remembered);
    rememberEncryptedDatasetKey(imported.name, remembered);
  };

  const importRemoteDatasetForJob = async (
    remoteRef: string,
    cache: Map<string, Promise<{ name: string; path: string }>>,
  ) => {
    const existing = cache.get(remoteRef);
    if (existing) return existing;
    const parsed = parseRemoteDatasetRef(remoteRef);
    if (!parsed) throw new Error('Invalid remote dataset reference');

    const promise = apiClient
      .post('/api/datasets/import-remote', {
        worker_id: parsed.workerID,
        datasetName: parsed.datasetName,
        project_id: projectID || undefined,
      })
      .then(res => {
        const importedName = res.data?.dataset?.name;
        const importedPath = res.data?.path || (importedName ? path.join(settings.DATASETS_FOLDER, importedName) : null);
        if (!importedName || !importedPath) {
          throw new Error('Remote dataset import did not return a local dataset path');
        }
        const imported = { name: importedName, path: importedPath };
        copyRememberedRemoteDatasetKey(remoteRef, imported);
        return imported;
      });
    cache.set(remoteRef, promise);
    return promise;
  };

  const importRemoteDatasetsForJobConfig = async (rawConfig: JobConfig, targetWorkerID: string) => {
    const nextConfig = objectCopy(rawConfig) as JobConfig;
    const importCache = new Map<string, Promise<{ name: string; path: string }>>();
    const datasetPathFields = [
      'folder_path',
      'dataset_path',
      'control_path',
      'control_path_1',
      'control_path_2',
      'control_path_3',
      'mask_path',
      'unconditional_path',
      'inpaint_path',
      'clip_image_path',
    ];

    for (const processConfig of nextConfig.config.process || []) {
      for (const dataset of processConfig.datasets || []) {
        for (const field of datasetPathFields) {
          const current = (dataset as any)[field];
          if (
            typeof current === 'string' &&
            shouldImportRemoteDatasetForWorker(
              current,
              targetWorkerID,
              datasetOptions.find(option => option.value === current)?.encrypted === true,
            )
          ) {
            const imported = await importRemoteDatasetForJob(current, importCache);
            (dataset as any)[field] = imported.path;
          } else if (Array.isArray(current)) {
            const nextValues = [];
            for (const value of current) {
              if (
                typeof value === 'string' &&
                shouldImportRemoteDatasetForWorker(
                  value,
                  targetWorkerID,
                  datasetOptions.find(option => option.value === value)?.encrypted === true,
                )
              ) {
                const imported = await importRemoteDatasetForJob(value, importCache);
                nextValues.push(imported.path);
              } else {
                nextValues.push(value);
              }
            }
            (dataset as any)[field] = nextValues;
          }
        }
      }
    }

    return nextConfig;
  };

  const applyComfyAutoInstallSetting = (rawConfig: JobConfig): JobConfig => {
    if (settings.COMFY_AUTO_INSTALL !== 'true') return rawConfig;

    const nextConfig = objectCopy(rawConfig) as JobConfig;
    for (const processConfig of nextConfig.config?.process ?? []) {
      const processSections = processConfig as Record<string, any>;
      for (const key of ['sample', 'first_sample', 'generate'] as const) {
        const generationConfig = processSections[key];
        if (generationConfig?.backend === 'comfy' && generationConfig?.comfy?.mode === 'managed') {
          generationConfig.comfy.managed_install = true;
        }
      }
    }
    return nextConfig;
  };

  const validateJobBeforeSave = (rawConfig: JobConfig): ValidationMessage[] => {
    const messages: ValidationMessage[] = [];
    const name = rawConfig.config?.name?.trim() || '';
    const processConfig = rawConfig.config?.process?.[0];
    const trainConfig = processConfig?.train;
    const modelConfig = processConfig?.model;
    const datasetsConfig = processConfig?.datasets || [];
    const sampleConfig = processConfig?.sample;

    if (!name) {
      messages.push({ level: 'error', message: 'Training name is required.' });
    }
    if (name === '.' || name.includes('..') || /[\\/]/.test(name)) {
      messages.push({ level: 'error', message: 'Training name cannot contain path separators or "..".' });
    }
    if (!workerID) {
      messages.push({ level: 'error', message: 'Select a worker before creating the job.' });
    }
    if (!gpuIDs) {
      messages.push({ level: 'error', message: 'Select a GPU before creating the job.' });
    }
    if (!processConfig) {
      messages.push({ level: 'error', message: 'Job config must include one process.' });
      return messages;
    }
    if (!modelConfig?.name_or_path?.trim()) {
      messages.push({ level: 'error', message: 'Select or enter a base model path.' });
    }
    const baseLoraPath = modelConfig?.base_lora_path?.trim();
    if (baseLoraPath) {
      if (modelConfig?.inference_lora_path?.trim()) {
        messages.push({
          level: 'error',
          message: 'Base LoRA Path cannot be used with sample-time Inference LoRA Path.',
        });
      }
      const baseLoraName = baseLoraPath.split(/[\\/]/).pop() || baseLoraPath;
      if (/\.[^./\\]+$/.test(baseLoraName) && !baseLoraName.toLowerCase().endsWith('.safetensors')) {
        messages.push({ level: 'error', message: 'Base LoRA Path must be a .safetensors adapter.' });
      }
      const baseLoraStrength = Number(modelConfig?.base_lora_strength ?? 1.0);
      if (!Number.isFinite(baseLoraStrength)) {
        messages.push({ level: 'error', message: 'Base LoRA Strength must be a finite number.' });
      }
    }
    if (!datasetsConfig.length) {
      messages.push({ level: 'error', message: 'Add at least one dataset.' });
    }

    const unresolvedDatasets = datasetsConfig.filter(dataset => {
      return !dataset.folder_path || dataset.folder_path === defaultDatasetConfig.folder_path;
    });
    if (unresolvedDatasets.length > 0) {
      messages.push({ level: 'error', message: 'Select a target dataset for every dataset entry.' });
    }

    datasetsConfig.forEach((dataset, index) => {
      const datasetOption = datasetOptions.find(option => option.value === dataset.folder_path);
      if (datasetOption?.encrypted) {
        const remembered =
          getRememberedEncryptedDatasetKey(dataset.folder_path) ||
          (datasetOption.name ? getRememberedEncryptedDatasetKey(datasetOption.name) : null);
        if (!remembered && !parseRemoteDatasetRef(dataset.folder_path)) {
          messages.push({
            level: 'warning',
            message: `Dataset ${index + 1} is encrypted. Unlock it before starting or resuming this job.`,
          });
        }
      }
      if (!dataset.is_reg && (!dataset.resolution || dataset.resolution.length === 0)) {
        messages.push({ level: 'error', message: `Dataset ${index + 1} needs at least one resolution.` });
      }
      if ((dataset.num_repeats ?? 1) < 1) {
        messages.push({ level: 'error', message: `Dataset ${index + 1} repeats must be at least 1.` });
      }
    });

    if (!trainConfig?.auto_train && (!trainConfig?.steps || trainConfig.steps < 1)) {
      messages.push({ level: 'error', message: 'Training steps must be at least 1.' });
    }
    if (!trainConfig?.batch_size || trainConfig.batch_size < 1) {
      messages.push({ level: 'error', message: 'Batch size must be at least 1.' });
    }
    if (!trainConfig?.gradient_accumulation || trainConfig.gradient_accumulation < 1) {
      messages.push({ level: 'error', message: 'Gradient accumulation must be at least 1.' });
    }
    if (trainConfig?.lr == null || trainConfig.lr < 0) {
      messages.push({ level: 'error', message: 'Learning rate must be zero or greater.' });
    }

    const watermarkConfig = processConfig?.watermark;
    if (watermarkConfig?.enabled) {
      const archName = `${modelConfig?.arch ?? ''}`.split(':')[0];
      const arch = modelArchs.find(option => option.name === archName);
      const networkType = `${processConfig?.network?.type ?? ''}`.toLowerCase();
      if (arch?.group === 'audio' || arch?.group === 'video') {
        messages.push({ level: 'error', message: 'AuthenLoRA watermarking requires an image LoRA job.' });
      }
      if (!processConfig?.network) {
        messages.push({ level: 'error', message: 'AuthenLoRA watermarking requires a LoRA network.' });
      }
      if (!['lora', 'locon', 'lycoris', 'lokr'].includes(networkType)) {
        messages.push({ level: 'error', message: 'AuthenLoRA watermarking supports LoRA, LoCon, LyCORIS, and LoKr networks.' });
      }
      if (trainConfig?.loss_type === 'mean_flow' || trainConfig?.do_guidance_loss) {
        messages.push({ level: 'error', message: 'AuthenLoRA watermarking currently supports the standard image LoRA loss path.' });
      }
      if (!watermarkConfig.codec_path?.trim()) {
        messages.push({ level: 'error', message: 'AuthenLoRA watermarking requires a local codec path.' });
      }
      const builtinMsgBits = AUTHENLORA_BUILTIN_CODEC_BITS[watermarkConfig.codec_path?.trim() || ''];
      if (builtinMsgBits && watermarkConfig.msg_bits !== builtinMsgBits) {
        messages.push({ level: 'error', message: `AuthenLoRA ${builtinMsgBits}-bit built-in codec requires Message bits to be ${builtinMsgBits}.` });
      }
      if (!watermarkConfig.msg_bits || watermarkConfig.msg_bits < 1) {
        messages.push({ level: 'error', message: 'AuthenLoRA message bits must be greater than 0.' });
      }
      if (!watermarkConfig.mapper_rank || watermarkConfig.mapper_rank < 1) {
        messages.push({ level: 'error', message: 'AuthenLoRA mapper rank must be greater than 0.' });
      }
      if (watermarkConfig.mapper_lr < 0) {
        messages.push({ level: 'error', message: 'AuthenLoRA mapper learning rate must be zero or greater.' });
      }
      if (watermarkConfig.watermark_loss_weight < 0 || watermarkConfig.style_loss_weight < 0) {
        messages.push({ level: 'error', message: 'AuthenLoRA loss weights must be zero or greater.' });
      }
      if (watermarkConfig.zero_message_probability < 0 || watermarkConfig.zero_message_probability > 1) {
        messages.push({ level: 'error', message: 'AuthenLoRA zero message chance must be between 0 and 1.' });
      }
      if (watermarkConfig.verify_every < 0) {
        messages.push({ level: 'error', message: 'AuthenLoRA verify every must be zero or greater.' });
      }
      const secret = watermarkConfig.secret?.trim();
      if (secret && (secret.length !== watermarkConfig.msg_bits || /[^01]/.test(secret))) {
        messages.push({ level: 'error', message: 'AuthenLoRA secret bits must be binary and match Message bits.' });
      }
    }

    const samplingDisabled = trainConfig?.disable_sampling === true;
    const samplePrompts = sampleConfig?.samples || [];
    if (!samplingDisabled && samplePrompts.length === 0) {
      messages.push({ level: 'warning', message: 'No sample prompts are configured.' });
    }
    if (!samplingDisabled && samplePrompts.some(sample => !sample.prompt?.trim())) {
      messages.push({ level: 'warning', message: 'One or more sample prompts are blank.' });
    }

    return messages;
  };

  const saveJob = async () => {
    if (status === 'saving') return;
    const jobConfigWithSettings = applyComfyAutoInstallSetting(jobConfig);
    const validation = validateJobBeforeSave(jobConfigWithSettings);
    setValidationMessages(validation);
    if (validation.some(message => message.level === 'error')) {
      setStatus('idle');
      return;
    }
    setStatus('saving');

    try {
      const preparedJobConfig = await importRemoteDatasetsForJobConfig(jobConfigWithSettings, workerID);
      setJobConfig(preparedJobConfig);
      const res = await apiClient.post('/api/jobs', {
        id: runId,
        name: preparedJobConfig.config.name,
        worker_id: workerID,
        gpu_ids: gpuIDs,
        job_config: preparedJobConfig,
        project_id: projectID || undefined,
      });
      setStatus('success');
      setValidationMessages([]);
      if (projectID) {
        router.push(`/projects/${encodeURIComponent(projectID)}/runs/${encodeURIComponent(res.data.id || runId)}`);
      } else if (runId) {
        router.push(`/jobs/${runId}`);
      } else {
        router.push(`/jobs/${res.data.id}`);
      }
    } catch (error: any) {
      setStatus('error');
      if (error.response?.status === 409) {
        setValidationMessages([
          {
            level: 'error',
            message: error?.response?.data?.error || 'Training name already exists in this workspace. Choose a different name.',
          },
        ]);
      } else {
        setValidationMessages([
          { level: 'error', message: error?.response?.data?.error || 'Failed to save job. Please try again.' },
        ]);
      }
      console.log('Error saving training:', error);
    } finally {
      setTimeout(() => {
        setStatus('idle');
      }, 2000);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    saveJob();
  };

  const validationErrors = validationMessages.filter(message => message.level === 'error');
  const validationWarnings = validationMessages.filter(message => message.level === 'warning');
  const workerOptions = [
    { value: 'local', label: 'Local worker' },
    ...workers.filter(worker => worker.enabled).map(worker => ({ value: worker.id, label: worker.name })),
  ];
  const trainerValue = `${jobConfig?.config.process[0].type}`;
  const trainerLabel = jobTypeOptions.find(option => option.value === trainerValue)?.label || 'LoRA Trainer';
  const workerLabel = workerOptions.find(option => option.value === workerID)?.label || 'Local worker';
  const handleJobTypeChange = (value: string) => {
    const currentOption = jobTypeOptions.find(option => option.value === jobConfig?.config.process[0].type);
    if (currentOption && currentOption.onDeactivate) {
      setJobConfig(currentOption.onDeactivate(objectCopy(jobConfig)));
    }
    const option = jobTypeOptions.find(option => option.value === value);
    if (option) {
      if (option.onActivate) {
        setJobConfig(option.onActivate(objectCopy(jobConfig)));
      }
      jobTypeOptions.forEach(opt => {
        if (opt.value !== option.value && opt.onDeactivate) {
          setJobConfig(opt.onDeactivate(objectCopy(jobConfig)));
        }
      });
    }
    setJobConfig(value, 'config.process[0].type');
  };
  const transformAdvancedConfig = (parsed: any) => {
    try {
      parsed.config.process[0].sqlite_db_path = './aitk_db.db';
      parsed.config.process[0].training_folder = settings.TRAINING_FOLDER;
      parsed.config.process[0].device = 'cuda';
      parsed.config.process[0].performance_log_every = 10;
    } catch (e) {
      console.warn(e);
    }
    return migrateJobConfig(parsed);
  };
  const renderAdvancedConfigEditor = () => (
    <AdvancedConfigEditor config={jobConfig} setConfig={setJobConfig} transformOnParse={transformAdvancedConfig} />
  );
  const validationSummary =
    validationMessages.length > 0 ? (
      <PageNotice
        tone={validationErrors.length > 0 ? 'danger' : 'warning'}
        title={validationErrors.length > 0 ? 'Fix these issues before saving' : 'Review before saving'}
      >
        <ul className="list-disc space-y-1 pl-4">
          {validationErrors.map((message, index) => (
            <li key={`error-${index}`}>{message.message}</li>
          ))}
          {validationWarnings.map((message, index) => (
            <li key={`warning-${index}`}>{message.message}</li>
          ))}
        </ul>
      </PageNotice>
    ) : null;

  return (
    <>
      <TopBar className="h-16 !overflow-hidden border-gray-900 bg-gray-950 px-4">
        <div className="flex min-w-0 items-center gap-3">
          <Button className="operator-icon-button h-9 w-9 flex-none" onClick={() => history.back()} title="Back">
            <ChevronLeft className="h-5 w-5" />
          </Button>
          <h1 className="truncate text-lg font-semibold text-gray-100">
            {runId ? 'Edit Training Job' : projectID ? 'New Project Training Job' : 'New Training Job'}
          </h1>
        </div>
        <div className="flex-1"></div>
        <div className="hidden min-w-40 md:block">
          <SelectInput
            value={workerID}
            onChange={value => {
              setWorkerID(value);
              setGpuIDs(null);
            }}
            options={workerOptions}
          />
        </div>
        <div className="hidden min-w-44 md:block">
          <SelectInput value={trainerValue} onChange={handleJobTypeChange} options={jobTypeOptions} />
        </div>
        {showAdvancedView && (
          <div className="hidden min-w-32 lg:block">
            <SelectInput
              value={`${gpuIDs}`}
              onChange={value => setGpuIDs(value)}
              options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
            />
          </div>
        )}
        {showAdvancedView && (
          <Button className="operator-button hidden py-1 lg:inline-flex" onClick={handleImportConfig}>
            Import Config
          </Button>
        )}
        <div className="flex-none">
          <Button
            className="operator-button h-9 border-gray-700 bg-gray-900 px-3 py-1 text-gray-100"
            onClick={() => {
              setRawConfigOpen(false);
              setShowAdvancedView(!showAdvancedView);
            }}
            title={showAdvancedView ? 'Show Simple' : 'Show Advanced'}
          >
            <SlidersHorizontal className="h-4 w-4" />
            <span className="hidden sm:inline">{showAdvancedView ? 'Simple' : 'Advanced'}</span>
          </Button>
        </div>
        <div className="flex-none">
          <Button
            className="operator-button h-9 border-emerald-800 bg-emerald-600/90 px-3 py-1 font-semibold text-gray-950 hover:bg-emerald-500 sm:px-5"
            onClick={() => saveJob()}
            disabled={status === 'saving'}
          >
            <Sparkles className="h-4 w-4" />
            <span className="hidden sm:inline">{status === 'saving' ? 'Saving...' : runId ? 'Update Job' : 'Create Job'}</span>
          </Button>
        </div>
      </TopBar>

      <input
        ref={fileInputRef}
        type="file"
        accept=".yaml,.yml,.json,.jsonc"
        style={{ display: 'none' }}
        onChange={handleFileSelected}
      />

      {showAdvancedView ? (
        <div className="absolute top-0 left-0 h-full w-full overflow-auto pt-16">
          {validationSummary && <div className="px-3 pt-3 sm:px-4">{validationSummary}</div>}
          {renderAdvancedConfigEditor()}
        </div>
      ) : (
        <MainContent className="bg-gray-950 px-0 pt-16 sm:px-0">
          <ErrorBoundary
            fallback={
              <div className="flex h-64 items-center justify-center border border-red-300 bg-red-100 text-lg font-medium text-red-600 dark:border-red-700 dark:bg-red-900/20 dark:text-red-400">
                Advanced job detected. Please switch to advanced view to continue.
              </div>
            }
          >
            <SimpleJob
              jobConfig={jobConfig}
              setJobConfig={setJobConfig}
              status={status}
              handleSubmit={handleSubmit}
              runId={runId}
              gpuIDs={gpuIDs}
              setGpuIDs={setGpuIDs}
              gpuList={gpuList}
              datasetOptions={datasetOptions}
              validationMessages={validationMessages}
              workerLabel={workerLabel}
              trainerLabel={trainerLabel}
              onOpenAdvanced={() => setShowAdvancedView(true)}
              onOpenRawConfig={() => setRawConfigOpen(true)}
              projectID={projectID}
              isLoading={
                !isSettingsLoaded || !isGPUInfoLoaded || workerStatus === 'loading' || datasetFetchStatus !== 'success'
              }
              comfyAutoInstall={settings.COMFY_AUTO_INSTALL === 'true'}
            />
          </ErrorBoundary>

          <div className="pt-20"></div>
        </MainContent>
      )}

      {rawConfigOpen && !showAdvancedView && (
        <div
          className="fixed inset-0 z-50 flex bg-black/65 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-labelledby="raw-config-title"
        >
          <button
            type="button"
            className="absolute inset-0 cursor-default"
            onClick={() => setRawConfigOpen(false)}
            aria-label="Close raw config"
          />
          <section
            id="raw-config-drawer"
            className="relative ml-auto flex h-full w-full flex-col border-l border-gray-800 bg-gray-950 shadow-2xl sm:w-[min(920px,calc(100vw-72px))]"
          >
            <header className="flex h-16 flex-none items-center gap-3 border-b border-gray-900 px-4">
              <div className="flex h-9 w-9 items-center justify-center border border-gray-800 bg-gray-900 text-cyan-200">
                <TerminalSquare className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <h2 id="raw-config-title" className="truncate text-base font-semibold text-gray-100">
                  Raw config
                </h2>
                <p className="truncate text-xs text-gray-500">
                  Editing this YAML updates the current job draft. Close returns to the same workspace position.
                </p>
              </div>
              <Button className="operator-button hidden h-9 px-3 py-1 sm:inline-flex" onClick={() => setRawConfigOpen(false)}>
                Close drawer
              </Button>
              <Button
                className="operator-icon-button h-9 w-9 flex-none"
                onClick={() => setRawConfigOpen(false)}
                title="Close raw config"
              >
                <X className="h-4 w-4" />
              </Button>
            </header>
            <div className="min-h-0 flex-1">{renderAdvancedConfigEditor()}</div>
          </section>
        </div>
      )}
    </>
  );
}
