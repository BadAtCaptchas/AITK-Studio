'use client';

import { useEffect, useRef, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { defaultJobConfig, defaultDatasetConfig, migrateJobConfig } from './jobConfig';
import { jobTypeOptions } from './options';
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
import { FaChevronLeft } from 'react-icons/fa';
import SimpleJob from './SimpleJob';
import AdvancedConfigEditor from '@/components/AdvancedConfigEditor';
import ErrorBoundary from '@/components/ErrorBoundary';
import { apiClient } from '@/utils/api';
import { TrainingAdvisorPanel } from '@/components/TrainingAdvisorPanel';
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

const isDev = process.env.NODE_ENV === 'development';

type ValidationMessage = {
  level: 'error' | 'warning';
  message: string;
};

export default function TrainingForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const runId = searchParams.get('id');
  const cloneId = searchParams.get('cloneId');
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const [workerID, setWorkerID] = useState('local');
  const { settings, isSettingsLoaded } = useSettings();
  const { workers, status: workerStatus } = useWorkers();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo(null, null, workerID);
  const { datasets, status: datasetFetchStatus } = useDatasetList({ includeRemote: true });
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
        value: source === 'remote' ? ref : path.join(settings.DATASETS_FOLDER, dataset.name),
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
      });
      setStatus('success');
      setValidationMessages([]);
      if (runId) {
        router.push(`/jobs/${runId}`);
      } else {
        router.push(`/jobs/${res.data.id}`);
      }
    } catch (error: any) {
      setStatus('error');
      if (error.response?.status === 409) {
        setValidationMessages([{ level: 'error', message: 'Training name already exists. Choose a different name.' }]);
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
      <TopBar>
        <div>
          <Button className="operator-icon-button" onClick={() => history.back()} title="Back">
            <FaChevronLeft />
          </Button>
        </div>
        <div className="min-w-0">
          <h1 className="truncate text-base font-semibold">{runId ? 'Edit Training Job' : 'New Training Job'}</h1>
        </div>
        <div className="flex-1"></div>
        {showAdvancedView && (
          <>
            <div className="min-w-40">
              <SelectInput
                value={workerID}
                onChange={value => {
                  setWorkerID(value);
                  setGpuIDs(null);
                }}
                options={[
                  { value: 'local', label: 'Local worker' },
                  ...workers.filter(worker => worker.enabled).map(worker => ({ value: worker.id, label: worker.name })),
                ]}
              />
            </div>
            <div className="mx-1 h-6 border-r border-gray-800"></div>
            <div className="min-w-32">
              <SelectInput
                value={`${gpuIDs}`}
                onChange={value => setGpuIDs(value)}
                options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              />
            </div>
            <div className="mx-1 h-6 border-r border-gray-800"></div>
            <div>
              <Button className="operator-button py-1" onClick={handleImportConfig}>
                Import Config
              </Button>
            </div>
            <div className="mx-1 h-6 border-r border-gray-800"></div>
          </>
        )}
        {!showAdvancedView && (
          <>
            <div className="min-w-40">
              <SelectInput
                value={workerID}
                onChange={value => {
                  setWorkerID(value);
                  setGpuIDs(null);
                }}
                options={[
                  { value: 'local', label: 'Local worker' },
                  ...workers.filter(worker => worker.enabled).map(worker => ({ value: worker.id, label: worker.name })),
                ]}
              />
            </div>
            <div className="mx-1 h-6 border-r border-gray-800"></div>
            <div className="min-w-44">
              <SelectInput
                value={`${jobConfig?.config.process[0].type}`}
                onChange={value => {
                  // undo current job type changes
                  const currentOption = jobTypeOptions.find(
                    option => option.value === jobConfig?.config.process[0].type,
                  );
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
                }}
                options={jobTypeOptions}
              />
            </div>
            <div className="mx-1 h-6 border-r border-gray-800"></div>
          </>
        )}

        <div className="pr-2">
          <Button
            className="operator-button py-1"
            onClick={() => setShowAdvancedView(!showAdvancedView)}
          >
            {showAdvancedView ? 'Show Simple' : 'Show Advanced'}
          </Button>
        </div>
        <div>
          <Button
            className="operator-button border-emerald-800 bg-emerald-950/60 py-1 text-emerald-100 hover:bg-emerald-900"
            onClick={() => saveJob()}
            disabled={status === 'saving'}
          >
            {status === 'saving' ? 'Saving...' : runId ? 'Update Job' : 'Create Job'}
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
        <div className="pt-[48px] absolute top-0 left-0 w-full h-full overflow-auto">
          {validationSummary && <div className="px-3 pt-3 sm:px-4">{validationSummary}</div>}
          <AdvancedConfigEditor
            config={jobConfig}
            setConfig={setJobConfig}
            transformOnParse={(parsed: any) => {
              try {
                parsed.config.process[0].sqlite_db_path = './aitk_db.db';
                parsed.config.process[0].training_folder = settings.TRAINING_FOLDER;
                parsed.config.process[0].device = 'cuda';
                parsed.config.process[0].performance_log_every = 10;
              } catch (e) {
                console.warn(e);
              }
              return migrateJobConfig(parsed);
            }}
          />
        </div>
      ) : (
        <MainContent>
          {validationSummary && <div className="mb-4">{validationSummary}</div>}
          <div className="mb-6">
            <TrainingAdvisorPanel jobConfig={jobConfig} gpuIDs={gpuIDs} />
          </div>
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
              isLoading={
                !isSettingsLoaded || !isGPUInfoLoaded || workerStatus === 'loading' || datasetFetchStatus !== 'success'
              }
              comfyAutoInstall={settings.COMFY_AUTO_INSTALL === 'true'}
            />
          </ErrorBoundary>

          <div className="pt-20"></div>
        </MainContent>
      )}
    </>
  );
}
