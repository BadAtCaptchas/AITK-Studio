'use client';
import { reportWorkflowError } from '@/components/WorkflowFeedback';
import {
  isEditableTrainingConfig,
  validateTrainingConfig,
  type ValidationMessage,
  type TrainingFieldTarget,
} from '@/utils/trainingValidation';
import { useEffect, useRef, useState } from 'react';
import { readTrainingDraft, useTrainingDraft } from '@/hooks/useTrainingDraft';
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
import { Button, Dialog, DialogPanel, DialogTitle } from '@headlessui/react';
import { ChevronLeft, SlidersHorizontal, Sparkles, TerminalSquare, X } from 'lucide-react';
import SimpleJob from './SimpleJob';
import AdvancedConfigEditor from '@/components/AdvancedConfigEditor';
import ErrorBoundary from '@/components/ErrorBoundary';
import { apiClient } from '@/utils/api';
import type { SelectOption } from '@/types';
import { PageNotice } from '@/components/OperatorPrimitives';
import { getRememberedEncryptedDatasetKey, rememberEncryptedDatasetKey } from '@/utils/encryptedDatasets';
import { normalizeDetectedCaptionExt } from '@/utils/jobDatasetDefaults';
import {
  makeRemoteDatasetRef,
  parseRemoteDatasetRef,
  remoteDatasetRememberKey,
  shouldImportRemoteDatasetForWorker,
} from '@/utils/remoteDatasetRefs';
import { AUTHENLORA_BUILTIN_CODEC_BITS } from '@/utils/authenloraCodecs';
import { getValidationConfigErrors } from '@/utils/validationConfig';
const isDev = process.env.NODE_ENV === 'development';
export function TrainingFormContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const runId = searchParams.get('id');
  const cloneId = searchParams.get('cloneId');
  const draftKey = runId || (cloneId ? `clone:${cloneId}` : 'new');
  const appliedDataset = useRef<string | null>(null);
  const restoredDraft = useRef(readTrainingDraft(draftKey)).current;
  const [gpuIDs, setGpuIDs] = useState<string | null>(restoredDraft?.gpuIDs ?? null);
  const [workerID, setWorkerID] = useState(restoredDraft?.workerID || 'local');
  const { settings, isSettingsLoaded, settingsStatus, settingsError, refreshSettings } = useSettings();
  const legacyView = settings.TRAINING_LEGACY_VIEW === 'true';
  const { workers, status: workerStatus, refreshWorkers } = useWorkers();
  const { gpuList, isGPUInfoLoaded, status: gpuStatus, refreshGpuInfo } = useGPUInfo(null, null, workerID);
  const { datasets, status: datasetFetchStatus, refreshDatasets } = useDatasetList({ includeRemote: true });
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
  const [rawEditorValid, setRawEditorValid] = useState(true);
  const closeRawConfig = () => {
    if (!rawEditorValid)
      reportWorkflowError('Invalid raw changes were not applied. Your last valid configuration is retained.');
    setRawConfigOpen(false);
    setRawEditorValid(true);
  };
  const [rawConfigOpen, setRawConfigOpen] = useState(false);
  const [jobConfig, setJobConfig] = useNestedState<JobConfig>(
    restoredDraft?.config || objectCopy(migrateJobConfig(defaultJobConfig)),
  );
  const draft = useTrainingDraft(draftKey, { config: jobConfig, workerID, gpuIDs });
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
        let parsed: unknown;
        if (file.name.endsWith('.json') || file.name.endsWith('.jsonc')) {
          parsed = JSON.parse(text.replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, ''));
        } else {
          parsed = YAML.parse(text);
        }
        if (!isEditableTrainingConfig(parsed))
          throw new Error(
            'Config needs a training name, model, training settings, samples, and datasets with resolutions.',
          );
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
        reportWorkflowError('Failed to parse config file. Please check the file format.');
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
    const requestedDataset = searchParams.get('dataset');
    const chosenDataset = datasetOptions.find(
      option =>
        option.value === requestedDataset || option.ref === requestedDataset || option.name === requestedDataset,
    );
    if (requestedDataset && chosenDataset && appliedDataset.current !== requestedDataset) {
      appliedDataset.current = requestedDataset;
      setJobConfig((previous: JobConfig) =>
        setNestedValue(previous, chosenDataset.value, 'config.process[0].datasets[0].folder_path'),
      );
    }
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
  }, [datasets, settings, isSettingsLoaded, datasetFetchStatus, searchParams, runId, cloneId]);
  // clone existing job
  useEffect(() => {
    if (cloneId && !restoredDraft) {
      apiClient
        .get(`/api/jobs?id=${cloneId}`)
        .then(res => res.data)
        .then(data => {
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
    if (runId && !restoredDraft) {
      apiClient
        .get(`/api/jobs?id=${runId}`)
        .then(res => res.data)
        .then(data => {
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
  const copyRememberedRemoteDatasetKey = (
    remoteRef: string,
    imported: {
      name: string;
      path: string;
    },
  ) => {
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
    cache: Map<
      string,
      Promise<{
        name: string;
        path: string;
      }>
    >,
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
        const importedPath =
          res.data?.path || (importedName ? path.join(settings.DATASETS_FOLDER, importedName) : null);
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
    const importCache = new Map<
      string,
      Promise<{
        name: string;
        path: string;
      }>
    >();
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
  const validateJobBeforeSave = (config: JobConfig) =>
    validateTrainingConfig(config, {
      workerID,
      gpuIDs,
      datasetOptions,
      unselectedDatasetPath: defaultDatasetConfig.folder_path,
      isNonImageModel: modelArchs.some(
        arch =>
          arch.name === config.config.process[0]?.model.arch &&
          (arch.group === 'audio' || arch.group === 'video' || arch.isVideoModel),
      ),
      isDatasetUnlocked: (datasetPath, name) =>
        Boolean(getRememberedEncryptedDatasetKey(datasetPath) || (name && getRememberedEncryptedDatasetKey(name))),
    });
  const saveJob = async () => {
    if (!isSettingsLoaded || settingsStatus === 'error') {
      reportWorkflowError('Load settings before saving this training run.');
      return;
    }
    if (status === 'saving' || !rawEditorValid) return;
    const jobConfigWithSettings = applyComfyAutoInstallSetting(jobConfig);
    const validation = validateJobBeforeSave(jobConfigWithSettings);
    setValidationMessages([]);
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
      draft.markSaved();
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
        setValidationMessages([
          {
            level: 'error',
            target: { step: 'job-basics', label: 'Training name' },
            message:
              error?.response?.data?.error ||
              'Training name already exists in this workspace. Choose a different name.',
          },
        ]);
      } else {
        setValidationMessages([
          {
            level: 'error',
            target: { step: 'job-review', label: '' },
            message: error?.response?.data?.error || 'Failed to save job. Please try again.',
          },
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
  const currentFindings: ValidationMessage[] = [
    ...(!rawEditorValid
      ? [
          {
            level: 'error' as const,
            message: 'Fix the raw configuration before saving.',
            target: { step: 'raw', label: 'Raw configuration' },
          },
        ]
      : []),
    ...(!isSettingsLoaded || settingsStatus === 'error'
      ? [
          {
            level: 'error' as const,
            message: settingsError || 'Loading training settings…',
            target: { step: 'settings', label: 'Settings' },
          },
        ]
      : []),
    ...validationMessages,
    ...validateJobBeforeSave(jobConfig),
  ];
  const [focusTarget, setFocusTarget] = useState<(TrainingFieldTarget & { nonce: number }) | null>(null);
  const revealFinding = (message: ValidationMessage) => {
    if (message.target.step === 'settings') {
      void refreshSettings();
      return;
    }
    if (message.target.step === 'raw') {
      setRawConfigOpen(true);
      return;
    }
    setShowAdvancedView(false);
    setFocusTarget({ ...message.target, nonce: Date.now() });
  };
  useEffect(() => {
    setValidationMessages([]);
  }, [jobConfig, workerID, gpuIDs]);
  const validationErrors = currentFindings.filter(message => message.level === 'error');
  const validationWarnings = currentFindings.filter(message => message.level === 'warning');
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
  const transformAdvancedConfig = (parsed: unknown) => {
    if (!isEditableTrainingConfig(parsed))
      throw new Error(
        'Config needs a training name, model, training settings, samples, and datasets with resolutions.',
      );
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
    <AdvancedConfigEditor
      config={jobConfig}
      setConfig={setJobConfig}
      transformOnParse={transformAdvancedConfig}
      onValidityChange={setRawEditorValid}
    />
  );
  const validationSummary =
    currentFindings.length > 0 ? (
      <PageNotice
        tone={validationErrors.length > 0 ? 'danger' : 'warning'}
        title={validationErrors.length > 0 ? 'Fix these issues before saving' : 'Review before saving'}
      >
        <ul className="list-disc space-y-1 pl-4">
          {validationErrors.map((message, index) => (
            <li key={`error-${index}`}>
              <button
                type="button"
                className="text-left underline underline-offset-2"
                onClick={() => revealFinding(message)}
              >
                {message.message}
              </button>
            </li>
          ))}
          {validationWarnings.map((message, index) => (
            <li key={`warning-${index}`}>
              <button
                type="button"
                className="text-left underline underline-offset-2"
                onClick={() => revealFinding(message)}
              >
                {message.message}
              </button>
            </li>
          ))}
        </ul>
      </PageNotice>
    ) : null;
  return (
    <>
      <TopBar className="h-16 border-gray-900 bg-gray-950 px-4">
        <div className="flex min-w-0 items-center gap-3">
          <Button
            className="operator-icon-button h-9 w-9 flex-none"
            onClick={() => draft.leaveSetup(() => router.back())}
            title="Back"
          >
            <ChevronLeft className="h-5 w-5" />
          </Button>
          <h1 className="truncate text-sm font-semibold text-gray-100 sm:text-lg">
            {runId ? 'Edit training' : 'New training'}
          </h1>
        </div>
        <div className="flex-1"></div>
        {(showAdvancedView || legacyView) && (
          <div className={legacyView ? 'min-w-40' : 'hidden min-w-40 md:block'}>
            <SelectInput
              value={workerID}
              onChange={value => {
                setWorkerID(value);
                setGpuIDs(null);
              }}
              options={workerOptions}
            />
          </div>
        )}
        <div className="w-32 shrink-0 sm:w-44">
          <SelectInput value={trainerValue} onChange={handleJobTypeChange} options={jobTypeOptions} />
        </div>
        {(showAdvancedView || legacyView) && (
          <div className={legacyView ? 'min-w-32' : 'hidden min-w-32 lg:block'}>
            <SelectInput
              value={`${gpuIDs}`}
              onChange={value => setGpuIDs(value)}
              options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
            />
          </div>
        )}
        {(showAdvancedView || legacyView) && (
          <Button className="operator-button hidden py-1 lg:inline-flex" onClick={handleImportConfig}>
            Import Config
          </Button>
        )}
        <div className="flex-none">
          <Button
            className="operator-button h-9 border-gray-700 bg-gray-900 px-3 py-1 text-gray-100"
            onClick={() => {
              closeRawConfig();
              setShowAdvancedView(!showAdvancedView);
            }}
            title={showAdvancedView ? 'Show Simple' : 'Show Advanced'}
          >
            <SlidersHorizontal className="h-4 w-4" />
            <span className="hidden sm:inline">
              {showAdvancedView ? (legacyView ? 'Legacy' : 'Guided') : 'Advanced'}
            </span>
          </Button>
        </div>
        {(showAdvancedView || legacyView) && (
          <div className="flex-none">
            <Button
              className="operator-button-primary h-10 px-3 sm:px-5"
              onClick={() => saveJob()}
              disabled={status === 'saving' || validationErrors.length > 0}
            >
              <Sparkles className="h-4 w-4" />
              <span className="hidden sm:inline">
                {status === 'saving' ? 'Saving...' : runId ? 'Update Job' : 'Create Job'}
              </span>
            </Button>
          </div>
        )}
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
          {(settingsError || gpuStatus === 'error' || datasetFetchStatus === 'error' || workerStatus === 'error') && (
            <div className="px-4 pb-4">
              <PageNotice tone="danger" title="Some setup information is unavailable">
                {settingsError && (
                  <p>
                    {settingsError}{' '}
                    <button type="button" className="underline" onClick={() => void refreshSettings()}>
                      Retry settings
                    </button>
                  </p>
                )}
                {gpuStatus === 'error' && (
                  <p>
                    Compute information could not be loaded.{' '}
                    <button type="button" className="underline" onClick={() => void refreshGpuInfo()}>
                      Retry compute
                    </button>
                  </p>
                )}
                {datasetFetchStatus === 'error' && (
                  <p>
                    Datasets could not be loaded.{' '}
                    <button type="button" className="underline" onClick={() => refreshDatasets()}>
                      Retry datasets
                    </button>
                  </p>
                )}
                {workerStatus === 'error' && (
                  <p>
                    Remote workers could not be loaded.{' '}
                    <button type="button" className="underline" onClick={refreshWorkers}>
                      Retry workers
                    </button>
                  </p>
                )}
              </PageNotice>
            </div>
          )}
          <ErrorBoundary
            fallback={
              <div className="flex h-64 items-center justify-center border border-red-300 bg-red-100 text-lg font-medium text-red-600 dark:border-red-700 dark:bg-red-900/20 dark:text-red-400">
                Advanced job detected. Please switch to advanced view to continue.
              </div>
            }
          >
            <SimpleJob
              legacyView={legacyView}
              jobConfig={jobConfig}
              setJobConfig={setJobConfig}
              status={status}
              handleSubmit={handleSubmit}
              runId={runId}
              gpuIDs={gpuIDs}
              setGpuIDs={setGpuIDs}
              gpuList={gpuList}
              datasetOptions={datasetOptions}
              validationMessages={currentFindings}
              focusTarget={focusTarget}
              onRevealFinding={revealFinding}
              computeControls={
                <SelectInput
                  label="Compute"
                  value={workerID}
                  onChange={value => {
                    setWorkerID(value);
                    setGpuIDs(null);
                  }}
                  options={workerOptions}
                />
              }
              workerLabel={workerLabel}
              trainerLabel={trainerLabel}
              onOpenAdvanced={() => setShowAdvancedView(true)}
              onOpenRawConfig={() => setRawConfigOpen(true)}
              isLoading={
                settingsStatus === 'loading' ||
                !isGPUInfoLoaded ||
                workerStatus === 'loading' ||
                ['idle', 'loading'].includes(datasetFetchStatus)
              }
              comfyAutoInstall={settings.COMFY_AUTO_INSTALL === 'true'}
            />
          </ErrorBoundary>

          <div className="pt-20"></div>
        </MainContent>
      )}

      {rawConfigOpen && !showAdvancedView && (
        <Dialog open onClose={closeRawConfig} className="fixed inset-0 z-50 flex bg-black/65 backdrop-blur-sm">
          <button
            type="button"
            className="absolute inset-0 cursor-default"
            onClick={closeRawConfig}
            aria-label="Close raw config"
          />
          <DialogPanel
            id="raw-config-drawer"
            className="relative ml-auto flex h-full w-full flex-col border-l border-gray-800 bg-gray-950 shadow-2xl sm:w-[min(920px,calc(100vw-72px))]"
          >
            <header className="flex h-16 flex-none items-center gap-3 border-b border-gray-900 px-4">
              <div className="flex h-9 w-9 items-center justify-center border border-gray-800 bg-gray-900 text-brand-200">
                <TerminalSquare className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <DialogTitle id="raw-config-title" className="truncate text-base font-semibold text-gray-100">
                  Raw config
                </DialogTitle>
                <p className="truncate text-xs text-gray-500">
                  Editing this YAML updates the current job draft. Close returns to the same workspace position.
                </p>
              </div>
              <Button className="operator-button hidden h-9 px-3 py-1 sm:inline-flex" onClick={closeRawConfig}>
                Close drawer
              </Button>
              <Button
                className="operator-icon-button h-9 w-9 flex-none"
                onClick={closeRawConfig}
                title="Close raw config"
              >
                <X className="h-4 w-4" />
              </Button>
            </header>
            <div className="min-h-0 flex-1">{renderAdvancedConfigEditor()}</div>
          </DialogPanel>
        </Dialog>
      )}
    </>
  );
}
