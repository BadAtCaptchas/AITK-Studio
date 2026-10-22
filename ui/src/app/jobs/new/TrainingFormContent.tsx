'use client';
import { configContractErrors } from '@/domain/configContract';
import { reportWorkflowError } from '@/components/WorkflowFeedback';
import {
  isEditableTrainingConfig,
  validateTrainingConfig,
  type ValidationMessage,
  type TrainingFieldTarget,
} from '@/utils/trainingValidation';
import { useEffect, useRef, useState } from 'react';
import { readTrainingDraft, restoreTrainingDraft, useTrainingDraft } from '@/hooks/useTrainingDraft';
import { readStoredTrainingDraft, removeStoredTrainingDraft, type StoredTrainingDraft } from '@/utils/trainingDraftStorage';
import useHydrated from '@/hooks/useHydrated';
import { useSearchParams, useRouter } from 'next/navigation';
import { defaultJobConfig, defaultDatasetConfig, migrateJobConfig } from './jobConfig';
import { jobTypeOptions, modelArchs } from './options';
import type { JobConfig } from '@/types';
import { objectCopy } from '@/utils/basic';
import { saveTrainingJob, trainingJobSaveError } from '@/utils/trainingJobSave';
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
import { groupModelOptions } from '@/domain/modelOptions';
import {
  getMigrationTargets,
  migrateTrainingModel,
  migrationSourceError,
  prepareFreshTrainingConfig,
  suggestMigrationName,
  validateTrainingMigration,
  type TrainingMigrationDraft,
} from '@/domain/trainingModelMigration';
type MigrationState =
  | { phase: 'loading' }
  | { phase: 'ready'; draft: TrainingMigrationDraft }
  | { phase: 'error'; message: string };
const isDev = process.env.NODE_ENV === 'development';
export function TrainingFormContent() {
  const hydrated = useHydrated();
  const searchParams = useSearchParams();
  const migrationId = searchParams.get('migrateId');
  const runId = migrationId ? null : searchParams.get('id');
  const cloneId = migrationId ? null : searchParams.get('cloneId');
  const key = migrationId ? `migrate:${migrationId}` : runId ? `edit:${runId}` : cloneId ? `clone:${cloneId}` : 'new';
  return hydrated ? <TrainingDraftRecovery key={key} draftKey={key} /> : <div className="p-6 text-gray-400">Loading training setup…</div>;
}

function TrainingDraftRecovery({ draftKey }: { draftKey: string }) {
  const [recovery, setRecovery] = useState<{ phase: 'ready' } | { phase: 'recover'; stored: StoredTrainingDraft }>(() => {
    const stored = readTrainingDraft(draftKey) ? undefined : readStoredTrainingDraft(draftKey);
    return stored ? { phase: 'recover', stored } : { phase: 'ready' };
  });
  if (recovery.phase === 'recover') return (
    <section className="m-6 max-w-xl rounded-lg border border-gray-700 bg-gray-900 p-6" aria-label="Recover training draft">
      <h1 className="text-lg font-semibold">Recover unsaved training setup?</h1>
      <p className="mt-2 text-gray-300">A draft for {recovery.stored.draft.config.config.name || 'this run'} is available from this tab. Credentials and unlocked dataset contents are not stored.</p>
      <div className="mt-5 flex gap-3">
        <button className="rounded bg-emerald-600 px-4 py-2" onClick={() => { restoreTrainingDraft(draftKey, recovery.stored); setRecovery({ phase: 'ready' }); }}>Restore draft</button>
        <button className="rounded border border-gray-600 px-4 py-2" onClick={() => { removeStoredTrainingDraft(draftKey); setRecovery({ phase: 'ready' }); }}>Discard draft</button>
      </div>
    </section>
  );
  return <TrainingFormEditor draftKey={draftKey} />;
}

function TrainingFormEditor({ draftKey }: { draftKey: string }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const migrationId = searchParams.get('migrateId');
  const runId = migrationId ? null : searchParams.get('id');
  const cloneId = migrationId ? null : searchParams.get('cloneId');
  const appliedDataset = useRef<string | null>(null);
  const [restoredDraft] = useState(() => readTrainingDraft(draftKey));
  const [migration, setMigration] = useState<MigrationState>(() =>
    restoredDraft?.migration ? { phase: 'ready', draft: restoredDraft.migration } : { phase: 'loading' },
  );
  const [gpuIDs, setGpuIDs] = useState<string | null>(restoredDraft?.gpuIDs ?? null);
  const [workerID, setWorkerID] = useState(restoredDraft?.workerID || 'local');
  const { settings, isSettingsLoaded, settingsStatus, settingsError, refreshSettings } = useSettings();
  const legacyView = settings.TRAINING_LEGACY_VIEW === 'true';
  const { workers, status: workerStatus, refreshWorkers } = useWorkers();
  const { gpuData, gpuList, isGPUInfoLoaded, status: gpuStatus, refreshGpuInfo } = useGPUInfo(null, null, workerID);
  const { datasets, status: datasetFetchStatus, refreshDatasets } = useDatasetList({ includeRemote: true });
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
  const draft = useTrainingDraft(draftKey, {
    config: jobConfig,
    workerID,
    gpuIDs,
    ...(migrationId && migration.phase === 'ready' ? { migration: migration.draft } : {}),
  });
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [saveError, setSaveError] = useState<string | null>(null);
  const savingRef = useRef(false);
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
        if (migrationId && parsed.config.process[0].model.arch !== jobConfig.config.process[0].model.arch)
          throw new Error(
            'Choose the destination architecture in the migration form before importing its configuration.',
          );
        // Set required fields (same pattern as AdvancedJob.handleChange)
        try {
          parsed.config.process[0].sqlite_db_path = './aitk_db.db';
          parsed.config.process[0].training_folder = settings.TRAINING_FOLDER;
          parsed.config.process[0].device = gpuData?.isMac ? 'mps' : 'cuda';
          parsed.config.process[0].performance_log_every = 10;
        } catch (err) {
          console.warn('Could not set required fields on imported config:', err);
        }
        if (!migrationId) migrateJobConfig(parsed);
        setJobConfig(parsed);
      } catch (err) {
        console.error('Failed to parse config file:', err);
        reportWorkflowError(
          err instanceof Error ? err.message : 'Failed to parse config file. Please check the file format.',
        );
      }
    };
    reader.readAsText(file);
    // Reset so the same file can be re-imported
    e.target.value = '';
  };
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
  useEffect(() => {
    if (!isSettingsLoaded || datasetFetchStatus !== 'success') return;
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
    if (!migrationId || restoredDraft?.migration) return;
    let canceled = false;
    apiClient
      .get<unknown>(`/api/jobs?id=${encodeURIComponent(migrationId)}`)
      .then(({ data }) => {
        if (canceled) return;
        if (
          !data ||
          typeof data !== 'object' ||
          !('job_type' in data) ||
          data.job_type !== 'train' ||
          !('job_config' in data) ||
          typeof data.job_config !== 'string'
        )
          throw new Error('The source training job could not be loaded.');
        const source: unknown = JSON.parse(data.job_config);
        const sourceError = migrationSourceError(source);
        if (sourceError || !isEditableTrainingConfig(source))
          throw new Error(sourceError || 'Invalid source training configuration.');
        const prepared = prepareFreshTrainingConfig(source);
        if (prepared.config.config.process[0].type === 'ui_trainer')
          prepared.config.config.process[0].type = 'diffusion_trainer';
        prepared.config.config.name = suggestMigrationName(source.config.name);
        setJobConfig(prepared.config);
        setGpuIDs('gpu_ids' in data && typeof data.gpu_ids === 'string' ? data.gpu_ids : null);
        setWorkerID('worker_id' in data && typeof data.worker_id === 'string' ? data.worker_id : 'local');
        setMigration({ phase: 'ready', draft: { source, changes: prepared.changes } });
      })
      .catch((cause: unknown) => {
        if (!canceled)
          setMigration({
            phase: 'error',
            message: cause instanceof Error ? cause.message : 'Failed to load the source job.',
          });
      });
    return () => {
      canceled = true;
    };
  }, [migrationId, restoredDraft, setJobConfig]);

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
  if (isGPUInfoLoaded) {
    if (gpuIDs === null) {
      setGpuIDs(gpuList.length > 0 ? `${gpuList[0].index}` : '0');
    }
  }
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
  const migrationFindings = (config: JobConfig): ValidationMessage[] => {
    if (!migrationId) return [];
    if (migration.phase !== 'ready')
      return [
        {
          level: 'error',
          message: migration.phase === 'error' ? migration.message : 'Loading the source training job...',
          target: { step: 'job-basics', label: 'Source job' },
        },
      ];
    return validateTrainingMigration(config, migration.draft.source);
  };
  const changeMigrationModel = (arch: string, modelPath?: string) => {
    if (migration.phase !== 'ready') return;
    try {
      const result = migrateTrainingModel(jobConfig, arch, modelPath);
      const sourceName = migration.draft.source.config.name;
      if (
        [
          suggestMigrationName(sourceName),
          suggestMigrationName(sourceName, jobConfig.config.process[0].model.arch),
        ].includes(jobConfig.config.name)
      ) {
        result.config.config.name = suggestMigrationName(sourceName, arch);
      }
      setJobConfig(result.config);
      setMigration({
        phase: 'ready',
        draft: {
          ...migration.draft,
          changes: [
            ...new Map([...migration.draft.changes, ...result.changes].map(change => [change.path, change])).values(),
          ],
        },
      });
    } catch (cause) {
      reportWorkflowError(cause instanceof Error ? cause.message : 'Could not migrate this model.');
    }
  };
  const saveJob = async () => {
    if (!isSettingsLoaded || settingsStatus === 'error') {
      reportWorkflowError('Load settings before saving this training run.');
      return;
    }
    if (savingRef.current || !rawEditorValid) return;
    const freshConfig = migrationId ? prepareFreshTrainingConfig(jobConfig).config : jobConfig;
    const jobConfigWithSettings = applyComfyAutoInstallSetting(freshConfig);
    const validation = [...validateJobBeforeSave(jobConfigWithSettings), ...migrationFindings(jobConfigWithSettings)];
    setValidationMessages(validation);
    if (validation.some(message => message.level === 'error')) {
      setStatus('idle');
      return;
    }
    savingRef.current = true;
    setSaveError(null);
    setStatus('saving');
    try {
      const preparedJobConfig = await importRemoteDatasetsForJobConfig(jobConfigWithSettings, workerID);
      setJobConfig(preparedJobConfig);
      const saved = await saveTrainingJob({
        ...(migrationId ? {} : { id: runId }),
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
        router.push(`/jobs/${encodeURIComponent(saved.id)}`);
      }
    } catch (error: unknown) {
      const failure = trainingJobSaveError(error);
      setStatus('error');
      setSaveError(failure.message);
      if (failure.conflict) {
        setValidationMessages([
          {
            level: 'error',
            target: { step: 'job-basics', label: 'Training name' },
            message: failure.message,
          },
        ]);
      }
    } finally {
      savingRef.current = false;
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
    ...migrationFindings(jobConfig),
    ...validateJobBeforeSave(jobConfig),
    ...configContractErrors(jobConfig, { deviceBackend: gpuData ? (gpuData.isMac ? 'mps' : 'cuda') : undefined }).map(
      message => ({ level: 'error' as const, message, target: { step: 'raw', label: 'Raw configuration' } }),
    ),
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
    setFocusTarget(previous => ({ ...message.target, nonce: (previous?.nonce ?? 0) + 1 }));
  };
  const [validatedInput, setValidatedInput] = useState({ jobConfig, workerID, gpuIDs });
  if (
    validatedInput.jobConfig !== jobConfig ||
    validatedInput.workerID !== workerID ||
    validatedInput.gpuIDs !== gpuIDs
  ) {
    setValidatedInput({ jobConfig, workerID, gpuIDs });
    setValidationMessages([]);
  }
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
    if (migrationId && parsed.config.process[0].model.arch !== jobConfig.config.process[0].model.arch)
      throw new Error(
        'Choose the destination model using the migration selector before editing its raw configuration.',
      );
    try {
      parsed.config.process[0].sqlite_db_path = './aitk_db.db';
      parsed.config.process[0].training_folder = settings.TRAINING_FOLDER;
      parsed.config.process[0].device = gpuData?.isMac ? 'mps' : 'cuda';
      parsed.config.process[0].performance_log_every = 10;
    } catch (e) {
      console.warn(e);
    }
    return migrationId ? parsed : migrateJobConfig(parsed);
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
  const migrationOptions =
    migration.phase === 'ready' ? groupModelOptions(getMigrationTargets(migration.draft.source)) : [];
  const migrationNotice = migrationId ? (
    <div className="px-4 py-4">
      <PageNotice tone={migration.phase === 'error' ? 'danger' : 'info'} title="Migrate to another model">
        {migration.phase === 'loading' && <p>Loading the source training job...</p>}
        {migration.phase === 'error' && <p role="alert">{migration.message}</p>}
        {migration.phase === 'ready' && (
          <>
            <p>
              Create a separate run from <strong>{migration.draft.source.config.name}</strong>. Training starts at step
              0. The original run and its outputs are retained.
            </p>
            <p className="mt-1">
              Choose another model architecture or model path. Shared settings are kept; required adjustments appear
              below.
            </p>
            {showAdvancedView && (
              <div className="mt-3 max-w-lg">
                <SelectInput
                  label="Destination model"
                  value={jobConfig.config.process[0].model.arch}
                  options={migrationOptions}
                  onChange={changeMigrationModel}
                />
              </div>
            )}
            {migration.draft.changes.length > 0 && (
              <details className="mt-3">
                <summary className="cursor-pointer">Automatic adjustments ({migration.draft.changes.length})</summary>
                <ul className="mt-2 space-y-2 text-sm">
                  {migration.draft.changes.map(change => (
                    <li key={change.path}>
                      <code>{change.path.replace('config.process[0].', '')}</code>: {change.reason}
                      <span className="block break-words text-gray-300">
                        {change.before} → {change.after}
                      </span>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </>
        )}
      </PageNotice>
    </div>
  ) : null;
  return (
    <div className="contents" {...draft.formTracking}>
      <TopBar className="training-topbar h-16 border-gray-900 bg-gray-950 px-4">
        <div className="flex min-w-0 items-center gap-3">
          <Button
            className="operator-icon-button h-9 w-9 flex-none"
            onClick={() => draft.leaveSetup(() => router.back())}
            title="Back"
          >
            <ChevronLeft className="h-5 w-5" />
          </Button>
          <h1 className="truncate text-sm font-semibold text-gray-100 sm:text-lg">
            {migrationId ? 'Migrate Training Job' : runId ? 'Edit Training Job' : 'New Training Job'}
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
          <Button className="operator-button py-1" onClick={handleImportConfig}>
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
            <span className="hidden sm:inline">{showAdvancedView ? (legacyView ? 'Form' : 'Guided') : 'Advanced'}</span>
          </Button>
        </div>
        {(showAdvancedView || legacyView) && (
          <div className="flex-none">
            <Button
              className="operator-button-primary h-10 px-3 sm:px-5"
              aria-label={runId ? 'Update Job' : 'Create Job'}
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
        <div className="training-content absolute top-0 left-0 h-full w-full overflow-auto pt-16">
          {migrationNotice}
          {saveError && (
            <div role="alert" className="px-4 pt-4">
              <PageNotice tone="danger" title="Could not save job">
                {saveError}
              </PageNotice>
            </div>
          )}
          {validationSummary && <div className="px-3 pt-3 sm:px-4">{validationSummary}</div>}
          {renderAdvancedConfigEditor()}
        </div>
      ) : (
        <MainContent className="training-content bg-gray-950 px-0 pt-16 sm:px-0">
          {migrationNotice}
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
              workerId={workerID}
              legacyView={legacyView}
              jobConfig={jobConfig}
              setJobConfig={setJobConfig}
              status={status}
              saveError={saveError}
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
              migrationModelOptions={migrationId ? migrationOptions : undefined}
              onMigrationModelChange={migrationId ? changeMigrationModel : undefined}
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
    </div>
  );
}
