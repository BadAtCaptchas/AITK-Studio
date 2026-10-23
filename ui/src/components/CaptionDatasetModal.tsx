'use client';
import { reportWorkflowError } from '@/components/WorkflowFeedback';
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Modal } from '@/components/Modal';
import { createGlobalState } from 'react-global-hooks';
import { CaptionJobConfig } from '@/types';
import { defaultCaptionJobConfig } from '@/helpers/captionJobConfig';
import { objectCopy } from '@/utils/basic';
import { useNestedState } from '@/utils/hooks';
import { isMac } from '@/helpers/basic';
import useGPUInfo from '@/hooks/useGPUInfo';
import { apiClient } from '@/utils/api';
import { v4 as uuidv4 } from 'uuid';
import { isStartJobError, startJob } from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import CaptionSimpleJob from '@/components/CaptionSimpleJob';
import AdvancedConfigEditor from '@/components/AdvancedConfigEditor';
import { SelectInput } from '@/components/formInputs';
import { Loader2 } from 'lucide-react';
import { defaultIdeogramJsonCaptionPrompt } from '@/helpers/captionOptions';
import { parseCaptionJobConfig, parseCaptionJobSource } from '@/utils/captionJobSource';
export interface CaptionDatasetModalState {
  datasetPath: string;
  jobId?: string | null;
  cloneId?: string | null;
  encryptedDatasetKeyB64?: string | null;
  datasetName?: string | null;
  rootCaption?: string | null;
  preset?: 'ideogram_json' | null;
  onClose?: () => void;
}
export const captionDatasetModalState = createGlobalState<CaptionDatasetModalState | null>(null);
export const openCaptionDatasetModal = (
  datasetPath: string,
  onClose?: () => void,
  options?: {
    jobId?: string | null;
    cloneId?: string | null;
    encryptedDatasetKeyB64?: string | null;
    datasetName?: string | null;
    rootCaption?: string | null;
    preset?: 'ideogram_json' | null;
  },
) => {
  captionDatasetModalState.set({
    datasetPath,
    onClose,
    jobId: options?.jobId ?? null,
    cloneId: options?.cloneId ?? null,
    encryptedDatasetKeyB64: options?.encryptedDatasetKeyB64 ?? null,
    datasetName: options?.datasetName ?? null,
    rootCaption: options?.rootCaption !== undefined ? (options.rootCaption ?? '') : undefined,
    preset: options?.preset ?? null,
  });
};
export const CaptionDatasetModal: React.FC = () => {
  const [modalInfo, setModalInfo] = captionDatasetModalState.use();
  if (!modalInfo) return null;
  return (
    <CaptionDatasetDialog
      key={JSON.stringify([modalInfo.jobId, modalInfo.cloneId, modalInfo.datasetPath])}
      modalInfo={modalInfo}
      onClose={() => setModalInfo(null)}
    />
  );
};

function initialCaptionConfig(modalInfo: CaptionDatasetModalState): CaptionJobConfig {
  const config = objectCopy(defaultCaptionJobConfig);
  const caption = config.config.process[0].caption;
  caption.path_to_caption = modalInfo.datasetPath;
  if (!modalInfo.jobId && !modalInfo.cloneId && modalInfo.rootCaption !== undefined) {
    caption.system_prompt = modalInfo.rootCaption ?? '';
  }
  if (modalInfo.preset === 'ideogram_json') {
    caption.output_format = 'ideogram_json';
    caption.caption_prompt = defaultIdeogramJsonCaptionPrompt;
    caption.caption_extension = 'json';
    caption.source_caption_extension = 'txt';
    caption.convert_destination = 'copy';
    caption.max_res = 1024;
    caption.max_new_tokens = 2048;
  }
  return config;
}

type CaptionLaunchCheckpoint = {
  id: string;
  name: string;
  fingerprint: string;
  gpuIds: string | null;
  jobRef: string;
  startKey: string;
  startState: 'saved' | 'uncertain' | 'started';
};

function CaptionDatasetDialog({ modalInfo, onClose }: { modalInfo: CaptionDatasetModalState; onClose: () => void }) {
  const [jobConfig, setJobConfig] = useNestedState<CaptionJobConfig>(initialCaptionConfig(modalInfo));
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const [existingJobName, setExistingJobName] = useState<string | null>(null);
  const [sourceStatus, setSourceStatus] = useState<'loading' | 'ready' | 'error'>(
    modalInfo.jobId || modalInfo.cloneId ? 'loading' : 'ready',
  );
  const [sourceAttempt, setSourceAttempt] = useState(0);
  const [activeTab, setActiveTab] = useState<'simple' | 'advanced'>('simple');
  const [hasOpenedConfig, setHasOpenedConfig] = useState(false);
  const [isConfigValid, setIsConfigValid] = useState(true);
  const [createName] = useState(() => uuidv4());
  const preparedCopyRef = useRef<{ source: string; path: string } | null>(null);
  const [launchCheckpoint, setLaunchCheckpoint] = useState<CaptionLaunchCheckpoint | null>(null);
  const clearLaunchError = useRef<(() => void) | null>(null);
  const [allowDurableEncryptedResume, setAllowDurableEncryptedResume] = useState(false);
  const { gpuList, isGPUInfoLoaded } = useGPUInfo(null, null, 'local');
  const isSavingRef = useRef(false);
  const systemPromptTouchedRef = useRef(false);
  const mountedRef = useRef(false);
  const [isSaving, setIsSaving] = useState(false);
  const inputsLocked = isSaving || (!!launchCheckpoint && launchCheckpoint.startState !== 'saved');
  const showGPUSelect = !isMac();
  const isLoadingExistingJob = sourceStatus === 'loading';
  const showLoadingOverlay = isLoadingExistingJob || isSaving;
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);
  const applyRootCaption = useCallback(
    (systemPrompt: string | null | undefined) => {
      if (!modalInfo || modalInfo.jobId || modalInfo.cloneId || systemPromptTouchedRef.current) return;
      if (systemPrompt == null) return;
      setJobConfig(systemPrompt, 'config.process[0].caption.system_prompt');
    },
    [modalInfo, setJobConfig],
  );
  useEffect(() => {
    if (!modalInfo || modalInfo.jobId || modalInfo.cloneId) return;
    const controller = new AbortController();
    if (modalInfo.rootCaption !== undefined) {
      applyRootCaption(modalInfo.rootCaption);
      return;
    }
    if (!modalInfo.datasetName) return;
    apiClient
      .post<unknown>(
        '/api/datasets/root-caption',
        {
          datasetName: modalInfo.datasetName,
        },
        { signal: controller.signal },
      )
      .then(res => {
        if (controller.signal.aborted) return;
        const data = res.data;
        if (
          data &&
          typeof data === 'object' &&
          'found' in data &&
          data.found === true &&
          'systemPrompt' in data &&
          typeof data.systemPrompt === 'string'
        ) {
          applyRootCaption(data.systemPrompt);
        }
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, [applyRootCaption, modalInfo]);
  // A failed source read must never make defaults available for overwriting an existing job.
  useEffect(() => {
    const sourceId = modalInfo.jobId || modalInfo.cloneId;
    if (!sourceId) return;
    const controller = new AbortController();
    apiClient
      .get<unknown>('/api/jobs', { params: { id: sourceId }, signal: controller.signal })
      .then(res => {
        if (controller.signal.aborted) return;
        const source = parseCaptionJobSource(res.data);
        if (!modalInfo.jobId) source.config.config.name = `${source.config.config.name}_copy`;
        setGpuIDs(source.gpuIds);
        setExistingJobName(modalInfo.jobId ? source.name : null);
        setJobConfig(source.config);
        setSourceStatus('ready');
      })
      .catch(() => {
        if (!controller.signal.aborted) setSourceStatus('error');
      });
    return () => controller.abort();
  }, [modalInfo, sourceAttempt, setJobConfig]);
  if (isGPUInfoLoaded) {
    if (gpuIDs === null && gpuList.length > 0) {
      setGpuIDs(`${gpuList[0].index}`);
    }
  }
  const handleClose = () => {
    if (modalInfo?.onClose) {
      modalInfo.onClose();
    }
    onClose();
  };
  const saveJob = async () => {
    if (isSavingRef.current || sourceStatus !== 'ready' || !isConfigValid) return;
    clearLaunchError.current?.();
    clearLaunchError.current = null;
    if (!modalInfo?.datasetPath) {
      clearLaunchError.current = reportWorkflowError('Dataset path is missing. Please try again.');
      return;
    }
    if (jobConfig.config.process[0].type === 'OpenRouterCaptioner' && modalInfo.encryptedDatasetKeyB64) {
      clearLaunchError.current = reportWorkflowError(
        'OpenRouter captioning is not supported for encrypted datasets. Caption an unencrypted copy, then encrypt it afterward if needed.',
      );
      return;
    }
    isSavingRef.current = true;
    setIsSaving(true);
    let stage: 'prepare' | 'save' | 'start' | 'queue' = 'prepare';
    let checkpoint = launchCheckpoint;
    try {
      if (!checkpoint || checkpoint.startState === 'saved') {
        const jobConfigToSave = objectCopy(jobConfig);
        const captionConfig = jobConfigToSave.config.process[0].caption;
        let jobRef = modalInfo.datasetPath;
        const shouldCopyDataset =
          (captionConfig.output_format === 'ideogram_json' || captionConfig.output_format === 'json') &&
          captionConfig.convert_destination === 'copy' &&
          !modalInfo.jobId;
        if (shouldCopyDataset) {
          if (preparedCopyRef.current?.source !== modalInfo.datasetPath) {
            const { data } = await apiClient.post<unknown>('/api/datasets/copy', {
              datasetPath: modalInfo.datasetPath,
              suffix: 'json_captions',
            });
            if (
              !data ||
              typeof data !== 'object' ||
              !('path' in data) ||
              typeof data.path !== 'string' ||
              !data.path.trim()
            ) {
              throw new Error('Invalid dataset copy response');
            }
            preparedCopyRef.current = { source: modalInfo.datasetPath, path: data.path };
          }
          jobRef = preparedCopyRef.current.path;
          captionConfig.path_to_caption = jobRef;
        }
        const fingerprint = JSON.stringify({ config: jobConfigToSave, jobRef, gpuIDs });
        stage = 'save';
        if (!checkpoint || checkpoint.fingerprint !== fingerprint) {
          const name = checkpoint?.name || existingJobName || createName;
          const { data } = await apiClient.post<unknown>('/api/jobs', {
            id: checkpoint?.id || modalInfo.jobId || null,
            name,
            gpu_ids: gpuIDs,
            job_config: jobConfigToSave,
            job_type: 'caption',
            job_ref: jobRef,
          });
          if (!data || typeof data !== 'object' || !('id' in data) || typeof data.id !== 'string' || !data.id.trim()) {
            throw new Error('Invalid caption job response');
          }
          checkpoint = {
            id: data.id,
            name,
            fingerprint,
            gpuIds: gpuIDs,
            jobRef,
            startKey: uuidv4(),
            startState: 'saved',
          };
          if (mountedRef.current) setLaunchCheckpoint(checkpoint);
        }
      }
      if (checkpoint.startState !== 'started') {
        stage = 'start';
        // Persist the command identity before sending: a lost response is not a rejection.
        checkpoint = { ...checkpoint, startState: 'uncertain' };
        if (mountedRef.current) setLaunchCheckpoint(checkpoint);
        await startJob(
          checkpoint.id,
          modalInfo.encryptedDatasetKeyB64
            ? [{ datasetPath: checkpoint.jobRef, keyB64: modalInfo.encryptedDatasetKeyB64 }]
            : undefined,
          { durableEncryptedDatasetKeys: allowDurableEncryptedResume, idempotencyKey: checkpoint.startKey },
        );
        checkpoint = { ...checkpoint, startState: 'started' };
        if (mountedRef.current) setLaunchCheckpoint(checkpoint);
      }
      stage = 'queue';
      await startQueue(checkpoint.gpuIds || '');
      if (mountedRef.current) handleClose();
    } catch (error: unknown) {
      if (!mountedRef.current) return;
      const uncertainStart = stage === 'start' && (!isStartJobError(error) || error.outcome === 'unknown');
      if (stage === 'start' && checkpoint && !uncertainStart) {
        // A conclusively rejected/failed command is finished. A deliberate retry
        // gets a new identity; only an uncertain result must replay its old key.
        checkpoint = { ...checkpoint, startKey: uuidv4(), startState: 'saved' };
        setLaunchCheckpoint(checkpoint);
      }
      const messages = {
        prepare: 'Failed to prepare caption dataset. Please try again.',
        save: 'Failed to save job. Please try again.',
        start: uncertainStart
          ? 'Captioning may already have started. Retry to confirm or complete the start before editing.'
          : 'Job saved, but captioning could not start. Retry to start the saved job.',
        queue: 'Caption start was accepted, but the queue could not start. Retry to start the queue.',
      };
      let detail: string | undefined;
      if (error && typeof error === 'object' && 'response' in error) {
        const response = error.response;
        if (response && typeof response === 'object' && 'data' in response) {
          const data = response.data;
          if (data && typeof data === 'object' && 'error' in data && typeof data.error === 'string')
            detail = data.error;
        }
      }
      // startJob exposes the actionable API reason as an Error, without the transport object.
      if (!detail && stage === 'start' && error instanceof Error) detail = error.message;
      clearLaunchError.current = reportWorkflowError(`${messages[stage]}${detail ? ` ${detail}` : ''}`);
    } finally {
      isSavingRef.current = false;
      if (mountedRef.current) setIsSaving(false);
    }
  };
  const tabButtonClass = (tab: 'simple' | 'advanced') =>
    `px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
      activeTab === tab
        ? 'border-brand-500 text-brand-400'
        : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-600'
    }`;
  return (
    <Modal isOpen onClose={handleClose} title="Caption Dataset" size={activeTab === 'advanced' ? 'xl' : 'lg'}>
      <div className="relative space-y-4 text-gray-200">
        {showLoadingOverlay && (
          <div className="absolute -left-6 -right-6 -top-4 -bottom-4 z-10 flex items-center justify-center backdrop-blur-sm bg-gray-900/40">
            <Loader2 className="w-10 h-10 text-brand-400 animate-spin" />
          </div>
        )}
        {sourceStatus === 'error' && (
          <div role="alert" className="rounded border border-rose-700 p-3 text-sm">
            <p>Could not load caption job.</p>
            <button
              type="button"
              className="operator-button mt-2"
              aria-label="Retry loading caption job"
              onClick={() => {
                setSourceStatus('loading');
                setSourceAttempt(attempt => attempt + 1);
              }}
            >
              Retry
            </button>
          </div>
        )}
        <div className="flex items-center border-b border-gray-700 -mt-2" inert={inputsLocked}>
          <button type="button" className={tabButtonClass('simple')} onClick={() => setActiveTab('simple')}>
            Simple
          </button>
          <button
            type="button"
            className={tabButtonClass('advanced')}
            onClick={() => {
              setHasOpenedConfig(true);
              setActiveTab('advanced');
            }}
          >
            Advanced
          </button>
          <div className="flex-1" />
          {activeTab === 'advanced' && showGPUSelect && (
            <div className="pb-2">
              <SelectInput
                value={`${gpuIDs}`}
                onChange={value => setGpuIDs(value)}
                options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              />
            </div>
          )}
        </div>
        <form
          onSubmit={e => {
            e.preventDefault();
            saveJob();
          }}
        >
          {sourceStatus === 'ready' && (
            <fieldset className="min-w-0" disabled={inputsLocked} inert={inputsLocked}>
              {activeTab === 'simple' &&
                (isConfigValid ? (
                  <CaptionSimpleJob
                    jobConfig={jobConfig}
                    setJobConfig={setJobConfig}
                    datasetPath={modalInfo?.datasetPath || ''}
                    encryptedDatasetKeyB64={modalInfo?.encryptedDatasetKeyB64}
                    gpuIDs={gpuIDs}
                    setGpuIDs={setGpuIDs}
                    gpuList={gpuList}
                    showGPUSelect={showGPUSelect}
                    onSystemPromptChange={() => {
                      systemPromptTouchedRef.current = true;
                    }}
                  />
                ) : (
                  <div role="alert" className="rounded border border-rose-700 p-3 text-sm">
                    <p>Fix the configuration in Advanced before editing this form or saving. Your text is retained.</p>
                    <button type="button" className="operator-button mt-2" onClick={() => setActiveTab('advanced')}>
                      Return to Advanced
                    </button>
                  </div>
                ))}
              {hasOpenedConfig && (
                <div className={activeTab === 'advanced' ? 'h-[60vh] mt-2' : 'hidden'}>
                  <AdvancedConfigEditor
                    config={jobConfig}
                    setConfig={setJobConfig}
                    transformOnParse={parseCaptionJobConfig}
                    onValidityChange={setIsConfigValid}
                    active={activeTab === 'advanced'}
                    readOnly={inputsLocked}
                  />
                </div>
              )}
            </fieldset>
          )}

          <div className="mt-6 flex justify-end space-x-3">
            {modalInfo?.encryptedDatasetKeyB64 && (
              <label className="mr-auto flex max-w-md items-start gap-2 text-sm text-gray-300">
                <input
                  type="checkbox"
                  className="mt-1"
                  checked={allowDurableEncryptedResume}
                  disabled={inputsLocked}
                  onChange={e => setAllowDurableEncryptedResume(e.target.checked)}
                />
                <span>
                  Allow durable encrypted resume
                  <span className="block text-xs text-gray-500">
                    Stores a wrapped dataset key on the server until the job completes or is deleted.
                  </span>
                </span>
              </label>
            )}
            <button type="button" className="operator-button" onClick={handleClose}>
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSaving || sourceStatus !== 'ready' || !isConfigValid}
              className="operator-button-primary"
            >
              {isSaving
                ? 'Starting…'
                : launchCheckpoint?.startState === 'started'
                  ? 'Retry queue start'
                  : launchCheckpoint?.startState === 'uncertain'
                    ? 'Retry caption start'
                    : launchCheckpoint
                      ? 'Save & retry start'
                      : 'Save & start captioning'}
            </button>
          </div>
        </form>
      </div>
    </Modal>
  );
}
