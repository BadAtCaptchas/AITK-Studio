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
import { startJob } from '@/utils/jobs';
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
  const [allowDurableEncryptedResume, setAllowDurableEncryptedResume] = useState(false);
  const { gpuList, isGPUInfoLoaded } = useGPUInfo(null, null, 'local');
  const isSavingRef = useRef(false);
  const systemPromptTouchedRef = useRef(false);
  const mountedRef = useRef(false);
  const [isSaving, setIsSaving] = useState(false);
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
    if (!modalInfo?.datasetPath) {
      reportWorkflowError('Dataset path is missing. Please try again.');
      return;
    }
    if (jobConfig.config.process[0].type === 'OpenRouterCaptioner' && modalInfo.encryptedDatasetKeyB64) {
      reportWorkflowError(
        'OpenRouter captioning is not supported for encrypted datasets. Caption an unencrypted copy, then encrypt it afterward if needed.',
      );
      return;
    }
    isSavingRef.current = true;
    setIsSaving(true);
    const isEdit = !!modalInfo.jobId;
    const jobConfigToSave = objectCopy(jobConfig);
    let jobRef = modalInfo.datasetPath;
    try {
      const captionConfig = jobConfigToSave.config.process[0].caption;
      const outputFormat = captionConfig.output_format || 'text';
      const shouldCopyDataset =
        (outputFormat === 'ideogram_json' || outputFormat === 'json') &&
        captionConfig.convert_destination === 'copy' &&
        !isEdit;
      if (shouldCopyDataset) {
        const copied = await apiClient
          .post('/api/datasets/copy', {
            datasetPath: modalInfo.datasetPath,
            suffix: 'json_captions',
          })
          .then(res => res.data);
        if (!copied?.path) {
          throw new Error('Dataset copy failed');
        }
        captionConfig.path_to_caption = copied.path;
        jobRef = copied.path;
      }
    } catch (error: any) {
      reportWorkflowError(error?.response?.data?.error || error?.message || 'Failed to prepare caption dataset.');
      isSavingRef.current = false;
      setIsSaving(false);
      return;
    }
    apiClient
      .post('/api/jobs', {
        id: isEdit ? modalInfo.jobId : null,
        name: isEdit && existingJobName ? existingJobName : uuidv4(),
        gpu_ids: gpuIDs,
        job_config: jobConfigToSave,
        job_type: 'caption',
        job_ref: jobRef,
      })
      .then(async res => {
        const jobId = res.data.id;
        await startJob(
          jobId,
          modalInfo.encryptedDatasetKeyB64
            ? [{ datasetPath: jobRef, keyB64: modalInfo.encryptedDatasetKeyB64 }]
            : undefined,
          { durableEncryptedDatasetKeys: allowDurableEncryptedResume },
        );
        // start the queue as well
        await startQueue(gpuIDs || '');
        if (mountedRef.current) {
          isSavingRef.current = false;
          setIsSaving(false);
          handleClose();
        }
      })
      .catch(error => {
        if (!mountedRef.current) return;
        if (error.response?.status === 409) {
          reportWorkflowError('A caption job for this dataset already exists. Please check your jobs list.');
        } else {
          reportWorkflowError('Failed to save job. Please try again.');
        }
        console.log('Error saving training:', error);
        isSavingRef.current = false;
        setIsSaving(false);
      });
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
        <div className="flex items-center border-b border-gray-700 -mt-2">
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
            <>
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
                    readOnly={isSaving}
                  />
                </div>
              )}
            </>
          )}

          <div className="mt-6 flex justify-end space-x-3">
            {modalInfo?.encryptedDatasetKeyB64 && (
              <label className="mr-auto flex max-w-md items-start gap-2 text-sm text-gray-300">
                <input
                  type="checkbox"
                  className="mt-1"
                  checked={allowDurableEncryptedResume}
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
              {isSaving ? 'Starting…' : 'Save & start captioning'}
            </button>
          </div>
        </form>
      </div>
    </Modal>
  );
}
