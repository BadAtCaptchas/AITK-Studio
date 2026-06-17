'use client';
import React, { useState, useEffect, useRef } from 'react';
import { Modal } from '@/components/Modal';
import { createGlobalState } from 'react-global-hooks';
import { useFromNull } from '@/hooks/useFromNull';
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

export interface CaptionDatasetModalState {
  datasetPath: string;
  projectID?: string | null;
  jobId?: string | null;
  cloneId?: string | null;
  encryptedDatasetKeyB64?: string | null;
  preset?: 'ideogram_json' | null;
  onClose?: () => void;
}

export const captionDatasetModalState = createGlobalState<CaptionDatasetModalState | null>(null);

export const openCaptionDatasetModal = (
  datasetPath: string,
  onClose?: () => void,
  options?: {
    projectID?: string | null;
    jobId?: string | null;
    cloneId?: string | null;
    encryptedDatasetKeyB64?: string | null;
    preset?: 'ideogram_json' | null;
  },
) => {
  captionDatasetModalState.set({
    datasetPath,
    projectID: options?.projectID ?? null,
    onClose,
    jobId: options?.jobId ?? null,
    cloneId: options?.cloneId ?? null,
    encryptedDatasetKeyB64: options?.encryptedDatasetKeyB64 ?? null,
    preset: options?.preset ?? null,
  });
};

export const CaptionDatasetModal: React.FC = () => {
  const [modalInfo, setModalInfo] = captionDatasetModalState.use();
  const [jobConfig, setJobConfig] = useNestedState<CaptionJobConfig>(objectCopy(defaultCaptionJobConfig));
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const [existingJobName, setExistingJobName] = useState<string | null>(null);
  const [hasLoadedExistingJob, setHasLoadedExistingJob] = useState(false);
  const [activeTab, setActiveTab] = useState<'simple' | 'advanced'>('simple');
  const [allowDurableEncryptedResume, setAllowDurableEncryptedResume] = useState(false);
  const open = modalInfo !== null;
  const { gpuList, isGPUInfoLoaded } = useGPUInfo(null, null, 'local', { enabled: open });
  const isSavingRef = useRef(false);
  const [isSaving, setIsSaving] = useState(false);
  const showGPUSelect = !isMac();
  const isLoadingExistingJob = !!(modalInfo?.jobId || modalInfo?.cloneId) && !hasLoadedExistingJob;
  const showLoadingOverlay = isLoadingExistingJob || isSaving;

  useFromNull(() => {
    // reset the state
    setJobConfig(objectCopy(defaultCaptionJobConfig));
    setActiveTab('simple');
    setExistingJobName(null);
    setAllowDurableEncryptedResume(false);
    // set the path_to_caption
    if (modalInfo?.datasetPath) {
      setJobConfig(modalInfo.datasetPath, 'config.process[0].caption.path_to_caption');
    }
    if (modalInfo?.preset === 'ideogram_json') {
      setJobConfig('ideogram_json', 'config.process[0].caption.output_format');
      setJobConfig(defaultIdeogramJsonCaptionPrompt, 'config.process[0].caption.caption_prompt');
      setJobConfig('json', 'config.process[0].caption.caption_extension');
      setJobConfig('txt', 'config.process[0].caption.source_caption_extension');
      setJobConfig('copy', 'config.process[0].caption.convert_destination');
      setJobConfig(1024, 'config.process[0].caption.max_res');
      setJobConfig(2048, 'config.process[0].caption.max_new_tokens');
    }
  }, [modalInfo]);

  // clone existing caption job
  useEffect(() => {
    if (modalInfo?.cloneId) {
      apiClient
        .get(`/api/jobs?id=${modalInfo.cloneId}`)
        .then(res => res.data)
        .then(data => {
          setGpuIDs(data.gpu_ids);
          const newJobConfig = JSON.parse(data.job_config);
          newJobConfig.config.name = `${newJobConfig.config.name}_copy`;
          setJobConfig(newJobConfig);
        })
        .catch(error => console.error('Error fetching caption job:', error))
        .finally(() => setHasLoadedExistingJob(true));
    }
  }, [modalInfo?.cloneId]);

  // load existing caption job for editing
  useEffect(() => {
    if (modalInfo?.jobId) {
      apiClient
        .get(`/api/jobs?id=${modalInfo.jobId}`)
        .then(res => res.data)
        .then(data => {
          setGpuIDs(data.gpu_ids);
          setExistingJobName(data.name);
          setJobConfig(JSON.parse(data.job_config));
        })
        .catch(error => console.error('Error fetching caption job:', error))
        .finally(() => setHasLoadedExistingJob(true));
    }
  }, [modalInfo?.jobId]);

  useEffect(() => {
    if (isGPUInfoLoaded) {
      if (gpuIDs === null && gpuList.length > 0) {
        setGpuIDs(`${gpuList[0].index}`);
      }
    }
  }, [gpuList, isGPUInfoLoaded]);

  const handleClose = () => {
    if (modalInfo?.onClose) {
      modalInfo.onClose();
    }
    setHasLoadedExistingJob(false);
    setModalInfo(null);
  };

  const saveJob = async () => {
    if (isSavingRef.current) return;
    if (!modalInfo?.datasetPath) {
      alert('Dataset path is missing. Please try again.');
      return;
    }
    if (jobConfig.config.process[0].type === 'OpenRouterCaptioner' && modalInfo.encryptedDatasetKeyB64) {
      alert('OpenRouter captioning is not supported for encrypted datasets. Caption an unencrypted copy, then encrypt it afterward if needed.');
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
            project_id: modalInfo.projectID || undefined,
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
      alert(error?.response?.data?.error || error?.message || 'Failed to prepare caption dataset.');
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
        project_id: modalInfo.projectID || undefined,
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
        isSavingRef.current = false;
        setIsSaving(false);
        handleClose();
      })
      .catch(error => {
        if (error.response?.status === 409) {
          alert('A caption job for this dataset already exists. Please check your jobs list.');
        } else {
          alert('Failed to save job. Please try again.');
        }
        console.log('Error saving training:', error);
        isSavingRef.current = false;
        setIsSaving(false);
      });
  };

  const tabButtonClass = (tab: 'simple' | 'advanced') =>
    `px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
      activeTab === tab
        ? 'border-blue-500 text-blue-400'
        : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-600'
    }`;

  return (
    <Modal isOpen={open} onClose={handleClose} title="Caption Dataset" size={activeTab === 'advanced' ? 'xl' : 'lg'}>
      <div className="relative space-y-4 text-gray-200">
        {showLoadingOverlay && (
          <div className="absolute -left-6 -right-6 -top-4 -bottom-4 z-10 flex items-center justify-center backdrop-blur-sm bg-gray-900/40">
            <Loader2 className="w-10 h-10 text-blue-400 animate-spin" />
          </div>
        )}
        <div className="flex items-center border-b border-gray-700 -mt-2">
          <button type="button" className={tabButtonClass('simple')} onClick={() => setActiveTab('simple')}>
            Simple
          </button>
          <button type="button" className={tabButtonClass('advanced')} onClick={() => setActiveTab('advanced')}>
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
          {activeTab === 'simple' ? (
            <CaptionSimpleJob
              jobConfig={jobConfig}
              setJobConfig={setJobConfig}
              datasetPath={modalInfo?.datasetPath || ''}
              encryptedDatasetKeyB64={modalInfo?.encryptedDatasetKeyB64}
              gpuIDs={gpuIDs}
              setGpuIDs={setGpuIDs}
              gpuList={gpuList}
              showGPUSelect={showGPUSelect}
            />
          ) : (
            <div className="h-[60vh] mt-2">
              <AdvancedConfigEditor config={jobConfig} setConfig={setJobConfig} />
            </div>
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
            <button
              type="button"
              className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
              onClick={handleClose}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              Add to Queue
            </button>
          </div>
        </form>
      </div>
    </Modal>
  );
};
