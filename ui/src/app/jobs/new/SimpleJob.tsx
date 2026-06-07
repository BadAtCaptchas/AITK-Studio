'use client';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  modelArchs,
  ModelArch,
  groupedModelOptions,
  quantizationOptions,
  defaultQtype,
  jobTypeOptions,
  SampleTags,
} from './options';
import { defaultDatasetConfig } from './jobConfig';
import {
  ComfyMode,
  ComfyOnError,
  GenerationBackend,
  GroupedSelectOption,
  JobConfig,
  SelectOption,
  TrainingPhaseConfig,
} from '@/types';
import { objectCopy, tagsToObj, objToTags } from '@/utils/basic';
import {
  TextInput,
  TextAreaInput,
  SelectInput,
  Checkbox,
  FormGroup,
  NumberInput,
  SliderInput,
} from '@/components/formInputs';
import Card from '@/components/Card';
import { X, Copy, Loader2, Shuffle, Upload } from 'lucide-react';
import AddSingleImageModal, { openAddImageModal } from '@/components/AddSingleImageModal';
import SampleControlImage from '@/components/SampleControlImage';
import { FlipHorizontal2, FlipVertical2 } from 'lucide-react';
import { handleModelArchChange } from './utils';
import { applySelectedDatasetDefaults } from '@/utils/jobDatasetDefaults';
import { IoFlaskSharp } from 'react-icons/io5';
import { isMac } from '@/helpers/basic';
import { getLayerOffloadingMemoryProfile } from '@/utils/memoryProfiles';
import TrainingPhasesEditor from './TrainingPhasesEditor';
import { apiClient } from '@/utils/api';
import { getRememberedEncryptedDatasetKey } from '@/utils/encryptedDatasets';
import { parseRemoteDatasetRef } from '@/utils/remoteDatasetRefs';

type Props = {
  jobConfig: JobConfig;
  setJobConfig: (value: any, key: string) => void;
  status: 'idle' | 'saving' | 'success' | 'error';
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  runId: string | null;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  datasetOptions: Array<
    SelectOption & { encrypted?: boolean; name?: string; source?: 'local' | 'remote'; worker_id?: string; ref?: string }
  >;
  isLoading?: boolean;
  comfyAutoInstall?: boolean;
};

const isDev = process.env.NODE_ENV === 'development';
const segaDistillArchs = new Set(['flux2', 'flux2_klein_4b', 'flux2_klein_9b', 'zimage']);
const layerOffloadingBackendOptions: SelectOption[] = [
  { value: 'block', label: 'Block' },
  { value: 'legacy', label: 'Legacy' },
];

const generationBackendOptions: SelectOption[] = [
  { value: 'native', label: 'Native' },
  { value: 'comfy', label: 'ComfyUI' },
];

const comfyModeOptions: SelectOption[] = [
  { value: 'external', label: 'External' },
  { value: 'managed', label: 'Managed' },
];

const comfyOnErrorOptions: SelectOption[] = [
  { value: 'fail', label: 'Fail' },
  { value: 'native', label: 'Native fallback' },
  { value: 'skip', label: 'Skip' },
];

export default function SimpleJob({
  jobConfig,
  setJobConfig,
  handleSubmit,
  status,
  runId,
  gpuIDs,
  setGpuIDs,
  gpuList,
  datasetOptions,
  isLoading,
  comfyAutoInstall = false,
}: Props) {
  const [randomPromptLoadingIndex, setRandomPromptLoadingIndex] = useState<number | null>(null);
  const [encryptedKeyRefreshKey, setEncryptedKeyRefreshKey] = useState(0);
  const baseLoraFileInputRef = useRef<HTMLInputElement | null>(null);
  const [baseLoraUploadStatus, setBaseLoraUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [baseLoraUploadMessage, setBaseLoraUploadMessage] = useState('');

  const modelArch = useMemo(() => {
    return modelArchs.find(a => a.name === jobConfig.config.process[0].model.arch) as ModelArch;
  }, [jobConfig.config.process[0].model.arch]);

  const jobType = useMemo(() => {
    return jobTypeOptions.find(j => j.value === jobConfig.config.process[0].type);
  }, [jobConfig.config.process[0].type]);

  const disableSections = useMemo(() => {
    let sections: string[] = [];
    if (modelArch?.disableSections) {
      sections = sections.concat(modelArch.disableSections);
    }
    if (jobType?.disableSections) {
      sections = sections.concat(jobType.disableSections);
    }
    return sections;
  }, [modelArch, jobType]);

  const isVideoModel = !!(modelArch?.group === 'video');
  const networkType = jobConfig.config.process[0].network?.type ?? 'lora';
  const supportsNormalNetworkDropout = networkType !== 'lokr';
  const isAudioModel = !!(modelArch?.group === 'audio');
  const autoTrain = !!jobConfig.config.process[0].train.auto_train;
  const trainConfig = jobConfig.config.process[0].train;
  const modelArchName = jobConfig.config.process[0].model.arch;
  const supportsSegaDistill = segaDistillArchs.has(modelArchName.split(':')[0]);
  const segaDistillEnabled = !!trainConfig.sega_distill;
  const canEnableSegaDistill = supportsSegaDistill && networkType === 'lora';
  const showSegaDistill = supportsSegaDistill || segaDistillEnabled;

  const setSegaDefaults = () => {
    if (trainConfig.sega_distill_weight === undefined) setJobConfig(1.0, 'config.process[0].train.sega_distill_weight');
    if (trainConfig.sega_distill_base_resolution === undefined) {
      setJobConfig(1024, 'config.process[0].train.sega_distill_base_resolution');
    }
    if (trainConfig.sega_distill_strength === undefined) setJobConfig(1.0, 'config.process[0].train.sega_distill_strength');
    if (trainConfig.sega_distill_min_scale === undefined) setJobConfig(0.5, 'config.process[0].train.sega_distill_min_scale');
    if (trainConfig.sega_distill_max_scale === undefined) setJobConfig(2.0, 'config.process[0].train.sega_distill_max_scale');
    if (trainConfig.sega_distill_on_reg === undefined) setJobConfig(false, 'config.process[0].train.sega_distill_on_reg');
  };

  const handleSegaDistillToggle = (enabled: boolean) => {
    if (enabled && !canEnableSegaDistill) return;
    setJobConfig(enabled, 'config.process[0].train.sega_distill');
    if (!enabled) return;
    setSegaDefaults();
    setJobConfig(false, 'config.process[0].train.diff_output_preservation');
    setJobConfig(false, 'config.process[0].train.blank_prompt_preservation');
    setJobConfig(undefined, 'config.process[0].train.do_differential_guidance');
    setJobConfig(undefined, 'config.process[0].train.differential_guidance_scale');
    setJobConfig(undefined, 'config.process[0].train.do_guidance_loss');
  };

  const handleTrainingStepsChange = (value: number | null) => {
    if (autoTrain) return;
    const requestedSteps = Math.max(1, Number(value ?? 1));
    const phases = jobConfig.config.process[0].train.phases;
    if (!phases?.length) {
      setJobConfig(requestedSteps, 'config.process[0].train.steps');
      return;
    }

    const nextPhases: TrainingPhaseConfig[] = phases.map(phase => ({
      ...phase,
      optimizer_params: phase.optimizer_params ? { ...phase.optimizer_params } : undefined,
      lr_scheduler_params: phase.lr_scheduler_params ? { ...phase.lr_scheduler_params } : undefined,
      auto_advance: phase.auto_advance ? { ...phase.auto_advance } : undefined,
    }));
    const currentTotal = nextPhases.reduce((sum, phase) => sum + Math.max(1, Number(phase.steps) || 1), 0);
    const lastPhase = nextPhases[nextPhases.length - 1];
    lastPhase.steps = Math.max(1, Math.round((Number(lastPhase.steps) || 1) + requestedSteps - currentTotal));
    const synchronizedTotal = nextPhases.reduce((sum, phase) => sum + Math.max(1, Number(phase.steps) || 1), 0);
    setJobConfig(nextPhases, 'config.process[0].train.phases');
    setJobConfig(synchronizedTotal, 'config.process[0].train.steps');
  };

  const taggedSampleArr: Record<string, any>[] | null = useMemo(() => {
    if (!modelArch) return null;
    if (!modelArch.sampleTags) return null;
    if (!jobConfig.config.process[0].sample.samples) return null;
    let sampleArr: any[] = [];
    for (let i = 0; i < jobConfig.config.process[0].sample.samples.length; i++) {
      const taggedPrompt = jobConfig.config.process[0].sample.samples[i].prompt;
      const tagsObj = tagsToObj(taggedPrompt);
      sampleArr.push(tagsObj);
    }
    return sampleArr;
  }, [modelArch, jobConfig.config.process[0].sample.samples]);

  const modelArchTagSections: SampleTags[] | null = useMemo(() => {
    if (!modelArch?.sampleTags) return null;
    const maxPerGroup = 5;
    let sections: SampleTags[] = [];
    let subSection: SampleTags = {};
    for (const [tagKey, tag] of Object.entries(modelArch.sampleTags)) {
      if ((tag.full && Object.keys(subSection).length > 0) || Object.keys(subSection).length >= maxPerGroup) {
        // reset the sub section build if the next tag is full or max per group is reached
        sections.push(subSection);
        subSection = {};
      }
      subSection[tagKey] = tag;
      if (tag.full) {
        // if the tag is full, push the section immediately and reset the sub section build
        sections.push(subSection);
        subSection = {};
      }
    }
    if (Object.keys(subSection).length > 0) {
      sections.push(subSection);
    }
    return sections.length > 0 ? sections : null;
  }, [modelArch]);

  const randomPromptDatasets = useMemo(() => {
    return jobConfig.config.process[0].datasets
      .map(dataset => {
        const option = datasetOptions.find(item => item.value === dataset.folder_path);
        const encrypted = option?.encrypted === true;
        const remote = option?.source === 'remote' || !!parseRemoteDatasetRef(dataset.folder_path);
        const keyB64 = encrypted
          ? getRememberedEncryptedDatasetKey(dataset.folder_path) ||
            (option?.name ? getRememberedEncryptedDatasetKey(option.name) : null)
          : null;

        return {
          folderPath: dataset.folder_path,
          captionExt: dataset.caption_ext,
          defaultCaption: dataset.default_caption,
          encrypted,
          remote,
          keyB64,
          label: option?.label || dataset.folder_path,
        };
      })
      .filter(dataset => dataset.folderPath && dataset.folderPath !== defaultDatasetConfig.folder_path && !dataset.remote);
  }, [datasetOptions, encryptedKeyRefreshKey, jobConfig.config.process[0].datasets]);

  const accessibleRandomPromptDatasets = useMemo(
    () => randomPromptDatasets.filter(dataset => !dataset.encrypted || dataset.keyB64),
    [randomPromptDatasets],
  );

  const randomPromptDisabledReason = useMemo(() => {
    if (randomPromptDatasets.length === 0) return 'Select a dataset before importing a random prompt';
    if (accessibleRandomPromptDatasets.length === 0) {
      return 'Unlock the encrypted dataset before importing random prompts';
    }
    return 'Import random prompt from dataset';
  }, [accessibleRandomPromptDatasets.length, randomPromptDatasets.length]);

  const canImportRandomPrompt = randomPromptDatasets.length > 0 && accessibleRandomPromptDatasets.length > 0;

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const refreshRememberedKeys = () => setEncryptedKeyRefreshKey(value => value + 1);
    window.addEventListener('focus', refreshRememberedKeys);
    document.addEventListener('visibilitychange', refreshRememberedKeys);
    return () => {
      window.removeEventListener('focus', refreshRememberedKeys);
      document.removeEventListener('visibilitychange', refreshRememberedKeys);
    };
  }, []);

  const setSamplePromptValue = (sampleIndex: number, prompt: string) => {
    if (modelArch?.sampleTags && taggedSampleArr?.[sampleIndex]) {
      const tagKey =
        (modelArch.sampleTags.CAPTION && 'CAPTION') ||
        (modelArch.sampleTags.PROMPT && 'PROMPT') ||
        Object.entries(modelArch.sampleTags).find(
          ([, tag]) => tag.type === 'text' || tag.type === 'multiline',
        )?.[0];

      if (tagKey) {
        setJobConfig(
          objToTags({ ...taggedSampleArr[sampleIndex], [tagKey]: prompt }),
          `config.process[0].sample.samples[${sampleIndex}].prompt`,
        );
        return;
      }
    }

    setJobConfig(prompt, `config.process[0].sample.samples[${sampleIndex}].prompt`);
  };

  const importRandomPromptFromDataset = async (sampleIndex: number) => {
    const datasets = accessibleRandomPromptDatasets.map(dataset => ({
      folderPath: dataset.folderPath,
      captionExt: dataset.captionExt,
      defaultCaption: dataset.defaultCaption,
    }));
    const encryptedDatasetKeys = accessibleRandomPromptDatasets
      .filter(dataset => dataset.encrypted && dataset.keyB64)
      .map(dataset => ({
        datasetPath: dataset.folderPath,
        keyB64: dataset.keyB64 as string,
      }));

    if (datasets.length === 0) {
      alert(randomPromptDisabledReason);
      return;
    }

    setRandomPromptLoadingIndex(sampleIndex);
    try {
      const response = await apiClient.post('/api/datasets/randomPrompt', { datasets, encryptedDatasetKeys });
      const prompt = typeof response.data?.prompt === 'string' ? response.data.prompt.trim() : '';
      if (!prompt) {
        alert('No captions were found in the configured datasets.');
        return;
      }
      setSamplePromptValue(sampleIndex, prompt);
    } catch (error: any) {
      alert(error?.response?.data?.error || 'Could not import a random prompt from the dataset.');
    } finally {
      setRandomPromptLoadingIndex(current => (current === sampleIndex ? null : current));
    }
  };

  const handleBaseLoraUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.safetensors')) {
      setBaseLoraUploadStatus('error');
      setBaseLoraUploadMessage('Base LoRA upload must be a .safetensors file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setBaseLoraUploadStatus('uploading');
    setBaseLoraUploadMessage('');
    try {
      const response = await apiClient.post('/api/generate/loras/upload', formData);
      const uploaded = response.data?.lora;
      if (!uploaded?.path) {
        throw new Error('Upload did not return a LoRA path.');
      }
      setJobConfig(uploaded.path, 'config.process[0].model.base_lora_path');
      const triggerWords = Array.isArray(uploaded.triggerWords) ? uploaded.triggerWords.filter(Boolean) : [];
      setBaseLoraUploadStatus('success');
      setBaseLoraUploadMessage(
        triggerWords.length > 0
          ? `Uploaded. Trigger metadata: ${triggerWords.join(', ')}`
          : 'Uploaded. No trigger metadata found.',
      );
    } catch (error: any) {
      setBaseLoraUploadStatus('error');
      setBaseLoraUploadMessage(error?.response?.data?.error || error?.message || 'Could not upload Base LoRA.');
    } finally {
      if (baseLoraFileInputRef.current) {
        baseLoraFileInputRef.current.value = '';
      }
    }
  };

  const numTrainingCols = useMemo(() => {
    let count = 4;
    if (!disableSections.includes('train.diff_output_preservation')) {
      count += 1;
    }
    if (showSegaDistill) {
      count += 1;
    }
    return count;
  }, [disableSections, showSegaDistill]);

  let trainingBarClass = 'grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4';

  if (numTrainingCols == 5) {
    trainingBarClass = 'grid grid-cols-1 gap-3 md:grid-cols-3 lg:grid-cols-5';
  }
  if (numTrainingCols == 6) {
    trainingBarClass = 'grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-6';
  }

  const transformerQuantizationOptions: GroupedSelectOption[] | SelectOption[] = useMemo(() => {
    const hasARA = modelArch?.accuracyRecoveryAdapters && Object.keys(modelArch.accuracyRecoveryAdapters).length > 0;
    if (!hasARA) {
      return quantizationOptions;
    }
    let newQuantizationOptions = [
      {
        label: 'Standard',
        options: [quantizationOptions[0], quantizationOptions[1]],
      },
    ];

    // add ARAs if they exist for the model
    let ARAs: SelectOption[] = [];
    if (modelArch.accuracyRecoveryAdapters) {
      for (const [label, value] of Object.entries(modelArch.accuracyRecoveryAdapters)) {
        ARAs.push({ value, label });
      }
    }
    if (ARAs.length > 0) {
      newQuantizationOptions.push({
        label: 'Accuracy Recovery Adapters',
        options: ARAs,
      });
    }

    let additionalQuantizationOptions: SelectOption[] = [];
    // add the quantization options if they are not already included
    for (let i = 2; i < quantizationOptions.length; i++) {
      const option = quantizationOptions[i];
      additionalQuantizationOptions.push(option);
    }
    if (additionalQuantizationOptions.length > 0) {
      newQuantizationOptions.push({
        label: 'Additional Quantization Options',
        options: additionalQuantizationOptions,
      });
    }
    return newQuantizationOptions;
  }, [modelArch]);

  const layerOffloadingMemoryProfile = useMemo(
    () => getLayerOffloadingMemoryProfile(jobConfig.config.process[0].model.arch),
    [jobConfig.config.process[0].model.arch],
  );

  const showGPUSelect = !isMac();

  let numDatasetCols = 4;
  let numSampleTopCols = 4;
  let datasetStyleClass = 'grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4';
  let sampleTopStyleClass = 'grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4';
  if (isVideoModel) {
    numSampleTopCols += 1;
  }
  if (isAudioModel) {
    numDatasetCols -= 1;
    numSampleTopCols -= 1;
  }
  if (numDatasetCols == 3) {
    datasetStyleClass = 'grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3';
  }
  if (numSampleTopCols == 5) {
    sampleTopStyleClass = 'grid grid-cols-1 gap-3 md:grid-cols-3 lg:grid-cols-5';
  }
  if (numSampleTopCols == 3) {
    sampleTopStyleClass = 'grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3';
  }
  return (
    <>
      <form
        onSubmit={handleSubmit}
        className={`relative space-y-4 ${isLoading ? 'pointer-events-none opacity-50' : ''}`}
      >
        <input
          ref={baseLoraFileInputRef}
          type="file"
          accept=".safetensors"
          className="hidden"
          onChange={event => {
            const file = event.target.files?.[0];
            if (file) void handleBaseLoraUpload(file);
          }}
        />
        {isLoading && (
          <div className="absolute inset-0 z-50 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-gray-400 border-t-blue-500" />
              <span className="text-sm text-gray-400">Loading...</span>
            </div>
          </div>
        )}
        <Card title="Setup">
          <div className="grid grid-cols-1 gap-x-6 gap-y-4 lg:grid-cols-[1fr_1.4fr_1fr]">
            <section className="min-w-0 space-y-2">
              <h3 className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Job</h3>
              <TextInput
                label="Training Name"
                value={jobConfig.config.name}
                docKey="config.name"
                onChange={value => setJobConfig(value, 'config.name')}
                placeholder="Enter training name"
                disabled={runId !== null}
                required
              />
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
                {showGPUSelect && (
                  <SelectInput
                    label="GPU ID"
                    value={`${gpuIDs}`}
                    docKey="gpuids"
                    onChange={value => setGpuIDs(value)}
                    options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
                  />
                )}
                {disableSections.includes('trigger_word') ? null : (
                  <TextInput
                    label="Trigger Word"
                    value={jobConfig.config.process[0].trigger_word || ''}
                    docKey="config.process[0].trigger_word"
                    onChange={(value: string | null) => {
                      if (value?.trim() === '') {
                        value = null;
                      }
                      setJobConfig(value, 'config.process[0].trigger_word');
                    }}
                    placeholder=""
                    required
                  />
                )}
              </div>
            </section>

            <section className="min-w-0 space-y-2 border-t border-gray-800 pt-3 lg:border-l lg:border-t-0 lg:pl-6 lg:pt-0">
              <h3 className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Model</h3>
              <div className="grid grid-cols-1 gap-2 xl:grid-cols-[0.8fr_1.2fr]">
                <SelectInput
                  label="Model Architecture"
                  value={jobConfig.config.process[0].model.arch}
                  onChange={value => {
                    handleModelArchChange(jobConfig.config.process[0].model.arch, value, jobConfig, setJobConfig);
                  }}
                  options={groupedModelOptions}
                />
                <TextInput
                  label="Name or Path"
                  value={jobConfig.config.process[0].model.name_or_path}
                  docKey="config.process[0].model.name_or_path"
                  onChange={(value: string | null) => {
                    if (value?.trim() === '') {
                      value = null;
                    }
                    setJobConfig(value, 'config.process[0].model.name_or_path');
                  }}
                  placeholder=""
                  required
                />
              </div>
              {modelArch?.additionalSections?.includes('model.assistant_lora_path') && (
                <TextInput
                  label="Training Adapter Path"
                  value={jobConfig.config.process[0].model.assistant_lora_path ?? ''}
                  docKey="config.process[0].model.assistant_lora_path"
                  onChange={(value: string | undefined) => {
                    if (value?.trim() === '') {
                      value = undefined;
                    }
                    setJobConfig(value, 'config.process[0].model.assistant_lora_path');
                  }}
                  placeholder=""
                />
              )}
              <div className="grid grid-cols-1 gap-2 xl:grid-cols-[1.2fr_0.45fr]">
                <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2">
                  <TextInput
                    label="Base LoRA Path"
                    value={jobConfig.config.process[0].model.base_lora_path ?? ''}
                    docKey="config.process[0].model.base_lora_path"
                    onChange={(value: string | undefined) => {
                      if (value?.trim() === '') {
                        value = undefined;
                      }
                      setJobConfig(value, 'config.process[0].model.base_lora_path');
                    }}
                    placeholder=""
                  />
                  <button
                    type="button"
                    className="operator-icon-button mt-7 h-9 w-9"
                    onClick={() => baseLoraFileInputRef.current?.click()}
                    disabled={baseLoraUploadStatus === 'uploading'}
                    title="Upload Base LoRA"
                    aria-label="Upload Base LoRA"
                  >
                    {baseLoraUploadStatus === 'uploading' ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Upload className="h-4 w-4" />
                    )}
                  </button>
                </div>
                <NumberInput
                  label="Strength"
                  value={jobConfig.config.process[0].model.base_lora_strength ?? 1.0}
                  onChange={value => setJobConfig(value ?? 1.0, 'config.process[0].model.base_lora_strength')}
                  placeholder="eg. 1.0"
                  required
                />
              </div>
              {baseLoraUploadMessage && (
                <div
                  className={`border px-3 py-2 text-xs ${
                    baseLoraUploadStatus === 'error'
                      ? 'border-red-900 bg-red-950/30 text-red-300'
                      : 'border-gray-800 bg-gray-950 text-gray-300'
                  }`}
                >
                  {baseLoraUploadMessage}
                </div>
              )}
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 pt-1">
                {modelArch?.additionalSections?.includes('model.low_vram') && (
                  <Checkbox
                    label="Low VRAM"
                    checked={jobConfig.config.process[0].model.low_vram}
                    onChange={value => {
                      setJobConfig(value, 'config.process[0].model.low_vram');
                      if (value) {
                        setJobConfig(true, 'config.process[0].sample.keep_low_vram_for_samples');
                      }
                    }}
                  />
                )}
                {modelArch?.additionalSections?.includes('model.qie.match_target_res') && (
                  <Checkbox
                    label="Match Target Res"
                    docKey="model.qie.match_target_res"
                    checked={jobConfig.config.process[0].model.model_kwargs.match_target_res}
                    onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.match_target_res')}
                  />
                )}
                {modelArch?.additionalSections?.includes('model.layer_offloading') && !isMac() && (
                  <Checkbox
                    label={
                      <>
                        Layer Offloading <IoFlaskSharp className="inline text-yellow-500" name="Experimental" />{' '}
                      </>
                    }
                    checked={jobConfig.config.process[0].model.layer_offloading || false}
                    onChange={value => {
                      setJobConfig(value, 'config.process[0].model.layer_offloading');
                      if (value) {
                        const model = jobConfig.config.process[0].model;
                        if (model.layer_offloading_backend === undefined) {
                          setJobConfig(layerOffloadingMemoryProfile.backend, 'config.process[0].model.layer_offloading_backend');
                        }
                        if (model.layer_offloading_transformer_percent === undefined) {
                          setJobConfig(
                            layerOffloadingMemoryProfile.transformerPercent,
                            'config.process[0].model.layer_offloading_transformer_percent',
                          );
                        }
                        if (model.layer_offloading_text_encoder_percent === undefined) {
                          setJobConfig(
                            layerOffloadingMemoryProfile.textEncoderPercent,
                            'config.process[0].model.layer_offloading_text_encoder_percent',
                          );
                        }
                      }
                    }}
                    docKey="model.layer_offloading"
                  />
                )}
              </div>
              {modelArch?.additionalSections?.includes('model.layer_offloading') &&
                !isMac() &&
                jobConfig.config.process[0].model.layer_offloading && (
                  <div className="grid grid-cols-1 gap-2 pt-2 sm:grid-cols-3">
                    <SelectInput
                      label="Offload Backend"
                      value={jobConfig.config.process[0].model.layer_offloading_backend ?? layerOffloadingMemoryProfile.backend}
                      onChange={value => setJobConfig(value, 'config.process[0].model.layer_offloading_backend')}
                      options={layerOffloadingBackendOptions}
                    />
                    <SliderInput
                      label="Transformer Offload %"
                      value={Math.round(
                        (jobConfig.config.process[0].model.layer_offloading_transformer_percent ?? 1) * 100,
                      )}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].model.layer_offloading_transformer_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                    />
                    <SliderInput
                      label="Text Encoder Offload %"
                      value={Math.round(
                        (jobConfig.config.process[0].model.layer_offloading_text_encoder_percent ?? 1) * 100,
                      )}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].model.layer_offloading_text_encoder_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                    />
                  </div>
                )}
            </section>

            <section className="min-w-0 space-y-2 border-t border-gray-800 pt-3 lg:border-l lg:border-t-0 lg:pl-6 lg:pt-0">
              <h3 className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Target</h3>
              <SelectInput
                label="Target Type"
                value={networkType}
                onChange={value => {
                  setJobConfig(value, 'config.process[0].network.type');
                  if (value === 'lokr') {
                    setJobConfig(undefined, 'config.process[0].network.dropout');
                    if (jobConfig.config.process[0].train.sega_distill) {
                      setJobConfig(false, 'config.process[0].train.sega_distill');
                    }
                  }
                }}
                options={[
                  { value: 'lora', label: 'LoRA' },
                  { value: 'lokr', label: 'LoKr' },
                ]}
              />
              {networkType == 'lokr' && (
                <SelectInput
                  label="LoKr Factor"
                  value={`${jobConfig.config.process[0].network?.lokr_factor ?? -1}`}
                  onChange={value => setJobConfig(parseInt(value), 'config.process[0].network.lokr_factor')}
                  options={[
                    { value: '-1', label: 'Auto' },
                    { value: '4', label: '4' },
                    { value: '8', label: '8' },
                    { value: '16', label: '16' },
                    { value: '32', label: '32' },
                  ]}
                />
              )}
              {networkType == 'lora' && (
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
                  <NumberInput
                    label="Linear Rank"
                    value={jobConfig.config.process[0].network?.linear ?? null}
                    onChange={value => {
                      console.log('onChange', value);
                      setJobConfig(value, 'config.process[0].network.linear');
                      setJobConfig(value, 'config.process[0].network.linear_alpha');
                    }}
                    placeholder="eg. 16"
                    min={1}
                    max={1024}
                    required
                  />
                  {disableSections.includes('network.conv') ? null : (
                    <NumberInput
                      label="Conv Rank"
                      value={jobConfig.config.process[0].network?.conv ?? null}
                      onChange={value => {
                        console.log('onChange', value);
                        setJobConfig(value, 'config.process[0].network.conv');
                        setJobConfig(value, 'config.process[0].network.conv_alpha');
                      }}
                      placeholder="eg. 16"
                      min={0}
                      max={1024}
                    />
                  )}
                </div>
              )}
              {supportsNormalNetworkDropout && (
                <NumberInput
                  label="Network Dropout"
                  value={jobConfig.config.process[0].network?.dropout ?? null}
                  onChange={value => setJobConfig(value ?? undefined, 'config.process[0].network.dropout')}
                  placeholder="eg. 0.05"
                  min={0}
                  max={1}
                />
              )}
            </section>
          </div>
        </Card>

        <Card title="Model runtime and saving" collapsible>
          <div className="grid grid-cols-1 gap-x-6 gap-y-4 md:grid-cols-2 xl:grid-cols-4">
            {disableSections.includes('model.quantize') ? null : (
              <section className="min-w-0 space-y-2">
                <h3 className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Quantization</h3>
                <SelectInput
                  label="Transformer"
                  value={jobConfig.config.process[0].model.quantize ? jobConfig.config.process[0].model.qtype : ''}
                  onChange={value => {
                    if (value === '') {
                      setJobConfig(false, 'config.process[0].model.quantize');
                      value = defaultQtype;
                    } else {
                      setJobConfig(true, 'config.process[0].model.quantize');
                    }
                    setJobConfig(value, 'config.process[0].model.qtype');
                  }}
                  options={transformerQuantizationOptions}
                />
                {!disableSections.includes('model.quantize_te') && (
                  <SelectInput
                    label="Text Encoder"
                    value={
                      jobConfig.config.process[0].model.quantize_te ? jobConfig.config.process[0].model.qtype_te : ''
                    }
                    onChange={value => {
                      if (value === '') {
                        setJobConfig(false, 'config.process[0].model.quantize_te');
                        value = defaultQtype;
                      } else {
                        setJobConfig(true, 'config.process[0].model.quantize_te');
                      }
                      setJobConfig(value, 'config.process[0].model.qtype_te');
                    }}
                    options={quantizationOptions}
                  />
                )}
              </section>
            )}
            {modelArch?.additionalSections?.includes('model.multistage') && (
              <section className="min-w-0 space-y-2">
                <h3 className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Multistage</h3>
                <FormGroup label="Stages to Train" docKey={'model.multistage'}>
                  <Checkbox
                    label="High Noise"
                    checked={jobConfig.config.process[0].model.model_kwargs?.train_high_noise || false}
                    onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.train_high_noise')}
                  />
                  <Checkbox
                    label="Low Noise"
                    checked={jobConfig.config.process[0].model.model_kwargs?.train_low_noise || false}
                    onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.train_low_noise')}
                  />
                </FormGroup>
                <NumberInput
                  label="Switch Every"
                  value={jobConfig.config.process[0].train.switch_boundary_every}
                  onChange={value => setJobConfig(value, 'config.process[0].train.switch_boundary_every')}
                  placeholder="eg. 1"
                  docKey={'train.switch_boundary_every'}
                  min={1}
                  required
                />
              </section>
            )}
            {!disableSections.includes('slider') && (
              <section className="min-w-0 space-y-2">
                <h3 className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Slider</h3>
                <TextInput
                  label="Target Class"
                  value={jobConfig.config.process[0].slider?.target_class ?? ''}
                  onChange={value => setJobConfig(value, 'config.process[0].slider.target_class')}
                  placeholder="eg. person"
                />
                <TextInput
                  label="Positive Prompt"
                  value={jobConfig.config.process[0].slider?.positive_prompt ?? ''}
                  onChange={value => setJobConfig(value, 'config.process[0].slider.positive_prompt')}
                  placeholder="eg. person who is happy"
                />
                <TextInput
                  label="Negative Prompt"
                  value={jobConfig.config.process[0].slider?.negative_prompt ?? ''}
                  onChange={value => setJobConfig(value, 'config.process[0].slider.negative_prompt')}
                  placeholder="eg. person who is sad"
                />
                <TextInput
                  label="Anchor Class"
                  value={jobConfig.config.process[0].slider?.anchor_class ?? ''}
                  onChange={value => setJobConfig(value, 'config.process[0].slider.anchor_class')}
                  placeholder=""
                />
              </section>
            )}
            <section className="min-w-0 space-y-2">
              <h3 className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Saving</h3>
              <SelectInput
                label="Data Type"
                value={jobConfig.config.process[0].save.dtype}
                onChange={value => setJobConfig(value, 'config.process[0].save.dtype')}
                options={[
                  { value: 'bf16', label: 'BF16' },
                  { value: 'fp16', label: 'FP16' },
                  { value: 'fp32', label: 'FP32' },
                ]}
              />
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-1">
                <NumberInput
                  label="Save Every"
                  value={jobConfig.config.process[0].save.save_every}
                  onChange={value => setJobConfig(value, 'config.process[0].save.save_every')}
                  placeholder="eg. 250"
                  min={1}
                  required
                />
                <NumberInput
                  label="Max Step Saves to Keep"
                  value={jobConfig.config.process[0].save.max_step_saves_to_keep}
                  onChange={value => setJobConfig(value, 'config.process[0].save.max_step_saves_to_keep')}
                  placeholder="eg. 4"
                  min={1}
                  required
                />
              </div>
            </section>
          </div>
        </Card>
        <div>
          <Card title="Training">
            <div className={trainingBarClass}>
              <div>
                <NumberInput
                  label="Batch Size"
                  value={jobConfig.config.process[0].train.batch_size}
                  onChange={value => setJobConfig(value, 'config.process[0].train.batch_size')}
                  placeholder="eg. 4"
                  min={1}
                  required
                />
                <NumberInput
                  label="Gradient Accumulation"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.gradient_accumulation}
                  onChange={value => setJobConfig(value, 'config.process[0].train.gradient_accumulation')}
                  placeholder="eg. 1"
                  min={1}
                  required
                />
                {!autoTrain && (
                  <NumberInput
                    label="Steps"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.steps}
                    onChange={handleTrainingStepsChange}
                    placeholder="eg. 2000"
                    min={1}
                    required
                  />
                )}
              </div>
              <div>
                <SelectInput
                  label="Optimizer"
                  value={jobConfig.config.process[0].train.optimizer}
                  onChange={value => setJobConfig(value, 'config.process[0].train.optimizer')}
                  options={[
                    { value: 'adafactor', label: 'Adafactor' },
                    { value: 'adam', label: 'Adam' },
                    { value: 'adamw', label: 'AdamW' },
                    { value: 'adamw8bit', label: 'AdamW8Bit' },
                    { value: 'automagic', label: 'Automagic' },
                    { value: 'automagic2', label: 'Automagic v2' },
                    { value: 'prodigyopt', label: 'Prodigy' },
                    { value: 'prodigy8bit', label: 'Prodigy8Bit' },
                  ]}
                />
                <NumberInput
                  label="Learning Rate"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.lr}
                  onChange={value => setJobConfig(value, 'config.process[0].train.lr')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                />
                <NumberInput
                  label="Weight Decay"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.optimizer_params.weight_decay}
                  onChange={value => setJobConfig(value, 'config.process[0].train.optimizer_params.weight_decay')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                />
              </div>
              <div>
                {disableSections.includes('train.timestep_type') ? null : (
                  <SelectInput
                    label="Timestep Type"
                    value={jobConfig.config.process[0].train.timestep_type}
                    disabled={disableSections.includes('train.timestep_type') || false}
                    onChange={value => setJobConfig(value, 'config.process[0].train.timestep_type')}
                    options={[
                      { value: 'sigmoid', label: 'Sigmoid' },
                      { value: 'linear', label: 'Linear' },
                      { value: 'shift', label: 'Shift' },
                      { value: 'weighted', label: 'Weighted' },
                    ]}
                  />
                )}
                <SelectInput
                  label="Timestep Bias"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.content_or_style}
                  onChange={value => setJobConfig(value, 'config.process[0].train.content_or_style')}
                  options={[
                    { value: 'balanced', label: 'Balanced' },
                    { value: 'content', label: 'High Noise' },
                    { value: 'style', label: 'Low Noise' },
                  ]}
                />
                <SelectInput
                  label="Loss Type"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.loss_type}
                  onChange={value => setJobConfig(value, 'config.process[0].train.loss_type')}
                  options={[
                    { value: 'mse', label: 'Mean Squared Error' },
                    { value: 'mae', label: 'Mean Absolute Error' },
                    { value: 'wavelet', label: 'Wavelet' },
                    { value: 'stepped', label: 'Stepped Recovery' },
                  ]}
                />
                {modelArch?.additionalSections?.includes('train.audio_loss_multiplier') && (
                  <NumberInput
                    label="Audio Loss Multiplier"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.audio_loss_multiplier ?? 1.0}
                    onChange={value => setJobConfig(value, 'config.process[0].train.audio_loss_multiplier')}
                    placeholder="eg. 1.0"
                    docKey={'train.audio_loss_multiplier'}
                    min={0}
                  />
                )}
              </div>
              <div>
                <FormGroup label="EMA (Exponential Moving Average)">
                  <Checkbox
                    label="Use EMA"
                    className="pt-1"
                    checked={jobConfig.config.process[0].train.ema_config?.use_ema || false}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.use_ema')}
                  />
                </FormGroup>
                {jobConfig.config.process[0].train.ema_config?.use_ema && (
                  <NumberInput
                    label="EMA Decay"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.ema_config?.ema_decay as number}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.ema_decay')}
                    placeholder="eg. 0.99"
                    min={0}
                  />
                )}

                <FormGroup label="Text Encoder Optimizations" className="pt-2">
                  {!disableSections.includes('train.unload_text_encoder') && (
                    <Checkbox
                      label="Unload TE"
                      checked={jobConfig.config.process[0].train.unload_text_encoder || false}
                      docKey={'train.unload_text_encoder'}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.unload_text_encoder');
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.cache_text_embeddings');
                        }
                      }}
                    />
                  )}
                  <Checkbox
                    label="Cache Text Embeddings"
                    checked={jobConfig.config.process[0].train.cache_text_embeddings || false}
                    docKey={'train.cache_text_embeddings'}
                    onChange={value => {
                      setJobConfig(value, 'config.process[0].train.cache_text_embeddings');
                      if (value) {
                        setJobConfig(false, 'config.process[0].train.unload_text_encoder');
                      }
                    }}
                  />
                </FormGroup>
              </div>
              <div>
                {disableSections.includes('train.diff_output_preservation') ||
                disableSections.includes('train.blank_prompt_preservation') ? null : (
                  <FormGroup label="Regularization">
                    <></>
                  </FormGroup>
                )}
                {disableSections.includes('train.diff_output_preservation') ? null : (
                  <>
                    <Checkbox
                      label="Differential Output Preservation"
                      docKey={'train.diff_output_preservation'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.diff_output_preservation || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.diff_output_preservation');
                        if (value && jobConfig.config.process[0].train.blank_prompt_preservation) {
                          // only one can be enabled at a time
                          setJobConfig(false, 'config.process[0].train.blank_prompt_preservation');
                        }
                        if (value && jobConfig.config.process[0].train.sega_distill) {
                          setJobConfig(false, 'config.process[0].train.sega_distill');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.diff_output_preservation && (
                      <>
                        <NumberInput
                          label="DOP Loss Multiplier"
                          className="pt-2"
                          value={jobConfig.config.process[0].train.diff_output_preservation_multiplier as number}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.diff_output_preservation_multiplier')
                          }
                          placeholder="eg. 1.0"
                          min={0}
                        />
                        <TextInput
                          label="DOP Preservation Class"
                          className="pt-2 pb-4"
                          value={jobConfig.config.process[0].train.diff_output_preservation_class as string}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.diff_output_preservation_class')
                          }
                          placeholder="eg. woman"
                        />
                      </>
                    )}
                  </>
                )}
                {disableSections.includes('train.blank_prompt_preservation') ? null : (
                  <>
                    <Checkbox
                      label="Blank Prompt Preservation"
                      docKey={'train.blank_prompt_preservation'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.blank_prompt_preservation || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.blank_prompt_preservation');
                        if (value && jobConfig.config.process[0].train.diff_output_preservation) {
                          // only one can be enabled at a time
                          setJobConfig(false, 'config.process[0].train.diff_output_preservation');
                        }
                        if (value && jobConfig.config.process[0].train.sega_distill) {
                          setJobConfig(false, 'config.process[0].train.sega_distill');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.blank_prompt_preservation && (
                      <>
                        <NumberInput
                          label="BPP Loss Multiplier"
                          className="pt-2"
                          value={
                            (jobConfig.config.process[0].train.blank_prompt_preservation_multiplier as number) || 1.0
                          }
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.blank_prompt_preservation_multiplier')
                          }
                          placeholder="eg. 1.0"
                          min={0}
                        />
                      </>
                    )}
                  </>
                )}
              </div>
              {showSegaDistill && (
                <div>
                  <FormGroup label="SEGA Distillation" docKey="train.sega_distill">
                    <Checkbox
                      label="SEGA Distillation"
                      docKey="train.sega_distill"
                      checked={segaDistillEnabled}
                      disabled={!segaDistillEnabled && !canEnableSegaDistill}
                      onChange={handleSegaDistillToggle}
                    />
                  </FormGroup>
                  {segaDistillEnabled && (
                    <>
                      <NumberInput
                        label="Weight"
                        className="pt-2"
                        docKey="train.sega_distill_weight"
                        value={trainConfig.sega_distill_weight ?? 1.0}
                        onChange={value => setJobConfig(value ?? 1.0, 'config.process[0].train.sega_distill_weight')}
                        min={0.000001}
                      />
                      <NumberInput
                        label="Base Resolution"
                        className="pt-2"
                        docKey="train.sega_distill_base_resolution"
                        value={trainConfig.sega_distill_base_resolution ?? 1024}
                        onChange={value =>
                          setJobConfig(value ?? 1024, 'config.process[0].train.sega_distill_base_resolution')
                        }
                        min={1}
                      />
                      <NumberInput
                        label="Strength"
                        className="pt-2"
                        docKey="train.sega_distill_strength"
                        value={trainConfig.sega_distill_strength ?? 1.0}
                        onChange={value => setJobConfig(value ?? 1.0, 'config.process[0].train.sega_distill_strength')}
                        min={0}
                      />
                      <div className="grid grid-cols-2 gap-3 pt-2">
                        <NumberInput
                          label="Min Scale"
                          docKey="train.sega_distill_scale"
                          value={trainConfig.sega_distill_min_scale ?? 0.5}
                          onChange={value => setJobConfig(value ?? 0.5, 'config.process[0].train.sega_distill_min_scale')}
                          min={0.01}
                        />
                        <NumberInput
                          label="Max Scale"
                          docKey="train.sega_distill_scale"
                          value={trainConfig.sega_distill_max_scale ?? 2.0}
                          onChange={value => setJobConfig(value ?? 2.0, 'config.process[0].train.sega_distill_max_scale')}
                          min={0.01}
                        />
                      </div>
                      <Checkbox
                        label="Apply To Reg"
                        className="pt-3"
                        checked={trainConfig.sega_distill_on_reg ?? false}
                        onChange={value => setJobConfig(value, 'config.process[0].train.sega_distill_on_reg')}
                        docKey="train.sega_distill_on_reg"
                      />
                    </>
                  )}
                </div>
              )}
            </div>
            <TrainingPhasesEditor
              train={jobConfig.config.process[0].train}
              network={jobConfig.config.process[0].network}
              currentArch={jobConfig.config.process[0].model.arch}
              setJobConfig={setJobConfig}
              disableTimestepType={disableSections.includes('train.timestep_type')}
              modelArchName={modelArch?.name}
              defaultAutoTrainingProfileId={modelArch?.defaultAutoTrainingProfileId}
            />
          </Card>
        </div>
        <div>
          <Card title="Advanced" collapsible>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <Checkbox
                  label="Do Differential Guidance"
                  docKey={'train.do_differential_guidance'}
                  className="pt-1"
                  checked={jobConfig.config.process[0].train.do_differential_guidance || false}
                  onChange={value => {
                    let newValue = value == false ? undefined : value;
                    setJobConfig(newValue, 'config.process[0].train.do_differential_guidance');
                    if (!newValue) {
                      setJobConfig(undefined, 'config.process[0].train.differential_guidance_scale');
                    } else if (
                      jobConfig.config.process[0].train.differential_guidance_scale === undefined ||
                      jobConfig.config.process[0].train.differential_guidance_scale === null
                    ) {
                      // set default differential guidance scale to 3.0
                      setJobConfig(3.0, 'config.process[0].train.differential_guidance_scale');
                    }
                    if (newValue && jobConfig.config.process[0].train.sega_distill) {
                      setJobConfig(false, 'config.process[0].train.sega_distill');
                    }
                  }}
                />
                {jobConfig.config.process[0].train.differential_guidance_scale && (
                  <>
                    <NumberInput
                      label="Differential Guidance Scale"
                      className="pt-2"
                      value={(jobConfig.config.process[0].train.differential_guidance_scale as number) || 3.0}
                      onChange={value => setJobConfig(value, 'config.process[0].train.differential_guidance_scale')}
                      placeholder="eg. 3.0"
                      min={0}
                    />
                  </>
                )}
              </div>
            </div>
          </Card>
        </div>
        <div>
          <Card title="Datasets">
            <>
              {jobConfig.config.process[0].datasets.map((dataset, i) => (
                <div key={i} className="relative border border-gray-800 bg-gray-950/50 p-3">
                  <div className="absolute top-2 right-2 flex gap-1">
                    <button
                      type="button"
                      onClick={() => {
                        const duplicated = objectCopy(dataset);
                        const datasets = [...jobConfig.config.process[0].datasets];
                        datasets.splice(i + 1, 0, duplicated);
                        setJobConfig(datasets, 'config.process[0].datasets');
                      }}
                      className="operator-icon-button"
                      title="Duplicate Dataset"
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        setJobConfig(
                          jobConfig.config.process[0].datasets.filter((_, index) => index !== i),
                          'config.process[0].datasets',
                        )
                      }
                      className="operator-icon-button hover:border-rose-800 hover:bg-rose-950/60 hover:text-rose-100"
                      title="Remove Dataset"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                  <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-gray-400">Dataset {i + 1}</h2>
                  <div className={datasetStyleClass}>
                    <div>
                      <SelectInput
                        label="Target Dataset"
                        value={dataset.folder_path}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].folder_path`)}
                        options={datasetOptions}
                      />
                      {modelArch?.additionalSections?.includes('datasets.control_path') && (
                        <SelectInput
                          label="Control Dataset"
                          docKey="datasets.control_path"
                          value={dataset.control_path ?? ''}
                          className="pt-2"
                          onChange={value =>
                            setJobConfig(value == '' ? null : value, `config.process[0].datasets[${i}].control_path`)
                          }
                          options={[{ value: '', label: ' ' }, ...datasetOptions]}
                        />
                      )}
                      {modelArch?.additionalSections?.includes('datasets.multi_control_paths') && (
                        <>
                          <SelectInput
                            label="Control Dataset 1"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_1 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_1`,
                              )
                            }
                            options={[{ value: '', label: ' ' }, ...datasetOptions]}
                          />
                          <SelectInput
                            label="Control Dataset 2"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_2 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_2`,
                              )
                            }
                            options={[{ value: '', label: ' ' }, ...datasetOptions]}
                          />
                          <SelectInput
                            label="Control Dataset 3"
                            docKey="datasets.multi_control_paths"
                            value={dataset.control_path_3 ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(
                                value == '' ? null : value,
                                `config.process[0].datasets[${i}].control_path_3`,
                              )
                            }
                            options={[{ value: '', label: ' ' }, ...datasetOptions]}
                          />
                        </>
                      )}
                      <NumberInput
                        label="LoRA Weight"
                        value={dataset.network_weight}
                        className="pt-2"
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].network_weight`)}
                        placeholder="eg. 1.0"
                      />
                      <NumberInput
                        label="Num Repeats"
                        value={dataset.num_repeats || 1}
                        className="pt-2"
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].num_repeats`)}
                        placeholder="eg. 1"
                        docKey={'dataset.num_repeats'}
                      />
                    </div>
                    <div>
                      <TextInput
                        label="Default Caption"
                        value={dataset.default_caption}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].default_caption`)}
                        placeholder="eg. A photo of a cat"
                      />
                      <NumberInput
                        label="Caption Dropout Rate"
                        className="pt-2"
                        value={dataset.caption_dropout_rate}
                        onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].caption_dropout_rate`)}
                        placeholder="eg. 0.05"
                        min={0}
                        required
                      />
                      {modelArch?.additionalSections?.includes('datasets.num_frames') && !dataset.auto_frame_count && (
                        <NumberInput
                          label="Num Frames"
                          className="pt-2"
                          docKey="datasets.num_frames"
                          value={dataset.num_frames}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].num_frames`)}
                          placeholder="eg. 41"
                          min={1}
                          required
                        />
                      )}
                    </div>
                    <div>
                      <FormGroup label="Settings" className="">
                        <Checkbox
                          label="Cache Latents"
                          checked={dataset.cache_latents_to_disk || false}
                          onChange={value =>
                            setJobConfig(value, `config.process[0].datasets[${i}].cache_latents_to_disk`)
                          }
                        />
                        <Checkbox
                          label="Is Regularization"
                          checked={dataset.is_reg || false}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].is_reg`)}
                        />
                        {modelArch?.additionalSections?.includes('datasets.auto_frame_count') && (
                          <Checkbox
                            label="Auto Frame Count"
                            checked={dataset.auto_frame_count || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].auto_frame_count`)}
                            docKey="datasets.auto_frame_count"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.do_i2v') && (
                          <Checkbox
                            label="Do I2V"
                            checked={dataset.do_i2v || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].do_i2v`)}
                            docKey="datasets.do_i2v"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.do_audio') && (
                          <Checkbox
                            label="Do Audio"
                            checked={dataset.do_audio || false}
                            onChange={value => {
                              if (!value) {
                                setJobConfig(undefined, `config.process[0].datasets[${i}].do_audio`);
                              } else {
                                setJobConfig(value, `config.process[0].datasets[${i}].do_audio`);
                              }
                            }}
                            docKey="datasets.do_audio"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.audio_normalize') && (
                          <Checkbox
                            label="Audio Normalize"
                            checked={dataset.audio_normalize || false}
                            onChange={value => {
                              if (!value) {
                                setJobConfig(undefined, `config.process[0].datasets[${i}].audio_normalize`);
                              } else {
                                setJobConfig(value, `config.process[0].datasets[${i}].audio_normalize`);
                              }
                            }}
                            docKey="datasets.audio_normalize"
                          />
                        )}
                        {modelArch?.additionalSections?.includes('datasets.audio_preserve_pitch') && (
                          <Checkbox
                            label="Audio Preserve Pitch"
                            checked={dataset.audio_preserve_pitch || false}
                            onChange={value => {
                              if (!value) {
                                setJobConfig(undefined, `config.process[0].datasets[${i}].audio_preserve_pitch`);
                              } else {
                                setJobConfig(value, `config.process[0].datasets[${i}].audio_preserve_pitch`);
                              }
                            }}
                            docKey="datasets.audio_preserve_pitch"
                          />
                        )}
                      </FormGroup>
                      {!isAudioModel && (
                        <FormGroup label="Flipping" docKey={'datasets.flip'} className="mt-2">
                          <Checkbox
                            label={
                              <>
                                Flip X <FlipHorizontal2 className="inline-block w-4 h-4 ml-1" />
                              </>
                            }
                            checked={dataset.flip_x || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_x`)}
                          />
                          <Checkbox
                            label={
                              <>
                                Flip Y <FlipVertical2 className="inline-block w-4 h-4 ml-1" />
                              </>
                            }
                            checked={dataset.flip_y || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_y`)}
                          />
                        </FormGroup>
                      )}
                    </div>
                    {!isAudioModel && (
                      <div>
                        <FormGroup label="Resolutions" className="pt-2">
                          <div className="grid grid-cols-2 gap-2">
                            {[
                              [256, 512, 768, 1024],
                              [1280, 1328, 1536, 2048],
                            ].map(resGroup => (
                              <div key={resGroup[0]} className="space-y-2">
                                {resGroup.map(res => (
                                  <Checkbox
                                    key={res}
                                    label={res.toString()}
                                    checked={dataset.resolution.includes(res)}
                                    onChange={value => {
                                      const resolutions = dataset.resolution.includes(res)
                                        ? dataset.resolution.filter(r => r !== res)
                                        : [...dataset.resolution, res];
                                      setJobConfig(resolutions, `config.process[0].datasets[${i}].resolution`);
                                    }}
                                  />
                                ))}
                              </div>
                            ))}
                          </div>
                        </FormGroup>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <button
                type="button"
                onClick={() => {
                  const newDataset = applySelectedDatasetDefaults(objectCopy(defaultDatasetConfig), modelArch?.defaults);
                  // automaticallt add the controls for a new dataset
                  const controls = modelArch?.controls ?? [];
                  newDataset.controls = controls;
                  setJobConfig([...jobConfig.config.process[0].datasets, newDataset], 'config.process[0].datasets');
                }}
                className="operator-button w-full"
              >
                Add Dataset
              </button>
            </>
          </Card>
        </div>
        <div>
          <Card title="Sample generation" collapsible>
            <div className={sampleTopStyleClass}>
              <div>
                <NumberInput
                  label="Sample Every"
                  value={jobConfig.config.process[0].sample.sample_every}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_every')}
                  placeholder="eg. 250"
                  min={1}
                  required
                />
                <SelectInput
                  label="Sampler"
                  className="pt-2"
                  value={jobConfig.config.process[0].sample.sampler}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sampler')}
                  options={[
                    { value: 'flowmatch', label: 'FlowMatch' },
                    { value: 'ddpm', label: 'DDPM' },
                  ]}
                />
                <NumberInput
                  label="Guidance Scale"
                  value={jobConfig.config.process[0].sample.guidance_scale}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.guidance_scale')}
                  placeholder="eg. 1.0"
                  className="pt-2"
                  min={0}
                  required
                />
                <NumberInput
                  label="Sample Steps"
                  value={jobConfig.config.process[0].sample.sample_steps}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_steps')}
                  placeholder="eg. 1"
                  className="pt-2"
                  min={1}
                  required
                />
              </div>

              {!isAudioModel && (
                <div>
                  <NumberInput
                    label="Width"
                    value={jobConfig.config.process[0].sample.width}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.width')}
                    placeholder="eg. 1024"
                    min={0}
                    required
                  />
                  <NumberInput
                    label="Height"
                    value={jobConfig.config.process[0].sample.height}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.height')}
                    placeholder="eg. 1024"
                    className="pt-2"
                    min={0}
                    required
                  />
                  {isVideoModel && (
                    <div>
                      <NumberInput
                        label="Num Frames"
                        value={jobConfig.config.process[0].sample.num_frames}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.num_frames')}
                        placeholder="eg. 0"
                        className="pt-2"
                        min={0}
                        required
                      />
                      <NumberInput
                        label="FPS"
                        value={jobConfig.config.process[0].sample.fps}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.fps')}
                        placeholder="eg. 0"
                        className="pt-2"
                        min={0}
                        required
                      />
                    </div>
                  )}
                </div>
              )}

              <div>
                <NumberInput
                  label="Seed"
                  value={jobConfig.config.process[0].sample.seed}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.seed')}
                  placeholder="eg. 0"
                  min={0}
                  required
                />
                <Checkbox
                  label="Walk Seed"
                  className="pt-4 pl-2"
                  checked={jobConfig.config.process[0].sample.walk_seed}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.walk_seed')}
                />
              </div>
              <div>
                <FormGroup label="Advanced Sampling" className="pt-2">
                  <div>
                    <Checkbox
                      label="Skip First Sample"
                      className="pt-4"
                      checked={jobConfig.config.process[0].train.skip_first_sample || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.skip_first_sample');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="Force First Sample"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.force_first_sample || false}
                      docKey={'train.force_first_sample'}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.force_first_sample');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.skip_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="Disable Sampling"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.disable_sampling || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.disable_sampling');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="Keep Low VRAM During Samples"
                      className="pt-1"
                      checked={jobConfig.config.process[0].sample.keep_low_vram_for_samples || false}
                      docKey={'sample.keep_low_vram_for_samples'}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].sample.keep_low_vram_for_samples');
                      }}
                    />
                  </div>
                </FormGroup>
              </div>
            </div>
            {!isAudioModel && (
              <FormGroup label="Backend" className="pt-2">
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4">
                  <SelectInput
                    label="Generation Backend"
                    value={jobConfig.config.process[0].sample.backend ?? 'native'}
                    onChange={value => {
                      setJobConfig(value as GenerationBackend, 'config.process[0].sample.backend');
                      if (value === 'comfy' && !jobConfig.config.process[0].sample.comfy) {
                        setJobConfig(
                          { mode: 'external', workflow_name: 'auto', on_error: 'fail' },
                          'config.process[0].sample.comfy',
                        );
                      }
                    }}
                    options={generationBackendOptions}
                  />
                  {jobConfig.config.process[0].sample.backend === 'comfy' && (
                    <>
                      <SelectInput
                        label="Comfy Mode"
                        value={jobConfig.config.process[0].sample.comfy?.mode ?? 'external'}
                        onChange={value => {
                          setJobConfig(value as ComfyMode, 'config.process[0].sample.comfy.mode');
                          if (value === 'managed' && comfyAutoInstall) {
                            setJobConfig(true, 'config.process[0].sample.comfy.managed_install');
                          }
                        }}
                        options={comfyModeOptions}
                      />
                      <SelectInput
                        label="On Error"
                        value={jobConfig.config.process[0].sample.comfy?.on_error ?? 'fail'}
                        onChange={value => setJobConfig(value as ComfyOnError, 'config.process[0].sample.comfy.on_error')}
                        options={comfyOnErrorOptions}
                      />
                      <TextInput
                        label="Workflow"
                        value={jobConfig.config.process[0].sample.comfy?.workflow_name ?? 'auto'}
                        onChange={value => setJobConfig(value || 'auto', 'config.process[0].sample.comfy.workflow_name')}
                        placeholder="auto"
                      />
                      {(jobConfig.config.process[0].sample.comfy?.mode ?? 'external') === 'external' ? (
                        <TextInput
                          label="Comfy URL"
                          value={jobConfig.config.process[0].sample.comfy?.server_url ?? ''}
                          onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.server_url')}
                          placeholder="http://127.0.0.1:8188"
                        />
                      ) : (
                        <>
                          <Checkbox
                            label="Install Managed ComfyUI"
                            checked={comfyAutoInstall || (jobConfig.config.process[0].sample.comfy?.managed_install ?? false)}
                            onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.managed_install')}
                            disabled={comfyAutoInstall}
                          />
                          <TextInput
                            label="Comfy Root"
                            value={jobConfig.config.process[0].sample.comfy?.root ?? ''}
                            onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.root')}
                            placeholder=".aitk_comfy/ComfyUI"
                          />
                        </>
                      )}
                      <TextInput
                        label="Workflow JSON"
                        value={
                          typeof jobConfig.config.process[0].sample.comfy?.workflow === 'string'
                            ? jobConfig.config.process[0].sample.comfy?.workflow
                            : ''
                        }
                        onChange={value => setJobConfig(value || undefined, 'config.process[0].sample.comfy.workflow')}
                        placeholder="optional path"
                      />
                    </>
                  )}
                </div>
              </FormGroup>
            )}
            <FormGroup label={`Sample Prompts (${jobConfig.config.process[0].sample.samples.length})`} className="pt-2">
              <div></div>
            </FormGroup>
            {jobConfig.config.process[0].sample.samples.map((sample, i) => (
              <div key={i} className="mb-3 border border-gray-800 bg-gray-950/50 pl-3 pr-1">
                <div className="flex items-center space-x-2">
                  <div className="flex-1">
                    <div className="flex">
                      <div className="flex-1">
                        {modelArch?.sampleTags && taggedSampleArr && modelArchTagSections ? (
                          <>
                            {modelArchTagSections.map((sampleTagSection, sti) => (
                              <div key={sti} className="grid w-full lg:grid-flow-col lg:auto-cols-fr gap-4 mt-2">
                                {Object.entries(sampleTagSection).map(([tagKey, tag]) => (
                                  <div key={tagKey} className="mb-2">
                                    {tag.type === 'text' && (
                                      <TextInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                    {tag.type === 'multiline' && (
                                      <TextAreaInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                    {tag.type === 'number' && (
                                      <NumberInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                  </div>
                                ))}
                              </div>
                            ))}
                          </>
                        ) : (
                          <>
                            {modelArch?.hasMultiLinePrompts ? (
                              <TextAreaInput
                                label={`Prompt`}
                                value={sample.prompt}
                                onChange={value => setJobConfig(value, `config.process[0].sample.samples[${i}].prompt`)}
                                placeholder="Enter prompt"
                                required
                              />
                            ) : (
                              <TextInput
                                label={`Prompt`}
                                value={sample.prompt}
                                onChange={value => setJobConfig(value, `config.process[0].sample.samples[${i}].prompt`)}
                                placeholder="Enter prompt"
                                required
                              />
                            )}
                          </>
                        )}

                        <div className="grid w-full lg:grid-flow-col lg:auto-cols-fr gap-4 mt-2">
                          {!isAudioModel && (
                            <TextInput
                              label={`Width`}
                              value={sample.width ? `${sample.width}` : ''}
                              onChange={value => {
                                // remove any non-numeric characters
                                value = value.replace(/\D/g, '');
                                if (value === '') {
                                  // remove the key from the config if empty
                                  let newConfig = objectCopy(jobConfig);
                                  if (newConfig.config.process[0].sample.samples[i]) {
                                    delete newConfig.config.process[0].sample.samples[i].width;
                                    setJobConfig(
                                      newConfig.config.process[0].sample.samples,
                                      'config.process[0].sample.samples',
                                    );
                                  }
                                } else {
                                  const intValue = parseInt(value);
                                  if (!isNaN(intValue)) {
                                    setJobConfig(intValue, `config.process[0].sample.samples[${i}].width`);
                                  } else {
                                    console.warn('Invalid width value:', value);
                                  }
                                }
                              }}
                              placeholder={`${jobConfig.config.process[0].sample.width} (default)`}
                            />
                          )}
                          {!isAudioModel && (
                            <TextInput
                              label={`Height`}
                              value={sample.height ? `${sample.height}` : ''}
                              onChange={value => {
                                // remove any non-numeric characters
                                value = value.replace(/\D/g, '');
                                if (value === '') {
                                  // remove the key from the config if empty
                                  let newConfig = objectCopy(jobConfig);
                                  if (newConfig.config.process[0].sample.samples[i]) {
                                    delete newConfig.config.process[0].sample.samples[i].height;
                                    setJobConfig(
                                      newConfig.config.process[0].sample.samples,
                                      'config.process[0].sample.samples',
                                    );
                                  }
                                } else {
                                  const intValue = parseInt(value);
                                  if (!isNaN(intValue)) {
                                    setJobConfig(intValue, `config.process[0].sample.samples[${i}].height`);
                                  } else {
                                    console.warn('Invalid height value:', value);
                                  }
                                }
                              }}
                              placeholder={`${jobConfig.config.process[0].sample.height} (default)`}
                            />
                          )}
                          <TextInput
                            label={`Seed`}
                            value={sample.seed ? `${sample.seed}` : ''}
                            onChange={value => {
                              // remove any non-numeric characters
                              value = value.replace(/\D/g, '');
                              if (value === '') {
                                // remove the key from the config if empty
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].seed;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                const intValue = parseInt(value);
                                if (!isNaN(intValue)) {
                                  setJobConfig(intValue, `config.process[0].sample.samples[${i}].seed`);
                                } else {
                                  console.warn('Invalid seed value:', value);
                                }
                              }
                            }}
                            placeholder={`${jobConfig.config.process[0].sample.walk_seed ? jobConfig.config.process[0].sample.seed + i : jobConfig.config.process[0].sample.seed} (default)`}
                          />
                          <TextInput
                            label={`LoRA Scale`}
                            value={sample.network_multiplier ? `${sample.network_multiplier}` : ''}
                            onChange={value => {
                              // remove any non-numeric, - or . characters
                              value = value.replace(/[^0-9.-]/g, '');
                              if (value === '') {
                                // remove the key from the config if empty
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].network_multiplier;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                // set it as a string
                                setJobConfig(value, `config.process[0].sample.samples[${i}].network_multiplier`);
                                return;
                              }
                            }}
                            placeholder={`1.0 (default)`}
                          />
                        </div>
                      </div>
                      {modelArch?.additionalSections?.includes('datasets.multi_control_paths') && (
                        <FormGroup label="Control Images" className="pt-2 ml-4">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2 mt-2">
                            {['ctrl_img_1', 'ctrl_img_2', 'ctrl_img_3'].map((ctrlKey, ctrl_idx) => (
                              <SampleControlImage
                                key={ctrlKey}
                                instruction={`Add Control Image ${ctrl_idx + 1}`}
                                className=""
                                src={sample[ctrlKey as keyof typeof sample] as string}
                                onNewImageSelected={imagePath => {
                                  if (!imagePath) {
                                    let newSamples = objectCopy(jobConfig.config.process[0].sample.samples);
                                    delete newSamples[i][ctrlKey as keyof typeof sample];
                                    setJobConfig(newSamples, 'config.process[0].sample.samples');
                                  } else {
                                    setJobConfig(imagePath, `config.process[0].sample.samples[${i}].${ctrlKey}`);
                                  }
                                }}
                              />
                            ))}
                          </div>
                        </FormGroup>
                      )}
                      {modelArch?.additionalSections?.includes('sample.ctrl_img') && (
                        <SampleControlImage
                          className="mt-6 ml-4"
                          src={sample.ctrl_img}
                          onNewImageSelected={imagePath => {
                            if (!imagePath) {
                              let newSamples = objectCopy(jobConfig.config.process[0].sample.samples);
                              delete newSamples[i].ctrl_img;
                              setJobConfig(newSamples, 'config.process[0].sample.samples');
                            } else {
                              setJobConfig(imagePath, `config.process[0].sample.samples[${i}].ctrl_img`);
                            }
                          }}
                        />
                      )}
                    </div>
                    <div className="pb-4"></div>
                  </div>
                  <div className="flex flex-col items-center gap-2">
                    <button
                      type="button"
                      onClick={() => importRandomPromptFromDataset(i)}
                      disabled={randomPromptLoadingIndex !== null || !canImportRandomPrompt}
                      className="operator-icon-button border-0 disabled:cursor-not-allowed disabled:opacity-50"
                      title={randomPromptDisabledReason}
                      aria-label={randomPromptDisabledReason}
                    >
                      {randomPromptLoadingIndex === i ? (
                        <Loader2 className="h-5 w-5 animate-spin" />
                      ) : (
                        <Shuffle className="h-5 w-5" />
                      )}
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        setJobConfig(
                          jobConfig.config.process[0].sample.samples.filter((_, index) => index !== i),
                          'config.process[0].sample.samples',
                        )
                      }
                      className="operator-icon-button border-0"
                    >
                      <X />
                    </button>
                  </div>
                </div>
              </div>
            ))}
            <button
              type="button"
              onClick={() =>
                setJobConfig(
                  [...jobConfig.config.process[0].sample.samples, { prompt: '' }],
                  'config.process[0].sample.samples',
                )
              }
              className="operator-button w-full"
            >
              Add Prompt
            </button>
          </Card>
        </div>

        {status === 'success' && <p className="text-green-500 text-center">Training saved successfully!</p>}
        {status === 'error' && <p className="text-red-500 text-center">Error saving training. Please try again.</p>}
      </form>
      <AddSingleImageModal />
    </>
  );
}
