'use client';
import Link from 'next/link';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import {
  modelArchs,
  ModelArch,
  groupedModelOptions,
  quantizationOptions,
  defaultQtype,
  jobTypeOptions,
  SampleTags,
} from './options';
import {
  applyOrbit4LowVramDefaults,
  defaultDatasetConfig,
  defaultValidationConfig,
  defaultWatermarkConfig,
  isOrbit4LowVramProfile,
} from './jobConfig';
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
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Copy,
  Cpu,
  Database,
  FolderOpen,
  Gauge,
  Info,
  Layers3,
  ListChecks,
  Loader2,
  Save,
  Settings2,
  Shuffle,
  SlidersHorizontal,
  Sparkles,
  Target,
  TerminalSquare,
  Upload,
  X,
  Zap,
} from 'lucide-react';
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
import { normalizeDetectedCaptionExt } from '@/utils/jobDatasetDefaults';
import { setNestedValue } from '@/utils/hooks';
import { parseRemoteDatasetRef } from '@/utils/remoteDatasetRefs';
import { TrainingAdvisorPanel } from '@/components/TrainingAdvisorPanel';
import {
  AUTHENLORA_BUILTIN_CODEC_BITS,
  AUTHENLORA_CODEC_OPTIONS,
  getAuthenloraCodecSelectValue,
} from '@/utils/authenloraCodecs';
import { uploadLoraFile } from '@/utils/streamedUploads';
import { openDoc } from '@/components/DocModal';

type Props = {
  legacyView?: boolean;
  jobConfig: JobConfig;
  setJobConfig: (value: any, key?: string) => void;
  status: 'idle' | 'saving' | 'success' | 'error';
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  runId: string | null;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  datasetOptions: Array<
    SelectOption & {
      encrypted?: boolean;
      name?: string;
      source?: 'local' | 'remote';
      worker_id?: string;
      ref?: string;
      detectedCaptionExt?: string | null;
    }
  >;
  validationMessages?: Array<{ level: 'error' | 'warning'; message: string }>;
  workerLabel?: string;
  computeControls?: ReactNode;
  trainerLabel?: string;
  onOpenAdvanced?: () => void;
  onOpenRawConfig?: () => void;
  isLoading?: boolean;
  comfyAutoInstall?: boolean;
  projectID?: string | null;
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

const guidedStepItems = [
  { id: 'job-basics', title: 'Basics', detail: 'Name your job and choose a model.' },
  { id: 'job-dataset', title: 'Dataset', detail: 'Add the dataset you want to train on.' },
  { id: 'job-training', title: 'Training', detail: 'Set training length and core parameters.' },
  { id: 'job-samples', title: 'Samples', detail: 'Add prompts to generate previews.' },
  { id: 'job-validation', title: 'Validation', detail: 'Track deterministic loss on held-out images.' },
  { id: 'job-review', title: 'Review', detail: 'Review settings before starting.' },
];

export default function SimpleJob({
  legacyView = false,
  jobConfig,
  setJobConfig,
  handleSubmit,
  status,
  runId,
  gpuIDs,
  setGpuIDs,
  gpuList,
  datasetOptions,
  validationMessages = [],
  workerLabel = 'Local worker',
  computeControls,
  trainerLabel = 'LoRA Trainer',
  onOpenAdvanced,
  onOpenRawConfig,
  isLoading,
  comfyAutoInstall = false,
  projectID = null,
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

  const isVideoModel = Boolean(modelArch?.isVideoModel || modelArch?.group === 'video');
  const networkConfig = jobConfig.config.process[0].network;
  const networkType = networkConfig?.type ?? 'lora';
  const normalizedNetworkType = networkType.toLowerCase();
  const supportsNormalNetworkDropout = networkType !== 'lokr';
  const lokrFullMatrix = !!(networkConfig?.lokr_full_matrix || networkConfig?.lokr_full_rank);
  const isAudioModel = !!(modelArch?.group === 'audio');
  const autoTrain = !!jobConfig.config.process[0].train.auto_train;
  const trainConfig = jobConfig.config.process[0].train;
  const textEncoderTrainingEnabled = !!trainConfig.train_text_encoder;
  const modelArchName = jobConfig.config.process[0].model.arch;
  const supportsSegaDistill = segaDistillArchs.has(modelArchName.split(':')[0]);
  const segaDistillEnabled = !!trainConfig.sega_distill;
  const canEnableSegaDistill = supportsSegaDistill && networkType === 'lora';
  const showSegaDistill = supportsSegaDistill || segaDistillEnabled;
  const watermarkConfig = jobConfig.config.process[0].watermark ?? defaultWatermarkConfig;
  const watermarkEnabled = !!watermarkConfig.enabled;
  const selectedAuthenloraCodec = getAuthenloraCodecSelectValue(watermarkConfig.codec_path);
  const canEnableWatermark =
    !isAudioModel &&
    !isVideoModel &&
    !!networkConfig &&
    ['lora', 'locon', 'lycoris', 'lokr'].includes(normalizedNetworkType) &&
    trainConfig.loss_type !== 'mean_flow' &&
    !trainConfig.do_guidance_loss;

  const setSegaDefaults = () => {
    if (trainConfig.sega_distill_weight === undefined) setJobConfig(1.0, 'config.process[0].train.sega_distill_weight');
    if (trainConfig.sega_distill_base_resolution === undefined) {
      setJobConfig(1024, 'config.process[0].train.sega_distill_base_resolution');
    }
    if (trainConfig.sega_distill_strength === undefined)
      setJobConfig(1.0, 'config.process[0].train.sega_distill_strength');
    if (trainConfig.sega_distill_min_scale === undefined)
      setJobConfig(0.5, 'config.process[0].train.sega_distill_min_scale');
    if (trainConfig.sega_distill_max_scale === undefined)
      setJobConfig(2.0, 'config.process[0].train.sega_distill_max_scale');
    if (trainConfig.sega_distill_on_reg === undefined)
      setJobConfig(false, 'config.process[0].train.sega_distill_on_reg');
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

  const handleWatermarkToggle = (enabled: boolean) => {
    if (enabled && !canEnableWatermark) return;
    if (!jobConfig.config.process[0].watermark) {
      setJobConfig({ ...defaultWatermarkConfig, enabled, method: 'authenlora' }, 'config.process[0].watermark');
      return;
    }
    setJobConfig(enabled, 'config.process[0].watermark.enabled');
    setJobConfig('authenlora', 'config.process[0].watermark.method');
  };

  const handleWatermarkCodecChange = (value: string) => {
    if (value === 'custom') {
      if (selectedAuthenloraCodec !== 'custom') {
        setJobConfig('', 'config.process[0].watermark.codec_path');
      }
      return;
    }
    setJobConfig(value, 'config.process[0].watermark.codec_path');
    const msgBits = AUTHENLORA_BUILTIN_CODEC_BITS[value];
    if (msgBits) {
      setJobConfig(msgBits, 'config.process[0].watermark.msg_bits');
    }
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
    const sampleArr: any[] = [];
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
    const sections: SampleTags[] = [];
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
      .filter(
        dataset => dataset.folderPath && dataset.folderPath !== defaultDatasetConfig.folder_path && !dataset.remote,
      );
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

  useEffect(() => {
    if (!textEncoderTrainingEnabled) return;
    if (trainConfig.unload_text_encoder) {
      setJobConfig(false, 'config.process[0].train.unload_text_encoder');
    }
    if (trainConfig.cache_text_embeddings) {
      setJobConfig(false, 'config.process[0].train.cache_text_embeddings');
    }
    const datasets = jobConfig.config.process[0].datasets;
    if (datasets.some(dataset => dataset.cache_text_embeddings)) {
      setJobConfig(
        datasets.map(dataset => ({ ...dataset, cache_text_embeddings: false })),
        'config.process[0].datasets',
      );
    }
  }, [
    jobConfig.config.process[0].datasets,
    setJobConfig,
    textEncoderTrainingEnabled,
    trainConfig.cache_text_embeddings,
    trainConfig.unload_text_encoder,
  ]);

  const {
    arch: orbitProfileArch,
    low_vram: orbitProfileLowVram,
    qtype: orbitProfileQtype,
    qtype_te: orbitProfileTextEncoderQtype,
    quantize: orbitProfileQuantize,
    quantize_te: orbitProfileQuantizeTextEncoder,
  } = jobConfig.config.process[0].model;
  // Existing jobs count as already configured. A profile selected in this
  // editor receives the full defaults once, after which manual edits win.
  const orbitProfileAppliedRef = useRef(isOrbit4LowVramProfile(jobConfig));

  useEffect(() => {
    if (orbitProfileAppliedRef.current || !isOrbit4LowVramProfile(jobConfig)) return;
    orbitProfileAppliedRef.current = true;
    setJobConfig((current: JobConfig) => applyOrbit4LowVramDefaults(current, true));
  }, [
    jobConfig,
    orbitProfileArch,
    orbitProfileLowVram,
    orbitProfileQtype,
    orbitProfileTextEncoderQtype,
    orbitProfileQuantize,
    orbitProfileQuantizeTextEncoder,
    setJobConfig,
  ]);

  const setSamplePromptValue = (sampleIndex: number, prompt: string) => {
    if (modelArch?.sampleTags && taggedSampleArr?.[sampleIndex]) {
      const tagKey =
        (modelArch.sampleTags.CAPTION && 'CAPTION') ||
        (modelArch.sampleTags.PROMPT && 'PROMPT') ||
        Object.entries(modelArch.sampleTags).find(([, tag]) => tag.type === 'text' || tag.type === 'multiline')?.[0];

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
      const response = await apiClient.post('/api/datasets/randomPrompt', {
        datasets,
        encryptedDatasetKeys,
        ...(projectID ? { project_id: projectID } : {}),
      });
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

  const setDatasetPath = (datasetIndex: number, value: string) => {
    const selectedOption = datasetOptions.find(option => option.value === value);
    const detectedCaptionExt = normalizeDetectedCaptionExt(selectedOption?.detectedCaptionExt);

    setJobConfig((previous: JobConfig) => {
      let next = setNestedValue(previous, value, `config.process[0].datasets[${datasetIndex}].folder_path`);
      if (detectedCaptionExt) {
        next = setNestedValue(next, detectedCaptionExt, `config.process[0].datasets[${datasetIndex}].caption_ext`);
      }
      return next;
    });
  };

  const handleBaseLoraUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.safetensors')) {
      setBaseLoraUploadStatus('error');
      setBaseLoraUploadMessage('Base LoRA upload must be a .safetensors file.');
      return;
    }

    setBaseLoraUploadStatus('uploading');
    setBaseLoraUploadMessage('');
    try {
      const response = await uploadLoraFile(file);
      const uploaded = response.data?.lora;
      if (!uploaded?.path) {
        throw new Error('Upload did not return a LoRA path.');
      }
      const reused = response.data?.reused === true;
      setJobConfig(uploaded.path, 'config.process[0].model.base_lora_path');
      const triggerWords = Array.isArray(uploaded.triggerWords) ? uploaded.triggerWords.filter(Boolean) : [];
      setBaseLoraUploadStatus('success');
      setBaseLoraUploadMessage(
        triggerWords.length > 0
          ? `${reused ? 'Loaded existing upload' : 'Uploaded'}. Trigger metadata: ${triggerWords.join(', ')}`
          : `${reused ? 'Loaded existing upload' : 'Uploaded'}. No trigger metadata found.`,
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
    const newQuantizationOptions = [
      {
        label: 'Standard',
        options: [quantizationOptions[0], quantizationOptions[1]],
      },
    ];

    // add ARAs if they exist for the model
    const ARAs: SelectOption[] = [];
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

    const additionalQuantizationOptions: SelectOption[] = [];
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

  const processConfig = jobConfig.config.process[0];
  const datasetsConfig = processConfig.datasets || [];
  const sampleConfig = processConfig.sample;
  const validationConfig = processConfig.train.validation_config;
  const validationItems = Array.isArray(validationConfig?.validation_items) ? validationConfig.validation_items : [];
  const firstDataset = datasetsConfig[0] || defaultDatasetConfig;
  const firstSample = sampleConfig.samples?.[0] || { prompt: '' };
  const selectedDatasetOption = datasetOptions.find(option => option.value === firstDataset.folder_path);
  const selectedGpu = Array.isArray(gpuList) ? gpuList.find((gpu: any) => `${gpu.index}` === `${gpuIDs}`) : null;
  const samplePromptBlank = !processConfig.train.disable_sampling && !firstSample.prompt?.trim();
  const incompleteValidationItems = validationItems.filter(
    item => typeof item.image_path !== 'string' || !item.image_path.trim(),
  ).length;
  const unresolvedDataset =
    !datasetsConfig.length ||
    !firstDataset.folder_path ||
    firstDataset.folder_path === defaultDatasetConfig.folder_path;
  const localReadinessMessages = [
    ...(unresolvedDataset
      ? [{ level: 'error' as const, message: 'Select a target dataset before creating this job.' }]
      : []),
    ...(samplePromptBlank
      ? [{ level: 'warning' as const, message: 'No sample prompt is configured. Add one in Basics or Samples.' }]
      : []),
    ...(validationConfig && validationItems.length === 0
      ? [{ level: 'error' as const, message: 'Add at least one held-out image before enabling validation loss.' }]
      : []),
    ...(incompleteValidationItems > 0
      ? [
          {
            level: 'error' as const,
            message: `${incompleteValidationItems} validation item${incompleteValidationItems === 1 ? '' : 's'} still need an image.`,
          },
        ]
      : []),
  ];
  const readinessMessages = validationMessages.length > 0 ? validationMessages : localReadinessMessages;
  const readinessErrors = readinessMessages.filter(message => message.level === 'error');
  const readinessWarnings = readinessMessages.filter(message => message.level === 'warning');
  const stepItems = guidedStepItems;
  const [activeStepId, setActiveStepId] = useState(stepItems[0].id);
  const dtypeOptions = [
    { value: 'bf16', label: 'BF16' },
    { value: 'fp16', label: 'FP16' },
    { value: 'fp32', label: 'FP32' },
  ];
  const optimizerOptions = [
    { value: 'adafactor', label: 'Adafactor' },
    { value: 'adam', label: 'Adam' },
    { value: 'adamw', label: 'AdamW' },
    { value: 'adamw8bit', label: 'AdamW8Bit' },
    { value: 'automagic', label: 'Automagic' },
    { value: 'automagic2', label: 'Automagic v2' },
    { value: 'automagic3', label: 'Automagic v3' },
    { value: 'prodigyopt', label: 'Prodigy' },
    { value: 'prodigy8bit', label: 'Prodigy8Bit' },
  ];
  const timestepOptions = [
    { value: 'sigmoid', label: 'Sigmoid' },
    { value: 'linear', label: 'Linear' },
    { value: 'shift', label: 'Shift' },
    { value: 'weighted', label: 'Weighted' },
    { value: 'shifted_logit_normal', label: 'Shifted logit normal' },
  ];
  const timestepBiasOptions = [
    { value: 'balanced', label: 'Balanced' },
    { value: 'content', label: 'High Noise' },
    { value: 'style', label: 'Low Noise' },
  ];
  const lossOptions = [
    { value: 'mse', label: 'Mean Squared Error' },
    { value: 'mae', label: 'Mean Absolute Error' },
    { value: 'wavelet', label: 'Wavelet' },
    { value: 'stepped', label: 'Stepped Recovery' },
  ];
  const targetTypeOptions = [
    { value: 'lora', label: 'LoRA' },
    { value: 'dora', label: 'DoRA' },
    { value: 'lokr', label: 'LoKr' },
  ].filter(option => !modelArch?.allowedNetworkTypes || modelArch.allowedNetworkTypes.includes(option.value));

  const formatNumber = (value: number | null | undefined) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
    return Number(value).toLocaleString();
  };
  const formatMemory = (value: number | null | undefined) => {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '-';
    return `${(Number(value) / 1024).toFixed(1)} GB`;
  };
  const selectedGpuLabel = selectedGpu
    ? `GPU #${selectedGpu.index} - ${selectedGpu.name}`
    : gpuIDs
      ? `GPU #${gpuIDs}`
      : 'GPU not selected';
  const selectedGpuMemory = selectedGpu?.memory
    ? `${formatMemory(selectedGpu.memory.free)} free / ${formatMemory(selectedGpu.memory.total)}`
    : 'Telemetry loading';

  const setPrimarySamplePrompt = (prompt: string) => {
    if (!sampleConfig.samples?.length) {
      setJobConfig([{ prompt }], 'config.process[0].sample.samples');
      return;
    }
    setSamplePromptValue(0, prompt);
  };

  const addDataset = () => {
    const newDataset = applySelectedDatasetDefaults(objectCopy(defaultDatasetConfig), modelArch?.defaults);
    newDataset.controls = modelArch?.controls ?? [];
    setJobConfig([...datasetsConfig, newDataset], 'config.process[0].datasets');
  };

  const duplicateDataset = (datasetIndex: number) => {
    const duplicated = objectCopy(datasetsConfig[datasetIndex]);
    const nextDatasets = [...datasetsConfig];
    nextDatasets.splice(datasetIndex + 1, 0, duplicated);
    setJobConfig(nextDatasets, 'config.process[0].datasets');
  };

  const removeDataset = (datasetIndex: number) => {
    setJobConfig(
      datasetsConfig.filter((_, index) => index !== datasetIndex),
      'config.process[0].datasets',
    );
  };

  const renderSectionIntro = (title: string, detail: string) => (
    <div className="mb-5 border-b border-gray-900 pb-4">
      <h2 tabIndex={-1} className="text-2xl font-semibold text-gray-100 outline-none">
        {title}
      </h2>
      <p className="mt-1 text-sm text-gray-500">{detail}</p>
    </div>
  );

  const renderDisclosure = (
    id: string,
    title: string,
    detail: string,
    Icon: typeof Settings2,
    children: ReactNode,
    defaultOpen = false,
  ) =>
    legacyView ? (
      <section id={id} className="studio-legacy-card">
        <h3 className="mb-3 flex items-center gap-2 text-sm font-semibold text-gray-100">
          <Icon className="h-4 w-4 text-brand-400" />
          {title}
        </h3>
        {children}
      </section>
    ) : (
      <details id={id} className="group border-t border-gray-900 bg-gray-950/30" open={defaultOpen}>
        <summary className="flex cursor-pointer list-none items-center gap-3 px-4 py-3 text-sm hover:bg-gray-900/45">
          <Icon className="h-4 w-4 flex-none text-gray-300" />
          <span className="min-w-0 flex-1">
            <span className="font-semibold text-gray-100">{title}</span>
            <span className="ml-3 hidden text-gray-500 sm:inline">{detail}</span>
          </span>
          <ChevronDown className="h-4 w-4 flex-none text-gray-400 transition-transform group-open:rotate-180" />
        </summary>
        <div className="border-t border-gray-900 px-4 pb-5 pt-2">{children}</div>
      </details>
    );

  const activeStepIndex = stepItems.findIndex(step => step.id === activeStepId);
  const formRef = useRef<HTMLFormElement>(null);
  const didSelectStep = useRef(false);
  const selectStep = (id: string) => {
    didSelectStep.current = true;
    setActiveStepId(id);
    window.history.replaceState(null, '', `#${id}`);
  };
  const nextStep = () => {
    if (formRef.current && !formRef.current.reportValidity()) return;
    const next = stepItems[activeStepIndex + 1];
    if (next) selectStep(next.id);
  };
  useEffect(() => {
    const syncHash = () => {
      const id = window.location.hash.slice(1);
      if (guidedStepItems.some(step => step.id === id)) setActiveStepId(id);
    };
    syncHash();
    window.addEventListener('hashchange', syncHash);
    return () => window.removeEventListener('hashchange', syncHash);
  }, []);
  useEffect(() => {
    if (!didSelectStep.current) return;
    const heading = document.querySelector<HTMLElement>(`#${activeStepId} h2`);
    heading?.focus({ preventScroll: true });
    document.getElementById('training-step-navigation')?.scrollIntoView({ block: 'start', behavior: 'smooth' });
  }, [activeStepId]);

  return (
    <>
      <form
        ref={formRef}
        onSubmit={event => {
          if (legacyView || activeStepId === 'job-review') handleSubmit(event);
          else {
            event.preventDefault();
            nextStep();
          }
        }}
        className={`relative min-h-full bg-gray-950 ${legacyView ? 'studio-training-legacy' : ''} ${isLoading ? 'pointer-events-none opacity-50' : ''}`}
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
          <div className="absolute inset-0 z-50 flex items-center justify-center bg-gray-950/70">
            <div className="flex flex-col items-center gap-3 border border-gray-800 bg-gray-950 px-5 py-4">
              <Loader2 className="h-6 w-6 animate-spin text-brand-300" />
              <span className="text-sm text-gray-400">Loading training workspace...</span>
            </div>
          </div>
        )}

        <header className="studio-page-intro px-5 pt-8 sm:px-8">
          <h2>
            {legacyView
              ? 'Training configuration'
              : runId
                ? 'Refine your training setup.'
                : 'Start with the essentials.'}
          </h2>
          <p>
            {legacyView
              ? 'All options in one place. Configure your model, training, datasets, and samples.'
              : 'Choose your model. We will guide you through the rest.'}
          </p>
        </header>
        {!legacyView && (
          <nav id="training-step-navigation" aria-label="Training setup steps" className="studio-training-steps">
            {stepItems.map((step, index) => (
              <button
                key={step.id}
                type="button"
                onClick={() => selectStep(step.id)}
                aria-current={activeStepId === step.id ? 'step' : undefined}
              >
                <span>{index + 1}</span>
                {step.title}
              </button>
            ))}
          </nav>
        )}
        <div className="studio-training-layout">
          <div className="studio-training-sections min-w-0">
            <fieldset
              id="job-basics"
              hidden={!legacyView && activeStepId !== 'job-basics'}
              disabled={!legacyView && activeStepId !== 'job-basics'}
              className="studio-training-panel"
            >
              {!legacyView && (
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-brand-400">Step 1 of 6</p>
              )}
              {renderSectionIntro(
                legacyView ? 'Job & model' : 'Name your run',
                'Give your training run a name and choose your base model.',
              )}

              <div className="space-y-5">
                <div className="max-w-2xl">
                  <TextInput
                    label="Training name"
                    value={jobConfig.config.name}
                    docKey="config.name"
                    onChange={value => setJobConfig(value, 'config.name')}
                    placeholder="my_first_lora_v1"
                    disabled={runId !== null}
                    required
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    A name to identify this run inside the current workspace.
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <SelectInput
                    label="Model architecture"
                    value={processConfig.model.arch}
                    onChange={value => {
                      handleModelArchChange(processConfig.model.arch, value, jobConfig, setJobConfig);
                    }}
                    options={groupedModelOptions}
                  />
                  <SelectInput
                    label="Target type"
                    value={networkType}
                    onChange={value => {
                      setJobConfig(value, 'config.process[0].network.type');
                      if (value === 'lokr') {
                        setJobConfig(undefined, 'config.process[0].network.dropout');
                        if (processConfig.train.sega_distill)
                          setJobConfig(false, 'config.process[0].train.sega_distill');
                      }
                    }}
                    options={targetTypeOptions}
                  />
                </div>

                <TextInput
                  label="Model path"
                  value={processConfig.model.name_or_path}
                  docKey="config.process[0].model.name_or_path"
                  onChange={(value: string | null) => {
                    if (value?.trim() === '') value = null;
                    setJobConfig(value, 'config.process[0].model.name_or_path');
                  }}
                  placeholder="ostris/Flex.1-alpha"
                  required
                />

                {modelArch?.gateUrl && (
                  <button
                    type="button"
                    onClick={() => {
                      const gateUrl = modelArch.gateUrl;
                      openDoc({
                        title: 'Gated model',
                        description: (
                          <div className="space-y-3">
                            <p>
                              This model is gated on Hugging Face. Accept its terms on the model page before using it:
                            </p>
                            <p>
                              <a
                                href={gateUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-brand-300 underline hover:text-brand-200"
                              >
                                {gateUrl}
                              </a>
                            </p>
                            <p>
                              Create a Hugging Face{' '}
                              <a
                                href="https://huggingface.co/settings/tokens"
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-brand-300 underline hover:text-brand-200"
                              >
                                read token
                              </a>{' '}
                              and add it on the{' '}
                              <Link href="/settings" className="text-brand-300 underline hover:text-brand-200">
                                settings page
                              </Link>
                              .
                            </p>
                          </div>
                        ),
                      });
                    }}
                    className="flex w-full items-center gap-2 border border-brand-900 bg-brand-950/35 px-3 py-2 text-left text-sm text-brand-100 transition-colors hover:border-brand-800 hover:bg-brand-950/55"
                  >
                    <Info className="h-4 w-4 shrink-0 text-brand-300" />
                    <span>
                      Gated model. <span className="underline">Learn more.</span>
                    </span>
                  </button>
                )}

                {modelArch?.modelNotes && (
                  <button
                    type="button"
                    onClick={() => {
                      const notes = modelArch.modelNotes;
                      if (!notes) return;
                      openDoc({
                        title: `Notes - ${modelArch.label}`,
                        description: (
                          <div className="space-y-3">
                            <p>{notes.summary}</p>
                            {notes.link && (
                              <p>
                                <Link href={notes.link.href} className="text-brand-300 underline hover:text-brand-200">
                                  {notes.link.label}
                                </Link>
                              </p>
                            )}
                            {notes.codeBlock && (
                              <pre className="overflow-x-auto border border-gray-800 bg-gray-950 p-3 text-xs text-gray-300">
                                <code>{notes.codeBlock}</code>
                              </pre>
                            )}
                            {notes.paragraphs?.map(paragraph => <p key={paragraph}>{paragraph}</p>)}
                          </div>
                        ),
                      });
                    }}
                    className="flex w-full items-center gap-2 border border-amber-900 bg-amber-950/25 px-3 py-2 text-left text-sm text-amber-100 transition-colors hover:border-amber-800 hover:bg-amber-950/40"
                  >
                    <IoFlaskSharp className="h-4 w-4 shrink-0 text-amber-300" />
                    <span>
                      Experimental model. <span className="underline">Read model notes.</span>
                    </span>
                  </button>
                )}

                {modelArch?.additionalSections?.includes('model.assistant_lora_path') && (
                  <div className="max-w-2xl">
                    <TextInput
                      label="Training adapter path"
                      value={processConfig.model.assistant_lora_path ?? ''}
                      docKey="config.process[0].model.assistant_lora_path"
                      onChange={(value: string | null) => {
                        setJobConfig(value?.trim() ? value : undefined, 'config.process[0].model.assistant_lora_path');
                      }}
                      placeholder="Hugging Face repository/file path"
                    />
                  </div>
                )}

                {modelArch?.customModelSelectOptions?.map(customOption => (
                  <div className="max-w-2xl" key={customOption.label}>
                    <SelectInput
                      label={customOption.label}
                      value={customOption.getValue(jobConfig)}
                      doc={customOption.doc}
                      onChange={value => customOption.onChange(value, jobConfig, setJobConfig)}
                      options={customOption.options}
                    />
                  </div>
                ))}

                <div className="grid grid-cols-1 gap-5">
                  {disableSections.includes('trigger_word') ? null : (
                    <div>
                      <TextInput
                        label="Trigger word (optional)"
                        value={processConfig.trigger_word || ''}
                        docKey="config.process[0].trigger_word"
                        onChange={(value: string | null) => {
                          if (value?.trim() === '') value = null;
                          setJobConfig(value, 'config.process[0].trigger_word');
                        }}
                        placeholder="mytrigger"
                      />
                      <p className="mt-1 text-xs text-gray-500">The token that represents your concept.</p>
                    </div>
                  )}
                  {!processConfig.model.arch.startsWith('minimax_h3') && (
                    <details className="studio-disclosure" open={legacyView || undefined}>
                      <summary>Continue from an existing LoRA</summary>
                      <div>
                        <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2">
                          <TextInput
                            label="Base LoRA path (optional)"
                            value={processConfig.model.base_lora_path ?? ''}
                            docKey="config.process[0].model.base_lora_path"
                            onChange={(value: string | undefined) => {
                              if (value?.trim() === '') value = undefined;
                              setJobConfig(value, 'config.process[0].model.base_lora_path');
                            }}
                            placeholder="e.g. path/to/base_lora.safetensors"
                          />
                          <button
                            type="button"
                            className="operator-icon-button mt-7 h-8 w-8 border-gray-800"
                            onClick={() => baseLoraFileInputRef.current?.click()}
                            disabled={baseLoraUploadStatus === 'uploading'}
                            title="Upload or reuse Base LoRA"
                            aria-label="Upload or reuse Base LoRA"
                          >
                            {baseLoraUploadStatus === 'uploading' ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <FolderOpen className="h-4 w-4" />
                            )}
                          </button>
                        </div>
                        <p className="mt-1 text-xs text-gray-500">Use this to continue training from a base LoRA.</p>
                        {baseLoraUploadMessage && (
                          <div
                            className={`mt-2 border px-3 py-2 text-xs ${
                              baseLoraUploadStatus === 'error'
                                ? 'border-red-900 bg-red-950/30 text-red-300'
                                : 'border-gray-800 bg-gray-950 text-gray-300'
                            }`}
                          >
                            {baseLoraUploadMessage}
                          </div>
                        )}
                      </div>
                    </details>
                  )}
                </div>
              </div>
            </fieldset>

            <fieldset
              id="job-training"
              hidden={!legacyView && activeStepId !== 'job-training'}
              disabled={!legacyView && activeStepId !== 'job-training'}
              className="studio-training-panel"
            >
              {renderSectionIntro('Training', 'Set the pace and duration of your run.')}
              <div className="mb-6">
                {' '}
                <div>
                  <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-gray-100">
                    Quick training settings <Info className="h-3.5 w-3.5 text-gray-500" />
                  </div>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
                    {!autoTrain && (
                      <NumberInput
                        label="Steps"
                        value={processConfig.train.steps}
                        onChange={handleTrainingStepsChange}
                        placeholder="3000"
                        min={1}
                        required
                      />
                    )}
                    <NumberInput
                      label="Batch size"
                      value={processConfig.train.batch_size}
                      onChange={value => setJobConfig(value, 'config.process[0].train.batch_size')}
                      placeholder="1"
                      min={1}
                      required
                    />
                    <NumberInput
                      label="Gradient accumulation"
                      value={processConfig.train.gradient_accumulation}
                      onChange={value => setJobConfig(value, 'config.process[0].train.gradient_accumulation')}
                      placeholder="1"
                      min={1}
                      required
                    />
                    <NumberInput
                      label="Learning rate"
                      value={processConfig.train.lr}
                      onChange={value => setJobConfig(value, 'config.process[0].train.lr')}
                      placeholder="0.0001"
                      min={0}
                      required
                    />
                  </div>
                  <p className="mt-2 text-xs text-gray-500">
                    These defaults work well for most LoRA training runs. Adjust later in Advanced if needed.
                  </p>
                </div>
              </div>
              {renderDisclosure(
                'job-runtime',
                'Runtime & saving',
                'Data type, save frequency and checkpoints',
                Clock3,
                <div className="grid grid-cols-1 gap-x-5 gap-y-3 md:grid-cols-2 xl:grid-cols-4">
                  {disableSections.includes('model.quantize') ? null : (
                    <div className="space-y-2">
                      <h3 className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Quantization</h3>
                      <SelectInput
                        label="Transformer"
                        value={processConfig.model.quantize ? processConfig.model.qtype : ''}
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
                          value={processConfig.model.quantize_te ? processConfig.model.qtype_te : ''}
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
                    </div>
                  )}
                  <div className="space-y-2">
                    <h3 className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Saving</h3>
                    <SelectInput
                      label="Data type"
                      value={processConfig.save.dtype}
                      onChange={value => setJobConfig(value, 'config.process[0].save.dtype')}
                      options={dtypeOptions}
                    />
                    <NumberInput
                      label="Save every"
                      value={processConfig.save.save_every}
                      onChange={value => setJobConfig(value, 'config.process[0].save.save_every')}
                      placeholder="250"
                      min={1}
                      required
                    />
                    <NumberInput
                      label="Max saves to keep"
                      value={processConfig.save.max_step_saves_to_keep}
                      onChange={value => setJobConfig(value, 'config.process[0].save.max_step_saves_to_keep')}
                      placeholder="4"
                      min={1}
                      required
                    />
                  </div>
                  {modelArch?.additionalSections?.includes('model.low_vram') && (
                    <div className="space-y-2">
                      <h3 className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Memory</h3>
                      <Checkbox
                        label="Low VRAM"
                        checked={processConfig.model.low_vram}
                        onChange={value => {
                          setJobConfig(value, 'config.process[0].model.low_vram');
                          if (value) setJobConfig(true, 'config.process[0].sample.keep_low_vram_for_samples');
                        }}
                      />
                    </div>
                  )}
                  {modelArch?.additionalSections?.includes('model.model_kwargs.kv_cache') && (
                    <div className="space-y-2">
                      <h3 className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Edit</h3>
                      <Checkbox
                        label="KV Cache"
                        docKey="model.model_kwargs.kv_cache"
                        checked={!!processConfig.model.model_kwargs?.kv_cache}
                        onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.kv_cache')}
                      />
                    </div>
                  )}
                  {modelArch?.additionalSections?.includes('model.ideogram_skip_unconditional_transformer') && (
                    <div className="space-y-2">
                      <h3 className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">Experimental</h3>
                      <Checkbox
                        label="Skip unconditional transformer"
                        checked={!!processConfig.model.model_kwargs?.skip_unconditional_transformer_for_training}
                        onChange={value =>
                          setJobConfig(
                            value ? true : undefined,
                            'config.process[0].model.model_kwargs.skip_unconditional_transformer_for_training',
                          )
                        }
                      />
                    </div>
                  )}
                </div>,
              )}

              {renderDisclosure(
                'job-target',
                'LoRA target',
                'Modules to train and network settings',
                Target,
                <div className="grid grid-cols-1 gap-x-5 gap-y-3 md:grid-cols-2 xl:grid-cols-4">
                  <SelectInput
                    label="Target type"
                    value={networkType}
                    onChange={value => {
                      setJobConfig(value, 'config.process[0].network.type');
                      if (value === 'lokr') {
                        setJobConfig(undefined, 'config.process[0].network.dropout');
                        if (processConfig.train.sega_distill)
                          setJobConfig(false, 'config.process[0].train.sega_distill');
                      }
                    }}
                    options={targetTypeOptions}
                  />
                  {(networkType === 'lora' || networkType === 'dora') && (
                    <>
                      <NumberInput
                        label="Linear rank"
                        value={processConfig.network?.linear ?? null}
                        onChange={value => {
                          setJobConfig(value, 'config.process[0].network.linear');
                          setJobConfig(value, 'config.process[0].network.linear_alpha');
                        }}
                        placeholder="32"
                        min={1}
                        max={1024}
                        required
                      />
                      {disableSections.includes('network.conv') ? null : (
                        <NumberInput
                          label="Conv rank"
                          value={processConfig.network?.conv ?? null}
                          onChange={value => {
                            setJobConfig(value, 'config.process[0].network.conv');
                            setJobConfig(value, 'config.process[0].network.conv_alpha');
                          }}
                          placeholder="16"
                          min={0}
                          max={1024}
                        />
                      )}
                    </>
                  )}
                  {networkType === 'lokr' && (
                    <SelectInput
                      label="LoKr factor"
                      docKey="config.process[0].network.lokr_factor"
                      value={`${networkConfig?.lokr_factor ?? -1}`}
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
                  {supportsNormalNetworkDropout && (
                    <NumberInput
                      label="Network dropout"
                      value={processConfig.network?.dropout ?? null}
                      onChange={value => setJobConfig(value ?? undefined, 'config.process[0].network.dropout')}
                      placeholder="0.05"
                      min={0}
                      max={1}
                    />
                  )}
                </div>,
              )}

              {renderDisclosure(
                'job-optimizer',
                'Optimizer',
                'Optimizer, scheduler and loss function',
                Gauge,
                <div className="space-y-4">
                  <div className="grid grid-cols-1 gap-x-5 gap-y-3 md:grid-cols-2 xl:grid-cols-4">
                    <SelectInput
                      label="Optimizer"
                      value={processConfig.train.optimizer}
                      onChange={value => setJobConfig(value, 'config.process[0].train.optimizer')}
                      options={optimizerOptions}
                    />
                    <NumberInput
                      label="Weight decay"
                      value={processConfig.train.optimizer_params.weight_decay}
                      onChange={value => setJobConfig(value, 'config.process[0].train.optimizer_params.weight_decay')}
                      placeholder="0.0001"
                      min={0}
                      required
                    />
                    {disableSections.includes('train.timestep_type') ? null : (
                      <SelectInput
                        label="Timestep type"
                        value={processConfig.train.timestep_type}
                        onChange={value => setJobConfig(value, 'config.process[0].train.timestep_type')}
                        options={timestepOptions}
                      />
                    )}
                    <SelectInput
                      label="Timestep bias"
                      value={processConfig.train.content_or_style}
                      onChange={value => setJobConfig(value, 'config.process[0].train.content_or_style')}
                      options={timestepBiasOptions}
                    />
                    <SelectInput
                      label="Loss type"
                      value={processConfig.train.loss_type}
                      onChange={value => setJobConfig(value, 'config.process[0].train.loss_type')}
                      options={lossOptions}
                    />
                    <FormGroup label="EMA">
                      <Checkbox
                        label="Use EMA"
                        checked={processConfig.train.ema_config?.use_ema || false}
                        onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.use_ema')}
                      />
                    </FormGroup>
                    <FormGroup label="Text encoder">
                      {!disableSections.includes('train.train_text_encoder') && (
                        <Checkbox
                          label="Train TE"
                          checked={processConfig.train.train_text_encoder || false}
                          docKey={'train.train_text_encoder'}
                          onChange={value => {
                            setJobConfig(value, 'config.process[0].train.train_text_encoder');
                            if (value) {
                              setJobConfig(false, 'config.process[0].train.unload_text_encoder');
                              setJobConfig(false, 'config.process[0].train.cache_text_embeddings');
                              if (processConfig.datasets.some(dataset => dataset.cache_text_embeddings)) {
                                setJobConfig(
                                  processConfig.datasets.map(dataset => ({ ...dataset, cache_text_embeddings: false })),
                                  'config.process[0].datasets',
                                );
                              }
                            }
                          }}
                        />
                      )}
                      {textEncoderTrainingEnabled && (
                        <NumberInput
                          label="TE learning rate"
                          value={processConfig.train.text_encoder_lr ?? processConfig.train.lr ?? null}
                          docKey={'train.text_encoder_lr'}
                          onChange={value =>
                            setJobConfig(value ?? undefined, 'config.process[0].train.text_encoder_lr')
                          }
                          placeholder={`${processConfig.train.lr || 0.0001}`}
                          min={0}
                        />
                      )}
                      {!disableSections.includes('train.unload_text_encoder') && (
                        <Checkbox
                          label="Unload TE"
                          checked={processConfig.train.unload_text_encoder || false}
                          docKey={'train.unload_text_encoder'}
                          disabled={textEncoderTrainingEnabled}
                          onChange={value => {
                            setJobConfig(value, 'config.process[0].train.unload_text_encoder');
                            if (value) setJobConfig(false, 'config.process[0].train.cache_text_embeddings');
                          }}
                        />
                      )}
                      <Checkbox
                        label="Cache embeddings"
                        checked={processConfig.train.cache_text_embeddings || false}
                        docKey={'train.cache_text_embeddings'}
                        disabled={textEncoderTrainingEnabled}
                        onChange={value => {
                          setJobConfig(value, 'config.process[0].train.cache_text_embeddings');
                          if (value) setJobConfig(false, 'config.process[0].train.unload_text_encoder');
                        }}
                      />
                    </FormGroup>
                    {disableSections.includes('train.diff_output_preservation') ? null : (
                      <FormGroup label="Regularization">
                        <Checkbox
                          label="Differential Output Preservation"
                          docKey={'train.diff_output_preservation'}
                          checked={processConfig.train.diff_output_preservation || false}
                          onChange={value => {
                            setJobConfig(value, 'config.process[0].train.diff_output_preservation');
                            if (value && processConfig.train.blank_prompt_preservation)
                              setJobConfig(false, 'config.process[0].train.blank_prompt_preservation');
                            if (value && processConfig.train.sega_distill)
                              setJobConfig(false, 'config.process[0].train.sega_distill');
                          }}
                        />
                        {disableSections.includes('train.blank_prompt_preservation') ? null : (
                          <Checkbox
                            label="Blank Prompt Preservation"
                            docKey={'train.blank_prompt_preservation'}
                            checked={processConfig.train.blank_prompt_preservation || false}
                            onChange={value => {
                              setJobConfig(value, 'config.process[0].train.blank_prompt_preservation');
                              if (value && processConfig.train.diff_output_preservation)
                                setJobConfig(false, 'config.process[0].train.diff_output_preservation');
                              if (value && processConfig.train.sega_distill)
                                setJobConfig(false, 'config.process[0].train.sega_distill');
                            }}
                          />
                        )}
                      </FormGroup>
                    )}
                  </div>
                  {renderDisclosure(
                    'job-watermark-settings',
                    'Watermarking',
                    'AuthenLoRA mapper, secret bits, and save artifacts',
                    Target,
                    <div className="space-y-4">
                      <FormGroup label="AuthenLoRA">
                        <Checkbox
                          label="Enable watermarking"
                          checked={watermarkEnabled}
                          disabled={!canEnableWatermark}
                          onChange={handleWatermarkToggle}
                        />
                        <Checkbox
                          label="Bake on save"
                          checked={watermarkConfig.bake_on_save || false}
                          onChange={value => setJobConfig(value, 'config.process[0].watermark.bake_on_save')}
                        />
                      </FormGroup>
                      <div className="grid grid-cols-1 gap-x-5 gap-y-3 md:grid-cols-2 xl:grid-cols-4">
                        <SelectInput
                          label="Codec"
                          value={selectedAuthenloraCodec}
                          onChange={handleWatermarkCodecChange}
                          options={AUTHENLORA_CODEC_OPTIONS}
                        />
                        {selectedAuthenloraCodec === 'custom' && (
                          <TextInput
                            label="Codec path"
                            value={watermarkConfig.codec_path || ''}
                            onChange={value => setJobConfig(value, 'config.process[0].watermark.codec_path')}
                            placeholder="E:\\models\\authenlora_codec.pth"
                          />
                        )}
                        <NumberInput
                          label="Message bits"
                          value={watermarkConfig.msg_bits}
                          onChange={value =>
                            setJobConfig(
                              value ?? defaultWatermarkConfig.msg_bits,
                              'config.process[0].watermark.msg_bits',
                            )
                          }
                          min={1}
                        />
                        <TextInput
                          label="Secret bits"
                          value={watermarkConfig.secret || ''}
                          onChange={value => setJobConfig(value, 'config.process[0].watermark.secret')}
                          placeholder="blank for generated"
                        />
                        <NumberInput
                          label="Mapper rank"
                          value={watermarkConfig.mapper_rank}
                          onChange={value =>
                            setJobConfig(
                              value ?? defaultWatermarkConfig.mapper_rank,
                              'config.process[0].watermark.mapper_rank',
                            )
                          }
                          min={1}
                        />
                        <NumberInput
                          label="Mapper LR"
                          value={watermarkConfig.mapper_lr}
                          onChange={value =>
                            setJobConfig(
                              value ?? defaultWatermarkConfig.mapper_lr,
                              'config.process[0].watermark.mapper_lr',
                            )
                          }
                          min={0}
                        />
                        <NumberInput
                          label="Watermark loss"
                          value={watermarkConfig.watermark_loss_weight}
                          onChange={value =>
                            setJobConfig(
                              value ?? defaultWatermarkConfig.watermark_loss_weight,
                              'config.process[0].watermark.watermark_loss_weight',
                            )
                          }
                          min={0}
                        />
                        <NumberInput
                          label="Style loss"
                          value={watermarkConfig.style_loss_weight}
                          onChange={value =>
                            setJobConfig(
                              value ?? defaultWatermarkConfig.style_loss_weight,
                              'config.process[0].watermark.style_loss_weight',
                            )
                          }
                          min={0}
                        />
                        <NumberInput
                          label="Zero message chance"
                          value={watermarkConfig.zero_message_probability}
                          onChange={value =>
                            setJobConfig(
                              value ?? defaultWatermarkConfig.zero_message_probability,
                              'config.process[0].watermark.zero_message_probability',
                            )
                          }
                          min={0}
                          max={1}
                        />
                        <NumberInput
                          label="Verify every"
                          value={watermarkConfig.verify_every}
                          onChange={value =>
                            setJobConfig(
                              value ?? defaultWatermarkConfig.verify_every,
                              'config.process[0].watermark.verify_every',
                            )
                          }
                          min={0}
                        />
                      </div>
                    </div>,
                  )}
                  <TrainingPhasesEditor
                    train={processConfig.train}
                    network={processConfig.network}
                    currentArch={processConfig.model.arch}
                    setJobConfig={setJobConfig}
                    disableTimestepType={disableSections.includes('train.timestep_type')}
                    modelArchName={modelArch?.name}
                    defaultAutoTrainingProfileId={modelArch?.defaultAutoTrainingProfileId}
                  />
                </div>,
              )}

              {renderDisclosure(
                'job-sampling-settings',
                'Sampling',
                'Sampling method and generation settings',
                Layers3,
                <div className={sampleTopStyleClass}>
                  <NumberInput
                    label="Sample every"
                    value={sampleConfig.sample_every}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.sample_every')}
                    placeholder="250"
                    min={1}
                    required
                  />
                  <NumberInput
                    label="Sample Start Step"
                    value={sampleConfig.sample_start_step ?? 0}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.sample_start_step')}
                    placeholder="0"
                    className="pt-2"
                    min={0}
                    required
                  />
                  <SelectInput
                    label="Sampler"
                    value={sampleConfig.sampler}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.sampler')}
                    options={[
                      { value: 'flowmatch', label: 'FlowMatch' },
                      { value: 'ddpm', label: 'DDPM' },
                    ]}
                  />
                  <NumberInput
                    label="Guidance scale"
                    value={sampleConfig.guidance_scale}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.guidance_scale')}
                    placeholder="1.0"
                    min={0}
                    required
                  />
                  <NumberInput
                    label="Sample steps"
                    value={sampleConfig.sample_steps}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.sample_steps')}
                    placeholder="1"
                    min={1}
                    required
                  />
                  {!isAudioModel && (
                    <>
                      <NumberInput
                        label="Width"
                        value={sampleConfig.width}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.width')}
                        placeholder="1024"
                        min={0}
                        required
                      />
                      <NumberInput
                        label="Height"
                        value={sampleConfig.height}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.height')}
                        placeholder="1024"
                        min={0}
                        required
                      />
                    </>
                  )}
                  <NumberInput
                    label="Seed"
                    value={sampleConfig.seed}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.seed')}
                    placeholder="0"
                    min={0}
                    required
                  />
                  <FormGroup label="Advanced sampling">
                    <Checkbox
                      label="Walk seed"
                      checked={sampleConfig.walk_seed}
                      onChange={value => setJobConfig(value, 'config.process[0].sample.walk_seed')}
                    />
                    <Checkbox
                      label="Skip first sample"
                      checked={processConfig.train.skip_first_sample || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.skip_first_sample');
                        if (value) setJobConfig(false, 'config.process[0].train.force_first_sample');
                      }}
                    />
                    <Checkbox
                      label="Force first sample"
                      checked={processConfig.train.force_first_sample || false}
                      docKey={'train.force_first_sample'}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.force_first_sample');
                        if (value) setJobConfig(false, 'config.process[0].train.skip_first_sample');
                      }}
                    />
                    <Checkbox
                      label="Disable sampling"
                      checked={processConfig.train.disable_sampling || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.disable_sampling');
                        if (value) setJobConfig(false, 'config.process[0].train.force_first_sample');
                      }}
                    />
                  </FormGroup>
                </div>,
              )}

              <button
                type="button"
                onClick={() => {
                  (onOpenRawConfig ?? onOpenAdvanced)?.();
                }}
                aria-haspopup="dialog"
                aria-controls="raw-config-drawer"
                className="flex w-full items-center gap-3 border-t border-gray-900 bg-gray-950/30 px-4 py-3 text-left text-sm hover:bg-gray-900/45"
              >
                <TerminalSquare className="h-4 w-4 flex-none text-gray-300" />
                <span className="min-w-0 flex-1">
                  <span className="font-semibold text-gray-100">Raw config</span>
                  <span className="ml-3 hidden text-gray-500 sm:inline">
                    Open the YAML inspector without leaving this workspace
                  </span>
                </span>
                <span className="hidden text-xs text-brand-200 sm:inline">Open drawer</span>
              </button>
            </fieldset>

            <fieldset
              id="job-dataset"
              hidden={!legacyView && activeStepId !== 'job-dataset'}
              disabled={!legacyView && activeStepId !== 'job-dataset'}
              className="studio-training-panel"
            >
              {renderSectionIntro(
                'Dataset',
                'Choose what the model learns from and tune repeats, captions, and resolutions.',
              )}
              <div className="space-y-3">
                {datasetsConfig.map((dataset, i) => (
                  <div key={i} className="border border-gray-900 bg-gray-950/35 p-3">
                    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                      <div className="text-sm font-semibold text-gray-100">Dataset {i + 1}</div>
                      <div className="flex gap-1">
                        <button
                          type="button"
                          onClick={() => duplicateDataset(i)}
                          className="operator-icon-button h-7 w-7"
                          title="Duplicate dataset"
                        >
                          <Copy className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => removeDataset(i)}
                          className="operator-icon-button h-7 w-7 hover:border-rose-800 hover:bg-rose-950/60 hover:text-rose-100"
                          title="Remove dataset"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>
                    <div className={datasetStyleClass}>
                      <div>
                        <SelectInput
                          label="Target dataset"
                          value={dataset.folder_path}
                          onChange={value => setDatasetPath(i, value)}
                          options={datasetOptions}
                        />
                        <NumberInput
                          label="LoRA weight"
                          value={dataset.network_weight}
                          className="pt-2"
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].network_weight`)}
                          placeholder="1.0"
                        />
                        <NumberInput
                          label="Num repeats"
                          value={dataset.num_repeats || 1}
                          className="pt-2"
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].num_repeats`)}
                          placeholder="1"
                          docKey={'dataset.num_repeats'}
                        />
                        <NumberInput
                          label="Dataset batch size"
                          value={dataset.batch_size ?? null}
                          className="pt-2"
                          onChange={value =>
                            setJobConfig(value ?? undefined, `config.process[0].datasets[${i}].batch_size`)
                          }
                          placeholder={`${processConfig.train.batch_size}`}
                          min={1}
                          allowEmpty
                        />
                      </div>
                      <div>
                        <TextInput
                          label="Default caption"
                          value={dataset.default_caption}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].default_caption`)}
                          placeholder="A photo of a cat"
                        />
                        <NumberInput
                          label="Caption dropout"
                          className="pt-2"
                          value={dataset.caption_dropout_rate}
                          onChange={value =>
                            setJobConfig(value, `config.process[0].datasets[${i}].caption_dropout_rate`)
                          }
                          placeholder="0.05"
                          min={0}
                          required
                        />
                      </div>
                      <div>
                        <FormGroup label="Settings">
                          <Checkbox
                            label="Cache latents"
                            checked={dataset.cache_latents_to_disk || false}
                            onChange={value =>
                              setJobConfig(value, `config.process[0].datasets[${i}].cache_latents_to_disk`)
                            }
                          />
                          <Checkbox
                            label="Cache raw tensors"
                            checked={dataset.cache_tensors_to_disk || false}
                            onChange={value =>
                              setJobConfig(value, `config.process[0].datasets[${i}].cache_tensors_to_disk`)
                            }
                          />
                          {modelArch?.additionalSections?.includes('datasets.include_images_in_video_dataset') && (
                            <Checkbox
                              label="Include images with videos"
                              docKey="datasets.include_images_in_video_dataset"
                              checked={dataset.include_images_in_video_dataset || false}
                              onChange={value =>
                                setJobConfig(value, `config.process[0].datasets[${i}].include_images_in_video_dataset`)
                              }
                            />
                          )}
                          <Checkbox
                            label="Is regularization"
                            checked={dataset.is_reg || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].is_reg`)}
                          />
                          {!isAudioModel && (
                            <>
                              <Checkbox
                                label="Flip X"
                                checked={dataset.flip_x || false}
                                onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_x`)}
                              />
                              <Checkbox
                                label="Flip Y"
                                checked={dataset.flip_y || false}
                                onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_y`)}
                              />
                            </>
                          )}
                        </FormGroup>
                      </div>
                      {!isAudioModel && (
                        <div>
                          <FormGroup label="Resolutions">
                            <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                              {[512, 768, 1024, 1280, 1536, 2048].map(res => (
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
                          </FormGroup>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <button type="button" onClick={addDataset} className="operator-button w-full">
                  Add Dataset
                </button>
              </div>
            </fieldset>

            <fieldset
              id="job-samples"
              hidden={!legacyView && activeStepId !== 'job-samples'}
              disabled={!legacyView && activeStepId !== 'job-samples'}
              className="studio-training-panel"
            >
              {renderSectionIntro('Samples', 'Add prompts used to preview the LoRA during training.')}
              <div className="mb-6 space-y-5">
                {' '}
                <TextAreaInput
                  label="Sample prompt (optional)"
                  value={firstSample.prompt ?? ''}
                  onChange={setPrimarySamplePrompt}
                  placeholder="a photo of sks dog in the forest, cinematic lighting"
                  rows={3}
                />
                <p className="-mt-4 text-xs text-gray-500">
                  We will use this to generate previews during training. Add more prompts below.
                </p>
              </div>
              <div className="space-y-3">
                {sampleConfig.samples.map((sample, i) => (
                  <div key={i} className="border border-gray-900 bg-gray-950/35 p-3">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <div className="text-sm font-semibold text-gray-100">Sample prompt {i + 1}</div>
                      <div className="flex gap-1">
                        <button
                          type="button"
                          onClick={() => importRandomPromptFromDataset(i)}
                          disabled={randomPromptLoadingIndex !== null || !canImportRandomPrompt}
                          className="operator-icon-button h-7 w-7 disabled:opacity-50"
                          title={randomPromptDisabledReason}
                        >
                          {randomPromptLoadingIndex === i ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : (
                            <Shuffle className="h-3.5 w-3.5" />
                          )}
                        </button>
                        <button
                          type="button"
                          onClick={() =>
                            setJobConfig(
                              sampleConfig.samples.filter((_, index) => index !== i),
                              'config.process[0].sample.samples',
                            )
                          }
                          className="operator-icon-button h-7 w-7"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>
                    {modelArch?.hasMultiLinePrompts ? (
                      <TextAreaInput
                        label="Prompt"
                        value={sample.prompt}
                        onChange={value => setSamplePromptValue(i, value)}
                        placeholder="Enter prompt"
                        required
                      />
                    ) : (
                      <TextInput
                        label="Prompt"
                        value={sample.prompt}
                        onChange={value => setSamplePromptValue(i, value)}
                        placeholder="Enter prompt"
                        required
                      />
                    )}
                    <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                      {!isAudioModel && (
                        <>
                          <TextInput
                            label="Width"
                            value={sample.width ? `${sample.width}` : ''}
                            onChange={value =>
                              setJobConfig(
                                value ? parseInt(value.replace(/\D/g, '')) || undefined : undefined,
                                `config.process[0].sample.samples[${i}].width`,
                              )
                            }
                            placeholder={`${sampleConfig.width} default`}
                          />
                          <TextInput
                            label="Height"
                            value={sample.height ? `${sample.height}` : ''}
                            onChange={value =>
                              setJobConfig(
                                value ? parseInt(value.replace(/\D/g, '')) || undefined : undefined,
                                `config.process[0].sample.samples[${i}].height`,
                              )
                            }
                            placeholder={`${sampleConfig.height} default`}
                          />
                        </>
                      )}
                      <TextInput
                        label="Seed"
                        value={sample.seed ? `${sample.seed}` : ''}
                        onChange={value =>
                          setJobConfig(
                            value ? parseInt(value.replace(/\D/g, '')) || undefined : undefined,
                            `config.process[0].sample.samples[${i}].seed`,
                          )
                        }
                        placeholder={`${sampleConfig.seed} default`}
                      />
                      <TextInput
                        label="LoRA scale"
                        value={sample.network_multiplier ? `${sample.network_multiplier}` : ''}
                        onChange={value =>
                          setJobConfig(value || undefined, `config.process[0].sample.samples[${i}].network_multiplier`)
                        }
                        placeholder="1.0 default"
                      />
                    </div>
                  </div>
                ))}
                <button
                  type="button"
                  onClick={() =>
                    setJobConfig([...sampleConfig.samples, { prompt: '' }], 'config.process[0].sample.samples')
                  }
                  className="operator-button w-full"
                >
                  Add Prompt
                </button>
              </div>
            </fieldset>

            <fieldset
              id="job-validation"
              hidden={!legacyView && activeStepId !== 'job-validation'}
              disabled={!legacyView && activeStepId !== 'job-validation'}
              className="studio-training-panel"
            >
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="max-w-3xl">
                  {renderSectionIntro(
                    'Validation',
                    'Measure deterministic loss against images that are held out from the training dataset.',
                  )}
                  <p className="text-xs leading-5 text-gray-500">
                    Validation images are encoded once and evaluated with fixed noise, making the resulting{' '}
                    <span className="font-mono text-gray-300">val/loss</span> series comparable throughout the run. Do
                    not include these images in a training dataset.
                  </p>
                </div>
                <Checkbox
                  label="Enable validation loss"
                  checked={!!validationConfig}
                  onChange={enabled =>
                    setJobConfig(
                      enabled ? objectCopy(defaultValidationConfig) : undefined,
                      'config.process[0].train.validation_config',
                    )
                  }
                />
              </div>

              {validationConfig ? (
                <div className="mt-4 space-y-4">
                  <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
                    <NumberInput
                      label="Validate every"
                      value={validationConfig.validate_every_n_steps}
                      onChange={value =>
                        setJobConfig(
                          value ?? defaultValidationConfig.validate_every_n_steps,
                          'config.process[0].train.validation_config.validate_every_n_steps',
                        )
                      }
                      placeholder="1"
                      min={1}
                      required
                    />
                    <NumberInput
                      label="Validation resolution"
                      value={validationConfig.resolution}
                      onChange={value =>
                        setJobConfig(
                          value ?? defaultValidationConfig.resolution,
                          'config.process[0].train.validation_config.resolution',
                        )
                      }
                      placeholder="1024"
                      min={64}
                      required
                    />
                    <SelectInput
                      label="Validation sigmas"
                      value={(
                        validationConfig.validation_sigmas ??
                        defaultValidationConfig.validation_sigmas ?? [0.5]
                      ).join(',')}
                      onChange={value =>
                        setJobConfig(
                          value
                            .split(',')
                            .map(sigma => Number.parseFloat(sigma.trim()))
                            .filter(Number.isFinite),
                          'config.process[0].train.validation_config.validation_sigmas',
                        )
                      }
                      options={[
                        { value: '0.5', label: '0.5' },
                        { value: '1,0.5', label: '1.0, 0.5' },
                        { value: '1,0.66,0.33', label: '1.0, 0.66, 0.33' },
                        { value: '1,0.75,0.5,0.25', label: '1.0, 0.75, 0.5, 0.25' },
                      ]}
                    />
                  </div>

                  <div>
                    <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
                      Held-out images ({validationItems.length})
                    </div>
                    <div className="space-y-2">
                      {validationItems.map((item, index) => (
                        <div
                          key={index}
                          className="flex flex-col gap-3 border border-gray-900 bg-gray-950/35 p-3 sm:flex-row sm:items-center"
                        >
                          <SampleControlImage
                            instruction="Add image"
                            src={typeof item.image_path === 'string' ? item.image_path || null : null}
                            projectID={projectID}
                            onNewImageSelected={imagePath =>
                              setJobConfig(
                                imagePath ?? '',
                                `config.process[0].train.validation_config.validation_items[${index}].image_path`,
                              )
                            }
                          />
                          <TextInput
                            className="min-w-0 flex-1"
                            label={`Validation prompt ${index + 1}`}
                            value={typeof item.prompt === 'string' ? item.prompt : ''}
                            onChange={value =>
                              setJobConfig(
                                value,
                                `config.process[0].train.validation_config.validation_items[${index}].prompt`,
                              )
                            }
                            placeholder="A photo of [trigger]..."
                          />
                          <button
                            type="button"
                            onClick={() =>
                              setJobConfig(
                                validationItems.filter((_, itemIndex) => itemIndex !== index),
                                'config.process[0].train.validation_config.validation_items',
                              )
                            }
                            className="operator-icon-button h-8 w-8 flex-none self-end hover:border-rose-800 hover:bg-rose-950/60 hover:text-rose-100 sm:self-center"
                            title="Remove validation image"
                            aria-label={`Remove validation image ${index + 1}`}
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      ))}
                    </div>
                    <button
                      type="button"
                      onClick={() =>
                        setJobConfig(
                          [...validationItems, { image_path: '', prompt: '' }],
                          'config.process[0].train.validation_config.validation_items',
                        )
                      }
                      className="operator-button mt-3 w-full"
                    >
                      Add Validation Image
                    </button>
                  </div>
                </div>
              ) : (
                <div className="mt-4 border border-dashed border-gray-800 bg-gray-950/30 px-3 py-4 text-xs text-gray-600">
                  Validation is optional. Enable it to configure held-out images and cadence.
                </div>
              )}
            </fieldset>

            <fieldset
              id="job-review"
              hidden={!legacyView && activeStepId !== 'job-review'}
              disabled={!legacyView && activeStepId !== 'job-review'}
              className="studio-training-panel"
            >
              {renderSectionIntro(
                'Review',
                'Confirm the worker, dataset, training plan, and any warnings before creating the job.',
              )}
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <div className="border border-gray-900 bg-gray-950/35 p-3">
                  <h3 className="text-sm font-semibold text-gray-100">Training plan</h3>
                  <div className="mt-3 grid grid-cols-[1fr_auto] gap-y-2 text-sm">
                    <div className="text-gray-500">Trainer</div>
                    <div className="text-gray-200">{trainerLabel}</div>
                    <div className="text-gray-500">Architecture</div>
                    <div className="text-gray-200">{modelArch?.label || processConfig.model.arch}</div>
                    <div className="text-gray-500">Target</div>
                    <div className="text-gray-200">
                      {networkType === 'lokr' ? 'LoKr' : networkType === 'dora' ? 'DoRA' : 'LoRA'}
                    </div>
                    <div className="text-gray-500">Steps</div>
                    <div className="text-gray-200">{autoTrain ? 'Auto' : formatNumber(processConfig.train.steps)}</div>
                    <div className="text-gray-500">Batch size</div>
                    <div className="text-gray-200">{formatNumber(processConfig.train.batch_size)}</div>
                    <div className="text-gray-500">Learning rate</div>
                    <div className="text-gray-200">{processConfig.train.lr}</div>
                  </div>
                </div>

                <div className="border border-gray-900 bg-gray-950/35 p-3">
                  <h3 className="text-sm font-semibold text-gray-100">Worker and data</h3>
                  <div className="mt-3 grid grid-cols-[1fr_auto] gap-y-2 text-sm">
                    <div className="text-gray-500">Worker</div>
                    <div className="text-gray-200">{workerLabel}</div>
                    <div className="text-gray-500">GPU</div>
                    <div className="text-right text-gray-200">{selectedGpuLabel}</div>
                    <div className="text-gray-500">Dataset</div>
                    <div className="max-w-52 truncate text-right text-gray-200">
                      {selectedDatasetOption?.label || 'Not selected'}
                    </div>
                    <div className="text-gray-500">Saves every</div>
                    <div className="text-gray-200">{formatNumber(processConfig.save.save_every)} steps</div>
                    <div className="text-gray-500">Data type</div>
                    <div className="text-gray-200">{processConfig.save.dtype?.toUpperCase()}</div>
                  </div>
                </div>
              </div>

              <div className="mt-4 border border-gray-900 bg-gray-950/35 p-3">
                <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-gray-100">
                  {readinessErrors.length > 0 ? (
                    <AlertTriangle className="h-4 w-4 text-rose-300" />
                  ) : readinessWarnings.length > 0 ? (
                    <AlertTriangle className="h-4 w-4 text-amber-300" />
                  ) : (
                    <CheckCircle2 className="h-4 w-4 text-emerald-300" />
                  )}
                  {readinessErrors.length > 0
                    ? `Issues (${readinessErrors.length})`
                    : readinessWarnings.length > 0
                      ? `Warnings (${readinessWarnings.length})`
                      : 'No blocking issues'}
                </div>
                {readinessMessages.length > 0 ? (
                  <div className="space-y-2">
                    {readinessMessages.slice(0, 4).map((message, index) => (
                      <div
                        key={`${message.level}-${index}`}
                        className={`border px-2 py-2 text-xs ${message.level === 'error' ? 'border-rose-900 bg-rose-950/25 text-rose-200' : 'border-amber-900 bg-amber-950/20 text-amber-200'}`}
                      >
                        {message.message}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-gray-500">Your setup is ready to be queued.</p>
                )}
              </div>

              <button type="submit" disabled={status === 'saving'} className="operator-button-primary mt-5 h-11 w-full">
                <Sparkles className="h-4 w-4" />
                {status === 'saving' ? 'Saving...' : runId ? 'Update Job' : 'Create Job'}
              </button>
            </fieldset>

            {!legacyView && (
              <div className="mt-5 flex items-center justify-between gap-3">
                <button
                  type="button"
                  className="operator-button h-11 px-5"
                  disabled={activeStepIndex === 0}
                  onClick={() => selectStep(stepItems[activeStepIndex - 1].id)}
                >
                  Back
                </button>
                {activeStepIndex < stepItems.length - 1 && (
                  <button type="button" className="operator-button-primary h-11 px-6" onClick={nextStep}>
                    Continue to {stepItems[activeStepIndex + 1].title.toLowerCase()}{' '}
                    <ChevronDown className="h-4 w-4 -rotate-90" />
                  </button>
                )}
              </div>
            )}
          </div>
          <aside id="job-readiness-rail" className="min-w-0">
            <div className="operator-panel space-y-6 p-6">
              <h2 className="text-lg font-semibold text-gray-100">Run overview</h2>
              <div className="space-y-3">
                {computeControls || (
                  <div className="flex items-center gap-2 text-sm text-gray-400">
                    <Cpu className="h-4 w-4" />
                    {workerLabel}
                  </div>
                )}
                <p className="break-words text-sm text-gray-200">{selectedGpuLabel}</p>
                <p className="text-xs text-gray-400">{selectedGpuMemory}</p>
                {showGPUSelect && (
                  <SelectInput
                    label="GPU"
                    value={`${gpuIDs}`}
                    onChange={value => setGpuIDs(value)}
                    options={gpuList.map((gpu: { index: number }) => ({
                      value: `${gpu.index}`,
                      label: `GPU #${gpu.index}`,
                    }))}
                  />
                )}
              </div>
              <dl className="grid grid-cols-[1fr_minmax(0,1fr)] gap-x-4 gap-y-4 border-t border-gray-800 pt-6 text-sm">
                <dt className="text-gray-400">Model</dt>
                <dd className="text-right text-gray-100">{modelArch?.label || processConfig.model.arch}</dd>
                <dt className="text-gray-400">Target</dt>
                <dd className="text-right text-gray-100">
                  {networkType === 'lokr' ? 'LoKr' : networkType === 'dora' ? 'DoRA' : 'LoRA'}
                </dd>
                <dt className="text-gray-400">Dataset</dt>
                <dd className="break-words text-right text-gray-100">
                  {selectedDatasetOption?.label || (unresolvedDataset ? 'Not selected' : firstDataset.folder_path)}
                </dd>
                {(legacyView || activeStepIndex >= 2) && (
                  <>
                    <dt className="text-gray-400">Steps</dt>
                    <dd className="text-right text-gray-100">
                      {autoTrain ? 'Auto' : formatNumber(processConfig.train.steps)}
                    </dd>
                    <dt className="text-gray-400">Learning rate</dt>
                    <dd className="text-right text-gray-100">{processConfig.train.lr}</dd>
                  </>
                )}
              </dl>
              <p className="flex items-start gap-2 border-t border-gray-800 pt-5 text-xs leading-5 text-gray-400">
                <Info className="mt-0.5 h-4 w-4 shrink-0" />
                You can review everything before creating the job.
              </p>
            </div>
            <button
              type="button"
              className="mt-5 inline-flex items-center gap-2 px-3 py-2 text-sm text-gray-300 hover:text-brand-400"
              onClick={onOpenRawConfig ?? onOpenAdvanced}
            >
              <TerminalSquare className="h-4 w-4" />
              Open raw configuration
            </button>
            {(legacyView || activeStepId === 'job-review') && (
              <div className="operator-panel mt-5 p-5">
                <TrainingAdvisorPanel jobConfig={jobConfig} gpuIDs={gpuIDs} variant="rail" />
              </div>
            )}
          </aside>
        </div>
      </form>
      <AddSingleImageModal />
    </>
  );
}
