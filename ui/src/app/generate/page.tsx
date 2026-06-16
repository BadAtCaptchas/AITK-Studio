'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@headlessui/react';
import { ArrowRight, FileJson, ImagePlus, Layers, Loader2, Upload, Wand2, X } from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';
import {
  Checkbox,
  CreatableSelectInput,
  FormGroup,
  NumberInput,
  SelectInput,
  SliderInput,
  TextAreaInput,
  TextInput,
} from '@/components/formInputs';
import JobsTable from '@/components/JobsTable';
import useGPUInfo from '@/hooks/useGPUInfo';
import useSettings from '@/hooks/useSettings';
import { apiClient } from '@/utils/api';
import { startJob } from '@/utils/jobs';
import { getMediaUrl } from '@/utils/media';
import { startQueue } from '@/utils/queue';
import type { ComfyConfig, ComfyMode, ComfyOnError, GenerationBackend, ModelConfig, SelectOption } from '@/types';
import { groupedModelOptions, modelArchs, quantizationOptions } from '@/app/jobs/new/options';
import { PageNotice } from '@/components/OperatorPrimitives';
import { getLayerOffloadingMemoryProfile, type LayerOffloadingBackend } from '@/utils/memoryProfiles';

type GeneratedLora = {
  id: string;
  label: string;
  path: string;
  filename: string;
  source: 'job' | 'uploaded';
  jobId?: string;
  jobName?: string;
  jobStatus?: string;
  updatedAt: string;
  sizeBytes: number;
  triggerWords?: string[];
  triggerWordSource?: 'metadata' | 'user' | 'none';
  originalFilename?: string;
  model?: Partial<ModelConfig> & Record<string, unknown>;
};

type GeneratorModelConfig = ModelConfig & {
  dtype?: string;
  lora_path?: string;
  inference_lora_path?: string;
  vae_path?: string;
  refiner_name_or_path?: string;
  te_name_or_path?: string;
  extras_name_or_path?: string;
  quantize_kwargs?: ModelConfig['quantize_kwargs'];
  [key: string]: unknown;
};

type PromptImageSettings = {
  prompt: string;
  width?: number;
  height?: number;
  seed?: number;
  guidance_scale?: number;
  guidance?: number;
  sample_steps?: number;
  steps?: number;
  num_inference_steps?: number;
  sampler?: string;
  ext?: string;
  format?: string;
  neg?: string;
  negative_prompt?: string;
  prompt_2?: string;
  neg_2?: string;
  negative_prompt_2?: string;
  guidance_rescale?: number;
  network_multiplier?: number;
  [key: string]: unknown;
};

const dtypeOptions: SelectOption[] = [
  { value: 'bf16', label: 'bf16' },
  { value: 'float16', label: 'float16' },
  { value: 'float32', label: 'float32' },
];

const samplerOptions: SelectOption[] = [
  { value: 'flowmatch', label: 'flowmatch' },
  { value: 'ddpm', label: 'ddpm' },
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

const imageFormatOptions: SelectOption[] = [
  { value: 'png', label: 'PNG' },
  { value: 'jpg', label: 'JPG' },
  { value: 'webp', label: 'WEBP' },
  { value: 'jxl', label: 'JXL' },
];

const layerOffloadingBackendOptions: SelectOption[] = [
  { value: 'block', label: 'Block' },
  { value: 'legacy', label: 'Legacy' },
];

function getArchDefault(archName: string, key: string, fallback: unknown) {
  const arch = modelArchs.find(item => item.name === archName);
  const value = arch?.defaults?.[key];
  if (Array.isArray(value)) {
    return value[0] ?? fallback;
  }
  return value ?? fallback;
}

function getArchNumberDefault(archName: string, key: string, fallback: number) {
  const value = getArchDefault(archName, key, fallback);
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function archSupportsSection(archName: string, section: 'model.layer_offloading') {
  return Boolean(modelArchs.find(item => item.name === archName)?.additionalSections?.includes(section));
}

function getDefaultModelConfig(archName: string): GeneratorModelConfig {
  const memoryProfile = getLayerOffloadingMemoryProfile(archName);
  return {
    name_or_path: String(getArchDefault(archName, 'config.process[0].model.name_or_path', '')),
    arch: archName,
    quantize: Boolean(getArchDefault(archName, 'config.process[0].model.quantize', false)),
    quantize_te: Boolean(getArchDefault(archName, 'config.process[0].model.quantize_te', false)),
    qtype: 'qfloat8',
    qtype_te: 'qfloat8',
    low_vram: false,
    model_kwargs:
      (getArchDefault(archName, 'config.process[0].model.model_kwargs', {}) as Record<string, unknown>) || {},
    dtype: String(getArchDefault(archName, 'config.process[0].train.dtype', 'bf16')),
    layer_offloading: false,
    layer_offloading_backend: String(
      getArchDefault(archName, 'config.process[0].model.layer_offloading_backend', memoryProfile.backend),
    ) as LayerOffloadingBackend,
    layer_offloading_transformer_percent: getArchNumberDefault(
      archName,
      'config.process[0].model.layer_offloading_transformer_percent',
      memoryProfile.transformerPercent,
    ),
    layer_offloading_text_encoder_percent: getArchNumberDefault(
      archName,
      'config.process[0].model.layer_offloading_text_encoder_percent',
      memoryProfile.textEncoderPercent,
    ),
  };
}

function getDefaultSampler(archName: string) {
  return String(getArchDefault(archName, 'config.process[0].sample.sampler', 'flowmatch'));
}

function sanitizeJobName(value: string) {
  return value
    .trim()
    .replace(/[^a-zA-Z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 80);
}

function makeDefaultJobName() {
  const timestamp = new Date()
    .toISOString()
    .replaceAll('-', '')
    .replaceAll(':', '')
    .replaceAll('.', '')
    .replace('T', '')
    .replace('Z', '')
    .slice(0, 14);
  return `generate_${timestamp}`;
}

function joinPath(root: string, ...parts: string[]) {
  const separator = root.includes('\\') ? '\\' : '/';
  return [root.replace(/[\\/]+$/, ''), ...parts.map(part => part.replace(/^[\\/]+|[\\/]+$/g, ''))].join(separator);
}

function splitPrompts(value: string) {
  return value
    .split(/\r?\n/)
    .map(prompt => prompt.trim())
    .filter(Boolean);
}

function toFiniteNumber(value: unknown, fallback: number) {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function promptItemsFromText(value: string): PromptImageSettings[] {
  return splitPrompts(value).map(prompt => ({ prompt }));
}

function normalizePromptEntry(item: unknown): PromptImageSettings | null {
  if (typeof item === 'string') {
    const prompt = item.trim();
    return prompt ? { prompt } : null;
  }

  if (!item || typeof item !== 'object') {
    return null;
  }

  const raw = item as Record<string, unknown>;
  const prompt = String(raw.prompt ?? raw.text ?? raw.caption ?? '').trim();
  if (!prompt) {
    return null;
  }

  const normalized: PromptImageSettings = { ...raw, prompt };
  if (raw.guidance != null && raw.guidance_scale == null) normalized.guidance_scale = toFiniteNumber(raw.guidance, 4);
  if (raw.steps != null && raw.sample_steps == null) normalized.sample_steps = toFiniteNumber(raw.steps, 20);
  if (raw.num_inference_steps != null && raw.sample_steps == null) {
    normalized.sample_steps = toFiniteNumber(raw.num_inference_steps, 20);
  }
  if (raw.negative_prompt != null && raw.neg == null) normalized.neg = String(raw.negative_prompt);
  if (raw.format != null && raw.ext == null) normalized.ext = String(raw.format);
  return normalized;
}

function promptItemsFromJsonText(value: string): PromptImageSettings[] {
  const parsed = JSON.parse(value);
  let source: unknown = parsed;

  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    const parsedObject = parsed as Record<string, unknown>;
    if (Array.isArray(parsedObject.images)) source = parsedObject.images;
    else if (Array.isArray(parsedObject.prompts)) source = parsedObject.prompts;
    else if (Array.isArray(parsedObject.samples)) source = parsedObject.samples;
    else if (parsedObject.prompt != null) source = [parsedObject];
  }

  const items = Array.isArray(source) ? source : [source];
  return items.map(normalizePromptEntry).filter((item): item is PromptImageSettings => item !== null);
}

function cleanPromptImageSettings(item: PromptImageSettings) {
  const cleaned: PromptImageSettings = { prompt: item.prompt };
  Object.entries(item).forEach(([key, value]) => {
    if (key === 'prompt' || value == null || value === '') return;
    cleaned[key] = value;
  });
  return cleaned;
}

function getPromptNumber(item: PromptImageSettings, keys: string[], fallback: number) {
  for (const key of keys) {
    if (item[key] != null && item[key] !== '') {
      return toFiniteNumber(item[key], fallback);
    }
  }
  return fallback;
}

function getPromptString(item: PromptImageSettings, keys: string[], fallback: string) {
  for (const key of keys) {
    if (item[key] != null && item[key] !== '') {
      return String(item[key]);
    }
  }
  return fallback;
}

function cleanModelConfig(modelConfig: GeneratorModelConfig, useLora: boolean, loraPath: string) {
  const model: GeneratorModelConfig = {
    ...modelConfig,
    name_or_path: modelConfig.name_or_path.trim(),
    arch: modelConfig.arch,
    dtype: modelConfig.dtype || 'bf16',
    qtype: modelConfig.quantize ? modelConfig.qtype || 'qfloat8' : '',
    qtype_te: modelConfig.quantize_te ? modelConfig.qtype_te || 'qfloat8' : '',
    model_kwargs: modelConfig.model_kwargs || {},
  };

  delete model.assistant_lora_path;
  delete model.inference_lora_path;
  delete model.lora_path;

  if (useLora && loraPath.trim()) {
    model.lora_path = loraPath.trim();
  }

  return model;
}

function formatLoraSource(lora: GeneratedLora) {
  return lora.source === 'uploaded' ? 'Uploaded' : lora.jobName || 'Training job';
}

function formatMegabytes(bytes: number) {
  return `${Math.max(1, Math.round(bytes / 1024 / 1024))} MB`;
}

function promptHasAnyTrigger(prompt: string, triggerWords: string[]) {
  const lowerPrompt = prompt.toLowerCase();
  return triggerWords.some(word => lowerPrompt.includes(word.toLowerCase()));
}

export default function GeneratePage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const loraFileInputRef = useRef<HTMLInputElement | null>(null);
  const inlineAbortControllerRef = useRef<AbortController | null>(null);
  const statusResetTimeoutRef = useRef<number | null>(null);
  const { settings, isSettingsLoaded } = useSettings();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const [jobName, setJobName] = useState(makeDefaultJobName);
  const [modelConfig, setModelConfig] = useState<GeneratorModelConfig>(() => getDefaultModelConfig('flux'));
  const [useLora, setUseLora] = useState(false);
  const [loraPath, setLoraPath] = useState('');
  const [loras, setLoras] = useState<GeneratedLora[]>([]);
  const [loraStatus, setLoraStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [loraUploadStatus, setLoraUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const [loraUploadProgress, setLoraUploadProgress] = useState(0);
  const [loraUploadTriggerWords, setLoraUploadTriggerWords] = useState('');
  const [loraUploadMessage, setLoraUploadMessage] = useState('');
  const [prompts, setPrompts] = useState('photo of a cinematic portrait, detailed lighting');
  const [jsonPromptItems, setJsonPromptItems] = useState<PromptImageSettings[] | null>(null);
  const [importSummary, setImportSummary] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [width, setWidth] = useState<number | null>(1024);
  const [height, setHeight] = useState<number | null>(1024);
  const [seed, setSeed] = useState<number | null>(-1);
  const [guidanceScale, setGuidanceScale] = useState<number | null>(4);
  const [sampleSteps, setSampleSteps] = useState<number | null>(20);
  const [numRepeats, setNumRepeats] = useState<number | null>(1);
  const [sampler, setSampler] = useState(getDefaultSampler('flux'));
  const [imageFormat, setImageFormat] = useState('png');
  const [writePromptFile, setWritePromptFile] = useState(true);
  const [startImmediately, setStartImmediately] = useState(true);
  const [generationBackend, setGenerationBackend] = useState<GenerationBackend>('native');
  const [comfyMode, setComfyMode] = useState<ComfyMode>('external');
  const [comfyServerUrl, setComfyServerUrl] = useState('');
  const [comfyManagedInstall, setComfyManagedInstall] = useState(false);
  const [comfyRoot, setComfyRoot] = useState('');
  const [comfyWorkflowName, setComfyWorkflowName] = useState('auto');
  const [comfyWorkflowPath, setComfyWorkflowPath] = useState('');
  const [comfyOnError, setComfyOnError] = useState<ComfyOnError>('fail');
  const [status, setStatus] = useState<'idle' | 'saving' | 'generating' | 'error'>('idle');
  const [inlineImagePath, setInlineImagePath] = useState('');
  const [inlineError, setInlineError] = useState('');
  const [inlineMessage, setInlineMessage] = useState('');
  const [cancelRequested, setCancelRequested] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  useEffect(() => {
    if (isGPUInfoLoaded && gpuIDs === null) {
      setGpuIDs(gpuList.length > 0 ? `${gpuList[0].index}` : '0');
    }
  }, [gpuIDs, gpuList, isGPUInfoLoaded]);

  const refreshLoras = useCallback(async () => {
    setLoraStatus('loading');
    try {
      const res = await apiClient.get('/api/generate/loras');
      setLoras(res.data.loras || []);
      setLoraStatus('success');
    } catch (error) {
      console.error('Error fetching LoRAs:', error);
      setLoraStatus('error');
    }
  }, []);

  useEffect(() => {
    void refreshLoras();
  }, [refreshLoras]);

  useEffect(() => {
    return () => {
      inlineAbortControllerRef.current?.abort();
      inlineAbortControllerRef.current = null;
      if (statusResetTimeoutRef.current !== null) {
        window.clearTimeout(statusResetTimeoutRef.current);
      }
    };
  }, []);

  const loraOptions = useMemo<SelectOption[]>(
    () => loras.map(lora => ({ value: lora.path, label: lora.label })),
    [loras],
  );

  const selectedLora = useMemo(() => loras.find(lora => lora.path === loraPath), [loras, loraPath]);

  const currentPromptItems = useMemo(() => jsonPromptItems ?? promptItemsFromText(prompts), [jsonPromptItems, prompts]);

  const imageCount = useMemo(() => {
    const repeats = Math.max(1, Math.floor(numRepeats || 1));
    return currentPromptItems.length * repeats;
  }, [currentPromptItems, numRepeats]);
  const supportsLayerOffloading = useMemo(
    () => archSupportsSection(modelConfig.arch, 'model.layer_offloading') || Boolean(modelConfig.layer_offloading),
    [modelConfig.arch, modelConfig.layer_offloading],
  );
  const layerOffloadingMemoryProfile = useMemo(
    () => getLayerOffloadingMemoryProfile(modelConfig.arch),
    [modelConfig.arch],
  );

  const isBusy = status === 'saving' || status === 'generating';
  const isManagedComfyGeneration = generationBackend === 'comfy' && comfyMode === 'managed';
  const primaryButtonLabel =
    status === 'generating'
      ? isManagedComfyGeneration
        ? 'Preparing ComfyUI...'
        : 'Generating...'
      : status === 'saving'
        ? 'Creating...'
        : imageCount === 1
          ? 'Generate Image'
          : 'Create Job';

  const clearStatusResetTimeout = () => {
    if (statusResetTimeoutRef.current === null) return;
    window.clearTimeout(statusResetTimeoutRef.current);
    statusResetTimeoutRef.current = null;
  };

  const scheduleStatusIdle = () => {
    clearStatusResetTimeout();
    statusResetTimeoutRef.current = window.setTimeout(() => {
      setStatus('idle');
      statusResetTimeoutRef.current = null;
    }, 1500);
  };

  const applyLoraModelDefaults = (lora: GeneratedLora) => {
    if (!lora.model) return;
    setModelConfig(current => ({
      ...current,
      ...lora.model,
      name_or_path: String(lora.model?.name_or_path || current.name_or_path),
      arch: String(lora.model?.arch || current.arch),
      dtype: String((lora.model as GeneratorModelConfig).dtype || current.dtype || 'bf16'),
      model_kwargs: (lora.model?.model_kwargs as Record<string, unknown>) || current.model_kwargs || {},
      low_vram: current.low_vram,
      layer_offloading: current.layer_offloading,
      layer_offloading_backend: current.layer_offloading_backend,
      layer_offloading_transformer_percent: current.layer_offloading_transformer_percent,
      layer_offloading_text_encoder_percent: current.layer_offloading_text_encoder_percent,
    }));
    if (lora.model.arch) {
      setSampler(getDefaultSampler(String(lora.model.arch)));
    }
  };

  const handleLoraPathChange = (value: string) => {
    setLoraPath(value);
    const lora = loras.find(item => item.path === value);
    if (lora) {
      applyLoraModelDefaults(lora);
      setLoraUploadTriggerWords(lora.triggerWords?.join(', ') || '');
    }
  };

  const handleUseLoraChange = (checked: boolean) => {
    setUseLora(checked);
    if (checked && !loraPath && loras[0]) {
      setLoraPath(loras[0].path);
      applyLoraModelDefaults(loras[0]);
      setLoraUploadTriggerWords(loras[0].triggerWords?.join(', ') || '');
    }
  };

  const handleLoraUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.safetensors')) {
      setLoraUploadStatus('error');
      setLoraUploadMessage('LoRA upload must be a .safetensors file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    if (loraUploadTriggerWords.trim()) {
      formData.append('trigger_words', loraUploadTriggerWords.trim());
    }

    setLoraUploadStatus('uploading');
    setLoraUploadProgress(0);
    setLoraUploadMessage('');
    try {
      const res = await apiClient.post('/api/generate/loras/upload', formData, {
        onUploadProgress: progressEvent => {
          if (!progressEvent.total) return;
          setLoraUploadProgress(Math.round((progressEvent.loaded * 100) / progressEvent.total));
        },
      });
      const uploaded = res.data.lora as GeneratedLora;
      setLoras(current => [uploaded, ...current.filter(item => item.path !== uploaded.path)]);
      setUseLora(true);
      setLoraPath(uploaded.path);
      applyLoraModelDefaults(uploaded);
      if (!loraUploadTriggerWords.trim() && uploaded.triggerWords?.length) {
        setLoraUploadTriggerWords(uploaded.triggerWords.join(', '));
      }
      setLoraUploadStatus('success');
      setLoraUploadMessage(
        uploaded.triggerWords?.length
          ? `Uploaded with trigger: ${uploaded.triggerWords.join(', ')}`
          : 'Uploaded. No trigger metadata found.',
      );
      void refreshLoras();
    } catch (error: any) {
      console.error('Error uploading LoRA:', error);
      setLoraUploadStatus('error');
      setLoraUploadMessage(error.response?.data?.error || 'Failed to upload LoRA.');
    } finally {
      setLoraUploadProgress(0);
      if (loraFileInputRef.current) {
        loraFileInputRef.current.value = '';
      }
    }
  };

  const insertSelectedLoraTrigger = () => {
    const triggerWords = selectedLora?.triggerWords?.filter(Boolean) ?? [];
    if (triggerWords.length === 0) return;

    const triggerText = triggerWords.join(', ');
    setPrompts(current => {
      const lines = current.split(/\r?\n/);
      if (lines.length === 0 || lines.every(line => !line.trim())) return triggerText;
      return lines.map(line => {
        if (!line.trim() || promptHasAnyTrigger(line, triggerWords)) return line;
        return `${triggerText}, ${line}`;
      }).join('\n');
    });
    setJsonPromptItems(null);
    setImportSummary('');
  };

  const handleArchChange = (archName: string) => {
    setModelConfig(getDefaultModelConfig(archName));
    setSampler(getDefaultSampler(archName));
  };

  const handleLayerOffloadingChange = (checked: boolean) => {
    setModelConfig(current => ({
      ...current,
      layer_offloading: checked,
      layer_offloading_backend: current.layer_offloading_backend ?? layerOffloadingMemoryProfile.backend,
      layer_offloading_transformer_percent:
        current.layer_offloading_transformer_percent ?? layerOffloadingMemoryProfile.transformerPercent,
      layer_offloading_text_encoder_percent:
        current.layer_offloading_text_encoder_percent ?? layerOffloadingMemoryProfile.textEncoderPercent,
    }));
  };

  const handleLayerOffloadingPercentChange = (
    key: 'layer_offloading_transformer_percent' | 'layer_offloading_text_encoder_percent',
    value: number,
  ) => {
    setModelConfig(current => ({
      ...current,
      [key]: Math.max(0, Math.min(1, value * 0.01)),
    }));
  };

  const handlePromptTextChange = (value: string) => {
    setPrompts(value);
    setJsonPromptItems(null);
    setImportSummary('');
    setInlineError('');
    setInlineMessage('');
  };

  const handlePromptFileImport = async (file: File) => {
    try {
      const fileText = await file.text();
      const isJsonFile = file.name.toLowerCase().endsWith('.json') || file.type === 'application/json';
      if (isJsonFile) {
        const importedItems = promptItemsFromJsonText(fileText);
        if (importedItems.length === 0) {
          alert('No prompts were found in the JSON file.');
          return;
        }
        setJsonPromptItems(importedItems);
        setPrompts(importedItems.map(item => item.prompt).join('\n'));
        setImportSummary(`Loaded ${importedItems.length} JSON prompt${importedItems.length === 1 ? '' : 's'}.`);
      } else {
        const importedItems = promptItemsFromText(fileText);
        if (importedItems.length === 0) {
          alert('No prompts were found in the text file.');
          return;
        }
        setJsonPromptItems(null);
        setPrompts(importedItems.map(item => item.prompt).join('\n'));
        setImportSummary(`Loaded ${importedItems.length} text prompt${importedItems.length === 1 ? '' : 's'}.`);
      }
      setInlineImagePath('');
      setInlineError('');
      setInlineMessage('');
    } catch (error) {
      console.error('Error importing prompt file:', error);
      alert('Failed to import prompt file.');
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const getGenerationValidationErrors = (promptItems: PromptImageSettings[], model: GeneratorModelConfig) => {
    const errors: string[] = [];
    if (!isSettingsLoaded || !settings.TRAINING_FOLDER) {
      errors.push('Settings are still loading. Try again after the training folder is available.');
    }
    if (!gpuIDs) {
      errors.push('Select a GPU before generating.');
    }
    if (!model.name_or_path) {
      errors.push('Select a base model before generating.');
    }
    if (useLora && !loraPath.trim()) {
      errors.push('Select a LoRA or enter a LoRA path.');
    }
    if (promptItems.length === 0) {
      errors.push('Enter at least one prompt.');
    }
    if (!jobName.trim()) {
      errors.push('Job name is required.');
    }
    if (jobName.trim() === '.' || jobName.includes('..') || /[\\/]/.test(jobName)) {
      errors.push('Job name cannot contain path separators or "..".');
    }
    if (!width || width < 64 || width > 4096) {
      errors.push('Width must be between 64 and 4096.');
    }
    if (!height || height < 64 || height > 4096) {
      errors.push('Height must be between 64 and 4096.');
    }
    if (!sampleSteps || sampleSteps < 1 || sampleSteps > 200) {
      errors.push('Steps must be between 1 and 200.');
    }
    if (!numRepeats || numRepeats < 1 || numRepeats > 100) {
      errors.push('Images per prompt must be between 1 and 100.');
    }
    if (generationBackend === 'comfy' && comfyMode === 'external' && !comfyServerUrl.trim()) {
      errors.push('ComfyUI server URL is required for external mode.');
    }
    return errors;
  };

  const validateGeneration = (promptItems: PromptImageSettings[], model: GeneratorModelConfig) => {
    const errors = getGenerationValidationErrors(promptItems, model);
    setValidationErrors(errors);
    return errors.length === 0;
  };

  const buildComfyConfig = (): ComfyConfig => {
    const comfyAutoInstall = settings.COMFY_AUTO_INSTALL === 'true';
    const comfy: ComfyConfig = {
      mode: comfyMode,
      workflow_name: comfyWorkflowName.trim() || 'auto',
      on_error: comfyOnError,
    };
    if (comfyMode === 'external') {
      comfy.server_url = comfyServerUrl.trim();
    }
    if (comfyMode === 'managed') {
      comfy.managed_install = comfyAutoInstall || comfyManagedInstall;
      if (comfyRoot.trim()) comfy.root = comfyRoot.trim();
    }
    if (comfyWorkflowPath.trim()) {
      comfy.workflow = comfyWorkflowPath.trim();
    }
    return comfy;
  };

  const buildGenerateJobConfig = (
    promptItems: PromptImageSettings[],
    normalizedJobName: string,
    model: GeneratorModelConfig,
  ) => {
    const promptList = promptItems.map(item => item.prompt);

    const sampleItems = promptItems.map(item => ({
      prompt: item.prompt,
      width: getPromptNumber(item, ['width'], width || 1024),
      height: getPromptNumber(item, ['height'], height || 1024),
      neg: getPromptString(item, ['neg', 'negative_prompt'], negativePrompt),
      seed: getPromptNumber(item, ['seed'], seed ?? -1),
      guidance_scale: getPromptNumber(item, ['guidance_scale', 'guidance'], guidanceScale ?? 4),
      sample_steps: getPromptNumber(item, ['sample_steps', 'steps', 'num_inference_steps'], sampleSteps ?? 20),
    }));

    const outputFolder = joinPath(settings.TRAINING_FOLDER, normalizedJobName, 'samples');
    const backendConfig =
      generationBackend === 'comfy'
        ? {
            backend: 'comfy' as const,
            comfy: buildComfyConfig(),
          }
        : {
            backend: 'native' as const,
          };
    return {
      job: 'generate',
      config: {
        name: normalizedJobName,
        device: 'cuda',
        process: [
          {
            type: 'to_folder',
            output_folder: outputFolder,
            device: 'cuda',
            dtype: model.dtype || 'bf16',
            generate: {
              ...backendConfig,
              sampler,
              width: width || 1024,
              height: height || 1024,
              neg: negativePrompt,
              seed: seed ?? -1,
              guidance_scale: guidanceScale ?? 4,
              sample_steps: sampleSteps ?? 20,
              ext: imageFormat,
              prompt_file: writePromptFile,
              num_repeats: numRepeats || 1,
              prompts: promptList,
              images: promptItems.map(cleanPromptImageSettings),
            },
            sample: {
              ...backendConfig,
              sampler,
              sample_every: 1,
              width: width || 1024,
              height: height || 1024,
              samples: sampleItems,
              neg: negativePrompt,
              seed: seed ?? -1,
              walk_seed: false,
              guidance_scale: guidanceScale ?? 4,
              sample_steps: sampleSteps ?? 20,
              num_frames: 1,
              fps: 16,
            },
            model,
          },
        ],
      },
      meta: {
        name: '[name]',
        version: '1.0',
      },
    };
  };

  const createGenerateJob = async (promptItems = currentPromptItems) => {
    if (isBusy) return;
    const normalizedJobName = sanitizeJobName(jobName) || makeDefaultJobName();
    const model = cleanModelConfig(modelConfig, useLora, loraPath);
    const selectedGpuIDs = gpuIDs;

    if (!validateGeneration(promptItems, model)) return;
    if (!selectedGpuIDs) return;

    const jobConfig = buildGenerateJobConfig(promptItems, normalizedJobName, model);

    clearStatusResetTimeout();
    setStatus('saving');
    try {
      const res = await apiClient.post('/api/jobs', {
        name: normalizedJobName,
        worker_id: 'local',
        gpu_ids: selectedGpuIDs,
        job_type: 'generate',
        job_ref: useLora ? loraPath.trim() : model.name_or_path,
        job_config: jobConfig,
      });

      if (startImmediately) {
        await startJob(res.data.id);
        await startQueue(selectedGpuIDs, 'local');
      }
      router.push(`/jobs/${res.data.id}`);
    } catch (error: any) {
      console.error('Error creating generate job:', error);
      if (error.response?.status === 409) {
        setValidationErrors(['A job with this name already exists. Choose another name.']);
      } else {
        setValidationErrors([error.response?.data?.error || 'Failed to create generate job.']);
      }
      setStatus('error');
    } finally {
      scheduleStatusIdle();
    }
  };

  const cancelInlineGeneration = () => {
    if (status !== 'generating' || cancelRequested) return;
    setCancelRequested(true);
    inlineAbortControllerRef.current?.abort();
  };

  const generateInline = async (promptItems = currentPromptItems) => {
    if (isBusy) return;
    const normalizedJobName = sanitizeJobName(jobName) || makeDefaultJobName();
    const model = cleanModelConfig(modelConfig, useLora, loraPath);
    const selectedGpuIDs = gpuIDs;

    if (!validateGeneration(promptItems, model)) return;
    if (!selectedGpuIDs) return;
    if (imageCount !== 1) {
      setValidationErrors(['Multiple images must be created as a generate job.']);
      return;
    }

    const jobConfig = buildGenerateJobConfig(promptItems, normalizedJobName, model);
    const abortController = new AbortController();
    let generationCanceled = false;
    inlineAbortControllerRef.current = abortController;
    clearStatusResetTimeout();
    setCancelRequested(false);
    setStatus('generating');
    setInlineImagePath('');
    setInlineError('');
    setInlineMessage('');
    try {
      const res = await apiClient.post(
        '/api/generate/inline',
        {
          gpu_ids: selectedGpuIDs,
          job_config: jobConfig,
        },
        { signal: abortController.signal },
      );
      setInlineImagePath(res.data.imagePath || res.data.image_path || '');
    } catch (error: any) {
      if (abortController.signal.aborted || error?.code === 'ERR_CANCELED' || error?.name === 'CanceledError') {
        generationCanceled = true;
        setInlineMessage('Generation canceled.');
        setStatus('idle');
        return;
      }
      console.error('Error generating inline image:', error);
      const message = error.response?.data?.error || 'Failed to generate image.';
      setInlineError(message);
      setStatus('error');
    } finally {
      if (inlineAbortControllerRef.current === abortController) {
        inlineAbortControllerRef.current = null;
      }
      setCancelRequested(false);
      if (generationCanceled) {
        setStatus('idle');
      } else {
        scheduleStatusIdle();
      }
    }
  };

  const handleGenerate = async () => {
    if (imageCount === 1) {
      await generateInline(currentPromptItems);
    } else {
      await createGenerateJob(currentPromptItems);
    }
  };

  return (
    <>
      <TopBar>
        <div className="flex shrink-0 items-center gap-2">
          <Wand2 className="h-4 w-4 text-cyan-300" />
          <h1 className="text-base font-semibold">Generate Images</h1>
        </div>
        <div className="flex-1"></div>
        {gpuList.length > 0 && (
          <div className="mr-2 min-w-32">
            <SelectInput
              value={`${gpuIDs}`}
              onChange={value => setGpuIDs(value)}
              options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
            />
          </div>
        )}
        <Button
          className="operator-button border-emerald-800 bg-emerald-950/60 py-1 text-emerald-100 hover:bg-emerald-900"
          onClick={handleGenerate}
          disabled={isBusy || !isSettingsLoaded || !isGPUInfoLoaded}
          title={primaryButtonLabel}
          aria-label={primaryButtonLabel}
        >
          {isBusy ? <Wand2 className="h-4 w-4 animate-pulse" /> : <ImagePlus className="h-4 w-4" />}
          <span className="hidden sm:inline">{primaryButtonLabel}</span>
        </Button>
      </TopBar>

      <MainContent>
        {validationErrors.length > 0 && (
          <PageNotice tone="danger" title="Fix these generation settings" className="mb-4">
            <ul className="list-disc space-y-1 pl-4">
              {validationErrors.map((error, index) => (
                <li key={`${error}-${index}`}>{error}</li>
              ))}
            </ul>
          </PageNotice>
        )}
        <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,520px)_minmax(0,1fr)]">
          <form
            className="operator-panel space-y-4 p-3"
            onSubmit={event => {
              event.preventDefault();
              void handleGenerate();
            }}
          >
            <div className="flex items-center gap-2 border-b border-gray-800 pb-2">
              <Wand2 className="h-5 w-5 text-blue-400" />
              <h2 className="font-medium text-gray-100">Prompt</h2>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,.json,text/plain,application/json"
              className="hidden"
              onChange={event => {
                const file = event.target.files?.[0];
                if (file) void handlePromptFileImport(file);
              }}
            />

            <TextInput label="Job Name" value={jobName} onChange={setJobName} required />
            <div>
              <div className="mb-1 mt-2 flex items-center justify-between gap-2">
                <label className="block text-xs text-gray-300">Prompts</label>
                <Button
                  type="button"
                  className="operator-button shrink-0 px-2 py-1 text-xs"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isBusy}
                >
                  <Upload className="h-3.5 w-3.5" />
                  <span className="hidden sm:inline">Import</span>
                </Button>
              </div>
              <TextAreaInput value={prompts} onChange={handlePromptTextChange} rows={5} required />
              <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-gray-400">
                <span>
                  {imageCount} image{imageCount === 1 ? '' : 's'} requested
                </span>
                {importSummary && (
                  <span className="inline-flex items-center gap-1 border border-gray-800 bg-gray-950 px-2 py-1 text-gray-300">
                    <FileJson className="h-3.5 w-3.5" />
                    {importSummary}
                  </span>
                )}
                {imageCount > 1 && <span className="text-amber-300">Multiple images will be created as a job.</span>}
              </div>
            </div>
            <TextAreaInput label="Negative Prompt" value={negativePrompt} onChange={setNegativePrompt} rows={2} />

            <div className="grid grid-cols-2 gap-3">
              <NumberInput label="Width" value={width} onChange={setWidth} min={64} max={4096} />
              <NumberInput label="Height" value={height} onChange={setHeight} min={64} max={4096} />
              <NumberInput label="Seed" value={seed} onChange={setSeed} />
              <NumberInput label="Images per Prompt" value={numRepeats} onChange={setNumRepeats} min={1} max={100} />
              <NumberInput label="Guidance" value={guidanceScale} onChange={setGuidanceScale} min={0} max={30} />
              <NumberInput label="Steps" value={sampleSteps} onChange={setSampleSteps} min={1} max={200} />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <SelectInput label="Sampler" value={sampler} onChange={setSampler} options={samplerOptions} />
              <SelectInput label="Format" value={imageFormat} onChange={setImageFormat} options={imageFormatOptions} />
            </div>

            <FormGroup label="Run">
              <Checkbox label="Start job now" checked={startImmediately} onChange={setStartImmediately} />
              <Checkbox label="Write prompt files" checked={writePromptFile} onChange={setWritePromptFile} />
            </FormGroup>

            <FormGroup label="Backend">
              <SelectInput
                label="Generation Backend"
                value={generationBackend}
                onChange={value => setGenerationBackend(value as GenerationBackend)}
                options={generationBackendOptions}
              />
              {generationBackend === 'comfy' && (
                <div className="grid grid-cols-1 gap-3 pt-2">
                  <div className="grid grid-cols-2 gap-3">
                    <SelectInput
                      label="Comfy Mode"
                      value={comfyMode}
                      onChange={value => setComfyMode(value as ComfyMode)}
                      options={comfyModeOptions}
                    />
                    <SelectInput
                      label="On Error"
                      value={comfyOnError}
                      onChange={value => setComfyOnError(value as ComfyOnError)}
                      options={comfyOnErrorOptions}
                    />
                  </div>
                  {comfyMode === 'external' ? (
                    <TextInput
                      label="Comfy URL"
                      value={comfyServerUrl}
                      onChange={setComfyServerUrl}
                      placeholder="http://127.0.0.1:8188"
                    />
                  ) : (
                    <>
                      <Checkbox
                        label="Install Managed ComfyUI"
                        checked={settings.COMFY_AUTO_INSTALL === 'true' || comfyManagedInstall}
                        onChange={setComfyManagedInstall}
                        disabled={settings.COMFY_AUTO_INSTALL === 'true'}
                      />
                      <TextInput
                        label="Comfy Root"
                        value={comfyRoot}
                        onChange={setComfyRoot}
                        placeholder=".aitk_comfy/ComfyUI"
                      />
                    </>
                  )}
                  <div className="grid grid-cols-2 gap-3">
                    <TextInput
                      label="Workflow"
                      value={comfyWorkflowName}
                      onChange={setComfyWorkflowName}
                      placeholder="auto"
                    />
                    <TextInput
                      label="Workflow JSON"
                      value={comfyWorkflowPath}
                      onChange={setComfyWorkflowPath}
                      placeholder="optional path"
                    />
                  </div>
                </div>
              )}
            </FormGroup>
          </form>

          <div className="space-y-6">
            <div className="operator-panel p-3">
              <div className="mb-3 flex items-center gap-2 border-b border-gray-800 pb-2">
                <Layers className="h-5 w-5 text-amber-400" />
                <h2 className="font-medium text-gray-100">Model</h2>
              </div>

              <input
                ref={loraFileInputRef}
                type="file"
                accept=".safetensors"
                className="hidden"
                onChange={event => {
                  const file = event.target.files?.[0];
                  if (file) void handleLoraUpload(file);
                }}
              />

              <div className="mb-4 grid grid-cols-2 gap-2">
                <button
                  type="button"
                  aria-pressed={!useLora}
                  onClick={() => setUseLora(false)}
                    className={`border px-3 py-2 text-sm ${
                    !useLora
                      ? 'border-blue-500 bg-blue-500/10 text-gray-100'
                      : 'border-gray-700 bg-gray-950 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  Base model
                </button>
                <button
                  type="button"
                  aria-pressed={useLora}
                  onClick={() => handleUseLoraChange(true)}
                    className={`border px-3 py-2 text-sm ${
                    useLora
                      ? 'border-blue-500 bg-blue-500/10 text-gray-100'
                      : 'border-gray-700 bg-gray-950 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  LoRA
                </button>
              </div>

              <div className="space-y-3">
                {useLora && (
                  <div className="space-y-2">
                    <CreatableSelectInput
                      label="LoRA"
                      value={loraPath}
                      onChange={handleLoraPathChange}
                      options={loraOptions}
                      placeholder="Path or Hugging Face repo"
                    />
                    <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-2">
                      <TextInput
                        label="Trigger Words"
                        value={loraUploadTriggerWords}
                        onChange={setLoraUploadTriggerWords}
                        placeholder="optional"
                      />
                      <Button
                        type="button"
                        className="operator-button mt-7 h-9 px-3"
                        onClick={() => loraFileInputRef.current?.click()}
                        disabled={isBusy || loraUploadStatus === 'uploading'}
                        title="Upload LoRA"
                        aria-label="Upload LoRA"
                      >
                        {loraUploadStatus === 'uploading' ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Upload className="h-4 w-4" />
                        )}
                        <span className="hidden sm:inline">
                          {loraUploadStatus === 'uploading' ? `${loraUploadProgress}%` : 'Upload'}
                        </span>
                      </Button>
                    </div>
                    {loraUploadMessage && (
                      <div
                        className={`border px-3 py-2 text-xs ${
                          loraUploadStatus === 'error'
                            ? 'border-red-900 bg-red-950/30 text-red-300'
                            : 'border-gray-800 bg-gray-950 text-gray-300'
                        }`}
                      >
                        {loraUploadMessage}
                      </div>
                    )}
                  </div>
                )}
                {useLora && selectedLora && (
                  <div className="border border-gray-800 bg-gray-950 px-3 py-2 text-xs text-gray-400">
                    <div className="truncate">{selectedLora.path}</div>
                    <div className="mt-1 flex flex-wrap gap-3">
                      <span>{formatLoraSource(selectedLora)}</span>
                      <span>{formatMegabytes(selectedLora.sizeBytes)}</span>
                    </div>
                    {selectedLora.triggerWords && selectedLora.triggerWords.length > 0 ? (
                      <div className="mt-2 flex flex-wrap items-center gap-2">
                        <span className="text-gray-500">Trigger</span>
                        {selectedLora.triggerWords.map(word => (
                          <button
                            type="button"
                            key={word}
                            className="border border-gray-700 bg-gray-900 px-2 py-1 text-gray-200 hover:border-gray-500"
                            onClick={() => setLoraUploadTriggerWords(word)}
                          >
                            {word}
                          </button>
                        ))}
                        <Button type="button" className="operator-button px-2 py-1 text-xs" onClick={insertSelectedLoraTrigger}>
                          Insert
                        </Button>
                      </div>
                    ) : (
                      <div className="mt-2 text-gray-500">No trigger metadata</div>
                    )}
                  </div>
                )}
                {useLora && loraStatus === 'success' && loras.length === 0 && (
                  <div className="border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-400">
                    No local LoRA checkpoints found.
                  </div>
                )}
                {useLora && loraStatus === 'error' && (
                  <div className="border border-red-900 bg-red-950/30 px-3 py-2 text-sm text-red-300">
                    Could not load local LoRA checkpoints.
                  </div>
                )}

                <SelectInput
                  label="Architecture"
                  value={modelConfig.arch}
                  onChange={handleArchChange}
                  options={groupedModelOptions}
                />
                <CreatableSelectInput
                  label="Base Model"
                  value={modelConfig.name_or_path}
                  onChange={value => setModelConfig(current => ({ ...current, name_or_path: value }))}
                  options={[]}
                  placeholder="Path or Hugging Face repo"
                />

                <div className="grid grid-cols-2 gap-3">
                  <SelectInput
                    label="Dtype"
                    value={String(modelConfig.dtype || 'bf16')}
                    onChange={value => setModelConfig(current => ({ ...current, dtype: value }))}
                    options={dtypeOptions}
                  />
                  <SelectInput
                    label="Transformer Quantization"
                    value={modelConfig.quantize ? modelConfig.qtype : ''}
                    onChange={value =>
                      setModelConfig(current => ({
                        ...current,
                        quantize: value !== '',
                        qtype: value || 'qfloat8',
                      }))
                    }
                    options={quantizationOptions}
                  />
                  <SelectInput
                    label="Text Encoder Quantization"
                    value={modelConfig.quantize_te ? modelConfig.qtype_te : ''}
                    onChange={value =>
                      setModelConfig(current => ({
                        ...current,
                        quantize_te: value !== '',
                        qtype_te: value || 'qfloat8',
                      }))
                    }
                    options={quantizationOptions}
                  />
                  <div className="pt-7">
                    <Checkbox
                      label="Low VRAM"
                      checked={Boolean(modelConfig.low_vram)}
                      onChange={value => setModelConfig(current => ({ ...current, low_vram: value }))}
                    />
                  </div>
                  {supportsLayerOffloading && (
                    <div className="col-span-2 border border-gray-800 bg-gray-950 px-3 py-3">
                      <Checkbox
                        label="Layer Offloading"
                        checked={Boolean(modelConfig.layer_offloading)}
                        onChange={handleLayerOffloadingChange}
                      />
                      {modelConfig.layer_offloading && (
                        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
                          <SelectInput
                            label="Offload Backend"
                            value={modelConfig.layer_offloading_backend ?? layerOffloadingMemoryProfile.backend}
                            onChange={value =>
                              setModelConfig(current => ({
                                ...current,
                                layer_offloading_backend: value as LayerOffloadingBackend,
                              }))
                            }
                            options={layerOffloadingBackendOptions}
                          />
                          <SliderInput
                            label="Transformer Offload %"
                            value={Math.round(
                              toFiniteNumber(modelConfig.layer_offloading_transformer_percent, 1.0) * 100,
                            )}
                            onChange={value =>
                              handleLayerOffloadingPercentChange('layer_offloading_transformer_percent', value)
                            }
                            min={0}
                            max={100}
                            step={1}
                          />
                          <SliderInput
                            label="Text Encoder Offload %"
                            value={Math.round(
                              toFiniteNumber(modelConfig.layer_offloading_text_encoder_percent, 1.0) * 100,
                            )}
                            onChange={value =>
                              handleLayerOffloadingPercentChange('layer_offloading_text_encoder_percent', value)
                            }
                            min={0}
                            max={100}
                            step={1}
                          />
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div className="flex justify-end pt-2">
                  <Button
                    type="button"
                    className="operator-button border-emerald-800 bg-emerald-950/60 text-emerald-100 hover:bg-emerald-900"
                    onClick={handleGenerate}
                    disabled={isBusy || !isSettingsLoaded || !isGPUInfoLoaded}
                  >
                    {primaryButtonLabel}
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            <div className="operator-panel p-3">
              <div className="mb-3 flex items-center gap-2 border-b border-gray-800 pb-2">
                <ImagePlus className="h-5 w-5 text-green-400" />
                <h2 className="font-medium text-gray-100">Result</h2>
              </div>

              {status === 'generating' && (
                <div className="flex min-h-64 flex-col items-center justify-center gap-3 border border-gray-800 bg-gray-950 px-4 text-sm text-gray-300">
                  <div className="flex items-center">
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    {cancelRequested
                      ? 'Canceling generation'
                      : isManagedComfyGeneration
                        ? 'Preparing managed ComfyUI, then generating image'
                        : 'Generating image'}
                  </div>
                  {isManagedComfyGeneration && !cancelRequested && (
                    <p className="max-w-md text-center text-xs text-gray-500">
                      If managed ComfyUI is missing, AI Toolkit will download and install it before this image is generated.
                    </p>
                  )}
                  <Button
                    type="button"
                    onClick={cancelInlineGeneration}
                    disabled={cancelRequested}
                    className="operator-button border-red-800 bg-red-950 px-3 py-1.5 text-xs text-red-100 hover:bg-red-900 disabled:cursor-wait"
                  >
                    <X className="h-3.5 w-3.5" />
                    {cancelRequested ? 'Canceling...' : 'Cancel'}
                  </Button>
                </div>
              )}

              {status !== 'generating' && inlineImagePath && (
                <div className="overflow-hidden border border-gray-800 bg-gray-950">
                  <img
                    src={getMediaUrl(inlineImagePath)}
                    alt="Generated image"
                    className="max-h-[640px] w-full object-contain"
                  />
                  <div className="truncate border-t border-gray-800 px-3 py-2 text-xs text-gray-400">
                    {inlineImagePath}
                  </div>
                </div>
              )}

              {status !== 'generating' && !inlineImagePath && !inlineError && !inlineMessage && (
                <div className="flex min-h-64 items-center justify-center border border-dashed border-gray-800 bg-gray-950 px-4 text-center text-sm text-gray-500">
                  Single-image generations appear here. Multiple prompts or repeats create a generate job.
                </div>
              )}

              {inlineMessage && status !== 'generating' && (
                <div className="mt-3 border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-300">
                  {inlineMessage}
                </div>
              )}

              {inlineError && (
                <div className="mt-3 border border-red-900 bg-red-950/30 px-3 py-2 text-sm text-red-300">
                  {inlineError}
                </div>
              )}
            </div>

            <div className="operator-panel p-3">
              <h2 className="mb-4 font-medium text-gray-100">Generation Jobs</h2>
              <JobsTable job_type="generate" />
            </div>
          </div>
        </div>
      </MainContent>
    </>
  );
}
