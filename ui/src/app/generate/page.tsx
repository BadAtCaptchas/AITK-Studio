'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@headlessui/react';
import { ArrowRight, FileJson, ImagePlus, Layers, Loader2, Upload, Wand2 } from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';
import {
  Checkbox,
  CreatableSelectInput,
  FormGroup,
  NumberInput,
  SelectInput,
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
import type { ModelConfig, SelectOption } from '@/types';
import { groupedModelOptions, modelArchs, quantizationOptions } from '@/app/jobs/new/options';

type GeneratedLora = {
  id: string;
  label: string;
  path: string;
  filename: string;
  jobId: string;
  jobName: string;
  jobStatus: string;
  updatedAt: string;
  sizeBytes: number;
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

const imageFormatOptions: SelectOption[] = [
  { value: 'png', label: 'PNG' },
  { value: 'jpg', label: 'JPG' },
  { value: 'webp', label: 'WEBP' },
];

function getArchDefault(archName: string, key: string, fallback: unknown) {
  const arch = modelArchs.find(item => item.name === archName);
  const value = arch?.defaults?.[key];
  if (Array.isArray(value)) {
    return value[0] ?? fallback;
  }
  return value ?? fallback;
}

function getDefaultModelConfig(archName: string): GeneratorModelConfig {
  return {
    name_or_path: String(getArchDefault(archName, 'config.process[0].model.name_or_path', '')),
    arch: archName,
    quantize: Boolean(getArchDefault(archName, 'config.process[0].model.quantize', false)),
    quantize_te: Boolean(getArchDefault(archName, 'config.process[0].model.quantize_te', false)),
    qtype: 'qfloat8',
    qtype_te: 'qfloat8',
    low_vram: Boolean(getArchDefault(archName, 'config.process[0].model.low_vram', false)),
    model_kwargs:
      (getArchDefault(archName, 'config.process[0].model.model_kwargs', {}) as Record<string, unknown>) || {},
    dtype: String(getArchDefault(archName, 'config.process[0].train.dtype', 'bf16')),
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

export default function GeneratePage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const { settings, isSettingsLoaded } = useSettings();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const [jobName, setJobName] = useState(makeDefaultJobName);
  const [modelConfig, setModelConfig] = useState<GeneratorModelConfig>(() => getDefaultModelConfig('flux'));
  const [useLora, setUseLora] = useState(false);
  const [loraPath, setLoraPath] = useState('');
  const [loras, setLoras] = useState<GeneratedLora[]>([]);
  const [loraStatus, setLoraStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
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
  const [status, setStatus] = useState<'idle' | 'saving' | 'generating' | 'error'>('idle');
  const [inlineImagePath, setInlineImagePath] = useState('');
  const [inlineError, setInlineError] = useState('');

  useEffect(() => {
    if (isGPUInfoLoaded && gpuIDs === null) {
      setGpuIDs(gpuList.length > 0 ? `${gpuList[0].index}` : '0');
    }
  }, [gpuIDs, gpuList, isGPUInfoLoaded]);

  useEffect(() => {
    setLoraStatus('loading');
    apiClient
      .get('/api/generate/loras')
      .then(res => {
        setLoras(res.data.loras || []);
        setLoraStatus('success');
      })
      .catch(error => {
        console.error('Error fetching LoRAs:', error);
        setLoraStatus('error');
      });
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

  const isBusy = status === 'saving' || status === 'generating';
  const primaryButtonLabel =
    status === 'generating'
      ? 'Generating...'
      : status === 'saving'
        ? 'Creating...'
        : imageCount === 1
          ? 'Generate Image'
          : 'Create Job';

  const applyLoraModelDefaults = (lora: GeneratedLora) => {
    if (!lora.model) return;
    setModelConfig(current => ({
      ...current,
      ...lora.model,
      name_or_path: String(lora.model?.name_or_path || current.name_or_path),
      arch: String(lora.model?.arch || current.arch),
      dtype: String((lora.model as GeneratorModelConfig).dtype || current.dtype || 'bf16'),
      model_kwargs: (lora.model?.model_kwargs as Record<string, unknown>) || current.model_kwargs || {},
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
    }
  };

  const handleUseLoraChange = (checked: boolean) => {
    setUseLora(checked);
    if (checked && !loraPath && loras[0]) {
      setLoraPath(loras[0].path);
      applyLoraModelDefaults(loras[0]);
    }
  };

  const handleArchChange = (archName: string) => {
    setModelConfig(getDefaultModelConfig(archName));
    setSampler(getDefaultSampler(archName));
  };

  const handlePromptTextChange = (value: string) => {
    setPrompts(value);
    setJsonPromptItems(null);
    setImportSummary('');
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
    } catch (error) {
      console.error('Error importing prompt file:', error);
      alert('Failed to import prompt file.');
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const validateGeneration = (promptItems: PromptImageSettings[], model: GeneratorModelConfig) => {
    if (!isSettingsLoaded || !settings.TRAINING_FOLDER) {
      alert('Settings are still loading. Please try again.');
      return false;
    }
    if (!gpuIDs) {
      alert('Select a GPU before generating.');
      return false;
    }
    if (!model.name_or_path) {
      alert('Select a base model before generating.');
      return false;
    }
    if (useLora && !loraPath.trim()) {
      alert('Select a LoRA or enter a LoRA path.');
      return false;
    }
    if (promptItems.length === 0) {
      alert('Enter at least one prompt.');
      return false;
    }
    return true;
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
    return {
      job: 'generate',
      config: {
        name: normalizedJobName,
        process: [
          {
            type: 'to_folder',
            output_folder: outputFolder,
            device: 'cuda',
            dtype: model.dtype || 'bf16',
            generate: {
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
        alert('A job with this name already exists. Choose another name.');
      } else {
        alert('Failed to create generate job.');
      }
      setStatus('error');
    } finally {
      setTimeout(() => setStatus('idle'), 1500);
    }
  };

  const generateInline = async (promptItems = currentPromptItems) => {
    if (isBusy) return;
    const normalizedJobName = sanitizeJobName(jobName) || makeDefaultJobName();
    const model = cleanModelConfig(modelConfig, useLora, loraPath);
    const selectedGpuIDs = gpuIDs;

    if (!validateGeneration(promptItems, model)) return;
    if (!selectedGpuIDs) return;
    if (imageCount !== 1) {
      alert('Multiple images must be created as a generate job.');
      return;
    }

    const jobConfig = buildGenerateJobConfig(promptItems, normalizedJobName, model);
    setStatus('generating');
    setInlineImagePath('');
    setInlineError('');
    try {
      const res = await apiClient.post('/api/generate/inline', {
        gpu_ids: selectedGpuIDs,
        job_config: jobConfig,
      });
      setInlineImagePath(res.data.imagePath || res.data.image_path || '');
    } catch (error: any) {
      console.error('Error generating inline image:', error);
      const message = error.response?.data?.error || 'Failed to generate image.';
      setInlineError(message);
      setStatus('error');
    } finally {
      setTimeout(() => setStatus('idle'), 1500);
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
        <div>
          <h1 className="text-lg">Generate Images</h1>
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
          className="inline-flex items-center gap-2 rounded-md bg-green-600 px-3 py-1 text-white hover:bg-green-700 disabled:opacity-60"
          onClick={handleGenerate}
          disabled={isBusy || !isSettingsLoaded || !isGPUInfoLoaded}
        >
          {isBusy ? <Wand2 className="h-4 w-4 animate-pulse" /> : <ImagePlus className="h-4 w-4" />}
          {primaryButtonLabel}
        </Button>
      </TopBar>

      <MainContent>
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,520px)_minmax(0,1fr)]">
          <form
            className="space-y-5 rounded-lg border border-gray-800 bg-gray-900 p-4"
            onSubmit={event => {
              event.preventDefault();
              void handleGenerate();
            }}
          >
            <div className="flex items-center gap-2 border-b border-gray-800 pb-3">
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
              <div className="mb-1 mt-2 flex items-center justify-between gap-3">
                <label className="block text-xs text-gray-300">Prompts</label>
                <Button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-md border border-gray-700 bg-gray-950 px-2 py-1 text-xs text-gray-200 hover:border-gray-500 disabled:opacity-60"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isBusy}
                >
                  <Upload className="h-3.5 w-3.5" />
                  Import
                </Button>
              </div>
              <TextAreaInput value={prompts} onChange={handlePromptTextChange} rows={5} required />
              <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-gray-400">
                <span>
                  {imageCount} image{imageCount === 1 ? '' : 's'} requested
                </span>
                {importSummary && (
                  <span className="inline-flex items-center gap-1 rounded border border-gray-800 bg-gray-950 px-2 py-1 text-gray-300">
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
          </form>

          <div className="space-y-6">
            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <div className="mb-4 flex items-center gap-2 border-b border-gray-800 pb-3">
                <Layers className="h-5 w-5 text-amber-400" />
                <h2 className="font-medium text-gray-100">Model</h2>
              </div>

              <div className="mb-4 grid grid-cols-2 gap-2">
                <button
                  type="button"
                  aria-pressed={!useLora}
                  onClick={() => setUseLora(false)}
                  className={`rounded-md border px-3 py-2 text-sm ${
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
                  className={`rounded-md border px-3 py-2 text-sm ${
                    useLora
                      ? 'border-blue-500 bg-blue-500/10 text-gray-100'
                      : 'border-gray-700 bg-gray-950 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  Created LoRA
                </button>
              </div>

              <div className="space-y-3">
                {useLora && (
                  <CreatableSelectInput
                    label="LoRA"
                    value={loraPath}
                    onChange={handleLoraPathChange}
                    options={loraOptions}
                    placeholder="Path or Hugging Face repo"
                  />
                )}
                {useLora && selectedLora && (
                  <div className="rounded-md border border-gray-800 bg-gray-950 px-3 py-2 text-xs text-gray-400">
                    <div className="truncate">{selectedLora.path}</div>
                    <div className="mt-1 flex gap-3">
                      <span>{selectedLora.jobName}</span>
                      <span>{Math.max(1, Math.round(selectedLora.sizeBytes / 1024 / 1024))} MB</span>
                    </div>
                  </div>
                )}
                {useLora && loraStatus === 'success' && loras.length === 0 && (
                  <div className="rounded-md border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-400">
                    No local LoRA checkpoints found.
                  </div>
                )}
                {useLora && loraStatus === 'error' && (
                  <div className="rounded-md border border-red-900 bg-red-950/30 px-3 py-2 text-sm text-red-300">
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
                </div>

                <div className="flex justify-end pt-2">
                  <Button
                    type="button"
                    className="inline-flex items-center gap-2 rounded-md bg-green-600 px-3 py-2 text-sm text-white hover:bg-green-700 disabled:opacity-60"
                    onClick={handleGenerate}
                    disabled={isBusy || !isSettingsLoaded || !isGPUInfoLoaded}
                  >
                    {primaryButtonLabel}
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <div className="mb-4 flex items-center gap-2 border-b border-gray-800 pb-3">
                <ImagePlus className="h-5 w-5 text-green-400" />
                <h2 className="font-medium text-gray-100">Result</h2>
              </div>

              {status === 'generating' && (
                <div className="flex min-h-64 items-center justify-center rounded-md border border-gray-800 bg-gray-950 text-sm text-gray-300">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating image
                </div>
              )}

              {status !== 'generating' && inlineImagePath && (
                <div className="overflow-hidden rounded-md border border-gray-800 bg-gray-950">
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

              {status !== 'generating' && !inlineImagePath && !inlineError && (
                <div className="flex min-h-64 items-center justify-center rounded-md border border-dashed border-gray-800 bg-gray-950 px-4 text-center text-sm text-gray-500">
                  Single-image generations appear here. Multiple prompts or repeats create a generate job.
                </div>
              )}

              {inlineError && (
                <div className="mt-3 rounded-md border border-red-900 bg-red-950/30 px-3 py-2 text-sm text-red-300">
                  {inlineError}
                </div>
              )}
            </div>

            <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
              <h2 className="mb-4 font-medium text-gray-100">Generation Jobs</h2>
              <JobsTable job_type="generate" />
            </div>
          </div>
        </div>
      </MainContent>
    </>
  );
}
