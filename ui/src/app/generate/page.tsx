'use client';

import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@headlessui/react';
import { ArrowRight, ImagePlus, Layers, Wand2 } from 'lucide-react';
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
  const [status, setStatus] = useState<'idle' | 'saving' | 'error'>('idle');

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

  const createGenerateJob = async () => {
    if (status === 'saving') return;
    const promptList = splitPrompts(prompts);
    const normalizedJobName = sanitizeJobName(jobName) || makeDefaultJobName();
    const model = cleanModelConfig(modelConfig, useLora, loraPath);

    if (!isSettingsLoaded || !settings.TRAINING_FOLDER) {
      alert('Settings are still loading. Please try again.');
      return;
    }
    if (!gpuIDs) {
      alert('Select a GPU before generating.');
      return;
    }
    if (!model.name_or_path) {
      alert('Select a base model before generating.');
      return;
    }
    if (useLora && !loraPath.trim()) {
      alert('Select a LoRA or enter a LoRA path.');
      return;
    }
    if (promptList.length === 0) {
      alert('Enter at least one prompt.');
      return;
    }

    const sampleItems = promptList.map(prompt => ({
      prompt,
      width: width || 1024,
      height: height || 1024,
      neg: negativePrompt,
      seed: seed ?? -1,
      guidance_scale: guidanceScale ?? 4,
      sample_steps: sampleSteps ?? 20,
    }));

    const outputFolder = joinPath(settings.TRAINING_FOLDER, normalizedJobName, 'samples');
    const jobConfig = {
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

    setStatus('saving');
    try {
      const res = await apiClient.post('/api/jobs', {
        name: normalizedJobName,
        worker_id: 'local',
        gpu_ids: gpuIDs,
        job_type: 'generate',
        job_ref: useLora ? loraPath.trim() : model.name_or_path,
        job_config: jobConfig,
      });

      if (startImmediately) {
        await startJob(res.data.id);
        await startQueue(gpuIDs, 'local');
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
          onClick={createGenerateJob}
          disabled={status === 'saving' || !isSettingsLoaded || !isGPUInfoLoaded}
        >
          {status === 'saving' ? <Wand2 className="h-4 w-4 animate-pulse" /> : <ImagePlus className="h-4 w-4" />}
          {status === 'saving' ? 'Creating...' : 'Generate'}
        </Button>
      </TopBar>

      <MainContent>
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,520px)_minmax(0,1fr)]">
          <form
            className="space-y-5 rounded-lg border border-gray-800 bg-gray-900 p-4"
            onSubmit={event => {
              event.preventDefault();
              void createGenerateJob();
            }}
          >
            <div className="flex items-center gap-2 border-b border-gray-800 pb-3">
              <Wand2 className="h-5 w-5 text-blue-400" />
              <h2 className="font-medium text-gray-100">Prompt</h2>
            </div>

            <TextInput label="Job Name" value={jobName} onChange={setJobName} required />
            <TextAreaInput label="Prompts" value={prompts} onChange={setPrompts} rows={5} required />
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
                    onClick={createGenerateJob}
                    disabled={status === 'saving' || !isSettingsLoaded || !isGPUInfoLoaded}
                  >
                    {status === 'saving' ? 'Creating...' : 'Generate Images'}
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
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
