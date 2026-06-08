import React, { useEffect, useMemo, useState } from 'react';
import {
  Checkbox,
  CreatableSelectInput,
  FormGroup,
  SelectInput,
  TextAreaInput,
  TextInput,
} from '@/components/formInputs';
import { CaptionJobConfig } from '@/types';
import { handleCaptionerTypeChange } from '@/helpers/captionJobConfig';
import {
  captionerTypes,
  defaultImageCaptionPrompt,
  defaultIdeogramJsonCaptionPrompt,
  defaultQtype,
  groupedCaptionerTypes,
  legacyDefaultImageCaptionPrompt,
  maxNewTokensOptions,
  maxResOptions,
  quantizationOptions,
} from '@/helpers/captionOptions';
import useRemoteOllamaWorkers from '@/hooks/useRemoteOllamaWorkers';
import { apiClient } from '@/utils/api';

type Props = {
  jobConfig: CaptionJobConfig;
  setJobConfig: (value: any, key?: string) => void;
  datasetPath: string;
  encryptedDatasetKeyB64?: string | null;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  showGPUSelect: boolean;
};

type CaptionEstimate = {
  encrypted?: boolean;
  mediaCount: number | null;
  estimatedCostUsd: number | null;
  pricing?: {
    modelName: string;
    prompt: number;
    completion: number;
    source: string;
  } | null;
  assumptions?: {
    inputTokensPerFile: number;
    outputTokensPerFile: number;
  };
  error?: string;
};

const outputFormatOptions = [
  { value: 'text', label: 'Text captions' },
  { value: 'ideogram_json', label: 'Convert to JSON' },
];

const outputExtensionOptions = [
  { value: 'json', label: '.json' },
  { value: 'txt', label: '.txt' },
];

const destinationOptions = [
  { value: 'current', label: 'Overwrite current dataset' },
  { value: 'copy', label: 'Make dataset copy' },
];

function formatUsd(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return 'Unavailable';
  if (value === 0) return '$0.00';
  if (value < 0.01) return `<$0.01`;
  return `$${value.toFixed(2)}`;
}

const CaptionSimpleJob: React.FC<Props> = ({
  jobConfig,
  setJobConfig,
  datasetPath,
  encryptedDatasetKeyB64,
  gpuIDs,
  setGpuIDs,
  gpuList,
  showGPUSelect,
}) => {
  const selectedCaptionOption = captionerTypes.find(option => option.name === jobConfig.config.process[0].type);
  const additionalSections = selectedCaptionOption?.additionalSections || [];
  const usesLocalGpu = selectedCaptionOption?.usesGpu === true;
  const usesQuantization = selectedCaptionOption?.usesQuantization === true;
  const usesOpenRouter = selectedCaptionOption?.usesOpenRouter === true;
  const usesRemoteOllama = additionalSections.includes('caption.remote_ollama_worker_id');
  const captionConfig = jobConfig.config.process[0].caption;
  const outputFormat = captionConfig.output_format || 'text';
  const isJsonOutput = outputFormat === 'ideogram_json' || outputFormat === 'json';
  const { workers } = useRemoteOllamaWorkers();
  const enabledWorkers = useMemo(() => workers.filter(worker => worker.enabled), [workers]);
  const remoteOllamaOptions = useMemo(
    () => enabledWorkers.map(worker => ({ value: worker.id, label: worker.name })),
    [enabledWorkers],
  );
  const [estimate, setEstimate] = useState<CaptionEstimate | null>(null);
  const [estimateStatus, setEstimateStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  useEffect(() => {
    if (!usesRemoteOllama || captionConfig.remote_ollama_worker_id || enabledWorkers.length === 0) return;
    setJobConfig(enabledWorkers[0].id, 'config.process[0].caption.remote_ollama_worker_id');
  }, [captionConfig.remote_ollama_worker_id, enabledWorkers, setJobConfig, usesRemoteOllama]);

  useEffect(() => {
    if (!usesOpenRouter || !datasetPath || !captionConfig.model_name_or_path) {
      setEstimate(null);
      setEstimateStatus('idle');
      return;
    }

    const timeout = window.setTimeout(() => {
      setEstimateStatus('loading');
      apiClient
        .post('/api/caption/estimate', {
          datasetPath,
          provider: 'openrouter',
          model: captionConfig.model_name_or_path,
          extensions: captionConfig.extensions,
          captionExtension: captionConfig.caption_extension || (isJsonOutput ? 'json' : 'txt'),
          recaption: captionConfig.recaption === true,
          maxNewTokens: captionConfig.max_new_tokens,
          outputFormat,
        })
        .then(res => {
          setEstimate(res.data);
          setEstimateStatus('success');
        })
        .catch(error => {
          setEstimate(error?.response?.data || { error: 'Estimate unavailable', mediaCount: null, estimatedCostUsd: null });
          setEstimateStatus('error');
        });
    }, 250);

    return () => window.clearTimeout(timeout);
  }, [
    captionConfig.caption_extension,
    captionConfig.extensions,
    captionConfig.max_new_tokens,
    captionConfig.model_name_or_path,
    captionConfig.recaption,
    datasetPath,
    isJsonOutput,
    outputFormat,
    usesOpenRouter,
  ]);

  const setOutputFormat = (value: string) => {
    setJobConfig(value, 'config.process[0].caption.output_format');
    if (value === 'ideogram_json') {
      const currentPrompt = captionConfig.caption_prompt?.trim() || '';
      if (
        !currentPrompt ||
        currentPrompt === defaultImageCaptionPrompt ||
        currentPrompt === legacyDefaultImageCaptionPrompt ||
        currentPrompt === 'Describe this image in detail.'
      ) {
        setJobConfig(defaultIdeogramJsonCaptionPrompt, 'config.process[0].caption.caption_prompt');
      }
      if (!captionConfig.caption_extension) {
        setJobConfig('json', 'config.process[0].caption.caption_extension');
      }
      if (!captionConfig.source_caption_extension) {
        setJobConfig('txt', 'config.process[0].caption.source_caption_extension');
      }
      if (!captionConfig.convert_destination) {
        setJobConfig('current', 'config.process[0].caption.convert_destination');
      }
      if (!captionConfig.max_res || captionConfig.max_res < 1024) {
        setJobConfig(1024, 'config.process[0].caption.max_res');
      }
      if (!captionConfig.max_new_tokens || captionConfig.max_new_tokens < 2048) {
        setJobConfig(2048, 'config.process[0].caption.max_new_tokens');
      }
      return;
    }
    if (captionConfig.caption_extension === 'json') {
      setJobConfig('txt', 'config.process[0].caption.caption_extension');
    }
  };

  return (
    <div className="text-sm text-gray-400">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <div>
          <SelectInput
            label="Captioner Type"
            value={jobConfig.config.process[0].type}
            onChange={value => {
              handleCaptionerTypeChange(jobConfig.config.process[0].type, value, jobConfig, setJobConfig);
            }}
            options={groupedCaptionerTypes}
          />
        </div>
        {showGPUSelect && usesLocalGpu && (
          <div>
            <SelectInput
              label="GPU ID"
              value={`${gpuIDs}`}
              onChange={value => setGpuIDs(value)}
              options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
            />
          </div>
        )}
      </div>
      {usesOpenRouter && encryptedDatasetKeyB64 && (
        <div className="mt-4 rounded-md border border-amber-900 bg-amber-950/30 px-3 py-2 text-sm text-amber-200">
          OpenRouter cannot caption encrypted datasets. Use a copied unencrypted dataset, then encrypt the completed result.
        </div>
      )}
      {additionalSections.includes('caption.remote_ollama_worker_id') && (
        <div className="mt-4">
          <SelectInput
            label="Remote Ollama"
            value={captionConfig.remote_ollama_worker_id || ''}
            onChange={value => setJobConfig(value, 'config.process[0].caption.remote_ollama_worker_id')}
            options={remoteOllamaOptions}
            disabled={remoteOllamaOptions.length === 0}
          />
        </div>
      )}
      <div className="mt-4">
        <CreatableSelectInput
          label="Name or Path"
          value={jobConfig.config.process[0].caption.model_name_or_path}
          docKey="config.process[0].caption.model_name_or_path"
          onChange={(value: string | null) => {
            if (value?.trim() === '') {
              value = null;
            }
            setJobConfig(value, 'config.process[0].caption.model_name_or_path');
          }}
          placeholder=""
          options={selectedCaptionOption?.name_or_path_options || []}
          required
        />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <SelectInput
          label="Caption Output"
          value={outputFormat}
          onChange={setOutputFormat}
          options={outputFormatOptions}
        />
        {isJsonOutput && (
          <SelectInput
            label="Destination"
            value={captionConfig.convert_destination || 'current'}
            onChange={value => setJobConfig(value, 'config.process[0].caption.convert_destination')}
            options={destinationOptions}
          />
        )}
      </div>
      {isJsonOutput && (
        <div className="mt-4 rounded-md border border-amber-900 bg-amber-950/30 px-3 py-2 text-sm text-amber-200">
          Convert to JSON requires a vision model. Smaller models may produce invalid JSON, miss NSFW details, or create weak boxes.
        </div>
      )}
      {isJsonOutput && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <SelectInput
            label="Output Extension"
            value={captionConfig.caption_extension || 'json'}
            onChange={value => setJobConfig(value, 'config.process[0].caption.caption_extension')}
            options={outputExtensionOptions}
          />
          <TextInput
            label="Source Extension"
            value={captionConfig.source_caption_extension || 'txt'}
            onChange={value => setJobConfig(value.replace(/^\.+/, ''), 'config.process[0].caption.source_caption_extension')}
          />
          <div className="pt-7">
            <Checkbox
              label="Replace source prompts"
              checked={captionConfig.delete_source_caption === true}
              onChange={value => setJobConfig(value, 'config.process[0].caption.delete_source_caption')}
            />
          </div>
        </div>
      )}
      {additionalSections.includes('caption.model_name_or_path2') && (
        <div className="mt-4">
          <CreatableSelectInput
            label="Name or Path 2"
            value={jobConfig.config.process[0].caption.model_name_or_path2 || ''}
            onChange={(value: string | null) => {
              if (value?.trim() === '') {
                value = null;
              }
              setJobConfig(value, 'config.process[0].caption.model_name_or_path2');
            }}
            placeholder=""
            options={selectedCaptionOption?.name_or_path2_options || []}
          />
        </div>
      )}
      {additionalSections.includes('caption.fixed_caption') && (
        <div className="mt-4">
          <TextInput
            label="Fixed Caption"
            value={jobConfig.config.process[0].caption.fixed_caption || ''}
            onChange={value => {
              if (value?.trim() === '') {
                //@ts-ignore
                value = undefined;
              }
              setJobConfig(value, 'config.process[0].caption.fixed_caption');
            }}
            placeholder="Enter fixed caption (if you want the same caption for all audio files)"
          />
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <div>
          {usesQuantization && (
            <SelectInput
              label="Quantize"
              value={jobConfig.config.process[0].caption.quantize ? jobConfig.config.process[0].caption.qtype : ''}
              onChange={value => {
                if (value === '') {
                  setJobConfig(false, 'config.process[0].caption.quantize');
                  value = defaultQtype;
                } else {
                  setJobConfig(true, 'config.process[0].caption.quantize');
                }
                setJobConfig(value, 'config.process[0].caption.qtype');
              }}
              options={quantizationOptions}
            />
          )}
          {additionalSections.includes('caption.max_res') && (
            <div className={usesQuantization ? 'mt-4' : ''}>
              <SelectInput
                label="Max Resolution"
                value={`${jobConfig.config.process[0].caption.max_res || ''}`}
                onChange={value => {
                  const intVal = parseInt(value);
                  if (!isNaN(intVal)) {
                    setJobConfig(intVal, 'config.process[0].caption.max_res');
                  }
                }}
                options={maxResOptions}
              />
            </div>
          )}
          {additionalSections.includes('caption.max_new_tokens') && (
            <div className="mt-4">
              <SelectInput
                label="Max New Tokens"
                value={`${jobConfig.config.process[0].caption.max_new_tokens || ''}`}
                onChange={value => {
                  const intVal = parseInt(value);
                  if (!isNaN(intVal)) {
                    setJobConfig(intVal, 'config.process[0].caption.max_new_tokens');
                  }
                }}
                options={maxNewTokensOptions}
              />
            </div>
          )}
        </div>
        <div>
          <FormGroup label="Options">
            {usesLocalGpu && (
              <Checkbox
                label="Low VRAM"
                checked={jobConfig.config.process[0].caption.low_vram}
                onChange={value => setJobConfig(value, 'config.process[0].caption.low_vram')}
              />
            )}
            <Checkbox
              label="Recaption"
              checked={jobConfig.config.process[0].caption.recaption}
              onChange={value => setJobConfig(value, 'config.process[0].caption.recaption')}
            />
          </FormGroup>
        </div>
      </div>
      {additionalSections.includes('caption.caption_prompt') && (
        <div className="mt-4">
          <TextAreaInput
            label="Caption Prompt"
            value={jobConfig.config.process[0].caption.caption_prompt || ''}
            onChange={value => {
              setJobConfig(value, 'config.process[0].caption.caption_prompt');
            }}
            placeholder="Enter caption prompt"
          />
        </div>
      )}
      {additionalSections.includes('caption.system_prompt') && (
        <div className="mt-4">
          <TextAreaInput
            label="System Prompt"
            value={captionConfig.system_prompt || ''}
            onChange={value => setJobConfig(value, 'config.process[0].caption.system_prompt')}
            placeholder="Optional system prompt"
          />
        </div>
      )}
      {usesOpenRouter && (
        <div className="mt-4 rounded-md border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-300">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span>
              Dataset estimate:{' '}
              {estimate?.mediaCount == null ? 'unknown count' : `${estimate.mediaCount} file${estimate.mediaCount === 1 ? '' : 's'}`}
            </span>
            <span className="font-medium text-gray-100">
              {estimateStatus === 'loading' ? 'Estimating...' : formatUsd(estimate?.estimatedCostUsd)}
            </span>
          </div>
          {estimate?.pricing && (
            <div className="mt-1 text-xs text-gray-500">
              {estimate.pricing.modelName}: ${(estimate.pricing.prompt * 1_000_000).toFixed(2)}/M input, $
              {(estimate.pricing.completion * 1_000_000).toFixed(2)}/M output.
            </div>
          )}
          {estimate?.assumptions && (
            <div className="mt-1 text-xs text-gray-500">
              Assumes about {estimate.assumptions.inputTokensPerFile} input and {estimate.assumptions.outputTokensPerFile} output tokens per file.
            </div>
          )}
          {estimate?.error && <div className="mt-1 text-xs text-amber-300">{estimate.error}</div>}
        </div>
      )}
    </div>
  );
};

export default CaptionSimpleJob;
