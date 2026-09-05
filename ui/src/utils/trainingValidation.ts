import type { JobConfig } from '../types';
import { parseRemoteDatasetRef } from './remoteDatasetRefs';
import { AUTHENLORA_BUILTIN_CODEC_BITS } from './authenloraCodecs';
import { getValidationConfigErrors } from './validationConfig';

export type TrainingFieldTarget = { step: string; label: string; index?: number };
export type ValidationMessage = { level: 'error' | 'warning'; message: string; target: TrainingFieldTarget };
export type TrainingValidationContext = {
  workerID: string;
  isNonImageModel?: boolean;
  unselectedDatasetPath: string;
  gpuIDs: string | null;
  datasetOptions: Array<{ value: string; name?: string; encrypted?: boolean }>;
  isDatasetUnlocked?: (path: string, name?: string) => boolean;
};
export function validateTrainingConfig(
  rawConfig: JobConfig,
  {
    workerID,
    gpuIDs,
    datasetOptions,
    isNonImageModel = false,
    unselectedDatasetPath,
    isDatasetUnlocked = () => false,
  }: TrainingValidationContext,
): ValidationMessage[] {
  const messages: ValidationMessage[] = [];
  const name = rawConfig.config?.name?.trim() || '';
  const processConfig = rawConfig.config?.process?.[0];
  const trainConfig = processConfig?.train;
  const modelConfig = processConfig?.model;
  const datasetsConfig = processConfig?.datasets || [];
  const sampleConfig = processConfig?.sample;
  if (!name) {
    messages.push({
      target: { step: 'job-basics', label: 'Training name' },
      level: 'error',
      message: 'Training name is required.',
    });
  }
  if (name === '.' || name.includes('..') || /[\\/]/.test(name)) {
    messages.push({
      target: { step: 'job-basics', label: 'Training name' },
      level: 'error',
      message: 'Training name cannot contain path separators or "..".',
    });
  }
  if (!workerID) {
    messages.push({
      target: { step: 'job-basics', label: 'Compute' },
      level: 'error',
      message: 'Select a worker before creating the job.',
    });
  }
  if (!gpuIDs) {
    messages.push({
      target: { step: 'job-basics', label: 'GPU' },
      level: 'error',
      message: 'Select a GPU before creating the job.',
    });
  }
  if (!processConfig) {
    messages.push({
      target: { step: 'raw', label: 'Raw configuration' },
      level: 'error',
      message: 'Job config must include one process.',
    });
    return messages;
  }
  if (!modelConfig?.name_or_path?.trim()) {
    messages.push({
      target: { step: 'job-basics', label: 'Model path' },
      level: 'error',
      message: 'Select or enter a base model path.',
    });
  }
  if (modelConfig?.arch === 'minimax_h3') {
    if (modelConfig.layer_offloading) {
      messages.push({
        target: { step: 'raw', label: 'Raw configuration' },
        level: 'error',
        message: 'MiniMax H3 does not support layer offloading.',
      });
    }
    if (!modelConfig.quantize || modelConfig.qtype !== 'convrot8') {
      messages.push({
        target: { step: 'raw', label: 'Raw configuration' },
        level: 'error',
        message: 'MiniMax H3 requires the ConvRot int8 transformer checkpoint.',
      });
    }
    if (!modelConfig.quantize_te || modelConfig.qtype_te !== 'nvfp4') {
      messages.push({
        target: { step: 'raw', label: 'Raw configuration' },
        level: 'error',
        message: 'MiniMax H3 requires the NVFP4 text-encoder checkpoint.',
      });
    }
    if (trainConfig?.train_text_encoder) {
      messages.push({
        target: { step: 'raw', label: 'Raw configuration' },
        level: 'error',
        message: 'MiniMax H3 does not support training its packed text encoder.',
      });
    }
    if (processConfig.network?.type !== 'lora') {
      messages.push({
        target: { step: 'raw', label: 'Raw configuration' },
        level: 'error',
        message: 'MiniMax H3 currently supports LoRA training only.',
      });
    }
    if (modelConfig.base_lora_path?.trim()) {
      messages.push({
        target: { step: 'raw', label: 'Raw configuration' },
        level: 'error',
        message: 'MiniMax H3 cannot merge a Base LoRA into its packed checkpoint.',
      });
    }
    if (modelConfig.inference_lora_path?.trim()) {
      messages.push({
        target: { step: 'raw', label: 'Raw configuration' },
        level: 'error',
        message: 'MiniMax H3 does not support a separate sample-time Inference LoRA.',
      });
    }
  }
  const baseLoraPath = modelConfig?.base_lora_path?.trim();
  if (baseLoraPath) {
    if (modelConfig?.inference_lora_path?.trim()) {
      messages.push({
        target: { step: 'raw', label: 'Raw configuration' },
        level: 'error',
        message: 'Base LoRA Path cannot be used with sample-time Inference LoRA Path.',
      });
    }
    const baseLoraName = baseLoraPath.split(/[\\/]/).pop() || baseLoraPath;
    if (/\.[^./\\]+$/.test(baseLoraName) && !baseLoraName.toLowerCase().endsWith('.safetensors')) {
      messages.push({
        target: { step: 'job-basics', label: 'Base LoRA path (optional)' },
        level: 'error',
        message: 'Base LoRA Path must be a .safetensors adapter.',
      });
    }
    const baseLoraStrength = Number(modelConfig?.base_lora_strength ?? 1.0);
    if (!Number.isFinite(baseLoraStrength)) {
      messages.push({
        target: { step: 'raw', label: 'Raw configuration' },
        level: 'error',
        message: 'Base LoRA Strength must be a finite number.',
      });
    }
  }
  if (!datasetsConfig.length) {
    messages.push({
      target: { step: 'job-dataset', label: 'Target dataset' },
      level: 'error',
      message: 'Add at least one dataset.',
    });
  }
  const unresolvedDatasets = datasetsConfig.filter(dataset => {
    return !dataset.folder_path || dataset.folder_path === unselectedDatasetPath;
  });
  if (unresolvedDatasets.length > 0) {
    messages.push({
      target: { step: 'job-dataset', label: 'Target dataset' },
      level: 'error',
      message: 'Select a target dataset for every dataset entry.',
    });
  }
  datasetsConfig.forEach((dataset, index) => {
    const datasetOption = datasetOptions.find(option => option.value === dataset.folder_path);
    if (datasetOption?.encrypted) {
      const remembered = isDatasetUnlocked(dataset.folder_path, datasetOption.name);
      if (!remembered && !parseRemoteDatasetRef(dataset.folder_path)) {
        messages.push({
          target: { step: 'job-dataset', label: 'Target dataset', index },
          level: 'warning',
          message: `Dataset ${index + 1} is encrypted. Unlock it before starting or resuming this job.`,
        });
      }
    }
    if (!dataset.is_reg && (!dataset.resolution || dataset.resolution.length === 0)) {
      messages.push({
        target: { step: 'job-dataset', label: 'Resolutions', index },
        level: 'error',
        message: `Dataset ${index + 1} needs at least one resolution.`,
      });
    }
    if (!Number.isSafeInteger(dataset.num_repeats ?? 1) || (dataset.num_repeats ?? 1) < 1) {
      messages.push({
        target: { step: 'job-dataset', label: 'Num repeats', index },
        level: 'error',
        message: `Dataset ${index + 1} repeats must be at least 1.`,
      });
    }
  });
  if (!trainConfig?.auto_train && (!Number.isSafeInteger(trainConfig?.steps) || trainConfig.steps < 1)) {
    messages.push({
      target: { step: 'job-training', label: 'Steps' },
      level: 'error',
      message: 'Training steps must be at least 1.',
    });
  }
  if (!Number.isSafeInteger(trainConfig?.batch_size) || trainConfig.batch_size < 1) {
    messages.push({
      target: { step: 'job-training', label: 'Batch size' },
      level: 'error',
      message: 'Batch size must be at least 1.',
    });
  }
  if (!Number.isSafeInteger(trainConfig?.gradient_accumulation) || trainConfig.gradient_accumulation < 1) {
    messages.push({
      target: { step: 'job-training', label: 'Gradient accumulation' },
      level: 'error',
      message: 'Gradient accumulation must be at least 1.',
    });
  }
  if (!Number.isFinite(trainConfig?.lr) || trainConfig.lr < 0) {
    messages.push({
      target: { step: 'job-training', label: 'Learning rate' },
      level: 'error',
      message: 'Learning rate must be zero or greater.',
    });
  }
  for (const message of getValidationConfigErrors(trainConfig?.validation_config)) {
    const label = /resolution/i.test(message)
      ? 'Validation resolution'
      : /cadence/i.test(message)
        ? 'Validate every'
        : /sigmas/i.test(message)
          ? 'Validation sigmas'
          : 'Held-out images';
    messages.push({ target: { step: 'job-validation', label }, level: 'error', message });
  }
  const watermarkConfig = processConfig?.watermark;
  if (watermarkConfig?.enabled) {
    const networkType = `${processConfig?.network?.type ?? ''}`.toLowerCase();
    if (isNonImageModel) {
      messages.push({
        target: { step: 'job-training', label: 'Enable watermarking' },
        level: 'error',
        message: 'AuthenLoRA watermarking requires an image LoRA job.',
      });
    }
    if (!processConfig?.network) {
      messages.push({
        target: { step: 'job-training', label: 'Enable watermarking' },
        level: 'error',
        message: 'AuthenLoRA watermarking requires a LoRA network.',
      });
    }
    if (!['lora', 'locon', 'lycoris', 'lokr'].includes(networkType)) {
      messages.push({
        target: { step: 'job-training', label: 'Enable watermarking' },
        level: 'error',
        message: 'AuthenLoRA watermarking supports LoRA, LoCon, LyCORIS, and LoKr networks.',
      });
    }
    if (trainConfig?.loss_type === 'mean_flow' || trainConfig?.do_guidance_loss) {
      messages.push({
        target: { step: 'job-training', label: 'Enable watermarking' },
        level: 'error',
        message: 'AuthenLoRA watermarking currently supports the standard image LoRA loss path.',
      });
    }
    if (!watermarkConfig.codec_path?.trim()) {
      messages.push({
        target: { step: 'job-training', label: 'Codec path' },
        level: 'error',
        message: 'AuthenLoRA watermarking requires a local codec path.',
      });
    }
    const builtinMsgBits = AUTHENLORA_BUILTIN_CODEC_BITS[watermarkConfig.codec_path?.trim() || ''];
    if (builtinMsgBits && watermarkConfig.msg_bits !== builtinMsgBits) {
      messages.push({
        target: { step: 'job-training', label: 'Message bits' },
        level: 'error',
        message: `AuthenLoRA ${builtinMsgBits}-bit built-in codec requires Message bits to be ${builtinMsgBits}.`,
      });
    }
    if (!watermarkConfig.msg_bits || watermarkConfig.msg_bits < 1) {
      messages.push({
        target: { step: 'job-training', label: 'Message bits' },
        level: 'error',
        message: 'AuthenLoRA message bits must be greater than 0.',
      });
    }
    if (!watermarkConfig.mapper_rank || watermarkConfig.mapper_rank < 1) {
      messages.push({
        target: { step: 'job-training', label: 'Mapper rank' },
        level: 'error',
        message: 'AuthenLoRA mapper rank must be greater than 0.',
      });
    }
    if (watermarkConfig.mapper_lr < 0) {
      messages.push({
        target: { step: 'job-training', label: 'Mapper LR' },
        level: 'error',
        message: 'AuthenLoRA mapper learning rate must be zero or greater.',
      });
    }
    if (watermarkConfig.watermark_loss_weight < 0 || watermarkConfig.style_loss_weight < 0) {
      messages.push({
        target: { step: 'job-training', label: 'Watermark loss' },
        level: 'error',
        message: 'AuthenLoRA loss weights must be zero or greater.',
      });
    }
    if (watermarkConfig.zero_message_probability < 0 || watermarkConfig.zero_message_probability > 1) {
      messages.push({
        target: { step: 'job-training', label: 'Zero message chance' },
        level: 'error',
        message: 'AuthenLoRA zero message chance must be between 0 and 1.',
      });
    }
    if (watermarkConfig.verify_every < 0) {
      messages.push({
        target: { step: 'job-training', label: 'Verify every' },
        level: 'error',
        message: 'AuthenLoRA verify every must be zero or greater.',
      });
    }
    const secret = watermarkConfig.secret?.trim();
    if (secret && (secret.length !== watermarkConfig.msg_bits || /[^01]/.test(secret))) {
      messages.push({
        target: { step: 'job-training', label: 'Message bits' },
        level: 'error',
        message: 'AuthenLoRA secret bits must be binary and match Message bits.',
      });
    }
  }
  const samplingDisabled = trainConfig?.disable_sampling === true;
  const samplePrompts = sampleConfig?.samples || [];
  if (!samplingDisabled && samplePrompts.length === 0) {
    messages.push({
      target: { step: 'job-samples', label: 'Sample prompt (optional)' },
      level: 'warning',
      message: 'No sample prompts are configured.',
    });
  }
  if (!samplingDisabled && samplePrompts.some(sample => !sample.prompt?.trim())) {
    messages.push({
      target: { step: 'job-samples', label: 'Sample prompt (optional)' },
      level: 'warning',
      message: 'One or more sample prompts are blank.',
    });
  }
  return messages;
}

/** Validate the editable training structure before importing untrusted YAML/JSON. */
export function isEditableTrainingConfig(value: unknown): value is JobConfig {
  const record = (item: unknown): item is Record<string, unknown> =>
    Boolean(item) && typeof item === 'object' && !Array.isArray(item);
  const textFields = (item: Record<string, unknown>, fields: string[]) =>
    fields.every(key => item[key] == null || typeof item[key] === 'string');
  const numberFields = (item: Record<string, unknown>, fields: string[]) =>
    fields.every(key => item[key] == null || typeof item[key] === 'number');
  if (
    !record(value) ||
    !record(value.config) ||
    typeof value.config.name !== 'string' ||
    !Array.isArray(value.config.process) ||
    !value.config.process.length
  )
    return false;
  return value.config.process.every(process => {
    if (
      !record(process) ||
      !record(process.model) ||
      !record(process.train) ||
      !record(process.sample) ||
      !Array.isArray(process.datasets)
    )
      return false;
    if (
      typeof process.type !== 'string' ||
      typeof process.model.arch !== 'string' ||
      typeof process.model.name_or_path !== 'string'
    )
      return false;
    if (
      !textFields(process.model, ['base_lora_path', 'inference_lora_path']) ||
      !numberFields(process.train, ['steps', 'batch_size', 'gradient_accumulation', 'lr'])
    )
      return false;
    if (
      process.watermark != null &&
      (!record(process.watermark) || !textFields(process.watermark, ['codec_path', 'secret']))
    )
      return false;
    if (
      process.sample.samples != null &&
      (!Array.isArray(process.sample.samples) ||
        !process.sample.samples.every(sample => record(sample) && textFields(sample, ['prompt'])))
    )
      return false;
    return process.datasets.every(
      dataset =>
        record(dataset) &&
        typeof dataset.folder_path === 'string' &&
        Array.isArray(dataset.resolution) &&
        dataset.resolution.every(resolution => typeof resolution === 'number') &&
        numberFields(dataset, ['num_repeats', 'batch_size', 'caption_dropout_rate']),
    );
  });
}
