import type { CaptionJobConfig } from '../types';

const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const nonemptyString = (value: unknown): value is string => typeof value === 'string' && value.trim() !== '';
const finiteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const captionStringFields = [
  'device', 'dtype', 'qtype', 'remote_worker_id', 'remote_ollama_worker_id', 'system_prompt',
] as const;
const captionNullableStringFields = [
  'model_name_or_path2', 'caption_prompt', 'caption_format', 'fixed_caption', 'output_format',
  'source_caption_extension', 'convert_destination', 'caption_extension',
] as const;
const captionBooleanFields = [
  'quantize', 'low_vram', 'recaption', 'compile', 'thinking', 'extract_vocals_before_transcribe',
  'keep_timestamps', 'layer_offloading', 'delete_source_caption',
] as const;
const captionNumberFields = [
  'max_res', 'max_new_tokens', 'batch_size', 'layer_offloading_percent', 'temperature',
] as const;

function invalidConfig(): never {
  // Do not include source contents: configurations can contain private prompts and paths.
  throw new Error('The source job has an invalid caption configuration.');
}

/** Validate the editable shape while retaining runtime-defaulted historical omissions. */
function assertCaptionJobConfig(value: unknown): asserts value is CaptionJobConfig {
  if (!record(value) || !nonemptyString(value.job) || !record(value.config) ||
      typeof value.config.name !== 'string' || !Array.isArray(value.config.process) ||
      value.config.process.length === 0) invalidConfig();

  for (const process of value.config.process) {
    if (!record(process) || !nonemptyString(process.type) || !record(process.caption)) invalidConfig();
    for (const key of ['device', 'sqlite_db_path']) {
      if (process[key] !== undefined && typeof process[key] !== 'string') invalidConfig();
    }
    const caption = process.caption;
    if (typeof caption.model_name_or_path !== 'string' || typeof caption.path_to_caption !== 'string' ||
        !Array.isArray(caption.extensions) || caption.extensions.length === 0 ||
        !caption.extensions.every(nonemptyString)) invalidConfig();
    // Older saved jobs rely on Python defaults for these options. Validate supplied
    // values without adding defaults or dropping extension fields from the document.
    for (const key of captionStringFields) {
      if (caption[key] !== undefined && typeof caption[key] !== 'string') invalidConfig();
    }
    for (const key of captionNullableStringFields) {
      if (caption[key] !== undefined && caption[key] !== null && typeof caption[key] !== 'string') invalidConfig();
    }
    for (const key of captionBooleanFields) {
      if (caption[key] !== undefined && typeof caption[key] !== 'boolean') invalidConfig();
    }
    for (const key of captionNumberFields) {
      if (caption[key] !== undefined && !finiteNumber(caption[key])) invalidConfig();
    }
    if (caption.loras !== undefined && caption.loras !== null) {
      if (!Array.isArray(caption.loras)) invalidConfig();
      for (const lora of caption.loras) {
        if (!record(lora) || !nonemptyString(lora.path) ||
            (lora.name !== undefined && typeof lora.name !== 'string') ||
            (lora.strength !== undefined && !finiteNumber(lora.strength))) invalidConfig();
      }
    }
  }
}

/** Accept a parsed JSON/YAML document without rewriting fields or inventing a new job. */
export function parseCaptionJobConfig(value: unknown): CaptionJobConfig {
  assertCaptionJobConfig(value);
  // Raw YAML can contain circular aliases. Saving and editor copies use JSON,
  // so reject those before the document enters component state.
  try { JSON.stringify(value); } catch { invalidConfig(); }
  return value;
}

export function parseCaptionJobSource(value: unknown): {
  config: CaptionJobConfig;
  name: string;
  gpuIds: string | null;
} {
  if (!record(value) || typeof value.job_config !== 'string' || !nonemptyString(value.name) ||
      (value.gpu_ids !== null && typeof value.gpu_ids !== 'string') ||
      (value.job_type !== undefined && value.job_type !== 'caption')) {
    throw new Error('The source caption job could not be loaded.');
  }
  let parsed: unknown;
  try { parsed = JSON.parse(value.job_config); } catch {
    throw new Error('The source job contains invalid configuration JSON.');
  }
  return { config: parseCaptionJobConfig(parsed), name: value.name, gpuIds: value.gpu_ids };
}
