import type { JobConfig } from '../types';

export const preservationModes = ['none', 'replay', 'teacher', 'replay_teacher'] as const;
export type PreservationMode = (typeof preservationModes)[number];
export type KreaContinuationBase = 'raw' | 'turbo';
export type LoraContinuation = {
  version: 1;
  source_sha256: string;
  training_base: KreaContinuationBase;
  deployment_base: KreaContinuationBase;
  source_scale?: number;
  assistant_sha256?: string;
};
export type SourcePreservation =
  | { mode: 'none' }
  | { mode: Exclude<PreservationMode, 'none'>; new_batches: number; teacher_weight: number };
export type LoraInspection = {
  version: 1; status: 'compatible'; sha256: string; format: string; tensor_count: number;
  ranks: number[]; architecture: 'krea2'; topology_verified: boolean;
  size_bytes: number;
  modules: { name: string; rank: number; input: number; output: number; source_scale: number }[];
  declared_training_base: string | null;
};
const record = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
export const validSha256 = (v: unknown): v is string => typeof v === 'string' && /^[a-f0-9]{64}$/.test(v);
const positive = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v > 0;
const positiveInteger = (v: unknown): v is number => positive(v) && Number.isSafeInteger(v);
export function continuationSourceKey(process: JobConfig['config']['process'][number], workerID: string): string {
  const source = process.network?.continuation;
  return JSON.stringify([workerID, process.model.arch, process.model.name_or_path,
    source !== undefined, process.network?.pretrained_lora_path, source?.source_scale, source?.training_base]);
}

export function parseLoraInspection(value: unknown): LoraInspection {
  if (record(value) && value.status === 'unsupported' && typeof value.error === 'string') throw new Error(value.error);
  if (!record(value) || value.version !== 1 || value.status !== 'compatible' || !validSha256(value.sha256) ||
      typeof value.format !== 'string' || value.format.length > 128 || !positiveInteger(value.tensor_count) || !positiveInteger(value.size_bytes) || value.size_bytes > 2 * 1024 ** 3 || value.architecture !== 'krea2' ||
      value.topology_verified !== true || !Array.isArray(value.ranks) || !value.ranks.length || !value.ranks.every(positiveInteger) ||
      !Array.isArray(value.modules) || value.modules.length === 0 || value.modules.length > 10000 ||
      !value.modules.every(item => record(item) && typeof item.name === 'string' && item.name.length < 512 &&
        positiveInteger(item.rank) && positiveInteger(item.input) && positiveInteger(item.output) && positive(item.source_scale)) ||
      !(value.declared_training_base === null || typeof value.declared_training_base === 'string'))
    throw new Error('Invalid LoRA inspection response');
  return value as LoraInspection;
}

export function continuationConfigErrors(process: Record<string, unknown>): string[] {
  const network = record(process.network) ? process.network : {};
  const train = record(process.train) ? process.train : {};
  const model = record(process.model) ? process.model : {};
  const datasets = Array.isArray(process.datasets) ? process.datasets.filter(record) : [];
  const preservation = train.source_preservation;
  const config = network.continuation;
  const errors: string[] = [];
  const preserved = datasets.filter(item => item.role === 'source_preservation');
  if (config === undefined) {
    if (preserved.length || preservation !== undefined && (!record(preservation) || preservation.mode !== 'none'))
      errors.push('Source preservation requires an inspected continuation LoRA.');
    return errors;
  }
  if (!record(config) || config.version !== 1 || !validSha256(config.source_sha256) ||
      !['raw', 'turbo'].includes(String(config.training_base)) || !['raw', 'turbo'].includes(String(config.deployment_base)) ||
      config.source_scale !== undefined && !positive(config.source_scale) ||
      config.assistant_sha256 !== undefined && !validSha256(config.assistant_sha256))
    errors.push('Inspect the continuation LoRA and declare its training base and scaling.');
  if (network.type !== 'lora' || typeof network.pretrained_lora_path !== 'string' || !network.pretrained_lora_path.trim())
    errors.push('Continuation requires a standard LoRA source file.');
  const kwargs = record(model.model_kwargs) ? model.model_kwargs : {};
  if (!['krea2', 'krea2:turbo'].includes(String(model.arch)) || kwargs.kv_cache || kwargs.edit || kwargs.is_edit || kwargs.kroma ||
      String(model.name_or_path).toLowerCase().includes('kroma')) errors.push('Continuation supports Krea 2 Raw and Turbo text-to-image only.');
  if (record(config)) {
    const name = String(model.name_or_path).toLowerCase();
    if (['krea/krea-2-raw', 'krea/krea-2-turbo'].includes(name) && !name.endsWith(String(config.training_base)))
      errors.push('Selected checkpoint conflicts with the declared continuation training base.');
    if (config.training_base === 'turbo' && !model.assistant_lora_path) errors.push('Turbo continuation requires its training adapter.');
    if (config.training_base === 'raw' && model.assistant_lora_path) errors.push('Raw continuation does not use the Turbo training adapter.');
  }
  if (model.base_lora_path || model.lora_path || model.inference_lora_path || train.merge_network_on_save || train.train_text_encoder)
    errors.push('Continuation cannot merge another base LoRA, train the text encoder, or merge on save.');
  if (process.adapter || process.embedding || process.decorator || record(process.watermark) && process.watermark.enabled || train.train_unet === false)
    errors.push('Continuation trains only the imported denoiser adapter.');
  if (train.start_step !== undefined && train.start_step !== null && train.start_step !== 0)
    errors.push('A downloaded LoRA starts a new run at step zero.');
  if (!datasets.some(item => item.role !== 'source_preservation' && !item.is_reg)) errors.push('Add new training examples.');
  if (datasets.some(item => item.type !== undefined && item.type !== 'image' || item.num_frames !== undefined && item.num_frames !== 1 ||
      item.auto_frame_count || item.do_audio || item.control_path || item.control_from_same_folder || item.unconditional_path ||
      Array.isArray(item.controls) && item.controls.length))
    errors.push('Continuation requires text-to-image datasets without video, control or unconditional images.');
  if (preservation !== undefined && (!record(preservation) || !preservationModes.some(mode => mode === preservation.mode)))
    errors.push('Invalid source preservation mode.');
  const active = record(preservation) && preservation.mode !== 'none';
  if (active !== (preserved.length > 0)) errors.push('Select preservation data when preservation is enabled; remove it when disabled.');
  if (active && record(preservation)) {
    if (typeof preservation.new_batches !== 'number' || !Number.isInteger(preservation.new_batches) || preservation.new_batches < 1 || preservation.new_batches > 100 || !positive(preservation.teacher_weight))
      errors.push('Preservation requires 1–100 new batches per source batch and a positive teacher weight.');
    if (['diff_output_preservation', 'blank_prompt_preservation', 'sega_distill', 'do_prior_divergence', 'do_guidance_loss', 'inverted_mask_prior', 'do_differential_guidance'].some(key => train[key]) || train.loss_type === 'mean_flow' || datasets.some(item => item.is_reg || item.prior_reg))
      errors.push('Source preservation cannot be combined with other teacher losses or regularization datasets.');
    if (preserved.some(item => item.caption_dropout_rate || item.trigger_word || item.token_dropout_rate || item.shuffle_tokens ||
      Array.isArray(item.random_triggers) && item.random_triggers.length > 0 || item.use_short_captions || item.network_weight !== undefined && item.network_weight !== 1))
      errors.push('Preservation examples require unchanged captions and adapter weight 1.');
  }
  return errors;
}
