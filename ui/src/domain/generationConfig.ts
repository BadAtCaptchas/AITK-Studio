import type { ModelConfig } from '../types';
import { modelArchs } from './modelOptions';
import { getLayerOffloadingMemoryProfile, type LayerOffloadingBackend } from '../utils/memoryProfiles';
export type GeneratorModelConfig = ModelConfig & {
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
export function archSupportsSection(archName: string, section: 'model.layer_offloading') {
  return Boolean(modelArchs.find(item => item.name === archName)?.additionalSections?.includes(section));
}
export function getDefaultModelConfig(archName: string): GeneratorModelConfig {
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
export function getDefaultSampler(archName: string) {
  return String(getArchDefault(archName, 'config.process[0].sample.sampler', 'flowmatch'));
}
