import type { JobConfig } from '../types';
import { modelArchs, type ModelArch } from './modelOptions';
import capabilities from './modelCapabilities.json';
import { configContractErrors } from './configContract';
import { isEditableTrainingConfig, type ValidationMessage } from '../utils/trainingValidation';
import { getLayerOffloadingMemoryProfile, normalizeMemoryProfileArch } from '../utils/memoryProfiles';
import { expectedFluxGuidanceBypass } from '../utils/fluxGuidancePolicy';

export type MigrationChange = { path: string; reason: string; before: string; after: string };
export type TrainingMigrationDraft = { source: JobConfig; changes: MigrationChange[] };
export type TrainingModelMigration = { config: JobConfig; changes: MigrationChange[]; issues: ValidationMessage[] };

const prefix = 'config.process[0].';
const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const clone = <T>(value: T): T => structuredClone(value);
const parts = (path: string) => path.replace(/\[(\d+)\]/g, '.$1').split('.');
function read(value: unknown, path: string): unknown {
  for (const key of parts(path))
    value = Array.isArray(value) ? value[Number(key)] : record(value) ? value[key] : undefined;
  return value;
}
function write(root: unknown, path: string, value: unknown): void {
  const keys = parts(path);
  if (keys.some(key => ['__proto__', 'prototype', 'constructor'].includes(key)))
    throw new Error('Invalid configuration path');
  let node = root;
  for (const [index, key] of keys.entries()) {
    if (!record(node) && !Array.isArray(node)) throw new Error(`Invalid configuration at ${path}`);
    const last = index === keys.length - 1;
    let next = Array.isArray(node) ? node[Number(key)] : node[key];
    if (!last && !record(next) && !Array.isArray(next)) next = /^\d+$/.test(keys[index + 1]) ? [] : {};
    if (last) next = clone(value);
    if (Array.isArray(node)) {
      if (last && value === undefined) delete node[Number(key)];
      else node[Number(key)] = next;
    } else if (last && value === undefined) delete node[key];
    else node[key] = next;
    node = next;
  }
}
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
function describeChange(path: string, before: unknown, after: unknown, reason: string): MigrationChange {
  const describe = (value: unknown): string => {
    if (value === undefined || value === null || value === '') return 'Not set';
    if (/secret|password|credential|api_key|access_token/i.test(path)) return 'Configured';
    if (typeof value === 'boolean') return value ? 'On' : 'Off';
    if (typeof value === 'string' || typeof value === 'number') return String(value);
    if (Array.isArray(value) && value.every(item => typeof item === 'number')) return value.join(', ') || 'None';
    return 'Model-specific options';
  };
  return { path, reason, before: describe(before), after: describe(after) };
}
const choice = (name: string) => capabilities.choices.find(item => item.name === name);
function media(arch: ModelArch): string {
  // LLM conditioning may mix media; it must not be treated as an image trainer.
  return arch.group === 'llm' ? 'llm' : choice(arch.name)?.outputType || (arch.isVideoModel ? 'video' : arch.group);
}
function error(message: string, step = 'job-basics', label = 'Model architecture', index?: number): ValidationMessage {
  return { level: 'error', message, target: { step, label, ...(index === undefined ? {} : { index }) } };
}
function needsControlDataset(arch: ModelArch): boolean {
  return (
    arch.group === 'instruction' ||
    ['moss_image:edit', 'moss_image:base_edit', 'krea2:o_edit', 'krea2:o_edit_turbo'].includes(arch.name)
  );
}
function needsSampleReference(arch: ModelArch): boolean {
  return (
    needsControlDataset(arch) ||
    ['wan21_i2v', 'wan22_14b_i2v', 'ming_image_design_layer', 'qwen25_omni'].includes(arch.name.split(':')[0])
  );
}

export function migrationSourceError(value: unknown): string | null {
  if (!isEditableTrainingConfig(value)) return 'This job does not have an editable training configuration.';
  if (value.config.process.length !== 1) return 'Migration supports jobs with one training process.';
  const process = value.config.process[0];
  if (!['diffusion_trainer', 'ui_trainer', 'concept_slider'].includes(process.type))
    return 'This trainer does not support model migration.';
  if (!modelArchs.some(arch => arch.name === process.model.arch))
    return 'Migration is not available for this custom model architecture.';
  if (process.network !== undefined && (!record(process.network) || typeof process.network.type !== 'string'))
    return 'The source network configuration is invalid.';
  if (
    process.train.phases !== undefined &&
    (!Array.isArray(process.train.phases) || !process.train.phases.every(record))
  )
    return 'The source training phases are invalid.';
  if (process.model.model_kwargs !== undefined && !record(process.model.model_kwargs))
    return 'The source model arguments are invalid.';
  if (read(process, 'adapter') != null || read(process, 'embedding') != null)
    return 'Migration does not support custom adapter or embedding processes. Use a network training configuration.';
  if (!Array.isArray(process.sample.samples)) return 'The source job needs a sample list before migration.';
  return null;
}

export function getMigrationTargets(source: JobConfig): ModelArch[] {
  const from = modelArchs.find(arch => arch.name === source.config.process[0].model.arch);
  return from
    ? modelArchs.filter(arch => media(arch) === media(from) && choice(arch.name)?.tasks.includes('train'))
    : [];
}

export function suggestMigrationName(sourceName: string, targetArch?: string): string {
  const suffix = targetArch ? targetArch.replace(/[^a-zA-Z0-9_-]/g, '_') : 'migrated';
  return `${sourceName.slice(0, 178 - suffix.length)}_${suffix}`;
}

// Explicit weight-loading inputs are different from model-required training adapters.
const freshStartPaths = [
  'name',
  'train.start_step',
  'network.pretrained_lora_path',
  'network.continuation',
  'train.source_preservation',
  'network.network_weights',
  'model.lora_path',
  'model.base_lora_path',
  'model.inference_lora_path',
  'model.accuracy_recovery_adapter',
  'adapter.name_or_path',
] as const;
export function prepareFreshTrainingConfig(source: JobConfig): { config: JobConfig; changes: MigrationChange[] } {
  const config = clone(source);
  const changes: MigrationChange[] = [];
  for (let index = 0; index < config.config.process.length; index++) {
    config.config.process[index].datasets = config.config.process[index].datasets.filter(dataset => dataset.role !== 'source_preservation');
    for (const path of freshStartPaths) {
      const fullPath = `config.process[${index}].${path}`;
      if (read(config, fullPath) !== undefined) {
        const before = read(config, fullPath);
        write(config, fullPath, undefined);
        changes.push(
          describeChange(
            fullPath,
            before,
            undefined,
            'Start a separate run at step 0 without source training weights.',
          ),
        );
      }
    }
  }
  return { config, changes };
}

// These fields select architecture behavior, not suggested training hyperparameters.
const structuralDefaults = new Set([
  'train.noise_scheduler',
  'train.bypass_guidance_embedding',
  'network.transformer_only',
  'network.network_kwargs.ignore_if_contains',
  'network.network_kwargs.only_if_contains',
  'network.network_kwargs.full_if_contains',
  'sample.sampler',
  'sample.format',
  'sample.num_layers',
  'datasets[x].do_i2v',
  'datasets[x].do_audio',
]);
type RequiredSettings = Record<string, unknown>;
function requiredSettings(arch: ModelArch): RequiredSettings {
  const name = arch.name.split(':')[0];
  const required: RequiredSettings = {};
  if (name === 'ming_image_design' || name === 'ming_image_design_layer')
    Object.assign(required, {
      'train.batch_size': 1,
      'train.train_text_encoder': false,
      'model.quantize_te': false,
      'model.qtype': 'ming_fp8',
      'train.do_guidance_loss': false,
      'train.do_cfg': false,
      'train.do_differential_guidance': false,
      'train.standardize_images': false,
      'train.img_multiplier': 1,
      'network.all_layers': false,
      'network.network_kwargs.full_if_contains': [],
      'network.network_kwargs.full_train_in_out': false,
      'datasets[x].standardize_images': false,
      'datasets[x].num_frames': 1,
      'datasets[x].auto_frame_count': false,
      'datasets[x].do_audio': false,
      'sample.neg': '',
    });
  if (name === 'qwen25_omni') required['train.batch_size'] = 1;
  if (name === 'fibo2') Object.assign(required, { 'train.train_text_encoder': false, 'train.unload_text_encoder': false });
  if (name === 'ideogram4') required['train.train_text_encoder'] = false;
  if (name === 'qwen_image_edit' || name === 'boogu_image_edit') required['train.unload_text_encoder'] = false;
  if (arch.name.startsWith('moss_image:kromoss'))
    Object.assign(required, {
      'train.train_text_encoder': false,
      'network.all_layers': false,
      'network.network_kwargs.full_if_contains': [],
      'network.network_kwargs.full_train_in_out': false,
    });
  if (arch.name === 'moss_image:kromoss_turbo') required['train.merge_network_on_save'] = false;
  if (name === 'yue2')
    Object.assign(required, {
      'model.quantize': true,
      'model.qtype': 'convrot8',
      'model.quantize_te': false,
      'datasets[x].caption_dropout_rate': 0,
      'datasets[x].cache_latents_to_disk': true,
    });
  if (name.startsWith('minimax_h3'))
    Object.assign(required, {
      'model.quantize': true,
      'model.qtype': 'convrot8',
      'model.quantize_te': true,
      'model.qtype_te': 'nvfp4',
      'train.train_text_encoder': false,
    });
  for (const section of arch.disableSections || []) {
    if (section === 'network.conv') {
      required['network.conv'] = undefined;
      required['network.conv_alpha'] = undefined;
    } else if (section.startsWith('train.')) {
      // A disabled timestep selector may describe a fixed scheduler, rather than a boolean.
      const selected: unknown = arch.defaults?.[prefix + section]?.[0];
      required[section] = selected ?? (section === 'train.timestep_type' ? 'sigmoid' : false);
    } else if (section === 'model.quantize' || section === 'model.quantize_te') {
      required[section] = arch.defaults?.[prefix + section]?.[0] ?? false;
      const qtype = section === 'model.quantize' ? 'model.qtype' : 'model.qtype_te';
      if (arch.defaults?.[prefix + qtype]) required[qtype] = arch.defaults[prefix + qtype][0];
    }
  }
  return required;
}

export function migrateTrainingModel(source: unknown, targetArch: string, targetPath?: string): TrainingModelMigration {
  const sourceError = migrationSourceError(source);
  if (sourceError || !isEditableTrainingConfig(source))
    throw new Error(sourceError || 'Invalid training configuration');
  const from = modelArchs.find(arch => arch.name === source.config.process[0].model.arch)!;
  const to = getMigrationTargets(source).find(arch => arch.name === targetArch);
  if (!to) throw new Error('Select a supported destination model with the same media type.');
  const { config, changes } = prepareFreshTrainingConfig(source);
  const process = config.config.process[0];
  const switched = from.name !== to.name;
  const put = (path: string, value: unknown, reason = 'Required by the destination model.') => {
    const fullPath = prefix + path;
    const before = read(config, fullPath);
    if (same(before, value)) return;
    write(config, fullPath, value);
    changes.push(describeChange(fullPath, before, value, reason));
  };
  const expand = (path: string): string[] =>
    path.includes('datasets[x]')
      ? process.datasets.map((_, index) => path.replace('datasets[x]', `datasets[${index}]`))
      : [path];
  const selectedDefault = (path: string): unknown => to.defaults?.[prefix + path]?.[0];
  const applyDefaults = (arch: ModelArch, selected: boolean) => {
    for (const [key, pair] of Object.entries(arch.defaults || {})) {
      const path = key.slice(prefix.length);
      if (key.startsWith(prefix) && structuralDefaults.has(path)) {
        for (const expanded of expand(path)) put(expanded, pair[selected ? 0 : 1]);
      }
    }
  };
  if (switched) {
    // Rebuild model-specific options so hidden paths, kwargs and custom cache identities cannot leak across families.
    const nextModel: Record<string, unknown> = {
      arch: to.name,
      name_or_path: '',
      quantize: false,
      quantize_te: false,
      qtype: 'qfloat8',
      qtype_te: 'qfloat8',
      low_vram: false,
      model_kwargs: {},
    };
    for (const [key, pair] of Object.entries(to.defaults || {})) {
      if (key.startsWith(prefix + 'model.')) write(nextModel, key.slice((prefix + 'model.').length), pair[0]);
    }
    const oldModel = source.config.process[0].model;
    for (const key of ['quantize', 'quantize_te', 'qtype', 'qtype_te', 'dtype', 'low_vram'] as const) {
      if (oldModel[key] !== undefined) nextModel[key] = oldModel[key];
    }
    // Quantization adapters embedded in qtype strings belong to the source model.
    for (const key of ['qtype', 'qtype_te'] as const) {
      if (typeof nextModel[key] === 'string' && nextModel[key].includes('|'))
        nextModel[key] = selectedDefault(`model.${key}`) ?? 'qfloat8';
    }
    if (!to.additionalSections?.includes('model.low_vram')) nextModel.low_vram = false;
    const sharesFluxRuntime =
      normalizeMemoryProfileArch(from.name) === 'flux' && normalizeMemoryProfileArch(to.name) === 'flux';
    if (
      to.additionalSections?.includes('model.layer_offloading') &&
      (oldModel.layer_offloading || (sharesFluxRuntime && oldModel.layer_offloading === false))
    ) {
      const profile = getLayerOffloadingMemoryProfile(to.name);
      const backend = oldModel.layer_offloading_backend;
      Object.assign(nextModel, {
        layer_offloading: oldModel.layer_offloading,
        layer_offloading_backend:
          sharesFluxRuntime && (backend === 'block' || backend === 'legacy') ? backend : profile.backend,
        layer_offloading_transformer_percent:
          oldModel.layer_offloading_transformer_percent ?? profile.transformerPercent,
        layer_offloading_text_encoder_percent:
          oldModel.layer_offloading_text_encoder_percent ?? profile.textEncoderPercent,
      });
    }
    for (const key of new Set([...Object.keys(oldModel), ...Object.keys(nextModel)]))
      put(`model.${key}`, nextModel[key], 'Use destination model components and supported memory settings.');
    if (process.network)
      put('network.network_kwargs', {}, 'Layer names and filters belong to the source architecture.');
    applyDefaults(from, false);
    applyDefaults(to, true);
    // Whole-sample audio presets also carry example prompts; only transfer their format/sampler and missing options.
    const sampleDefaults: unknown = selectedDefault('sample');
    if (record(sampleDefaults))
      for (const [key, value] of Object.entries(sampleDefaults)) {
        if (
          !['samples', 'prompts'].includes(key) &&
          (['sampler', 'format'].includes(key) || read(process.sample, key) === undefined)
        )
          put(`sample.${key}`, value);
      }
    // Source-only architectural training features have no meaning on the destination.
    if (!to.additionalSections?.includes('train.ltx_strategy')) put('train.ltx_strategy', undefined);
    if (!to.additionalSections?.includes('train.audio_loss_multiplier')) put('train.audio_loss_multiplier', undefined);
  }
  if (targetPath !== undefined) put('model.name_or_path', targetPath.trim(), 'Use the selected base model.');
  for (const [path, value] of Object.entries(requiredSettings(to))) {
    if (path.startsWith('network.') && !process.network) continue;
    for (const expanded of expand(path)) put(expanded, value);
  }
  const allowed = to.allowedNetworkTypes || (to.name.split(':')[0] === 'zimage' ? ['lora', 'lokr'] : undefined);
  if (process.network && allowed?.length && !allowed.includes(process.network.type)) {
    const preferred = selectedDefault('network.type');
    put('network.type', typeof preferred === 'string' && allowed.includes(preferred) ? preferred : allowed[0]);
  }
  const quantization = [
    process.model.quantize ? process.model.qtype : '',
    process.model.quantize_te ? process.model.qtype_te : '',
  ];
  const convrot = quantization.some(type => type?.startsWith('convrot'));
  const packed = convrot || quantization.some(type => type?.startsWith('orbit') || type === 'nvfp4');
  if (process.network && packed) {
    const supported = convrot ? ['lora', 'lokr'] : ['lora', 'lokr', 'dora'];
    if (!supported.includes(process.network.type))
      put('network.type', 'lora', 'Packed quantization requires a supported adapter type.');
    put('network.all_layers', false, 'Packed base weights cannot contain full-weight trainable modules.');
    if (process.network.type === 'lokr') {
      put('network.lokr_full_rank', false, 'Packed quantization requires finite-rank LoKr.');
      put('network.lokr_weight_decompose', false, 'Packed quantization does not support LoKr weight decomposition.');
    }
  }
  if (process.network && !(process.network.linear > 0) && !(process.network.conv > 0)) {
    put(
      'network.linear',
      selectedDefault('network.linear') ?? 32,
      'The destination requires a positive linear rank when convolution adapters are unavailable.',
    );
  }
  if (process.network && !process.train.train_unet && !process.train.train_text_encoder)
    put('train.train_unet', true, 'Train the denoiser when the destination requires frozen text conditioning.');
  const baseArch = to.name.split(':')[0];
  if (!['flux2', 'flux2_klein_4b', 'flux2_klein_9b', 'zimage'].includes(baseArch) || process.network?.type !== 'lora')
    put('train.sega_distill', false);
  const guidance = expectedFluxGuidanceBypass(process.model);
  if (guidance !== null) put('train.bypass_guidance_embedding', guidance);
  if (process.train.bypass_guidance_embedding) put('train.do_guidance_loss', false);
  const contract = choice(to.name);
  const dtype = process.model.dtype ?? process.train.dtype;
  if (contract && dtype && !contract.precisions.includes(dtype)) {
    const preferred = selectedDefault('train.dtype');
    const replacement =
      typeof preferred === 'string' && contract.precisions.includes(preferred) ? preferred : contract.precisions[0];
    put('train.dtype', replacement);
    if (process.model.dtype !== undefined) put('model.dtype', replacement);
  }
  if (!process.model.low_vram) put('sample.keep_low_vram_for_samples', false);
  const ming = baseArch === 'ming_image_design' || baseArch === 'ming_image_design_layer';
  const normalizeDimensions = (path: string) => {
    for (const key of ['width', 'height']) {
      const value = read(config, prefix + `${path}.${key}`);
      if (typeof value !== 'number' || !Number.isFinite(value)) continue;
      const multiple = ming ? 16 : 1;
      const min = ming ? 16 : (contract?.dimensions.minimum ?? 1);
      const max = ming ? 8192 : (contract?.dimensions.maximum ?? 32768);
      put(
        `${path}.${key}`,
        Math.min(max, Math.max(min, Math.round(value / multiple) * multiple)),
        'Use dimensions supported by the destination model.',
      );
    }
  };
  const normalizeFrames = (path: string) => {
    const value = read(config, prefix + `${path}.num_frames`);
    if (typeof value !== 'number' || !Number.isFinite(value)) return;
    const minimax = baseArch.startsWith('minimax_h3');
    const multiple = minimax ? 17 : (contract?.frames.multipleAfterFirst ?? 1);
    const offset = minimax && value > 1 ? 5 : 1;
    const frames =
      media(to) === 'image'
        ? 1
        : value === 1
          ? 1
          : Math.max(offset, offset + Math.round((value - offset) / multiple) * multiple);
    put(
      `${path}.num_frames`,
      Math.min(contract?.frames.maximum ?? 100000, frames),
      'Use the destination model frame grid.',
    );
  };
  if (ming && !['linear', 'shift'].includes(process.train.timestep_type)) put('train.timestep_type', 'linear');
  // Apply fixed training requirements to phase overrides too, without changing phase length or learning rates.
  for (const [index, phase] of (process.train.phases || []).entries()) {
    for (const [path, value] of Object.entries(requiredSettings(to))) {
      if (path.startsWith('train.') && read(phase, path.slice(6)) !== undefined)
        put(`train.phases[${index}].${path.slice(6)}`, value);
    }
    if (ming && phase.timestep_type !== undefined && !['linear', 'shift'].includes(phase.timestep_type))
      put(`train.phases[${index}].timestep_type`, 'linear');
  }
  const multiControl = to.additionalSections?.includes('datasets.multi_control_paths');
  const singleControl = to.additionalSections?.includes('datasets.control_path');
  for (const [index, dataset] of process.datasets.entries()) {
    const path = `datasets[${index}]`;
    normalizeFrames(path);
    if (switched) put(`${path}.controls`, to.controls || []);
    if (multiControl && dataset.control_path && !read(dataset, 'control_path_1'))
      put(`${path}.control_path_1`, dataset.control_path, 'Reuse the existing control dataset.');
    if (singleControl && !multiControl && !dataset.control_path && read(dataset, 'control_path_1'))
      put(`${path}.control_path`, read(dataset, 'control_path_1'), 'Reuse the first control dataset.');
    if (multiControl) put(`${path}.control_path`, undefined, 'Use the destination multi-reference format.');
    else if (singleControl)
      for (const key of ['control_path_1', 'control_path_2', 'control_path_3'])
        put(`${path}.${key}`, undefined, 'Use the destination single-reference format.');
    if (!singleControl && !multiControl && switched)
      for (const key of ['control_path', 'control_path_1', 'control_path_2', 'control_path_3'])
        put(`${path}.${key}`, undefined, 'The destination does not use control datasets.');
    if (ming) {
      put(
        `${path}.resolution`,
        dataset.resolution.map(value => Math.min(8192, Math.max(16, Math.round(value / 16) * 16))),
        'Ming resolutions must be divisible by 16.',
      );
      if (read(dataset, 'batch_size') !== undefined) put(`${path}.batch_size`, 1);
      if (read(dataset, 'cache_text_embeddings') === true)
        put(
          `${path}.cache_text_embeddings`,
          false,
          'Ming cached conditioning cannot preserve arbitrary caption and crop settings.',
        );
    }
    if (media(to) === 'audio' && dataset.resolution.length > 1)
      put(
        `${path}.resolution`,
        [dataset.resolution[0]],
        'Audio training uses one bucket; multiple image resolutions duplicate audio examples.',
      );
  }
  const sampleMulti = to.additionalSections?.includes('sample.multi_ctrl_imgs');
  const sampleSingle = to.additionalSections?.includes('sample.ctrl_img');
  const samples = [process.sample, ...process.sample.samples];
  for (const [index, sample] of samples.entries()) {
    const path = index === 0 ? 'sample' : `sample.samples[${index - 1}]`;
    normalizeFrames(path);
    normalizeDimensions(path);
    if (sampleMulti && read(sample, 'ctrl_img') && !read(sample, 'ctrl_img_1') && !read(sample, 'ctrl_imgs'))
      put(`${path}.ctrl_img_1`, read(sample, 'ctrl_img'), 'Reuse the existing sample reference.');
    if (sampleSingle && !sampleMulti && !read(sample, 'ctrl_img')) {
      const refs = read(sample, 'ctrl_imgs');
      const first = read(sample, 'ctrl_img_1') || (Array.isArray(refs) ? refs[0] : undefined);
      if (first) put(`${path}.ctrl_img`, first, 'Reuse the first sample reference.');
    }
    if (sampleMulti) put(`${path}.ctrl_img`, undefined, 'Use the destination multi-reference format.');
    else if (sampleSingle)
      for (const key of ['ctrl_imgs', 'ctrl_img_1', 'ctrl_img_2', 'ctrl_img_3'])
        put(`${path}.${key}`, undefined, 'Use the destination single-reference format.');
    if (switched && !sampleSingle && !sampleMulti)
      for (const key of ['ctrl_img', 'ctrl_imgs', 'ctrl_img_1', 'ctrl_img_2', 'ctrl_img_3'])
        put(`${path}.${key}`, undefined, 'The destination does not use sample references.');
    if (ming && read(sample, 'neg')) put(`${path}.neg`, '', 'Ming does not support negative prompts.');
  }
  const uniqueChanges = [...new Map(changes.map(change => [change.path, change])).values()];
  return { config, changes: uniqueChanges, issues: validateTrainingMigration(config, source, false) };
}

export function validateTrainingMigration(
  config: JobConfig,
  source: JobConfig,
  requireDifferentModel = true,
): ValidationMessage[] {
  const issue = migrationSourceError(config);
  if (issue) return [error(issue)];
  const process = config.config.process[0];
  const target = getMigrationTargets(source).find(arch => arch.name === process.model.arch);
  if (!target) return [error('Select a supported destination model with the same media type.')];
  const issues = configContractErrors(config).map(message => error(message, 'raw', 'Raw configuration'));
  const original = source.config.process[0].model;
  if (
    requireDifferentModel &&
    original.arch === process.model.arch &&
    original.name_or_path === process.model.name_or_path
  )
    issues.push(error('Choose another model architecture or model path.'));
  if (process.type === 'concept_slider' && target.disableSections?.includes('slider'))
    issues.push(error('The destination does not support Concept Slider training. Select LoRA Trainer.'));
  if (!process.network && target.allowedNetworkTypes?.length)
    issues.push(error('The destination requires an adapter network. Select a supported target type.'));
  for (const [path, expected] of Object.entries(requiredSettings(target))) {
    if (path.startsWith('network.') && !process.network) continue;
    const paths = path.includes('datasets[x]')
      ? process.datasets.map((_, index) => path.replace('datasets[x]', `datasets[${index}]`))
      : [path];
    for (const expanded of paths) {
      const actual = read(process, expanded);
      // Optional absent fields keep the trainer's own defaults.
      if (actual !== undefined && !same(actual, expected))
        issues.push(
          error(
            `${expanded} is incompatible with ${target.label}. Choose the destination again to apply its required settings.`,
            'raw',
            'Raw configuration',
          ),
        );
    }
  }
  const controls = needsControlDataset(target);
  for (const [index, dataset] of process.datasets.entries()) {
    if (read(dataset, 'type') === 'layered_image' && target.name !== 'ming_image_design_layer')
      issues.push(
        error(
          `Dataset ${index + 1} contains layered images. Select a compatible model or replace this dataset.`,
          'job-dataset',
          'Target dataset',
          index,
        ),
      );
    if (controls && !dataset.is_reg && !['control_path', 'control_path_1'].some(key => Boolean(read(dataset, key))))
      issues.push(
        error(
          `Dataset ${index + 1} needs a control dataset for ${target.label}.`,
          'job-dataset',
          'Control path',
          index,
        ),
      );
  }
  // I2V may derive training controls from target videos, but still needs explicit sampling references.
  const needsSampleControl = needsSampleReference(target);
  if (needsSampleControl && !process.train.disable_sampling)
    for (const [index, sample] of process.sample.samples.entries()) {
      if (
        !['ctrl_img', 'ctrl_img_1', 'ctrl_imgs'].some(key => {
          const value = read(sample, key);
          return Array.isArray(value)
            ? value.some(item => typeof item === 'string' && item.trim())
            : typeof value === 'string' && Boolean(value.trim());
        })
      )
        issues.push(
          error(
            `Sample ${index + 1} needs a reference input for ${target.label}.`,
            'job-samples',
            'Control image',
            index,
          ),
        );
    }
  if (process.model.arch === 'ming_image_design_layer' && process.train.validation_config)
    issues.push(
      error(
        'Ming Design-Layer does not support single-image validation. Remove it or use layered previews.',
        'job-validation',
        'Validation',
      ),
    );
  return issues;
}
