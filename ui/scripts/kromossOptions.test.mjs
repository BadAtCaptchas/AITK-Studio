import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import ts from 'typescript';
import YAML from 'yaml';

const require = createRequire(import.meta.url);
const uiRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const src = path.join(uiRoot, 'src');
const cache = new Map();
function load(relative) {
  const filename = path.resolve(src, relative);
  if (cache.has(filename)) return cache.get(filename);
  if (filename.endsWith('.json')) return JSON.parse(fs.readFileSync(filename, 'utf8'));
  const result = { exports: {} };
  cache.set(filename, result.exports);
  const code = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  new Function('require', 'module', 'exports', code)(name => {
    if (name === '@/helpers/basic') return { isMac: () => false };
    if (name.startsWith('@/') || name.startsWith('.')) {
      const target = name.startsWith('@/') ? path.resolve(src, name.slice(2)) : path.resolve(path.dirname(filename), name);
      return load(target.endsWith('.json') ? target : `${target}.ts`);
    }
    return require(name);
  }, result, result.exports);
  return result.exports;
}

const { modelArchs } = load('domain/modelOptions.ts');
const { handleModelArchChange } = load('app/jobs/new/utils.ts');
const { defaultJobConfig } = load('app/jobs/new/jobConfig.ts');
const { getDefaultModelConfig, getDefaultSampler, getLoraModelConfig } = load('domain/generationConfig.ts');
const { configContractErrors } = load('domain/configContract.ts');
const variants = ['moss_image:kromoss', 'moss_image:kromoss_turbo'];
const distillationPath = 'lodestones/Kromoss/8-steps-dmd-lora.safetensors';
const vaePath = 'lodestones/Kromoss/vae/flux2-vae_small.safetensors';

// Exercise component handlers and the API's pure summary function without a browser or database.
function loadFunction(relative, name, bindings = {}) {
  const filename = path.join(src, relative);
  const source = ts.createSourceFile(filename, fs.readFileSync(filename, 'utf8'), ts.ScriptTarget.Latest, true);
  let expression;
  function visit(node) {
    if (ts.isFunctionDeclaration(node) && node.name?.text === name) expression = node.getText(source);
    if (ts.isVariableDeclaration(node) && node.name.getText(source) === name && node.initializer) {
      expression = node.initializer.getText(source);
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.ok(expression, `Missing function ${name}`);
  const code = ts.transpileModule(`const selected = ${expression};`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText;
  return new Function(...Object.keys(bindings), `${code}\nreturn selected;`)(...Object.values(bindings));
}

function setter(job) {
  return (value, key) => {
    const parts = key.replace(/\[(\d+)\]/g, '.$1').split('.');
    let node = job;
    for (const part of parts.slice(0, -1)) node = node[part] ??= {};
    node[parts.at(-1)] = structuredClone(value);
  };
}
function select(job, arch) {
  handleModelArchChange(job.config.process[0].model.arch, arch, job, setter(job));
  return job.config.process[0];
}

for (const arch of variants) {
  const turbo = arch.endsWith('_turbo');
  test(`${arch}: training, generation and example defaults agree`, () => {
    const job = structuredClone(defaultJobConfig);
    const p = select(job, arch);
    assert.equal(p.model.name_or_path, 'lodestones/Kromoss');
    assert.equal(p.model.dtype, 'bf16');
    assert.equal(p.model.vae_path, vaePath);
    assert.equal(p.model.vae_dtype, 'fp32');
    assert.equal(p.model.quantize, true);
    assert.equal(p.model.quantize_te, true);
    assert.equal(p.model.qtype, 'convrot8');
    assert.equal(p.model.qtype_te, 'convrot8');
    assert.equal(p.model.low_vram, true);
    assert.equal(p.model.layer_offloading, true);
    assert.equal(p.model.layer_offloading_backend, 'legacy');
    assert.equal(p.model.layer_offloading_transformer_percent, 1);
    assert.equal(p.model.layer_offloading_text_encoder_percent, 1);
    assert.deepEqual(p.model.model_kwargs, {
      sample_shift: 5,
      uncond_skip_layers: turbo ? [] : [22],
      ...(turbo ? { distillation_lora_path: distillationPath } : {}),
    });
    assert.equal(p.model.assistant_lora_path, undefined);
    assert.equal(p.model.inference_lora_path, undefined);
    assert.equal(p.model.base_lora_path, undefined);
    assert.equal(p.network.type, 'lora');
    assert.equal(p.network.linear, 16);
    assert.equal(p.network.linear_alpha, 16);
    assert.equal(p.network.lokr_full_rank, false);
    assert.equal(p.network.lokr_factor, -1);
    assert.equal(p.network.transformer_only, true);
    assert.equal(p.network.conv, undefined);
    assert.deepEqual(p.network.network_kwargs.ignore_if_contains, turbo ? [] : ['layers.22.']);
    assert.equal(p.train.batch_size, 1);
    assert.equal(p.train.train_text_encoder, false);
    assert.equal(p.train.bypass_guidance_embedding, false);
    assert.equal(p.train.unload_text_encoder, true);
    assert.equal(p.train.cache_text_embeddings, true);
    assert.equal(p.train.gradient_checkpointing, true);
    assert.equal(p.train.lr, turbo ? 1e-5 : 1e-4);
    assert.deepEqual(p.datasets[0].resolution, [1024]);
    assert.equal(p.datasets[0].caption_dropout_rate, 0);
    assert.equal(p.datasets[0].cache_latents_to_disk, true);
    assert.equal(p.datasets[0].cache_text_embeddings, true);
    assert.equal(p.sample.width, 1024);
    assert.equal(p.sample.height, 1024);
    assert.equal(p.sample.keep_low_vram_for_samples, true);
    assert.equal(p.sample.sample_steps, turbo ? 8 : 50);
    assert.equal(p.sample.guidance_scale, turbo ? 1 : 3);
    assert.equal(p.sample.neg, '');
    assert.deepEqual(configContractErrors(YAML.parse(YAML.stringify(job))), []);
    for (const type of ['lora', 'lokr']) {
      p.network.type = type;
      assert.deepEqual(configContractErrors(job), []);
    }
    p.network.type = 'dora';
    assert.ok(configContractErrors(job).some(error => error.includes('does not support network')));
    p.network.type = 'lora';

    const generation = getDefaultModelConfig(arch);
    for (const [key, value] of Object.entries(generation)) assert.deepEqual(value, p.model[key], key);
    const example = YAML.parse(fs.readFileSync(path.join(uiRoot, `../config/examples/train_lora_kromoss${turbo ? '_turbo' : ''}_16gb.yaml`), 'utf8'));
    assert.deepEqual(configContractErrors(example), []);
    const exampleProcess = example.config.process[0];
    for (const [key, value] of Object.entries(exampleProcess.model)) assert.deepEqual(value, p.model[key], key);
    for (const [key, value] of Object.entries(exampleProcess.network)) assert.deepEqual(value, p.network[key], key);
    for (const [key, value] of Object.entries(exampleProcess.datasets[0])) {
      if (!['folder_path', 'caption_ext'].includes(key)) assert.deepEqual(value, p.datasets[0][key], key);
    }
    for (const key of ['lr', 'dtype', 'batch_size', 'train_unet', 'train_text_encoder', 'gradient_checkpointing', 'noise_scheduler', 'cache_text_embeddings', 'unload_text_encoder', 'timestep_type']) {
      assert.equal(exampleProcess.train[key], p.train[key], key);
    }
    for (const key of ['sampler', 'format', 'keep_low_vram_for_samples', 'width', 'height', 'sample_steps', 'guidance_scale', 'neg']) {
      assert.equal(exampleProcess.sample[key], p.sample[key], key);
    }
  });
}

test('Kromoss selection clears edit references, random conditioning and auxiliary adapters', () => {
  for (const source of ['moss_image', 'moss_image:base', 'moss_image:edit', 'moss_image:base_edit', 'krea2:turbo', ...variants]) {
    for (const arch of variants) {
      if (source === arch) continue;
      const job = structuredClone(defaultJobConfig);
      const current = select(job, source);
      current.model.assistant_lora_path = 'old-assistant.safetensors';
      current.model.inference_lora_path = 'old-inference.safetensors';
      current.model.base_lora_path = 'old-base.safetensors';
      current.network.lokr_full_rank = true;
      current.network.lokr_factor = 8;
      current.datasets[0].random_crop = true;
      current.datasets[0].random_scale = true;
      current.datasets[0].random_triggers = ['random'];
      current.datasets[0].shuffle_tokens = true;
      current.datasets[0].augments = ['flip'];
      current.datasets[0].augmentations = [{ name: 'RandomCrop' }];
      current.sample.samples[0] = {
        prompt: 'Preserve this prompt', ctrl_img: 'old.png', ctrl_imgs: ['old.png'],
        ctrl_img_1: 'old.png', ctrl_img_2: 'older.png', ctrl_img_3: 'oldest.png',
      };
      const p = select(job, arch);
      assert.equal(p.model.model_kwargs.edit, undefined);
      assert.equal(p.model.model_kwargs.kv_cache, undefined);
      assert.equal(p.model.model_kwargs.distillation_lora_path, arch.endsWith('_turbo') ? distillationPath : undefined);
      assert.equal(p.model.assistant_lora_path, undefined);
      assert.equal(p.model.inference_lora_path, undefined);
      assert.equal(p.model.base_lora_path, undefined);
      assert.equal(p.network.lokr_full_rank, false);
      assert.equal(p.network.lokr_factor, -1);
      assert.equal(p.datasets[0].control_path, undefined);
      assert.equal(p.datasets[0].control_path_1, undefined);
      assert.equal(p.datasets[0].random_crop, false);
      assert.equal(p.datasets[0].random_scale, false);
      assert.equal(p.datasets[0].shuffle_tokens, false);
      assert.deepEqual(p.datasets[0].random_triggers, []);
      assert.deepEqual(p.datasets[0].augments, []);
      assert.deepEqual(p.datasets[0].augmentations, []);
      assert.deepEqual(p.sample.samples[0], { prompt: 'Preserve this prompt' });
    }
  }
});

test('switching away removes Kromoss VAE precision and distillation settings', () => {
  for (const arch of variants) {
    for (const target of ['flux', 'moss_image', 'moss_image:base', 'moss_image:edit', 'moss_image:base_edit']) {
      const job = structuredClone(defaultJobConfig);
      select(job, arch);
      const p = select(job, target);
      assert.equal(p.model.vae_path, undefined);
      assert.equal(p.model.vae_dtype, undefined);
      assert.equal(p.model.model_kwargs.distillation_lora_path, undefined);
      assert.equal(p.model.model_kwargs.sample_shift, undefined);
      assert.deepEqual(p.network.network_kwargs.ignore_if_contains, target.startsWith('moss_image') ? ['layers.22.'] : []);
    }
  }
});

test('reselecting a saved Kromoss variant preserves explicit local sources and user settings', () => {
  const job = structuredClone(defaultJobConfig);
  const p = select(job, 'moss_image:kromoss_turbo');
  p.model.name_or_path = 'D:/custom/kromoss.safetensors';
  p.model.vae_path = 'D:/custom/vae.safetensors';
  p.model.model_kwargs.distillation_lora_path = 'D:/custom/dmd.safetensors';
  p.model.model_kwargs.text_encoder_path = 'D:/custom/encoder';
  p.network.type = 'lokr';
  const before = structuredClone(job);
  select(job, p.model.arch);
  assert.deepEqual(job, before);
});

test('Kromoss presets expose only text-to-image controls and supported adapter types', () => {
  for (const arch of variants) {
    const choice = modelArchs.find(item => item.name === arch);
    assert.deepEqual(choice.allowedNetworkTypes, ['lora', 'lokr']);
    assert.deepEqual(choice.additionalSections, ['model.low_vram', 'model.layer_offloading']);
    assert.ok(choice.disableSections.includes('train.train_text_encoder'));
    assert.equal(choice.modelNotes.link.href, 'https://huggingface.co/lodestones/Kromoss');
    if (arch.endsWith('_turbo')) {
      assert.match(choice.modelNotes.summary, /experimental/i);
      assert.ok(choice.modelNotes.paragraphs.some(paragraph => /do not establish that distilled image quality is preserved/.test(paragraph)));
    }
  }
  assert.equal(getDefaultModelConfig('flux').vae_dtype, undefined);
  assert.equal(getDefaultModelConfig('flux').vae_path, undefined);
});

test('Generate library selection applies the selected variant sample defaults on architecture changes', () => {
  const page = 'app/generate/GeneratePageContent.tsx';
  const getModelSummary = loadFunction('app/api/generate/loras/route.ts', 'getModelSummary');
  for (const arch of variants) {
    for (const previousArch of ['flux', ...variants]) {
      const job = structuredClone(defaultJobConfig);
      select(job, arch);
      const lora = { model: JSON.parse(JSON.stringify(getModelSummary(job))), path: 'D:/training/adapter.safetensors' };
      assert.equal(lora.model.vae_dtype, 'fp32');
      assert.equal(lora.model.vae_path, vaePath);
      let model = getDefaultModelConfig(previousArch);
      let sampler = 'custom-sampler';
      let guidance = 2.5;
      let steps = 17;
      const applyArchSampleDefaults = loadFunction(page, 'applyArchSampleDefaults', {
        archs: modelArchs, getDefaultSampler,
        setSampler: value => { sampler = value; },
        setGuidanceScale: value => { guidance = value; },
        setSampleSteps: value => { steps = value; },
      });
      const applyLoraModelDefaults = loadFunction(page, 'applyLoraModelDefaults', {
        modelConfig: model,
        archs: modelArchs, getLoraModelConfig,
        setModelConfig: update => { model = update(model); },
        applyArchSampleDefaults,
      });
      applyLoraModelDefaults(lora);
      assert.equal(model.arch, arch);
      assert.equal(model.vae_dtype, 'fp32');
      assert.equal(model.model_kwargs.distillation_lora_path, arch.endsWith('_turbo') ? distillationPath : undefined);
      const changedArch = previousArch !== arch;
      assert.equal(sampler, changedArch ? 'flowmatch' : 'custom-sampler');
      assert.equal(guidance, changedArch ? (arch.endsWith('_turbo') ? 1 : 3) : 2.5);
      assert.equal(steps, changedArch ? (arch.endsWith('_turbo') ? 8 : 50) : 17);
    }
  }
});

test('Generate library architecture changes clear omitted Turbo settings and same-architecture swaps preserve explicit settings', () => {
  for (const arch of ['moss_image', 'moss_image:base', 'moss_image:edit', 'moss_image:base_edit', 'moss_image:kromoss', 'moss_image:kromoss_turbo']) {
    let model = getDefaultModelConfig('moss_image:kromoss_turbo');
    model.vae_path = 'D:/custom/vae.safetensors';
    model.model_kwargs = { ...model.model_kwargs, max_text_length: 384 };
    const previous = structuredClone(model);
    const applyLoraModelDefaults = loadFunction('app/generate/GeneratePageContent.tsx', 'applyLoraModelDefaults', {
      modelConfig: model, archs: modelArchs, getLoraModelConfig,
      setModelConfig: update => { model = update(model); },
      applyArchSampleDefaults: () => {},
    });
    applyLoraModelDefaults({ model: { arch, name_or_path: 'saved/model' } });
    assert.equal(model.name_or_path, 'saved/model');
    if (arch === previous.arch) {
      assert.equal(model.vae_path, previous.vae_path);
      assert.deepEqual(model.model_kwargs, previous.model_kwargs);
    } else {
      const defaults = getDefaultModelConfig(arch);
      assert.equal(model.vae_path, defaults.vae_path);
      assert.equal(model.vae_dtype, defaults.vae_dtype);
      assert.deepEqual(model.model_kwargs, defaults.model_kwargs);
      assert.equal(model.model_kwargs.distillation_lora_path, undefined);
    }
  }
});

test('Generate payload retains the fixed Turbo adapter and FP32 VAE with or without a selected user LoRA', () => {
  const cleanModelConfig = loadFunction('app/generate/GeneratePageContent.tsx', 'cleanModelConfig');
  const model = getDefaultModelConfig('moss_image:kromoss_turbo');
  model.assistant_lora_path = 'stale-assistant.safetensors';
  model.inference_lora_path = 'stale-inference.safetensors';
  model.lora_path = 'stale-user.safetensors';
  for (const enabled of [true, false]) {
    const payload = cleanModelConfig(model, enabled, ' D:/training/user-lora.safetensors ');
    assert.equal(payload.lora_path, enabled ? 'D:/training/user-lora.safetensors' : undefined);
    assert.equal(payload.model_kwargs.distillation_lora_path, distillationPath);
    assert.equal(payload.model_kwargs.sample_shift, 5);
    assert.deepEqual(payload.model_kwargs.uncond_skip_layers, []);
    assert.equal(payload.vae_path, vaePath);
    assert.equal(payload.vae_dtype, 'fp32');
    assert.equal(payload.assistant_lora_path, undefined);
    assert.equal(payload.inference_lora_path, undefined);
  }
});
