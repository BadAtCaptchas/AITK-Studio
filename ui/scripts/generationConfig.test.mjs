import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import ts from 'typescript';

const require = createRequire(import.meta.url);
function loadSource(relative, overrides = {}) {
  const filename = new URL(relative, import.meta.url);
  const code = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const evaluatedModule = { exports: {} };
  new Function('require', 'module', 'exports', code)(
    name => (Object.hasOwn(overrides, name) ? overrides[name] : require(name)),
    evaluatedModule,
    evaluatedModule.exports,
  );
  return evaluatedModule.exports;
}

const catalog = loadSource('../src/domain/modelOptions.ts', {
  './trainingDefaults': loadSource('../src/domain/trainingDefaults.ts'),
  '@/helpers/defaultSamples': loadSource('../src/helpers/defaultSamples.ts'),
  './modelCapabilities.json': require('../src/domain/modelCapabilities.json'),
});
const { getDefaultModelConfig, getLoraModelConfig } = loadSource('../src/domain/generationConfig.ts', {
  './modelOptions': catalog,
  '../utils/memoryProfiles': loadSource('../src/utils/memoryProfiles.ts'),
});
const memoryKeys = [
  'low_vram',
  'layer_offloading',
  'layer_offloading_backend',
  'layer_offloading_transformer_percent',
  'layer_offloading_text_encoder_percent',
];
function memorySettings(model) {
  return Object.fromEntries(memoryKeys.map(key => [key, model[key]]));
}

test('generation defaults keep memory-saving flags disabled when the catalog supplies no memory defaults', () => {
  for (const [arch, backend, transformerPercent, textEncoderPercent] of [
    ['custom_model', 'legacy', 1, 1],
    ['flux', 'block', 0.7, 0.5],
  ]) {
    const installed = [{ name: arch, label: 'Test model', group: 'image', defaults: {} }];
    assert.deepEqual(memorySettings(getDefaultModelConfig(arch, installed)), {
      low_vram: false,
      layer_offloading: false,
      layer_offloading_backend: backend,
      layer_offloading_transformer_percent: transformerPercent,
      layer_offloading_text_encoder_percent: textEncoderPercent,
    });
  }
});

test('both Kromoss generation presets retain their catalog offloading defaults', () => {
  for (const arch of ['moss_image:kromoss', 'moss_image:kromoss_turbo']) {
    const model = getDefaultModelConfig(arch);
    assert.deepEqual(memorySettings(model), {
      low_vram: true,
      layer_offloading: true,
      layer_offloading_backend: 'legacy',
      layer_offloading_transformer_percent: 1,
      layer_offloading_text_encoder_percent: 1,
    });
    assert.equal(model.vae_dtype, 'fp32');
    assert.equal(model.vae_path, 'lodestones/Kromoss/vae/flux2-vae_small.safetensors');
  }
});

test('same-architecture LoRA selection preserves all manual memory fields omitted by the saved model', () => {
  const current = {
    ...getDefaultModelConfig('flux'),
    low_vram: true,
    layer_offloading: true,
    layer_offloading_backend: 'legacy',
    layer_offloading_transformer_percent: 0.17,
    layer_offloading_text_encoder_percent: 0.23,
  };
  for (const saved of [{ arch: 'flux', name_or_path: 'saved/base' }, { name_or_path: 'saved/base' }]) {
    const selected = getLoraModelConfig(current, saved);
    assert.deepEqual(memorySettings(selected), memorySettings(current));
    assert.equal(selected.arch, 'flux');
    assert.equal(selected.name_or_path, 'saved/base');
  }
});

test('explicit saved memory values, including false and zero, take precedence for same and different architectures', () => {
  const savedMemory = {
    low_vram: false,
    layer_offloading: false,
    layer_offloading_backend: 'block',
    layer_offloading_transformer_percent: 0,
    layer_offloading_text_encoder_percent: 0,
  };
  for (const arch of ['flux', 'moss_image:kromoss_turbo']) {
    const current = {
      ...getDefaultModelConfig(arch),
      low_vram: true,
      layer_offloading: true,
      layer_offloading_backend: 'legacy',
      layer_offloading_transformer_percent: 1,
      layer_offloading_text_encoder_percent: 1,
    };
    const selected = getLoraModelConfig(current, { arch: 'flux', name_or_path: 'saved/base', ...savedMemory });
    assert.deepEqual(memorySettings(selected), savedMemory);
    assert.equal(selected.arch, 'flux');
    assert.equal(selected.name_or_path, 'saved/base');
  }
});

test('different-architecture LoRA selection resets omitted memory fields to the target defaults', () => {
  for (const [source, target] of [
    ['moss_image:kromoss_turbo', 'flux'],
    ['flux', 'moss_image:kromoss'],
    ['moss_image:kromoss', 'moss_image:kromoss_turbo'],
  ]) {
    const current = {
      ...getDefaultModelConfig(source),
      low_vram: false,
      layer_offloading: false,
      layer_offloading_backend: 'block',
      layer_offloading_transformer_percent: 0.13,
      layer_offloading_text_encoder_percent: 0.29,
    };
    const selected = getLoraModelConfig(current, { arch: target, name_or_path: 'saved/base' });
    assert.deepEqual(memorySettings(selected), memorySettings(getDefaultModelConfig(target)));
    assert.equal(selected.name_or_path, 'saved/base');
    assert.deepEqual(selected.model_kwargs, getDefaultModelConfig(target).model_kwargs);
  }
});

test('LoRA model selection leaves the current config and saved model unchanged', () => {
  for (const arch of ['moss_image:kromoss_turbo', 'flux']) {
    const current = structuredClone(getDefaultModelConfig('moss_image:kromoss_turbo'));
    const saved = { arch, name_or_path: 'saved/base', low_vram: false, model_kwargs: { custom_option: true } };
    const beforeCurrent = structuredClone(current);
    const beforeSaved = structuredClone(saved);
    Object.freeze(current.model_kwargs);
    Object.freeze(current);
    Object.freeze(saved.model_kwargs);
    Object.freeze(saved);
    const selected = getLoraModelConfig(current, saved);
    assert.notEqual(selected, current);
    assert.notEqual(selected, saved);
    assert.deepEqual(current, beforeCurrent);
    assert.deepEqual(saved, beforeSaved);
    assert.deepEqual(selected.model_kwargs, saved.model_kwargs);
  }
});
