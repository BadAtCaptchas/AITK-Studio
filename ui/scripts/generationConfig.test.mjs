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
const { getDefaultModelConfig, getDefaultGenerationSettings, getLoraModelConfig } = loadSource('../src/domain/generationConfig.ts', {
  './modelOptions': catalog,
  '../utils/memoryProfiles': loadSource('../src/utils/memoryProfiles.ts'),
});

function pageFunction(name, bindings) {
  const filename = new URL('../src/app/generate/GeneratePageContent.tsx', import.meta.url);
  const source = ts.createSourceFile(filename.pathname, fs.readFileSync(filename, 'utf8'), ts.ScriptTarget.Latest, true);
  let expression;
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(source) === name && node.initializer) expression = node.initializer.getText(source);
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.ok(expression, `Missing page handler ${name}`);
  const code = ts.transpileModule(`const handler = ${expression}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText;
  return new Function(...Object.keys(bindings), `${code}; return handler;`)(...Object.values(bindings));
}

test('Generate architecture changes reset a Turbo schedule through the actual page handler', () => {
  for (const target of ['flux', 'sdxl', 'sd15']) {
    let model = getDefaultModelConfig('boogu_image_turbo');
    const settings = { sampler: 'flowmatch', guidanceScale: 0, sampleSteps: 4, width: 1536, height: 768 };
    const applyArchSampleDefaults = pageFunction('applyArchSampleDefaults', {
      getDefaultGenerationSettings, archs: catalog.modelArchs,
      setSampler: value => { settings.sampler = value; },
      setGuidanceScale: value => { settings.guidanceScale = value; },
      setSampleSteps: value => { settings.sampleSteps = value; },
      setWidth: value => { settings.width = value; },
      setHeight: value => { settings.height = value; },
    });
    const handleArchChange = pageFunction('handleArchChange', {
      getDefaultModelConfig, archs: catalog.modelArchs, applyArchSampleDefaults,
      setModelConfig: value => { model = value; },
    });
    handleArchChange(target);
    assert.equal(model.arch, target);
    assert.deepEqual(settings, getDefaultGenerationSettings(target));
    assert.equal(settings.sampleSteps, 25);
    assert.equal(settings.guidanceScale, target === 'sd15' || target === 'sdxl' ? 6 : 4);
  }
});

test('generation defaults reset the entire schedule and resolution after a Turbo model', () => {
  assert.deepEqual(getDefaultGenerationSettings('boogu_image_turbo'), {
    sampler: 'flowmatch', guidanceScale: 0, sampleSteps: 4, width: 1024, height: 1024,
  });
  assert.deepEqual(getDefaultGenerationSettings('flux'), {
    sampler: 'flowmatch', guidanceScale: 4, sampleSteps: 25, width: 1024, height: 1024,
  });
  assert.deepEqual(getDefaultGenerationSettings('sd15'), {
    sampler: 'ddpm', guidanceScale: 6, sampleSteps: 25, width: 512, height: 512,
  });
  assert.equal(getDefaultGenerationSettings('sdxl').sampleSteps, 25);
});

test('generation preserves required auxiliary component repositories and VAE precision', () => {
  for (const arch of ['zeta_chroma', 'zimage_l2p']) {
    assert.equal(getDefaultModelConfig(arch).extras_name_or_path, 'Tongyi-MAI/Z-Image-Turbo');
  }
  assert.equal(getDefaultModelConfig('zimage:juggernaut_z').extras_name_or_path, 'Tongyi-MAI/Z-Image');
  const custom = [{ name: 'custom_model', defaults: {
    'config.process[0].model.te_name_or_path': ['custom/encoder'],
    'config.process[0].model.refiner_name_or_path': ['custom/refiner'],
    'config.process[0].model.vae_path': ['custom/vae'],
    'config.process[0].model.vae_dtype': ['fp32'],
  } }];
  const config = getDefaultModelConfig('custom_model', custom);
  assert.equal(config.te_name_or_path, 'custom/encoder');
  assert.equal(config.refiner_name_or_path, 'custom/refiner');
  assert.equal(config.vae_path, 'custom/vae');
  assert.equal(config.vae_dtype, 'fp32');
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
