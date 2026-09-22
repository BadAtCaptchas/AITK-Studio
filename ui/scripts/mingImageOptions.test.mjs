import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';
import { createRequire } from 'node:module';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const { configContractErrors } = require('../dist/src/domain/configContract.js');
const { applySelectedDatasetDefaults } = require('../dist/src/utils/jobDatasetDefaults.js');
const cache = new Map();
function loadSource(filename) {
  filename = path.resolve(filename);
  if (cache.has(filename)) return cache.get(filename);
  if (filename.endsWith('.json')) return JSON.parse(fs.readFileSync(filename, 'utf8'));
  const module = { exports: {} };
  cache.set(filename, module.exports);
  const source = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true },
  }).outputText;
  new Function('require', 'module', 'exports', source)(
    specifier => {
      if (!specifier.startsWith('.') && !specifier.startsWith('@/')) return require(specifier);
      const target = specifier.startsWith('@/')
        ? path.resolve('src', specifier.slice(2))
        : path.resolve(path.dirname(filename), specifier);
      return loadSource(/\.(json|ts)$/.test(target) ? target : target + '.ts');
    },
    module,
    module.exports,
  );
  cache.set(filename, module.exports);
  return module.exports;
}
const { modelArchs, getTransformerQuantizationOptions } = loadSource('src/domain/modelOptions.ts');

test('Ming presets select LoRA, fixed cached conditioning, RGBA PNG samples and proper dataset types', () => {
  for (const arch of ['ming_image_design', 'ming_image_design_layer']) {
    const preset = modelArchs.find(model => model.name === arch);
    assert.ok(preset);
    assert.deepEqual(preset.allowedNetworkTypes, ['lora']);
    assert.equal(preset.defaults['config.process[0].train.cache_text_embeddings'][0], true);
    assert.equal(preset.defaults['config.process[0].train.train_text_encoder'][0], false);
    assert.equal(preset.defaults['config.process[0].train.train_unet'][0], true);
    assert.equal(preset.defaults['config.process[0].model.qtype'][0], 'ming_fp8');
    assert.deepEqual(
      getTransformerQuantizationOptions(arch).map(option => option.value),
      ['', 'ming_fp8'],
    );
    assert.equal(preset.defaults['config.process[0].train.standardize_images'][0], false);
    assert.equal(preset.defaults['config.process[0].train.img_multiplier'][0], 1);
    assert.equal(preset.defaults['config.process[0].network.transformer_only'][0], true);
    assert.equal(preset.defaults['config.process[0].sample.format'][0], 'png');
    const dataset = applySelectedDatasetDefaults(
      { type: 'image', folder_path: 'unchanged', caption_dropout_rate: 0.05, token_dropout_rate: 0.1, random_triggers: ['old'], do_i2v: true },
      preset.defaults,
    );
    assert.equal(dataset.type, arch.endsWith('_layer') ? 'layered_image' : 'image');
    assert.equal(dataset.caption_dropout_rate, 0);
    assert.equal(dataset.token_dropout_rate, 0);
    assert.deepEqual(dataset.random_triggers, []);
    assert.equal(dataset.do_i2v, false);
    assert.equal(dataset.folder_path, 'unchanged');
    assert.equal(dataset.random_crop, false);
    assert.equal(dataset.standardize_images, false);
    assert.equal(dataset.shuffle_tokens, false);
    if (arch.endsWith('_layer')) assert.equal(preset.defaults['config.process[0].sample.num_layers'][0], 2);
  }
  assert.equal(
    getTransformerQuantizationOptions('flux').some(option => option.value === 'ming_fp8'),
    false,
  );
});

const config = (arch = 'ming_image_design_layer') => ({
  config: {
    process: [
      {
        type: 'diffusion_trainer',
        device: 'cuda:0',
        model: { arch, dtype: 'bf16' },
        train: { batch_size: 1 },
        network: { type: 'lora' },
        datasets: [{ type: arch.endsWith('_layer') ? 'layered_image' : 'image' }],
        sample: { num_layers: 2, neg: '', samples: [{ prompt: 'poster', num_layers: 3 }] },
      },
    ],
  },
});
test('Ming config accepts both supported arches and rejects invalid layer counts, negative prompts, datasets and networks', () => {
  for (const arch of ['ming_image_design', 'ming_image_design_layer'])
    assert.deepEqual(configContractErrors(config(arch)), []);
  for (const count of [0, 33, 1.5, '2', null]) {
    const value = config();
    value.config.process[0].sample.samples[0].num_layers = count;
    assert.ok(configContractErrors(value).some(error => error.includes('num_layers')));
  }
  for (const mutate of [
    process => {
      process.sample.neg = 'blur';
    },
    process => {
      process.sample.samples[0].neg = 'blur';
    },
    process => {
      process.network.type = 'lokr';
    },
    process => {
      process.datasets[0].type = 'image';
    },
    process => {
      process.train.batch_size = 2;
    },
    process => {
      process.train.train_text_encoder = true;
    },
    process => {
      process.sample.width = 513;
    },
    process => {
      process.sample.samples[0].height = 1025;
    },
  ]) {
    const value = config();
    mutate(value.config.process[0]);
    assert.ok(configContractErrors(value).length);
  }
});
