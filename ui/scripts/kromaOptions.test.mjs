import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import test from 'node:test';
import ts from 'typescript';
import YAML from 'yaml';

const require = createRequire(import.meta.url);
const src = path.resolve('src');
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
const { getDefaultModelConfig } = load('domain/generationConfig.ts');
const { configContractErrors } = load('domain/configContract.ts');
const { supportsKrea2TextFusionExclusion } = load('utils/krea2TextFusion.ts');
const variants = ['base', 'teacher', 'turbo_opd'];

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

for (const variant of variants) {
  const arch = `krea2:kroma_${variant}`;
  test(`${arch}: complete training and generation defaults survive YAML`, () => {
    const job = structuredClone(defaultJobConfig);
    const p = select(job, arch);
    assert.equal(p.model.name_or_path, 'lodestones/Kroma');
    assert.match(p.model.model_kwargs.checkpoint_filename, /\.safetensors$/);
    assert.equal(p.network.type, 'lora');
    assert.equal(p.network.linear, 16);
    assert.equal(p.train.train_text_encoder, false);
    assert.equal(p.train.unload_text_encoder, true);
    assert.equal(p.train.cache_text_embeddings, true);
    assert.deepEqual(p.network.network_kwargs.ignore_if_contains, ['txtfusion.', 'txtmlp']);
    assert.deepEqual(p.datasets[0].resolution, [512]);
    assert.equal(p.model.layer_offloading, true);
    assert.equal(p.model.layer_offloading_backend, 'legacy');
    assert.equal(p.sample.keep_low_vram_for_samples, true);
    assert.equal(p.sample.guidance_scale, variant === 'turbo_opd' ? 0 : 4);
    assert.equal(p.model.assistant_lora_path, undefined);
    assert.deepEqual(configContractErrors(YAML.parse(YAML.stringify(job))), []);
    assert.equal(supportsKrea2TextFusionExclusion(p.model, p.network), true);
    const generation = getDefaultModelConfig(arch);
    assert.deepEqual(generation.model_kwargs, p.model.model_kwargs);
    assert.equal(generation.layer_offloading, true);
    assert.equal(generation.qtype, 'float8');
    p.network.type = 'lokr';
    assert.ok(configContractErrors(job).some(error => error.includes('does not support network')));
    const example = YAML.parse(fs.readFileSync(`../config/examples/train_lora_kroma_${variant}_16gb.yaml`, 'utf8'));
    assert.deepEqual(configContractErrors(example), []);
    const exampleProcess = example.config.process[0];
    for (const [key, value] of Object.entries(exampleProcess.model)) assert.deepEqual(value, p.model[key]);
    for (const key of ['lr', 'dtype', 'cache_text_embeddings', 'unload_text_encoder', 'timestep_type']) {
      assert.equal(exampleProcess.train[key], p.train[key]);
    }
  });
}

test('Krea/Kroma transitions remove stale edit, checkpoint, schedule and assistant settings', () => {
  for (const start of ['krea2', 'krea2:turbo', 'krea2:o_edit', 'krea2:o_edit_turbo', ...variants.map(v => `krea2:kroma_${v}`)]) {
    for (const end of ['krea2', 'krea2:turbo', 'krea2:o_edit', 'krea2:o_edit_turbo', ...variants.map(v => `krea2:kroma_${v}`)]) {
      if (start === end) continue;
      const job = structuredClone(defaultJobConfig);
      select(job, start);
      const p = select(job, end);
      assert.equal(Boolean(p.model.model_kwargs?.edit), end.includes('o_edit'));
      assert.equal(Boolean(p.model.model_kwargs?.checkpoint_filename), end.includes('kroma'));
      assert.equal(p.model.model_kwargs?.schedule_mu, end.endsWith('kroma_turbo_opd') ? 1.15 : undefined);
      assert.equal(Boolean(p.model.assistant_lora_path), ['krea2:turbo', 'krea2:o_edit_turbo'].includes(end));
    }
  }
});

test('Turbo adapter is opt-in and switching away clears it; saved custom sources are untouched', () => {
  const job = structuredClone(defaultJobConfig);
  const p = select(job, 'krea2:kroma_turbo_opd');
  const control = modelArchs.find(a => a.name === p.model.arch).customModelSelectOptions[0];
  assert.equal(control.getValue(job), 'none');
  control.onChange('adapter', job, setter(job));
  assert.match(p.model.assistant_lora_path, /^ostris\/krea2_turbo_training_adapter\//);
  assert.equal(control.getValue(job), 'adapter');
  control.onChange('none', job, setter(job));
  assert.equal(p.model.assistant_lora_path, undefined);
  p.model.name_or_path = 'D:/custom/kroma.safetensors';
  p.model.model_kwargs.text_encoder_path = 'D:/custom/encoder';
  p.model.model_kwargs.vae_path = 'D:/custom/vae';
  const before = structuredClone(job);
  select(job, p.model.arch);
  assert.deepEqual(job, before);
  control.onChange('adapter', job, setter(job));
  select(job, 'krea2:kroma_teacher');
  assert.equal(p.model.assistant_lora_path, undefined);
});
