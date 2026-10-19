import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const src = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src');
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

const { defaultJobConfig, migrateJobConfig } = load('app/jobs/new/jobConfig.ts');
const { handleModelArchChange } = load('app/jobs/new/utils.ts');
const { hasTextEmbeddingCache, updateTextEncoderControls } = load('utils/textEncoderControls.ts');

function loadFunction(relative, name, bindings = {}) {
  const filename = path.resolve(src, relative);
  const source = ts.createSourceFile(filename, fs.readFileSync(filename, 'utf8'), ts.ScriptTarget.Latest, true);
  let expression;
  function visit(node) {
    if (ts.isFunctionDeclaration(node) && node.name?.text === name) expression = node.getText(source);
    if (ts.isVariableDeclaration(node) && node.name.getText(source) === name && node.initializer) expression = node.initializer.getText(source);
    ts.forEachChild(node, visit);
  }
  visit(source);
  assert.ok(expression, `Missing ${name}`);
  return evaluate(expression, bindings);
}
function evaluate(expression, bindings) {
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

test('legacy Auto Learn migration preserves automatic phases, total steps, and explicit phase caps', () => {
  const job = structuredClone(defaultJobConfig);
  const train = job.config.process[0].train;
  delete train.auto_train;
  train.auto_learn = true;
  train.phases = [
    { name: 'Warmup', lr: 0.0001 },
    { name: 'Detail', steps: 700, lr: 0.00001, auto_advance: { min_steps: 200, window: 50 } },
  ];
  const phases = structuredClone(train.phases);
  const steps = train.steps;
  const migrated = migrateJobConfig(job);
  assert.equal(migrated.config.process[0].train.auto_train, true);
  assert.equal(migrated.config.process[0].train.auto_learn, undefined);
  assert.equal(migrated.config.process[0].train.steps, steps);
  assert.deepEqual(migrated.config.process[0].train.phases, phases);
  assert.deepEqual(migrateJobConfig(structuredClone(migrated)), migrated);
});

test('explicit auto_train takes precedence over either legacy alias value and migration remains idempotent', () => {
  for (const [canonical, legacy, expected] of [
    [undefined, false, false], [false, true, false], [true, false, true], [undefined, undefined, false],
  ]) {
    const job = structuredClone(defaultJobConfig);
    const train = job.config.process[0].train;
    if (canonical === undefined) delete train.auto_train;
    else train.auto_train = canonical;
    if (legacy !== undefined) train.auto_learn = legacy;
    train.steps = 50;
    train.phases = [{ name: 'Warmup', steps: 20 }, { name: 'Detail', steps: 30 }];
    const migrated = migrateJobConfig(job);
    assert.equal(migrated.config.process[0].train.auto_train, expected);
    assert.equal(migrated.config.process[0].train.auto_learn, undefined);
    assert.equal(migrated.config.process[0].train.steps, 50);
    assert.deepEqual(migrated.config.process[0].train.phases.map(phase => phase.steps), [20, 30]);
    assert.deepEqual(migrateJobConfig(structuredClone(migrated)), migrated);
  }
});

test('editing automatic phases keeps explicit step caps and leaves uncapped phases uncapped', () => {
  const relative = 'app/jobs/new/TrainingPhasesEditor.tsx';
  const defaultAutoAdvance = loadFunction(relative, 'defaultAutoAdvance');
  const normalizePhase = loadFunction(relative, 'normalizePhase', { defaultAutoAdvance });
  assert.equal(normalizePhase({ name: 'Capped', steps: 700 }, 0, true).steps, 700);
  assert.equal(normalizePhase({ name: 'Uncapped' }, 0, true).steps, undefined);
});

test('effective cache status follows dataset overrides of the global default', () => {
  assert.equal(hasTextEmbeddingCache({ cache_text_embeddings: false }, [{ cache_text_embeddings: true }]), true);
  assert.equal(hasTextEmbeddingCache({ cache_text_embeddings: true }, [{}, {}]), true);
  assert.equal(hasTextEmbeddingCache({ cache_text_embeddings: true }, [{ cache_text_embeddings: false }]), false);
  assert.equal(hasTextEmbeddingCache({ cache_text_embeddings: false }, [{}, { cache_text_embeddings: false }]), false);
  assert.equal(hasTextEmbeddingCache({ cache_text_embeddings: false }, [{ cache_text_embeddings: true }, {}]), true);
});

function configWithMultipleDatasets() {
  const process = structuredClone(defaultJobConfig.config.process[0]);
  process.datasets = [true, false, undefined].map((cacheEnabled, index) => ({
    ...process.datasets[0], folder_path: `/datasets/${index}`, cache_text_embeddings: cacheEnabled,
  }));
  return process;
}

test('cache toggles synchronize every dataset and disabling cache cannot activate trigger-only conditioning', () => {
  for (const enabled of [false, true]) {
    const config = configWithMultipleDatasets();
    config.train.unload_text_encoder = true;
    const before = structuredClone(config);
    const result = updateTextEncoderControls(config, 'cache', enabled);
    assert.equal(result.train.cache_text_embeddings, enabled);
    assert.equal(result.train.unload_text_encoder, false);
    assert.deepEqual(result.datasets.map(dataset => dataset.cache_text_embeddings), [enabled, enabled, enabled]);
    assert.deepEqual(result.datasets.map(dataset => dataset.folder_path), ['/datasets/0', '/datasets/1', '/datasets/2']);
    assert.equal(hasTextEmbeddingCache(result.train, result.datasets), enabled);
    assert.deepEqual(config, before);
  }
});

test('trigger-only unloading clears caching and disabling unloading preserves an existing cache mode', () => {
  const config = configWithMultipleDatasets();
  config.train.cache_text_embeddings = true;
  const unloaded = updateTextEncoderControls(config, 'unload', true);
  assert.equal(unloaded.train.unload_text_encoder, true);
  assert.equal(unloaded.train.cache_text_embeddings, false);
  assert.ok(unloaded.datasets.every(dataset => dataset.cache_text_embeddings === false));
  const cached = updateTextEncoderControls(config, 'cache', true);
  const unchangedCache = updateTextEncoderControls(cached, 'unload', false);
  assert.equal(unchangedCache.train.unload_text_encoder, false);
  assert.equal(hasTextEmbeddingCache(unchangedCache.train, unchangedCache.datasets), true);
});

test('text encoder training clears every cache flag and rejects enabling cache or unload while training', () => {
  const config = configWithMultipleDatasets();
  config.train.cache_text_embeddings = true;
  config.train.unload_text_encoder = true;
  const trained = updateTextEncoderControls(config, 'training', true);
  assert.equal(trained.train.train_text_encoder, true);
  assert.equal(trained.train.cache_text_embeddings, false);
  assert.equal(trained.train.unload_text_encoder, false);
  assert.ok(trained.datasets.every(dataset => dataset.cache_text_embeddings === false));
  assert.deepEqual(updateTextEncoderControls(trained, 'cache', true), trained);
  assert.deepEqual(updateTextEncoderControls(trained, 'unload', true), trained);
  const frozen = updateTextEncoderControls(trained, 'training', false);
  assert.equal(frozen.train.train_text_encoder, false);
  assert.equal(hasTextEmbeddingCache(frozen.train, frozen.datasets), false);
  assert.equal(frozen.train.unload_text_encoder, false);
});

test('actual Kromoss cache checkbox clears the preset dataset cache and stale unload flag', () => {
  const job = structuredClone(defaultJobConfig);
  handleModelArchChange(job.config.process[0].model.arch, 'moss_image:kromoss', job, setter(job));
  const processConfig = job.config.process[0];
  const relative = 'app/jobs/new/SimpleJob.tsx';
  const handleTextEncoderControlChange = loadFunction(relative, 'handleTextEncoderControlChange', {
    processConfig, setJobConfig: setter(job), updateTextEncoderControls,
  });
  const filename = path.resolve(src, relative);
  const source = ts.createSourceFile(filename, fs.readFileSync(filename, 'utf8'), ts.ScriptTarget.Latest, true);
  const controls = new Map();
  function visit(node) {
    if (ts.isJsxSelfClosingElement(node)) {
      const attrs = node.attributes.properties;
      const label = attrs.find(attr => ts.isJsxAttribute(attr) && attr.name.getText(source) === 'label');
      if (label?.initializer && ts.isStringLiteral(label.initializer)) controls.set(label.initializer.text, attrs);
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
  function expression(label, prop) {
    return controls.get(label).find(attr => ts.isJsxAttribute(attr) && attr.name.getText(source) === prop).initializer.expression.getText(source);
  }
  const bindings = {
    processConfig, handleTextEncoderControlChange,
    textEmbeddingCacheEnabled: hasTextEmbeddingCache(processConfig.train, processConfig.datasets),
  };
  assert.equal(evaluate(expression('Cache embeddings', 'checked'), bindings), true);
  assert.equal(evaluate(expression('Unload TE', 'checked'), bindings), false);
  evaluate(expression('Cache embeddings', 'onChange'), bindings)(false);
  assert.equal(job.config.process[0].train.cache_text_embeddings, false);
  assert.equal(job.config.process[0].train.unload_text_encoder, false);
  assert.ok(job.config.process[0].datasets.every(dataset => dataset.cache_text_embeddings === false));
});
