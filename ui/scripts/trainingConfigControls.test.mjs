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
  return evaluate(expression.replace(/^export\s+/, ''), bindings);
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

const { configContractErrors } = load('domain/configContract.ts');
const { trainingPhaseErrors } = load('utils/trainingPhases.ts');

test('saved-job start and resume reject removed automatic settings before launch preparation', async () => {
  for (const status of ['stopped', 'completed', 'error', 'queued']) {
    const config = structuredClone(defaultJobConfig);
    config.config.process[0].train.auto_learn = true;
    const prepareJobStart = loadFunction('server/jobStart.ts', 'prepareJobStart', {
      isValidJobId: () => true,
      db: { jobs: { findById: async () => ({ status, worker_id: 'local', job_config: JSON.stringify(config) }) } },
      isLocalWorker: () => true,
      assertPythonRuntimeReady: async () => {},
      configContractErrors,
      failStart: (payload, code) => { throw Object.assign(new Error(payload.error), { payload, status: code }); },
      process,
    });
    await assert.rejects(prepareJobStart('saved-job', undefined, false, { allowQueued: true }), error => {
      assert.equal(error.status, 400);
      assert.equal(error.payload.code, 'INVALID_CONFIG');
      assert.match(error.message, /no longer supported/);
      return true;
    });
  }
});

test('already queued jobs reject automatic or invalid manual phases before filesystem work or spawning', async () => {
  for (const train of [
    { auto_train: true }, { auto_learn: true },
    { steps: 10, phases: [{ steps: 10, auto_advance: {} }] },
    { steps: 10, phases: [{ steps: 10, auto_advance: null }] },
    { steps: 10, phases: [{}] }, { steps: 20, phases: [{ steps: 10 }] },
  ]) {
    const updates = [];
    const startAndWatchJob = loadFunction('../cron/actions/startJob.ts', 'startAndWatchJob', {
      trainingPhaseErrors,
      getJobTrainingRoot: async () => assert.fail('Invalid phases must fail before filesystem preparation'),
      db: { jobs: { updateIf: async (_id, _expected, update) => { updates.push(update); } } },
      console: { error: () => {} },
    });
    await startAndWatchJob({ id: 'queued-job', attempt_id: 'attempt', job_config: JSON.stringify({ config: { process: [{ train }] } }) });
    assert.equal(updates.length, 1);
    assert.equal(updates[0].status, 'error');
    assert.equal(updates[0].info, `Error launching job: ${trainingPhaseErrors(train).join('\n')}`);
  }
});
test('removed automatic settings stay visible and fail validation without rewriting saved steps', () => {
  for (const raw of [
    { auto_train: true }, { auto_learn: true }, { auto_train: false, auto_learn: true },
    { phases: [{ name: 'Saved', steps: 750, auto_advance: null }] },
    { phases: [{ name: 'Saved', auto_advance: { type: 'loss_plateau' } }] },
  ]) {
    const job = structuredClone(defaultJobConfig);
    Object.assign(job.config.process[0].train, { steps: 750, phases: [{ name: 'Saved', steps: 750 }] }, raw);
    const before = structuredClone(job.config.process[0].train);
    const migrated = migrateJobConfig(job);
    assert.equal(migrated.config.process[0].train.steps, before.steps);
    assert.deepEqual(migrated.config.process[0].train.phases, before.phases);
    assert.ok(configContractErrors(migrated).some(error => error.includes('no longer supported')));
    assert.deepEqual(migrateJobConfig(structuredClone(migrated)), migrated);
  }
});

test('disabled legacy flags strip without changing finite manual phases or ordinary settings', () => {
  const job = structuredClone(defaultJobConfig);
  Object.assign(job.config.process[0].train, { auto_train: false, auto_learn: false, steps: 50,
    phases: [{ name: 'Warmup', steps: 20, lr: 0.0001 }, { name: 'Detail', steps: 30, loss_type: 'mae' }] });
  const migrated = migrateJobConfig(job);
  assert.equal(migrated.config.process[0].train.auto_train, undefined);
  assert.equal(migrated.config.process[0].train.auto_learn, undefined);
  assert.deepEqual(migrated.config.process[0].train.phases.map(phase => phase.steps), [20, 30]);
  assert.equal(migrated.config.process[0].train.steps, 50);
  assert.deepEqual(configContractErrors(migrated), []);
  assert.deepEqual(migrateJobConfig(structuredClone(migrated)), migrated);
});

test('manual phase validation rejects missing, nonfinite, fractional and mismatched steps', () => {
  for (const steps of [undefined, 0, -1, 1.5, Infinity, '20']) {
    const job = structuredClone(defaultJobConfig);
    job.config.process[0].train.phases = [{ name: 'Repair me', steps }];
    const migrated = migrateJobConfig(job);
    assert.equal(migrated.config.process[0].train.phases[0].steps, steps);
    assert.ok(configContractErrors(migrated).some(error => error.includes('positive integer')));
  }
  const job = structuredClone(defaultJobConfig);
  job.config.process[0].train.phases = [{ name: 'Finite', steps: 10 }];
  assert.ok(configContractErrors(job).some(error => error.includes('sum of manual phase steps (10)')));
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
