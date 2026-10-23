import assert from 'node:assert/strict';
import test from 'node:test';
import { loadTrainingSource } from './trainingMigrationFixture.mjs';

const { parseCaptionJobConfig, parseCaptionJobSource } = loadTrainingSource('utils/captionJobSource.ts');
const { defaultCaptionJobConfig } = loadTrainingSource('helpers/captionJobConfig.ts');
const fixture = () => structuredClone(defaultCaptionJobConfig);
const source = config => ({ name: 'stored-caption', gpu_ids: '0,2', job_type: 'caption', job_config: JSON.stringify(config) });

test('caption sources preserve original identity, GPU selection and all advanced fields without mutating the row', () => {
  const config = fixture();
  config.extra = { custom: ['keep', 3] };
  config.config.process[0].caption.future_option = { strength: 0.25 };
  config.config.process.push(structuredClone(config.config.process[0]));
  const row = source(config);
  const before = structuredClone(row);
  const parsed = parseCaptionJobSource(row);
  assert.deepEqual(parsed, { config, name: 'stored-caption', gpuIds: '0,2' });
  assert.deepEqual(row, before);
  parsed.config.config.name = 'edited';
  assert.deepEqual(row, before);
});

test('safe historical caption documents retain absent runtime-defaulted options without filling or dropping fields', () => {
  const config = { job: 'extension', config: { name: 'legacy', process: [{
    type: 'InstalledCustomCaptioner', caption: {
      model_name_or_path: 'fixture/model', path_to_caption: 'fixture/data', extensions: ['png'],
      loras: [{ path: 'fixture/adapter' }], future_option: { preserve: true },
    },
  }] } };
  const before = structuredClone(config);
  assert.equal(parseCaptionJobConfig(config), config);
  assert.deepEqual(config, before);
  const row = source(config);
  delete row.job_type;
  row.gpu_ids = null;
  const parsed = parseCaptionJobSource(row);
  assert.deepEqual(parsed.config, before);
  assert.equal(parsed.gpuIds, null);
  assert.equal('dtype' in parsed.config.config.process[0].caption, false);
});

test('malformed caption source metadata and JSON never become a default editable job', () => {
  const valid = source(fixture());
  for (const value of [null, [], {}, { ...valid, name: null }, { ...valid, name: '' },
    { ...valid, gpu_ids: undefined }, { ...valid, gpu_ids: 0 }, { ...valid, gpu_ids: [] },
    { ...valid, job_type: 'train' }, { ...valid, job_config: {} }, { ...valid, job_config: '{private-invalid' }]) {
    assert.throws(() => parseCaptionJobSource(value), /source/i);
  }
});

test('malformed caption document topology is rejected before mounting the form', () => {
  for (const value of [null, [], {}, { config: {} }, { job: 'extension', config: [] },
    { job: 'extension', config: { name: 'name', process: [] } },
    { job: 'extension', config: { name: {}, process: [] } }]) {
    assert.throws(() => parseCaptionJobConfig(value), /invalid caption configuration/i);
  }
  for (const process of [null, [], {}, { type: null, caption: {} }, { type: '', caption: {} },
    { type: 'Captioner', caption: null }, { type: 'Captioner', caption: [] }]) {
    const config = fixture();
    config.config.process.push(process);
    assert.throws(() => parseCaptionJobConfig(config), /invalid caption configuration/i);
  }
});

test('caption model, target and extension values require their editor-safe mandatory shapes', () => {
  for (const [key, values] of [
    ['model_name_or_path', [undefined, null, 2, {}]], ['path_to_caption', [undefined, null, [], {}]],
    ['extensions', [undefined, null, 'png', [], ['png', {}], ['']]],
  ]) {
    for (const value of values) {
      const config = fixture();
      config.config.process[0].caption[key] = value;
      assert.throws(() => parseCaptionJobConfig(config), /invalid caption configuration/i, key);
    }
  }
});

test('caption strings, booleans and finite numeric options reject object and mistyped values', () => {
  const strings = ['model_name_or_path2', 'device', 'dtype', 'qtype', 'caption_prompt', 'caption_format',
    'fixed_caption', 'remote_worker_id', 'remote_ollama_worker_id', 'system_prompt', 'output_format',
    'source_caption_extension', 'convert_destination', 'caption_extension'];
  const booleans = ['quantize', 'low_vram', 'recaption', 'compile', 'thinking',
    'extract_vocals_before_transcribe', 'keep_timestamps', 'layer_offloading', 'delete_source_caption'];
  const numbers = ['max_res', 'max_new_tokens', 'batch_size', 'layer_offloading_percent', 'temperature'];
  for (const [fields, invalid] of [[strings, {}], [booleans, 'false'], [numbers, '3'], [numbers, Infinity]]) {
    for (const key of fields) {
      const config = fixture();
      config.config.process[0].caption[key] = invalid;
      assert.throws(() => parseCaptionJobConfig(config), /invalid caption configuration/i, key);
    }
  }
  for (const key of ['device', 'sqlite_db_path']) {
    const config = fixture();
    config.config.process[0][key] = {};
    assert.throws(() => parseCaptionJobConfig(config), /invalid caption configuration/i);
  }
});

test('caption LoRA lists reject crash-prone shapes and retain valid optional historical fields', () => {
  const config = fixture();
  for (const loras of [{}, 'adapter', [null], [{}], [{ path: {} }], [{ path: 'a', name: {} }],
    [{ path: 'a', strength: NaN }], [{ path: 'a', strength: '1' }]]) {
    config.config.process[0].caption.loras = loras;
    assert.throws(() => parseCaptionJobConfig(config), /invalid caption configuration/i);
  }
  for (const loras of [undefined, null, [], [{ path: 'a' }], [{ path: 'a', name: 'Adapter', strength: 0 }]]) {
    config.config.process[0].caption.loras = loras;
    assert.equal(parseCaptionJobConfig(config).config.process[0].caption.loras, loras);
  }
});

test('historical nullable caption options remain exact while unsafe null values are rejected', () => {
  const config = fixture();
  const caption = config.config.process[0].caption;
  const nullable = ['model_name_or_path2', 'caption_prompt', 'caption_format', 'fixed_caption', 'output_format',
    'source_caption_extension', 'convert_destination', 'caption_extension'];
  for (const key of nullable) caption[key] = null;
  const before = structuredClone(config);
  assert.deepEqual(parseCaptionJobSource(source(config)).config, before);
  for (const key of ['system_prompt', 'dtype', 'qtype', 'model_name_or_path', 'path_to_caption']) {
    const invalid = fixture();
    invalid.config.process[0].caption[key] = null;
    assert.throws(() => parseCaptionJobConfig(invalid), /invalid caption configuration/i, key);
  }
});

test('unserializable raw caption values fail without leaking document contents', () => {
  const config = fixture();
  config.private_prompt = 'fixture-private-text';
  config.recursive = config;
  assert.throws(() => parseCaptionJobConfig(config), error =>
    /invalid caption configuration/.test(error.message) && !error.message.includes(config.private_prompt));
  delete config.recursive;
  config.future_option = 1n;
  assert.throws(() => parseCaptionJobConfig(config), /invalid caption configuration/i);
});
