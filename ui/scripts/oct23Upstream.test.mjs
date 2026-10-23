import assert from 'node:assert/strict';
import test from 'node:test';
import { migration, trainingFixture, loadTrainingSource } from './trainingMigrationFixture.mjs';
const { modelArchs } = loadTrainingSource('domain/modelOptions.ts');
const { validateTrainingConfig } = loadTrainingSource('utils/trainingValidation.ts');

test('Fibo 2.0 exposes optional references and its own runtime defaults', () => {
  const model = modelArchs.find(row => row.name === 'fibo2');
  assert.ok(model);
  assert.equal(model.defaults['config.process[0].model.name_or_path'][0], 'Comfy-Org/Fibo-2.0');
  assert.equal(model.defaults['config.process[0].sample.guidance_scale'][0], 5);
  assert.equal(model.defaults['config.process[0].sample.sample_steps'][0], 30);
  assert.ok(model.additionalSections.includes('datasets.multi_control_paths'));
  assert.ok(model.additionalSections.includes('sample.multi_ctrl_imgs'));
  assert.ok(model.disableSections.includes('train.train_text_encoder'));
});

test('Fibo migration keeps user data and tuning while enforcing frozen available conditioning', () => {
  const source = trainingFixture();
  source.config.process[0].train.train_text_encoder = true;
  source.config.process[0].train.unload_text_encoder = true;
  const original = structuredClone(source);
  const { config } = migration.migrateTrainingModel(source, 'fibo2');
  const p = config.config.process[0];
  assert.equal(p.train.train_text_encoder, false);
  assert.equal(p.train.unload_text_encoder, false);
  assert.equal(p.model.name_or_path, 'Comfy-Org/Fibo-2.0');
  assert.equal(p.train.lr, source.config.process[0].train.lr);
  assert.deepEqual(p.datasets, source.config.process[0].datasets);
  assert.deepEqual(source, original);
  p.train.train_text_encoder = true;
  const messages = validateTrainingConfig(config, { workerID: 'local', gpuIDs: '0', datasetOptions: [], unselectedDatasetPath: 'unset' });
  assert.ok(messages.some(message => message.level === 'error' && message.message.includes('Fibo 2.0')));
});
