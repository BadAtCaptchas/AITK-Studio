import assert from 'node:assert/strict';
import test from 'node:test';
import { loadTrainingSource, trainingFixture } from './trainingMigrationFixture.mjs';

const { parseTrainingSource } = loadTrainingSource('utils/trainingSource.ts');
const source = config => ({ job_type: 'train', job_config: JSON.stringify(config), gpu_ids: '2,3', worker_id: 'remote-worker' });

test('source loading preserves advanced configuration and remote targeting for edits', () => {
  const config = trainingFixture();
  config.config.process[0].model.model_kwargs = { custom_training_option: { schedule: [1, 3, 8] } };
  const row = source(config);
  const loaded = parseTrainingSource(row, 'edit');
  assert.equal(loaded.config.config.name, 'source_run');
  assert.deepEqual(loaded.config.config.process[0].train.optimizer_params, config.config.process[0].train.optimizer_params);
  assert.deepEqual(loaded.config.config.process[0].model.model_kwargs, config.config.process[0].model.model_kwargs);
  assert.equal(loaded.workerID, 'remote-worker');
  assert.equal(loaded.gpuIDs, '2,3');
  assert.equal(row.job_config, JSON.stringify(config));
});

test('cloning renames an independent parsed config and retains legacy migrations', () => {
  const config = trainingFixture();
  config.config.process[0].type = 'ui_trainer';
  config.config.process[0].sample.prompts = ['legacy prompt'];
  delete config.config.process[0].sample.samples;
  const loaded = parseTrainingSource(source(config), 'clone');
  assert.equal(loaded.config.config.name, 'source_run_copy');
  assert.equal(loaded.config.config.process[0].type, 'diffusion_trainer');
  assert.deepEqual(loaded.config.config.process[0].sample.samples, [{ prompt: 'legacy prompt' }]);
  assert.equal(config.config.name, 'source_run');
  assert.equal(config.config.process[0].type, 'ui_trainer');
});

test('malformed and non-training source responses never become an editable default job', () => {
  for (const invalid of [null, {}, { ...source(trainingFixture()), job_type: 'caption' },
    { ...source(trainingFixture()), job_config: '{invalid' }, { ...source(trainingFixture()), job_config: '{"config":{}}' }]) {
    assert.throws(() => parseTrainingSource(invalid, 'edit'), /source/i);
  }
});

test('malformed legacy prompt arrays are rejected before migration can produce unsafe sample values', () => {
  for (const prompts of [[42], [{ text: 'not a prompt string' }], 'not an array']) {
    const config = trainingFixture();
    delete config.config.process[0].sample.samples;
    config.config.process[0].sample.prompts = prompts;
    assert.throws(() => parseTrainingSource(source(config), 'edit'), /invalid training configuration/i);
  }
});
