import assert from 'node:assert/strict';
import test from 'node:test';
import { loadTrainingSource, trainingFixture, migration } from './trainingMigrationFixture.mjs';

const { modelArchs } = loadTrainingSource('domain/modelOptions.ts');
const { configContractErrors } = loadTrainingSource('domain/configContract.ts');
const { getLayerOffloadingMemoryProfile, supportsBlockLayerOffloading } = loadTrainingSource('utils/memoryProfiles.ts');
const { migrateJobConfig } = loadTrainingSource('app/jobs/new/jobConfig.ts');
const { handleModelArchChange } = loadTrainingSource('app/jobs/new/utils.ts');

const explicitOffload = {
  layer_offloading: true,
  layer_offloading_backend: 'legacy',
  layer_offloading_transformer_percent: 0.64,
  layer_offloading_text_encoder_percent: 0.42,
};

function setter(job) {
  return (value, key) => {
    const parts = key.replace(/\[(\d+)\]/g, '.$1').split('.');
    let node = job;
    for (const part of parts.slice(0, -1)) node = node[part] ??= {};
    node[parts.at(-1)] = structuredClone(value);
  };
}

test('Flex.1 resolves the Flux runtime block profile, including suffixed architectures', () => {
  for (const name of ['flex1', 'flex1:variant']) {
    assert.equal(supportsBlockLayerOffloading(name), true);
    assert.deepEqual(getLayerOffloadingMemoryProfile(name), getLayerOffloadingMemoryProfile('flux'));
  }
});

test('Flex.1 exposes supported offloading and accepts explicit configuration', () => {
  const config = trainingFixture();
  Object.assign(config.config.process[0].model, { arch: 'flex1', ...explicitOffload });
  assert.deepEqual(configContractErrors(config), []);
  assert.ok(modelArchs.find(model => model.name === 'flex1').additionalSections.includes('model.layer_offloading'));
});

test('Flex.1 Orbit defaults match its runtime without overwriting training settings or the source', () => {
  const source = trainingFixture();
  Object.assign(source.config.process[0].model, { arch: 'flex1', low_vram: true, quantize: true, qtype: 'orbit4' });
  const applied = migrateJobConfig(structuredClone(source));
  const model = applied.config.process[0].model;
  assert.equal(model.arch, 'flex1');
  assert.equal(model.layer_offloading, true);
  assert.equal(model.layer_offloading_backend, 'block');
  assert.equal(model.layer_offloading_transformer_percent, 0.7);
  assert.equal(model.layer_offloading_text_encoder_percent, 0.5);
  for (const [key, value] of Object.entries(source.config.process[0].train)) {
    assert.deepEqual(applied.config.process[0].train[key], value, key);
  }
  assert.equal(source.config.process[0].model.layer_offloading, undefined);
  assert.deepEqual(configContractErrors(applied), []);
});

test('Flex.1 additive Orbit migration retains explicit disabled offloading, backend and fractions', () => {
  const config = trainingFixture();
  Object.assign(config.config.process[0].model, {
    arch: 'flex1', low_vram: true, quantize: true, qtype: 'orbit4',
    ...explicitOffload, layer_offloading: false,
  });
  const migrated = migrateJobConfig(config).config.process[0].model;
  for (const [key, value] of Object.entries({ ...explicitOffload, layer_offloading: false })) {
    assert.equal(migrated[key], value, key);
  }
});

test('switching Flux to Flex.1 retains explicit supported offloading settings', () => {
  const config = trainingFixture();
  Object.assign(config.config.process[0].model, explicitOffload);
  handleModelArchChange('flux', 'flex1', config, setter(config));
  assert.equal(config.config.process[0].model.arch, 'flex1');
  for (const [key, value] of Object.entries(explicitOffload)) assert.equal(config.config.process[0].model[key], value, key);
});

test('switching to Flex.1 fills only missing memory settings using the supported profile', () => {
  const config = trainingFixture();
  handleModelArchChange('flux', 'flex1', config, setter(config));
  const model = config.config.process[0].model;
  assert.equal(model.layer_offloading, false);
  assert.equal(model.layer_offloading_backend, 'block');
  assert.equal(model.layer_offloading_transformer_percent, 0.7);
  assert.equal(model.layer_offloading_text_encoder_percent, 0.5);
});

test('model migration retains Flex.1 supported offload fractions and leaves the source untouched', () => {
  const source = trainingFixture();
  Object.assign(source.config.process[0].model, explicitOffload, { layer_offloading_backend: 'block' });
  const before = structuredClone(source);
  const { config, issues } = migration.migrateTrainingModel(source, 'flex1');
  assert.deepEqual(source, before);
  assert.deepEqual(issues, []);
  assert.equal(config.config.process[0].model.layer_offloading, true);
  assert.equal(config.config.process[0].model.layer_offloading_backend, 'block');
  assert.equal(config.config.process[0].model.layer_offloading_transformer_percent, 0.64);
  assert.equal(config.config.process[0].model.layer_offloading_text_encoder_percent, 0.42);
});

for (const [from, destination] of [['flux', 'flex1'], ['flex1', 'flux']]) {
  test(`model migration ${from} to ${destination} retains explicitly disabled Orbit offloading`, () => {
    const source = trainingFixture();
    Object.assign(source.config.process[0].model, explicitOffload, {
      arch: from, low_vram: true, quantize: true, qtype: 'orbit4', layer_offloading: false,
    });
    const before = structuredClone(source);
    const migrated = migration.migrateTrainingModel(source, destination);
    const result = migrateJobConfig(structuredClone(migrated.config));
    assert.deepEqual(migrated.issues, []);
    assert.deepEqual(source, before);
    for (const config of [migrated.config, result]) {
      assert.equal(config.config.process[0].model.arch, destination);
      for (const [key, value] of Object.entries({ ...explicitOffload, layer_offloading: false })) {
        assert.equal(config.config.process[0].model[key], value, key);
      }
    }
  });
  test(`model migration ${from} to ${destination} preserves the explicitly selected compatible legacy backend`, () => {
    const source = trainingFixture();
    Object.assign(source.config.process[0].model, explicitOffload, { arch: from });
    const result = migration.migrateTrainingModel(source, destination);
    assert.deepEqual(result.issues, []);
    for (const [key, value] of Object.entries(explicitOffload)) {
      assert.equal(result.config.config.process[0].model[key], value, key);
    }
  });
}

test('model migration outside the shared Flux runtime still chooses the destination offloading backend', () => {
  const source = trainingFixture();
  Object.assign(source.config.process[0].model, explicitOffload);
  const result = migration.migrateTrainingModel(source, 'flux2');
  assert.deepEqual(result.issues, []);
  assert.equal(result.config.config.process[0].model.layer_offloading, true);
  assert.equal(result.config.config.process[0].model.layer_offloading_backend, 'block');
  assert.equal(result.config.config.process[0].model.layer_offloading_transformer_percent, 0.64);
  assert.equal(result.config.config.process[0].model.layer_offloading_text_encoder_percent, 0.42);
});

test('Flex.2 remains outside the verified block offloading capability', () => {
  assert.equal(supportsBlockLayerOffloading('flex2'), false);
  assert.equal(modelArchs.find(model => model.name === 'flex2').additionalSections.includes('model.layer_offloading'), false);
});
