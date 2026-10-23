import assert from 'node:assert/strict';
import test from 'node:test';
import { loadTrainingSource, trainingFixture } from './trainingMigrationFixture.mjs';

const { configContractErrors } = loadTrainingSource('domain/configContract.ts');
const { migrateJobConfig } = loadTrainingSource('app/jobs/new/jobConfig.ts');
const { modelArchs } = loadTrainingSource('domain/modelOptions.ts');

test('Flux.1 Orbit offloading defaults are accepted and have corresponding form controls', () => {
  const config = trainingFixture();
  Object.assign(config.config.process[0].model, { low_vram: true, quantize: true, qtype: 'orbit4' });
  const applied = migrateJobConfig(config);
  assert.equal(applied.config.process[0].model.layer_offloading, true);
  assert.equal(applied.config.process[0].train.batch_size, 3);
  assert.deepEqual(configContractErrors(applied), []);
  assert.ok(modelArchs.find(model => model.name === 'flux').additionalSections.includes('model.layer_offloading'));
});
