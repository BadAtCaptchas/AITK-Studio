import assert from 'node:assert/strict';
import test from 'node:test';
import { loadTrainingSource, trainingFixture } from './trainingMigrationFixture.mjs';

const { modelArchs } = loadTrainingSource('domain/modelOptions.ts');
const { handleModelArchChange } = loadTrainingSource('app/jobs/new/utils.ts');

function setter(job) {
  return (value, key) => {
    const parts = key.replace(/\[(\d+)\]/g, '.$1').split('.');
    let node = job;
    for (const part of parts.slice(0, -1)) node = node[part] ??= {};
    node[parts.at(-1)] = structuredClone(value);
  };
}

for (const [from, destination] of [
  ['flux_kontext', 'flux'],
  ['flux_kontext', 'flex1'],
  ['flux', 'flux_kontext'],
]) {
  test(`${destination} exposes its runtime-supported Low VRAM control`, () => {
    assert.ok(modelArchs.find(model => model.name === destination).additionalSections.includes('model.low_vram'));
  });
  test(`switching ${from} to ${destination} does not clear explicit supported low-memory settings`, () => {
    const job = trainingFixture();
    job.config.process[0].model.arch = from;
    job.config.process[0].model.low_vram = true;
    job.config.process[0].sample.keep_low_vram_for_samples = true;
    handleModelArchChange(from, destination, job, setter(job));
    assert.equal(job.config.process[0].model.arch, destination);
    assert.equal(job.config.process[0].model.low_vram, true);
    assert.equal(job.config.process[0].sample.keep_low_vram_for_samples, true);
  });
}

test('Low VRAM capabilities retain supported offloading and exclude Flex.2', () => {
  assert.equal(modelArchs.find(model => model.name === 'flex2').additionalSections.includes('model.low_vram'), false);
  for (const [name, expected] of [['flux', true], ['flex1', true], ['flux_kontext', true], ['flex2', false]]) {
    assert.equal(modelArchs.find(model => model.name === name).additionalSections.includes('model.layer_offloading'), expected);
  }
});
