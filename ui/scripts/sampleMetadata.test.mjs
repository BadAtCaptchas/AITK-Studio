import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { loadTrainingSource } from './trainingMigrationFixture.mjs';

const { parseSampleMetadata, parseSampleMetadataMap, sampleIdentity, groupSampleRows, adjacentSample } = loadTrainingSource('utils/sampleMetadata.ts');
const { readSampleMetadata, deleteSampleMetadata } = loadTrainingSource('server/sampleMetadata.ts');
const metadata = (overrides = {}) => ({ version: 1, prompt: `original prompt ${overrides.prompt_index ?? 0}`, negative_prompt: '', seed: 97,
  width: 640, height: 512, guidance_scale: 4, guidance_rescale: 0, sample_steps: 20, network_multiplier: 1,
  num_frames: 1, fps: 15, step: 100, prompt_index: 0, sampler: 'flowmatch', iteration_id: 'first', ...overrides });

test('metadata boundaries reject malformed settings and ignore arbitrary untrusted fields', () => {
  assert.equal(parseSampleMetadata({ ...metadata(), api_token: 'secret' }).api_token, undefined);
  assert.equal(parseSampleMetadata(metadata({ negative_prompt: null })).negative_prompt, '');
  for (const value of [null, {}, metadata({ version: 2 }), metadata({ seed: -1 }), metadata({ guidance_scale: NaN }),
    metadata({ prompt_index: '0' }), metadata({ iteration_id: '../bad' })]) assert.equal(parseSampleMetadata(value), null);
  const remote = 'remote://mirror/img/encoded/sample.png';
  assert.deepEqual(Object.keys(parseSampleMetadataMap({ [remote]: metadata(), unrequested: metadata() }, [remote])), [remote]);
});

test('iteration grouping and navigation survive missing samples and changing prompt counts', () => {
  const samples = ['a.png', 'c.png', 'd.png', 'e.png', 'f.png'];
  const recorded = { 'a.png': metadata(), 'c.png': metadata({ prompt_index: 2 }),
    'd.png': metadata({ iteration_id: 'second', step: 200 }),
    'e.png': metadata({ iteration_id: 'second', step: 200, prompt_index: 1 }),
    'f.png': metadata({ iteration_id: 'third', step: 200, prompt_index: 2 }) };
  assert.deepEqual(groupSampleRows(samples, recorded), [['a.png', 'c.png'], ['d.png', 'e.png'], ['f.png']]);
  assert.equal(adjacentSample(samples, recorded, 'a.png', 'right'), 'c.png');
  assert.equal(adjacentSample(samples, recorded, 'c.png', 'right'), undefined);
  assert.equal(adjacentSample(samples, recorded, 'c.png', 'down'), 'f.png');
  assert.equal(adjacentSample(samples, recorded, 'd.png', 'up'), 'a.png');
});

test('legacy files use explicit filename identity without inventing steps from array positions', () => {
  const samples = ['1700000000000__000000100_0.png', '1700000000002__000000100_2.png', '1700000000003__000000200_0.png'];
  assert.deepEqual(groupSampleRows(samples, {}), [samples.slice(0, 2), samples.slice(2)]);
  assert.equal(sampleIdentity('unknown-name.png').step, null);
  assert.equal(sampleIdentity('unknown-name.png').promptIndex, null);
});

test('recorded prompt identity follows reordered prompts and refuses different controls or changed prompts', () => {
  const samples = ['cat.png', 'dog.png', 'inserted.png', 'dog-again.png', 'cat-again.png', 'changed.png'];
  const recorded = {
    'cat.png': metadata({ prompt: 'cat', prompt_index: 0 }),
    'dog.png': metadata({ prompt: 'dog', prompt_index: 1 }),
    'inserted.png': metadata({ iteration_id: 'second', prompt: 'bird', prompt_index: 0 }),
    'dog-again.png': metadata({ iteration_id: 'second', prompt: 'dog', prompt_index: 1 }),
    'cat-again.png': metadata({ iteration_id: 'second', prompt: 'cat', prompt_index: 2 }),
    'changed.png': metadata({ iteration_id: 'third', prompt: 'horse', prompt_index: 2 }),
  };
  assert.equal(adjacentSample(samples, recorded, 'cat.png', 'down'), 'cat-again.png');
  assert.equal(adjacentSample(samples, recorded, 'cat-again.png', 'up'), 'cat.png');
  assert.equal(adjacentSample(samples, recorded, 'cat-again.png', 'down'), undefined);
  recorded['cat.png'].prompt_identity = 'a'.repeat(64);
  recorded['cat-again.png'].prompt_identity = 'b'.repeat(64);
  assert.equal(adjacentSample(samples, recorded, 'cat.png', 'down'), undefined, 'different control identities must not align');
});

test('sidecar reads are bounded and malformed metadata degrades to legacy; deletion leaves media intact', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'aitk-sample-meta-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const media = path.join(root, 'sample.png');
  await fs.writeFile(media, 'fixture');
  await fs.writeFile(media + '.metadata.json', JSON.stringify(metadata()));
  assert.deepEqual(await readSampleMetadata(media), metadata());
  await fs.writeFile(media + '.metadata.json', '{bad');
  assert.equal(await readSampleMetadata(media), null);
  await fs.writeFile(media + '.metadata.json', ' '.repeat(512 * 1024 + 1));
  assert.equal(await readSampleMetadata(media), null);
  await deleteSampleMetadata(media);
  assert.equal(await fs.readFile(media, 'utf8'), 'fixture');
  await deleteSampleMetadata(media);
});
