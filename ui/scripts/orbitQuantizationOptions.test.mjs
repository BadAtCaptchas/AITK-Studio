import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

function readSource(relativePath) {
  return fs.readFileSync(path.join(process.cwd(), relativePath), 'utf8');
}

function exportedArrayBlock(source, exportName, nextExportName) {
  const start = source.indexOf(`export const ${exportName}`);
  assert.notEqual(start, -1, `missing ${exportName}`);
  const end = source.indexOf(`export const ${nextExportName}`, start);
  assert.notEqual(end, -1, `missing ${nextExportName}`);
  return source.slice(start, end);
}

test('stable Orbit and experimental ConvRot backends are exposed by the new-job selector', () => {
  const options = readSource('src/app/jobs/new/options.ts');
  const quantizationOptions = exportedArrayBlock(options, 'quantizationOptions', 'defaultQtype');

  assert.match(quantizationOptions, /\{ value: 'orbit4', label: 'AITK Orbit 4-bit' \}/);
  assert.doesNotMatch(quantizationOptions, /orbit(?:2|3|vq|_vq)/i);
  assert.match(quantizationOptions, /\{ value: 'convrot4', label: 'ConvRot NVFP4 W4A4 \(experimental, Blackwell\)' \}/);
  assert.match(quantizationOptions, /\{ value: 'convrot8', label: 'ConvRot W8A8 \(experimental, Ampere\+\)' \}/);
  for (const bits of [2, 3, 4, 5, 6, 7]) {
    assert.match(quantizationOptions, new RegExp(`value: 'convrotint${bits}'`));
  }
  assert.match(quantizationOptions, /value: 'convrotbitnet'/);
  assert.doesNotMatch(quantizationOptions, /convrotcomfyw4a4/);
  assert.match(quantizationOptions, /\{ value: 'qfloat8', label: 'float8 \(default\)' \}/);
});

test('captioner exposes general ConvRot choices but not the specialized Comfy export backend', () => {
  const options = readSource('src/helpers/captionOptions.ts');
  const quantizationOptions = exportedArrayBlock(options, 'quantizationOptions', 'maxResOptions');

  assert.match(quantizationOptions, /value: 'convrot4'/);
  assert.match(quantizationOptions, /value: 'convrot8'/);
  assert.match(quantizationOptions, /value: 'convrotint2'/);
  assert.match(quantizationOptions, /value: 'convrotbitnet'/);
  assert.doesNotMatch(quantizationOptions, /convrotcomfyw4a4/);
  assert.match(quantizationOptions, /value: 'float8', label: 'float8 \(default\)'/);
});

test('Orbit quantization kwargs and low-VRAM defaults are typed and additive', () => {
  const types = readSource('src/types.ts');
  const jobConfig = readSource('src/app/jobs/new/jobConfig.ts');
  const simpleJob = readSource('src/app/jobs/new/SimpleJob.tsx');

  assert.match(types, /kernel\?: 'auto' \| 'triton' \| 'torch'/);
  assert.match(types, /max_workspace_mb\?: number/);
  assert.match(types, /include\?: string\[\]/);
  assert.match(types, /exclude\?: string\[\]/);

  assert.match(jobConfig, /nextModel\.layer_offloading \?\?= true/);
  assert.match(jobConfig, /nextModel\.layer_offloading_backend \?\?= 'block'/);
  assert.match(jobConfig, /nextModel\.layer_offloading_transformer_percent \?\?= 0\.7/);
  assert.match(jobConfig, /nextModel\.layer_offloading_text_encoder_percent \?\?= 0\.5/);
  assert.match(jobConfig, /batch_size: 1/);
  assert.match(jobConfig, /gradient_checkpointing: true/);
  assert.match(jobConfig, /optimizer: 'adamw8bit'/);
  assert.match(jobConfig, /cache_text_embeddings: true/);
  assert.match(jobConfig, /unload_text_encoder: true/);
  assert.match(jobConfig, /cache_latents_to_disk: true/);
  assert.match(simpleJob, /orbitProfileAppliedRef\.current/);
  assert.match(simpleJob, /applyOrbit4LowVramDefaults\(current, true\)/);
});
