import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

function readSource(relativePath) {
  return fs.readFileSync(path.join(process.cwd(), relativePath), 'utf8');
}

function archBlock(source, name) {
  const start = source.indexOf(`name: '${name}'`);
  assert.notEqual(start, -1, `missing ${name} arch block`);
  const end = source.indexOf('\n  },', start);
  assert.notEqual(end, -1, `missing end of ${name} arch block`);
  return source.slice(start, end);
}

test('MiniMax H3 preset is experimental and pins its supported packed checkpoints', () => {
  const options = readSource('src/app/jobs/new/options.ts');
  const block = archBlock(options, 'minimax_h3');
  assert.match(block, /group: 'experimental'/);
  assert.match(block, /isVideoModel: true/);
  assert.match(block, /model\.qtype': \['convrot8'/);
  assert.match(block, /model\.qtype_te': \['nvfp4'/);
  assert.match(block, /model\.model_kwargs\.partition': \['fl2va_pruned'/);
  assert.match(block, /model\.model_kwargs\.max_text_length': \[512/);
  assert.match(block, /model\.layer_offloading': \[false, false\]/);
  assert.match(block, /model\.base_lora_path': \[undefined, undefined\]/);
  assert.match(block, /model\.inference_lora_path': \[undefined, undefined\]/);
  assert.match(block, /train\.train_text_encoder': \[false, false\]/);
  assert.match(block, /train\.unload_text_encoder': \[false, false\]/);
  assert.match(block, /network\.type': \['lora', 'lora'\]/);
  assert.match(block, /include_images_in_video_dataset': \[true, false\]/);
  assert.match(block, /minimax_h3_training_adapter_alpha\.safetensors/);
  assert.match(block, /disableSections: \[[^\]]*'model\.quantize'[^\]]*'model\.quantize_te'/);
  const additionalSections = block.slice(block.indexOf('additionalSections:'), block.indexOf('modelNotes:'));
  assert.doesNotMatch(additionalSections, /'model\.layer_offloading'/);
  assert.match(block, /allowedNetworkTypes: \['lora'\]/);
});

test('H3 notes and mixed-media control use typed plain option data', () => {
  const options = readSource('src/app/jobs/new/options.ts');
  const simpleJob = readSource('src/app/jobs/new/SimpleJob.tsx');
  const trainingForm = readSource('src/app/jobs/new/TrainingFormContent.tsx');
  const types = readSource('src/types.ts');
  assert.match(options, /export interface ModelNotes/);
  assert.match(options, /modelNotes\?: ModelNotes/);
  assert.match(simpleJob, /modelArch\?\.modelNotes/);
  assert.match(simpleJob, /Include images with videos/);
  assert.match(simpleJob, /processConfig\.model\.arch !== 'minimax_h3'/);
  assert.match(trainingForm, /MiniMax H3 cannot merge a Base LoRA/);
  assert.match(types, /include_images_in_video_dataset: boolean/);
  assert.match(types, /guidance_loss_schedule: 'constant' \| 'sigma'/);
});

test('README lists MiniMax H3 with its experimental support boundaries', () => {
  const readme = readSource('../README.md');
  assert.match(readme, /Comfy-Org\/MiniMax-H3/);
  assert.match(readme, /ConvRot8 DiT and NVFP4 text encoder/);
  assert.match(readme, /approximately 43 GB/);
  assert.match(readme, /`ref2va` I2V training remains disabled/);
});
