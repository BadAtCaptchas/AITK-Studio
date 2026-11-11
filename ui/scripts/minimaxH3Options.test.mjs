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

test('MiniMax H3 preset exposes current checkpoints and distillation choices', () => {
  const options = readSource('src/domain/modelOptions.ts');
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
  assert.match(block, /minimax_h3_training_adapter_v3\.safetensors/);
  assert.match(block, /train\.do_guidance_loss': \[false/);
  assert.match(block, /customModelSelectOptions/);
  assert.match(block, /Contrastive guidance/);
  assert.match(block, /Training adapter \(default\)/);
  assert.match(block, /disableSections: \[[^\]]*'model\.quantize'[^\]]*'model\.quantize_te'/);
  const additionalSections = block.slice(block.indexOf('additionalSections:'), block.indexOf('modelNotes:'));
  assert.match(block, /allowedNetworkTypes: \['lora'\]/);
});

test('H3 notes and mixed-media control use typed plain option data', () => {
  const options = readSource('src/domain/modelOptions.ts');
  const simpleJob = readSource('src/app/jobs/new/SimpleJob.tsx');
  const trainingForm = readSource('src/utils/trainingValidation.ts');
  const types = readSource('src/types.ts');
  assert.match(options, /export interface ModelNotes/);
  assert.match(options, /modelNotes\?: ModelNotes/);
  assert.match(simpleJob, /modelArch\?\.modelNotes/);
  assert.match(simpleJob, /Include images with videos/);
  assert.match(simpleJob, /!processConfig\.model\.arch\.startsWith\('minimax_h3'\)/);
  assert.match(trainingForm, /MiniMax H3 cannot merge a Base LoRA/);
  assert.match(types, /include_images_in_video_dataset: boolean/);
  assert.match(types, /guidance_loss_schedule: 'constant' \| 'sigma'/);
});

test('Ref2VA and LTX 2.5 are registered with current controls', () => {
  const options = readSource('src/domain/modelOptions.ts');
  const ref2va = archBlock(options, 'minimax_h3_ref2va');
  assert.match(ref2va, /minimax_h3_ref2va_training_adapter_v2\.safetensors/);
  assert.match(ref2va, /D-OPSD self-distillation/);
  assert.match(ref2va, /image_refs_as_video/);
  assert.match(ref2va, /image_ref_video_frames/);
  assert.match(options, /name: 'ltx2\.5'/);
  assert.match(options, /Lightricks\/LTX-2\.5/);
});

test('model catalog lists current MiniMax H3 support', () => {
  const readme = readSource('../docs/supported-models.md');
  assert.match(readme, /Comfy-Org\/MiniMax-H3/);
  assert.match(readme, /ConvRot\/NVFP4 prequantized components/);
  assert.match(readme, /approximately 43 GB/);
  assert.match(readme, /Ref2VA/);
  assert.match(readme, /17n\+5/);
});

test('P-H3 image uses Studio presets and distinct ConvRot8 text encoder defaults', () => {
  const block = archBlock(readSource('src/domain/modelOptions.ts'), 'p_h3_image');
  assert.match(block, /model\.qtype_te': \['convrot8'/);
  assert.match(block, /p_h3_image_training_adapter_v1/);
  assert.match(block, /sample\.sample_steps': \[8/);
  assert.match(block, /sample\.num_frames': \[1/);
  assert.match(block, /model\.layer_offloading': \[false/);
  const contract = JSON.parse(readSource('src/domain/modelCapabilities.json'));
  assert.equal(contract.models.find(row => row.arch === 'p_h3_image').className, 'PH3ImageModel');
  assert.equal(contract.choices.find(row => row.name === 'p_h3_image').outputType, 'image');
});

test('MiniMax H3 Image uses its six-step image schedule and safe Studio defaults', () => {
  const block = archBlock(readSource('src/domain/modelOptions.ts'), 'minimax_h3_image');
  assert.match(block, /Comfy-Org\/MiniMax-H3-Image/);
  assert.match(block, /model\.qtype': \['convrot8'/);
  assert.match(block, /model\.qtype_te': \['nvfp4'/);
  assert.match(block, /minimax_h3_image_training_adapter_v1/);
  assert.match(block, /sample\.sample_steps': \[6/);
  assert.match(block, /sample\.guidance_scale': \[1/);
  assert.match(block, /sample\.width': \[2048/);
  assert.match(block, /sample\.height': \[2048/);
  assert.match(block, /sample\.num_frames': \[1, 1\]/);
  assert.match(block, /datasets\[x\]\.num_frames': \[1, 1\]/);
  for (const setting of ['do_audio', 'auto_frame_count', 'do_i2v']) {
    assert.ok(block.includes(`datasets[x].${setting}': [false, undefined]`));
  }
  assert.match(block, /model\.layer_offloading': \[false, false\]/);
  assert.match(block, /model\.low_vram': \[true, false\]/);
  assert.match(block, /train\.train_text_encoder': \[false, false\]/);
  assert.match(block, /disableSections: \['network.conv', 'model.quantize', 'model.quantize_te', 'train.timestep_type', 'train.train_text_encoder', 'train.unload_text_encoder'\]/);
  assert.match(block, /allowedNetworkTypes: \['lora'\]/);

  const contract = JSON.parse(readSource('src/domain/modelCapabilities.json'));
  const models = contract.models.filter(row => row.arch === 'minimax_h3_image');
  assert.equal(models.length, 1);
  assert.equal(models[0].className, 'MinimaxH3ImageModel');
  assert.equal(models[0].module, 'extensions_built_in.diffusion_models.minimax_h3');
  const choices = contract.choices.filter(row => row.name === 'minimax_h3_image');
  assert.equal(choices.length, 1);
  const choice = choices[0];
  assert.equal(choice.outputType, 'image');
  assert.equal(choice.isVideoModel, false);
  assert.equal(choice.frames.maximum, 1);
  assert.equal(choice.offloading.layers, false);
  assert.ok(!choice.additionalSections.includes('model.layer_offloading'));
  assert.deepEqual(choice.allowedNetworkTypes, ['lora']);
  assert.deepEqual(choice.disableSections, ['network.conv', 'model.quantize', 'model.quantize_te', 'train.timestep_type', 'train.train_text_encoder', 'train.unload_text_encoder']);
});
