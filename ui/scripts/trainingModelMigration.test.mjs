import assert from 'node:assert/strict';
import test from 'node:test';
import { migration, trainingFixture, loadTrainingSource } from './trainingMigrationFixture.mjs';

const {
  migrateTrainingModel,
  prepareFreshTrainingConfig,
  validateTrainingMigration,
  getMigrationTargets,
  migrationSourceError,
} = migration;
const { modelArchs } = loadTrainingSource('domain/modelOptions.ts');

test('Flux to Z-Image preserves tuned training, datasets, prompts and phases without mutating the source', () => {
  const source = trainingFixture();
  const before = structuredClone(source);
  const { config, changes, issues } = migrateTrainingModel(source, 'zimage');
  const from = source.config.process[0],
    to = config.config.process[0];
  for (const key of [
    'steps',
    'lr',
    'optimizer',
    'optimizer_params',
    'batch_size',
    'gradient_accumulation',
    'phases',
    'timestep_type',
  ])
    assert.deepEqual(to.train[key], from.train[key], key);
  for (const key of ['linear', 'linear_alpha']) assert.equal(to.network[key], from.network[key]);
  for (const key of ['folder_path', 'resolution', 'caption_ext', 'caption_dropout_rate', 'num_repeats'])
    assert.deepEqual(to.datasets[0][key], from.datasets[0][key]);
  assert.deepEqual(to.sample.samples, from.sample.samples);
  assert.deepEqual(to.save, from.save);
  assert.deepEqual(to.logging, from.logging);
  assert.equal(to.model.name_or_path, 'Tongyi-MAI/Z-Image');
  assert.ok(changes.some(change => change.path.endsWith('model.name_or_path')));
  assert.deepEqual(issues, []);
  assert.deepEqual(source, before);
});

test('does not apply recommended rank, learning rate, resolution or sample-step defaults', () => {
  const source = trainingFixture();
  const migrated = migrateTrainingModel(source, 'krea2:kroma_turbo_opd').config.config.process[0];
  assert.equal(migrated.train.lr, 0.00037);
  assert.equal(migrated.network.linear, 64);
  assert.deepEqual(migrated.datasets[0].resolution, [640, 960]);
  assert.equal(migrated.sample.sample_steps, source.config.process[0].sample.sample_steps);
});

test('removes source weights, explicit step offsets, process output names and hidden model arguments', () => {
  const source = trainingFixture(),
    p = source.config.process[0];
  p.name = 'original-output';
  p.train.start_step = 1200;
  p.network.pretrained_lora_path = '/original/checkpoint.safetensors';
  p.network.network_weights = '/legacy/checkpoint.safetensors';
  p.model.base_lora_path = '/base.safetensors';
  p.model.inference_lora_path = '/sample.safetensors';
  p.model.model_kwargs = { checkpoint_filename: 'source.safetensors' };
  p.model.latent_space_version = 'source-cache';
  p.network.network_kwargs.only_if_contains = ['source.layers'];
  const result = migrateTrainingModel(source, 'zimage:turbo').config.config.process[0];
  for (const key of ['pretrained_lora_path', 'network_weights']) assert.equal(result.network[key], undefined);
  assert.equal(result.name, undefined);
  assert.equal(result.train.start_step, undefined);
  assert.equal(result.model.base_lora_path, undefined);
  assert.equal(result.model.inference_lora_path, undefined);
  assert.equal(result.model.latent_space_version, undefined);
  assert.equal(result.model.model_kwargs.checkpoint_filename, undefined);
  assert.match(result.model.assistant_lora_path, /zimage_turbo_training_adapter/);
  assert.deepEqual(result.network.network_kwargs.only_if_contains, []);
  const again = migrateTrainingModel({ ...source, config: { ...source.config, process: [result] } }, 'flux').config
    .config.process[0];
  assert.equal(again.model.assistant_lora_path, undefined);
  assert.equal(again.train.lr, p.train.lr);
  assert.equal(again.network.linear, p.network.linear);
});

test('same-architecture path changes preserve model options but start without pretrained weights', () => {
  const source = trainingFixture();
  source.config.process[0].model.model_kwargs = { custom_option: 7 };
  source.config.process[0].network.pretrained_lora_path = '/weights.safetensors';
  const result = migrateTrainingModel(source, 'flux', 'other/model').config;
  assert.deepEqual(result.config.process[0].model.model_kwargs, { custom_option: 7 });
  assert.equal(result.config.process[0].network.pretrained_lora_path, undefined);
  assert.deepEqual(validateTrainingMigration(result, source), []);
  assert.ok(validateTrainingMigration(source, source).some(issue => issue.message.includes('another model')));
});

test('fresh-start sanitization is idempotent and reapplies after raw edits', () => {
  const source = trainingFixture();
  const first = prepareFreshTrainingConfig(source);
  first.config.config.process[0].train.start_step = 500;
  first.config.config.process[0].network.network_weights = 'old.safetensors';
  const second = prepareFreshTrainingConfig(first.config);
  assert.equal(second.config.config.process[0].train.start_step, undefined);
  assert.equal(second.config.config.process[0].network.network_weights, undefined);
  assert.deepEqual(prepareFreshTrainingConfig(second.config), { config: second.config, changes: [] });
});

test('applies supported network and fixed conditioning requirements, including phase overrides', () => {
  const source = trainingFixture();
  source.config.process[0].network.type = 'lokr';
  source.config.process[0].train.phases[0].timestep_type = 'sigmoid';
  const result = migrateTrainingModel(source, 'ming_image_design');
  const p = result.config.config.process[0];
  assert.equal(p.network.type, 'lora');
  assert.equal(p.network.linear, 64);
  assert.equal(p.train.batch_size, 1);
  assert.equal(p.model.quantize_te, false);
  assert.equal(p.train.timestep_type, 'linear');
  assert.equal(p.train.phases[0].timestep_type, 'linear');
  assert.equal(p.train.phases[0].steps, 200);
});

test('control migrations reuse references and block missing inputs without replacing datasets', () => {
  const source = trainingFixture();
  const missing = migrateTrainingModel(source, 'qwen_image_edit');
  assert.ok(missing.issues.some(issue => issue.target.step === 'job-dataset'));
  assert.ok(missing.issues.some(issue => issue.target.step === 'job-samples'));
  source.config.process[0].datasets[0].control_path = 'remote://worker/control';
  source.config.process[0].sample.samples[0].ctrl_img = '/reference.png';
  const result = migrateTrainingModel(source, 'qwen_image_edit').config;
  assert.equal(result.config.process[0].datasets[0].folder_path, source.config.process[0].datasets[0].folder_path);
  assert.equal(result.config.process[0].datasets[0].control_path, 'remote://worker/control');
  assert.deepEqual(validateTrainingMigration(result, source), []);
});

test('rejects cross-media, unknown, malformed, and multi-process sources', () => {
  const source = trainingFixture();
  assert.throws(() => migrateTrainingModel(source, 'yue2'), /same media/);
  assert.throws(() => migrateTrainingModel(source, 'wan22_5b'), /same media/);
  assert.throws(() => migrateTrainingModel(null, 'zimage'), /editable/);
  source.config.process.push(structuredClone(source.config.process[0]));
  assert.match(migrationSourceError(source), /one training process/);
  source.config.process.pop();
  source.config.process[0].model.arch = 'custom_unregistered';
  assert.match(migrationSourceError(source), /custom/);
});

test('single/multiple reference mappings do not leave duplicate source aliases', () => {
  const source = trainingFixture();
  source.config.process[0].datasets[0].control_path = '/control';
  source.config.process[0].sample.samples[0].ctrl_img = '/reference.png';
  const multi = migrateTrainingModel(source, 'qwen_image_edit_plus').config;
  assert.equal(multi.config.process[0].datasets[0].control_path, undefined);
  assert.equal(multi.config.process[0].datasets[0].control_path_1, '/control');
  assert.equal(multi.config.process[0].sample.samples[0].ctrl_img, undefined);
  assert.equal(multi.config.process[0].sample.samples[0].ctrl_img_1, '/reference.png');
  const single = migrateTrainingModel(multi, 'qwen_image_edit').config;
  assert.equal(single.config.process[0].datasets[0].control_path, '/control');
  assert.equal(single.config.process[0].datasets[0].control_path_1, undefined);
  assert.equal(single.config.process[0].sample.samples[0].ctrl_img, '/reference.png');
  assert.equal(single.config.process[0].sample.samples[0].ctrl_img_1, undefined);
});

test('layered datasets cannot silently migrate to an ordinary image model', () => {
  const source = trainingFixture();
  source.config.process[0].model.arch = 'ming_image_design_layer';
  source.config.process[0].datasets[0].type = 'layered_image';
  const result = migrateTrainingModel(source, 'flux');
  assert.ok(result.issues.some(issue => issue.message.includes('layered images')));
  assert.equal(result.config.config.process[0].datasets[0].type, 'layered_image');
});

test('optional reference support does not require controls for text-to-image or text-to-video migration', () => {
  for (const name of ['flux2', 'flux2_klein_4b', 'qwen_image_2', 'ming_image_design']) {
    const result = migrateTrainingModel(trainingFixture(), name);
    assert.equal(
      result.issues.some(
        issue => issue.message.includes('needs a control') || issue.message.includes('needs a reference'),
      ),
      false,
      name,
    );
  }
  const video = trainingFixture();
  video.config.process[0].model.arch = 'wan22_5b';
  assert.deepEqual(migrateTrainingModel(video, 'ltx2').issues, []);
});

test('video and audio migration preserve shared settings and sample content', () => {
  const video = trainingFixture();
  video.config.process[0].model.arch = 'wan22_5b';
  video.config.process[0].sample.num_frames = 42;
  const migratedVideo = migrateTrainingModel(video, 'ltx2').config.config.process[0];
  assert.equal(migratedVideo.train.lr, 0.00037);
  assert.equal(migratedVideo.sample.num_frames, 41);
  assert.deepEqual(migratedVideo.sample.samples, video.config.process[0].sample.samples);
  const audio = trainingFixture();
  const audioArch = modelArchs.find(arch => arch.group === 'audio' && arch.name !== 'yue2');
  assert.ok(audioArch);
  audio.config.process[0].model.arch = audioArch.name;
  const migratedAudio = migrateTrainingModel(audio, 'yue2').config.config.process[0];
  assert.equal(migratedAudio.train.steps, 4200);
  assert.deepEqual(migratedAudio.sample.samples, audio.config.process[0].sample.samples);
  assert.equal(migratedAudio.model.qtype, 'convrot8');
  assert.equal(migratedAudio.datasets[0].caption_dropout_rate, 0);
  assert.equal(migratedAudio.datasets[0].resolution.length, 1);
});

test('every registered destination can be transformed without applying a blanket settings reset', () => {
  for (const target of modelArchs) {
    const source = trainingFixture();
    const peer = modelArchs.find(
      arch =>
        arch.name !== target.name &&
        getMigrationTargets({
          ...source,
          config: {
            ...source.config,
            process: [{ ...source.config.process[0], model: { ...source.config.process[0].model, arch: arch.name } }],
          },
        }).some(option => option.name === target.name),
    );
    if (!peer) continue;
    source.config.process[0].model.arch = peer.name;
    const result = migrateTrainingModel(source, target.name);
    assert.equal(result.config.config.process[0].train.lr, 0.00037, target.name);
    assert.equal(result.config.config.process[0].train.steps, 4200, target.name);
    assert.equal(result.config.config.process[0].datasets[0].folder_path, 'fixture/dataset', target.name);
  }
});
