import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { analyzeTrainingAdvisor } from '../dist/src/server/trainingAdvisor.js';

function makeDataset(files) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-advisor-'));
  for (const [relativePath, content] of Object.entries(files)) {
    const filePath = path.join(root, relativePath);
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, content);
  }
  return root;
}

function baseConfig(datasetPath) {
  return {
    job: 'extension',
    config: {
      name: 'advisor_test',
      process: [
        {
          type: 'diffusion_trainer',
          training_folder: 'output',
          device: 'cuda',
          performance_log_every: 10,
          network: {
            type: 'lora',
            linear: 32,
            linear_alpha: 32,
            conv: 16,
            conv_alpha: 16,
            lokr_full_rank: false,
            lokr_factor: -1,
            lokr_use_tucker: false,
            lokr_use_scalar: false,
            lokr_decompose_both: false,
            lokr_rank_dropout_scale: false,
            lokr_weight_decompose: false,
            lokr_wd_on_output: true,
            lokr_full_matrix: false,
            lokr_bypass_mode: false,
            lokr_rs_lora: false,
            lokr_unbalanced_factorization: false,
            lokr_legacy_factorization: false,
            network_kwargs: { ignore_if_contains: [] },
          },
          save: {
            dtype: 'bf16',
            save_every: 100,
            max_step_saves_to_keep: 4,
            save_format: 'diffusers',
            push_to_hub: false,
          },
          datasets: [
            {
              folder_path: datasetPath,
              mask_path: null,
              mask_min_value: 0.1,
              default_caption: '',
              caption_ext: 'txt',
              caption_dropout_rate: 0.05,
              cache_latents_to_disk: false,
              is_reg: false,
              network_weight: 1,
              resolution: [512],
              controls: [],
              shrink_video_to_frames: true,
              num_frames: 1,
              flip_x: false,
              flip_y: false,
              num_repeats: 1,
            },
          ],
          train: {
            batch_size: 1,
            steps: 300,
            gradient_accumulation: 1,
            train_unet: true,
            train_text_encoder: false,
            gradient_checkpointing: true,
            noise_scheduler: 'flowmatch',
            optimizer: 'adamw8bit',
            timestep_type: 'sigmoid',
            content_or_style: 'balanced',
            optimizer_params: { weight_decay: 0.0001 },
            unload_text_encoder: false,
            cache_text_embeddings: false,
            lr: 0.0001,
            skip_first_sample: false,
            force_first_sample: false,
            disable_sampling: false,
            dtype: 'bf16',
            diff_output_preservation: false,
            diff_output_preservation_multiplier: 1,
            diff_output_preservation_class: 'person',
            switch_boundary_every: 1,
            loss_type: 'mse',
          },
          logging: {
            log_every: 1,
            use_ui_logger: true,
            monitor_every: 1,
          },
          model: {
            name_or_path: 'ostris/Flex.1-alpha',
            quantize: true,
            qtype: 'qfloat8',
            quantize_te: true,
            qtype_te: 'qfloat8',
            arch: 'flex1',
            low_vram: false,
            model_kwargs: {},
          },
          sample: {
            sampler: 'flowmatch',
            sample_every: 100,
            width: 512,
            height: 512,
            samples: [{ prompt: 'test subject' }],
            neg: '',
            seed: 1,
            walk_seed: true,
            guidance_scale: 4,
            sample_steps: 20,
            num_frames: 1,
            fps: 1,
          },
        },
      ],
    },
    meta: { name: '[name]', version: '1.0' },
  };
}

function findingIds(result) {
  return new Set(result.findings.map(finding => finding.id));
}

test('preflight reports missing and empty captions', () => {
  const dataset = makeDataset({
    'one.jpg': 'image',
    'one.txt': '',
    'two.png': 'image',
  });

  const result = analyzeTrainingAdvisor(baseConfig(dataset), { scanFileLimit: 20 });
  const ids = findingIds(result);

  assert.ok(ids.has('dataset.0.captions.empty'));
  assert.ok(ids.has('dataset.0.captions.some_missing'));
  assert.equal(result.datasetStats?.mediaFiles, 2);
  assert.equal(result.datasetStats?.emptyCaptions, 1);
});

test('preflight reports dominant alternate caption extension', () => {
  const files = {};
  for (let index = 0; index < 10; index += 1) {
    files[`image_${index}.jpg`] = 'image';
    files[`image_${index}.sdxl`] = 'sdxl caption';
  }
  const dataset = makeDataset(files);

  const result = analyzeTrainingAdvisor(baseConfig(dataset), { scanFileLimit: 20 });
  const ids = findingIds(result);

  assert.ok(ids.has('dataset.0.caption_ext_mismatch'));
  assert.equal(result.datasetStats?.captionExtensionMismatches, 10);
});

test('preflight ignores mixed sidecar captions when configured extension has reasonable coverage', () => {
  const files = {};
  for (let index = 0; index < 10; index += 1) {
    files[`image_${index}.jpg`] = 'image';
    files[`image_${index}.json`] = 'json caption';
    files[`image_${index}.sdxl`] = 'sdxl caption';
  }
  for (let index = 0; index < 6; index += 1) {
    files[`image_${index}.txt`] = 'txt caption';
  }
  const dataset = makeDataset(files);

  const result = analyzeTrainingAdvisor(baseConfig(dataset), { scanFileLimit: 20 });
  const ids = findingIds(result);

  assert.ok(!ids.has('dataset.0.caption_ext_mismatch'));
  assert.equal(result.datasetStats?.mediaFiles, 10);
  assert.equal(result.datasetStats?.captionFiles, 6);
});

test('preflight reports placeholder and inaccessible datasets', () => {
  const placeholderResult = analyzeTrainingAdvisor(baseConfig('/path/to/images/folder'));
  const inaccessibleResult = analyzeTrainingAdvisor(baseConfig(path.join(os.tmpdir(), 'missing-advisor-dataset')));

  assert.ok(findingIds(placeholderResult).has('dataset.0.placeholder'));
  assert.ok(findingIds(inaccessibleResult).has('dataset.0.inaccessible'));
});

test('preflight reports phase step mismatches and risky learning rates', () => {
  const dataset = makeDataset({ 'one.jpg': 'image', 'one.txt': 'caption' });
  const config = baseConfig(dataset);
  const processConfig = config.config.process[0];
  processConfig.train.lr = 0.002;
  processConfig.train.phases = [
    { name: 'Warmup', steps: 100, lr: 0.0001 },
    { name: 'Detail', steps: 100, lr: 0.00005 },
  ];

  const result = analyzeTrainingAdvisor(config);
  const ids = findingIds(result);

  assert.ok(ids.has('train.lr.critical'));
  assert.ok(ids.has('phases.steps.mismatch'));
});

test('preflight skips generic learning rate warning for Prodigy optimizers', () => {
  const dataset = makeDataset({ 'one.jpg': 'image', 'one.txt': 'caption' });
  const config = baseConfig(dataset);
  const processConfig = config.config.process[0];
  processConfig.train.optimizer = 'prodigyopt';
  processConfig.train.lr = 1;

  const result = analyzeTrainingAdvisor(config);
  const ids = findingIds(result);

  assert.ok(!ids.has('train.lr.warning'));
  assert.ok(!ids.has('train.lr.critical'));
});

test('preflight keeps generic learning rate warning for non-Prodigy optimizers', () => {
  const dataset = makeDataset({ 'one.jpg': 'image', 'one.txt': 'caption' });
  const config = baseConfig(dataset);
  const processConfig = config.config.process[0];
  processConfig.train.optimizer = 'adamw8bit';
  processConfig.train.lr = 0.002;

  const result = analyzeTrainingAdvisor(config);
  const ids = findingIds(result);

  assert.ok(ids.has('train.lr.critical'));
});

test('preflight accepts auto training phases without fixed steps', () => {
  const dataset = makeDataset({ 'one.jpg': 'image', 'one.txt': 'caption' });
  const config = baseConfig(dataset);
  const processConfig = config.config.process[0];
  processConfig.train.auto_train = true;
  processConfig.train.phases = [
    { name: 'Shape', lr: 0.00002, auto_advance: { type: 'loss_plateau' } },
    { name: 'Detail', lr: 0.000005, auto_advance: { type: 'loss_plateau' } },
  ];

  const result = analyzeTrainingAdvisor(config);
  const ids = findingIds(result);

  assert.ok(!ids.has('phases.steps.mismatch'));
  assert.equal(result.findings.length, 0);
});

test('preflight reports invalid SEGA distillation configuration', () => {
  const dataset = makeDataset({ 'one.png': 'fake', 'one.txt': 'caption' });
  const config = baseConfig(dataset);
  const processConfig = config.config.process[0];
  processConfig.train.sega_distill = true;
  processConfig.train.diff_output_preservation = true;
  processConfig.train.do_prior_divergence = true;
  processConfig.train.inverted_mask_prior = true;
  processConfig.model.arch = 'flex1';
  processConfig.network.type = 'lokr';

  const result = analyzeTrainingAdvisor(config, { scanFileLimit: 20 });
  const ids = findingIds(result);

  assert.ok(ids.has('train.sega_distill.unsupported_arch'));
  assert.ok(ids.has('train.sega_distill.non_lora'));
  assert.ok(ids.has('train.sega_distill.conflicts'));
  assert.ok(ids.has('train.sega_distill.teacher_cost'));
});

test('preflight accepts compatible SEGA distillation configuration with teacher cost note', () => {
  const dataset = makeDataset({ 'one.png': 'fake', 'one.txt': 'caption' });
  const config = baseConfig(dataset);
  const processConfig = config.config.process[0];
  processConfig.train.sega_distill = true;
  processConfig.model.arch = 'zimage:turbo';
  processConfig.network.type = 'lora';

  const result = analyzeTrainingAdvisor(config, { scanFileLimit: 20 });
  const ids = findingIds(result);

  assert.ok(!ids.has('train.sega_distill.unsupported_arch'));
  assert.ok(!ids.has('train.sega_distill.non_lora'));
  assert.ok(!ids.has('train.sega_distill.conflicts'));
  assert.ok(ids.has('train.sega_distill.teacher_cost'));
});

test('preflight reports auto training phases that cannot advance', () => {
  const dataset = makeDataset({ 'one.jpg': 'image', 'one.txt': 'caption' });
  const config = baseConfig(dataset);
  const processConfig = config.config.process[0];
  processConfig.train.auto_train = true;
  processConfig.train.phases = [{ name: 'Shape', lr: 0.00002 }];

  const result = analyzeTrainingAdvisor(config);
  const ids = findingIds(result);

  assert.ok(ids.has('phases.0.auto_train.no_auto_advance'));
});

test('live advisor reports OOM, memory pressure, and plateau signals', () => {
  const dataset = makeDataset({ 'one.jpg': 'image', 'one.txt': 'caption' });
  const config = baseConfig(dataset);
  config.config.process[0].sample.sample_every = 1000;
  config.config.process[0].save.save_every = 1000;
  const lossPoints = Array.from({ length: 140 }, (_, step) => ({
    step,
    wall_time: step,
    value: 1 - Math.min(step, 20) * 0.001,
  }));

  const result = analyzeTrainingAdvisor(config, {
    scanDatasets: false,
    job: { id: 'job1', step: 180, status: 'running', speed_string: '' },
    metrics: {
      series: {
        'loss/loss': { key: 'loss/loss', points: lossPoints, latest: lossPoints[lossPoints.length - 1] },
        'train/oom_skipped': { key: 'train/oom_skipped', points: [{ step: 20, value: 1 }], latest: { step: 20, value: 1 } },
        'train/gpu_mem_used_pct': {
          key: 'train/gpu_mem_used_pct',
          points: [{ step: 180, value: 96 }],
          latest: { step: 180, value: 96 },
        },
      },
    },
  });
  const ids = findingIds(result);

  assert.ok(ids.has('live.oom_skips'));
  assert.ok(ids.has('live.gpu_memory.high'));
  assert.ok(ids.has('live.loss.plateau.no_auto_advance'));
});

test('clean preflight has no findings', () => {
  const dataset = makeDataset({ 'one.jpg': 'image', 'one.txt': 'caption' });
  const result = analyzeTrainingAdvisor(baseConfig(dataset));

  assert.equal(result.findings.length, 0);
  assert.equal(result.summary.text, 'No training quality issues found');
});
