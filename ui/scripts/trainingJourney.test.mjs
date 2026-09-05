import test from 'node:test';
import assert from 'node:assert/strict';
import { validateTrainingConfig, isEditableTrainingConfig } from '../dist/src/utils/trainingValidation.js';
import { nextHomeAction } from '../dist/src/utils/homeJourney.js';
import {
  assertGlobalPayload,
  hasObsoleteWorkspaceHeaders,
  isLegacyScopedRecord,
} from '../dist/src/utils/obsoleteWorkspaceGuard.js';

const config = () => ({
  config: {
    name: 'ceramics',
    process: [
      {
        type: 'diffusion_trainer',
        model: { arch: 'flux', name_or_path: 'local/model' },
        train: { steps: 100, batch_size: 1, gradient_accumulation: 1, lr: 0.0001, disable_sampling: true },
        network: { type: 'lora' },
        datasets: [{ folder_path: '/data/ceramics', resolution: [512], num_repeats: 1 }],
        sample: { samples: [] },
      },
    ],
  },
});
const context = {
  workerID: 'local',
  gpuIDs: '0',
  datasetOptions: [{ value: '/data/ceramics' }],
  unselectedDatasetPath: 'SELECT_DATASET',
};

test('valid training config passes without changing advanced values', () => {
  const value = config();
  const before = structuredClone(value);
  assert.deepEqual(validateTrainingConfig(value, context), []);
  assert.equal(isEditableTrainingConfig(value), true);
  assert.deepEqual(value, before);
});

test('invalid numeric training values identify the guided control', () => {
  for (const [field, label] of [
    ['steps', 'Steps'],
    ['batch_size', 'Batch size'],
    ['gradient_accumulation', 'Gradient accumulation'],
    ['lr', 'Learning rate'],
  ]) {
    for (const number of [NaN, Infinity, -1]) {
      const value = config();
      value.config.process[0].train[field] = number;
      const finding = validateTrainingConfig(value, context).find(item => item.target.label === label);
      assert.equal(finding?.level, 'error');
      assert.equal(finding.target.step, 'job-training');
    }
  }
});

test('dataset findings target the affected row', () => {
  const value = config();
  value.config.process[0].datasets.push({ folder_path: 'SELECT_DATASET', resolution: [], num_repeats: 0 });
  const findings = validateTrainingConfig(value, context);
  assert.ok(findings.some(item => item.target.label === 'Target dataset'));
  assert.ok(findings.some(item => item.target.label === 'Resolutions' && item.target.index === 1));
  assert.ok(findings.some(item => item.target.label === 'Num repeats' && item.target.index === 1));
});

test('raw imports refuse malformed editable structures', () => {
  for (const value of [null, [], {}, { config: { name: 5, process: [] } }])
    assert.equal(isEditableTrainingConfig(value), false);
  const value = config();
  value.config.process[0].model.arch = 7;
  assert.equal(isEditableTrainingConfig(value), false);
});

test('Home prioritizes active training, attention, then data preparation', () => {
  const dataset = { name: 'ceramics', path: '/data/ceramics', itemCount: 8, missingCaptionCount: 2, encrypted: false };
  assert.equal(nextHomeAction([], []).label, 'Import data');
  assert.match(nextHomeAction([], [dataset]).description, /not quality/);
  assert.equal(
    nextHomeAction(
      [
        { id: 'failed', status: 'error', step: 2 },
        { id: 'active', status: 'running', step: 3 },
      ],
      [dataset],
    ).job.id,
    'active',
  );
  assert.equal(nextHomeAction([{ id: 'failed', status: 'error', step: 2 }], [dataset]).stage, 'review');
  assert.equal(nextHomeAction([], [{ ...dataset, missingCaptionCount: null }]).label, 'Review dataset');
  assert.equal(nextHomeAction([], [{ ...dataset, missingCaptionCount: 0 }]).label, 'Set up training');
});

test('obsolete JSON, queries, form fields, references and streamed headers fail closed', () => {
  for (const value of [
    { project_id: null },
    { projectID: 'old' },
    { source_project_id: 'old' },
    { destination_project_id: 'old' },
    { config: { datasets: [{ folder_path: 'aitk-project://old/datasets/cats' }] } },
    new URLSearchParams('scope=all'),
  ])
    assert.throws(() => assertGlobalPayload(value), /removed/);
  const form = new FormData();
  form.set('project_id', 'old');
  assert.throws(() => assertGlobalPayload(form), /removed/);
  assert.equal(hasObsoleteWorkspaceHeaders(new Headers({ 'X-AITK-Project-ID': 'old' })), true);
  assert.doesNotThrow(() => assertGlobalPayload({ datasetName: 'cats', worker_id: 'worker', scope: 'global' }));
  assert.equal(isLegacyScopedRecord({ project_id: 'old' }), true);
  assert.equal(isLegacyScopedRecord({ project_id: null }), false);
});
