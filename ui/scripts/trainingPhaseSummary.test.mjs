import assert from 'node:assert/strict';
import test from 'node:test';
import { buildTrainingPhaseSummary, hasTrainingPhases } from '../dist/src/utils/trainingPhaseSummary.js';

const fixedTrain = {
  phases: [
    { name: 'Teach structure', steps: 300 },
    { name: 'Stabilize concept', steps: 200 },
    { name: 'Refine detail', steps: 500 },
  ],
};

test('fixed phases derive the current phase from global step', () => {
  const summary = buildTrainingPhaseSummary(fixedTrain, 350);

  assert.equal(summary?.source, 'config');
  assert.equal(summary?.index, 1);
  assert.equal(summary?.count, 3);
  assert.equal(summary?.name, 'Stabilize concept');
  assert.equal(summary?.phaseStep, 50);
  assert.equal(summary?.phaseSteps, 200);
  assert.equal(summary?.progress, 25);
});

test('live metrics override config fallback', () => {
  const summary = buildTrainingPhaseSummary(fixedTrain, 50, {
    index: { value: 2 },
    name: { value: null, value_text: 'Metric phase' },
    step: { value: 17 },
    reason: { value: null, value_text: 'loss_plateau' },
  });

  assert.equal(summary?.source, 'metrics');
  assert.equal(summary?.index, 2);
  assert.equal(summary?.name, 'Metric phase');
  assert.equal(summary?.phaseStep, 17);
  assert.equal(summary?.phaseSteps, 500);
  assert.equal(summary?.reason, 'loss_plateau');
});

test('auto-train phases use live metric phase when fixed steps are omitted', () => {
  const summary = buildTrainingPhaseSummary(
    {
      auto_train: true,
      phases: [{ name: 'Teach subject' }, { name: 'Stabilize' }, { name: 'Polish style' }],
    },
    900,
    {
      index: { value: 1 },
      step: { value: 42 },
    },
  );

  assert.equal(summary?.source, 'metrics');
  assert.equal(summary?.telemetryPending, false);
  assert.equal(summary?.index, 1);
  assert.equal(summary?.name, 'Stabilize');
  assert.equal(summary?.phaseStep, 42);
  assert.equal(summary?.phaseSteps, null);
  assert.equal(summary?.progress, null);
});

test('auto-train phases without metrics fall back to first phase and mark telemetry pending', () => {
  const summary = buildTrainingPhaseSummary(
    {
      auto_train: true,
      phases: [{ name: 'Teach subject' }, { name: 'Stabilize' }],
    },
    900,
  );

  assert.equal(summary?.source, 'config');
  assert.equal(summary?.telemetryPending, true);
  assert.equal(summary?.index, 0);
  assert.equal(summary?.name, 'Teach subject');
  assert.equal(summary?.phaseStep, null);
  assert.equal(summary?.phaseSteps, null);
});

test('jobs without phases return no phase summary', () => {
  assert.equal(hasTrainingPhases({ phases: [] }), false);
  assert.equal(hasTrainingPhases({ auto_train: true }), false);
  assert.equal(buildTrainingPhaseSummary({ auto_train: true }, 100), null);
  assert.equal(buildTrainingPhaseSummary(null, 100), null);
});
