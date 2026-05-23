export interface TrainingPhaseSummaryPhase {
  name?: string | null;
  steps?: number | string | null;
}

export interface TrainingPhaseSummaryTrainConfig {
  auto_train?: boolean | null;
  auto_learn?: boolean | null;
  phases?: TrainingPhaseSummaryPhase[] | null;
}

export interface TrainingPhaseMetricPoint {
  value?: number | null;
  value_text?: string | null;
}

export interface TrainingPhaseMetricInputs {
  index?: TrainingPhaseMetricPoint | null;
  name?: TrainingPhaseMetricPoint | null;
  step?: TrainingPhaseMetricPoint | null;
  reason?: TrainingPhaseMetricPoint | null;
}

export interface TrainingPhaseSummary {
  index: number;
  count: number;
  name: string;
  phaseStep: number | null;
  phaseSteps: number | null;
  progress: number | null;
  isAutoTrain: boolean;
  reason: string | null;
  source: 'metrics' | 'config';
  telemetryPending: boolean;
}

function numericMetric(point?: TrainingPhaseMetricPoint | null) {
  const value = point?.value;
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value;
}

function textMetric(point?: TrainingPhaseMetricPoint | null) {
  const value = point?.value_text?.trim();
  return value || null;
}

function phaseSteps(phase: TrainingPhaseSummaryPhase | undefined) {
  const value = Number(phase?.steps);
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : null;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function indexForFixedStep(phases: TrainingPhaseSummaryPhase[], step: number) {
  const safeStep = Math.max(0, Math.floor(Number.isFinite(step) ? step : 0));
  let cursor = 0;
  for (let index = 0; index < phases.length; index++) {
    const steps = phaseSteps(phases[index]);
    if (steps == null) return index;
    cursor += steps;
    if (safeStep < cursor) return index;
  }
  return Math.max(0, phases.length - 1);
}

function phaseStartStep(phases: TrainingPhaseSummaryPhase[], index: number) {
  let start = 0;
  for (let i = 0; i < index; i++) {
    start += phaseSteps(phases[i]) ?? 0;
  }
  return start;
}

export function hasTrainingPhases(train?: TrainingPhaseSummaryTrainConfig | null) {
  const phases = Array.isArray(train?.phases) ? train.phases : [];
  return phases.length > 0;
}

export function buildTrainingPhaseSummary(
  train: TrainingPhaseSummaryTrainConfig | null | undefined,
  globalStep: number,
  metrics: TrainingPhaseMetricInputs = {},
): TrainingPhaseSummary | null {
  const phases = Array.isArray(train?.phases) ? train.phases : [];
  if (!phases.length) return null;

  const safeGlobalStep = Math.max(0, Math.floor(Number.isFinite(globalStep) ? globalStep : 0));
  const isAutoTrain = Boolean(train?.auto_train || train?.auto_learn);
  const metricIndex = numericMetric(metrics.index);
  const hasMetricIndex = metricIndex !== null;
  const fallbackIndex = isAutoTrain ? 0 : indexForFixedStep(phases, safeGlobalStep);
  const index = clamp(Math.round(hasMetricIndex ? metricIndex : fallbackIndex), 0, phases.length - 1);
  const source = hasMetricIndex ? 'metrics' : 'config';
  const metricStep = numericMetric(metrics.step);
  const steps = phaseSteps(phases[index]);
  const fallbackStep = isAutoTrain ? null : Math.max(0, safeGlobalStep - phaseStartStep(phases, index));
  const phaseStep = metricStep === null ? fallbackStep : Math.max(0, Math.floor(metricStep));
  const progress = steps !== null && phaseStep !== null ? clamp((phaseStep / steps) * 100, 0, 100) : null;
  const name = textMetric(metrics.name) || phases[index]?.name?.trim() || `Phase ${index + 1}`;
  const reason = textMetric(metrics.reason);

  return {
    index,
    count: phases.length,
    name,
    phaseStep,
    phaseSteps: steps,
    progress,
    isAutoTrain,
    reason,
    source,
    telemetryPending: isAutoTrain && source === 'config',
  };
}
