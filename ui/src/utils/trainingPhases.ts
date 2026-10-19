const isRecord = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);

const migrationMessage =
  'Auto Learn and plateau advancement are no longer supported. Remove train.auto_train, train.auto_learn, and phase auto_advance settings; give every phase positive integer steps and set train.steps to their total.';

/** Validate imported jobs before either saving or launching them. */
export function trainingPhaseErrors(train: unknown): string[] {
  if (!isRecord(train)) return [];
  const errors: string[] = [];
  if (train.auto_train || train.auto_learn) errors.push(migrationMessage);
  if (train.phases === undefined || train.phases === null) return errors;
  if (!Array.isArray(train.phases)) return [...errors, 'train.phases must be an array of finite manual phases.'];
  let total = 0;
  let validSteps = true;
  for (const [index, phase] of train.phases.entries()) {
    if (isRecord(phase) && 'auto_advance' in phase && !errors.includes(migrationMessage)) errors.push(migrationMessage);
    if (!isRecord(phase) || typeof phase.steps !== 'number' || !Number.isSafeInteger(phase.steps) || phase.steps < 1) {
      errors.push(
        `train.phases[${index}].steps must be a positive integer. Set a finite duration for every manual phase.`,
      );
      validSteps = false;
    } else {
      total += phase.steps;
    }
  }
  if (train.phases.length && validSteps && train.steps !== total) {
    errors.push(`train.steps must equal the sum of manual phase steps (${total}).`);
  }
  return errors;
}

/** Only disabled legacy flags are safe to discard without changing behavior. */
export function stripDisabledAutomaticTraining(train: unknown): void {
  if (!isRecord(train)) return;
  if (train.auto_train === false) delete train.auto_train;
  if (train.auto_learn === false) delete train.auto_learn;
}
