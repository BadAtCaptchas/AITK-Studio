function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

export function getValidationConfigErrors(value: unknown): string[] {
  if (value == null) return [];
  if (!isRecord(value)) return ['Validation settings must be an object.'];

  const errors: string[] = [];
  const validationItems = value.validation_items;
  if (!Array.isArray(validationItems) || validationItems.length === 0) {
    errors.push('Add at least one held-out image before enabling validation loss.');
  } else {
    validationItems.forEach((item, index) => {
      if (!isRecord(item) || typeof item.image_path !== 'string' || !item.image_path.trim()) {
        errors.push(`Validation image ${index + 1} is required.`);
      }
      if (isRecord(item) && item.prompt != null && typeof item.prompt !== 'string') {
        errors.push(`Validation prompt ${index + 1} must be text.`);
      }
    });
  }

  const resolution = value.resolution;
  if (!Number.isInteger(resolution) || (resolution as number) < 64) {
    errors.push('Validation resolution must be a whole number of at least 64.');
  }

  const cadence = value.validate_every_n_steps;
  if (!Number.isInteger(cadence) || (cadence as number) < 1) {
    errors.push('Validation cadence must be a whole number of at least 1 step.');
  }

  const sigmas = value.validation_sigmas;
  if (sigmas != null) {
    if (
      !Array.isArray(sigmas) ||
      sigmas.length === 0 ||
      sigmas.some(sigma => typeof sigma !== 'number' || !Number.isFinite(sigma) || sigma < 0 || sigma > 1)
    ) {
      errors.push('Validation sigmas must contain one or more finite values between 0 and 1.');
    }
  }

  return errors;
}

export function getJobValidationConfigErrors(jobConfig: unknown): string[] {
  if (!isRecord(jobConfig) || !isRecord(jobConfig.config)) return [];
  const processes = jobConfig.config.process;
  if (!Array.isArray(processes)) return [];

  return processes.flatMap((processConfig, index) => {
    if (!isRecord(processConfig) || !isRecord(processConfig.train)) return [];
    const errors = getValidationConfigErrors(processConfig.train.validation_config);
    if (processes.length <= 1) return errors;
    return errors.map(error => `Process ${index + 1}: ${error}`);
  });
}
