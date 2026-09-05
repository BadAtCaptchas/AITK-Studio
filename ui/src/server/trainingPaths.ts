import type { Job } from '../types';
import { getTrainingFolder } from './settings';

/** Existing runs keep the configured output root saved with their configuration. */
export async function getJobTrainingRoot(job: Pick<Job, 'job_config'>): Promise<string> {
  try {
    const parsed: unknown = JSON.parse(job.job_config);
    if (parsed && typeof parsed === 'object' && 'config' in parsed) {
      const config = parsed.config;
      if (config && typeof config === 'object' && 'process' in config && Array.isArray(config.process)) {
        const first: unknown = config.process[0];
        if (
          first &&
          typeof first === 'object' &&
          'training_folder' in first &&
          typeof first.training_folder === 'string' &&
          first.training_folder.trim()
        )
          return first.training_folder;
      }
    }
  } catch {
    /* Older or incomplete configurations use the configured global root. */
  }
  return getTrainingFolder();
}
