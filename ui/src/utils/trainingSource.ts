import { migrateJobConfig } from '../app/jobs/new/jobConfig';
import { isEditableTrainingConfig } from './trainingValidation';
import type { TrainingDraft } from './trainingDraftStorage';

/** Validate the source before mounting an editor so defaults cannot be saved over it. */
export function parseTrainingSource(data: unknown, mode: 'edit' | 'clone'): TrainingDraft {
  if (!data || typeof data !== 'object' || !('job_type' in data) || data.job_type !== 'train' ||
      !('job_config' in data) || typeof data.job_config !== 'string') {
    throw new Error('The source training job could not be loaded.');
  }
  let config: unknown;
  try {
    config = JSON.parse(data.job_config);
  } catch {
    throw new Error('The source job contains invalid configuration JSON.');
  }
  if (!isEditableTrainingConfig(config)) throw new Error('The source job has an invalid training configuration.');
  const prepared = migrateJobConfig(config);
  if (mode === 'clone') prepared.config.name = `${prepared.config.name}_copy`;
  return {
    config: prepared,
    gpuIDs: 'gpu_ids' in data && typeof data.gpu_ids === 'string' ? data.gpu_ids : null,
    workerID: 'worker_id' in data && typeof data.worker_id === 'string' && data.worker_id ? data.worker_id : 'local',
  };
}
