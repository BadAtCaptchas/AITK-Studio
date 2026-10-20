import { apiClient } from './api';
import type { Job, JobConfig } from '../types';

export type TrainingJobSaveRequest = Pick<Job, 'name' | 'worker_id'> & {
  id?: string | null;
  gpu_ids: string | null;
  job_config: JobConfig;
};

export const TRAINING_JOB_SAVE_TIMEOUT_MS = 30_000;
// Updates may also need to resolve dataset paths and save an existing worker job.
const TRAINING_JOB_UPDATE_TIMEOUT_MS = 90_000;
export const TRAINING_JOB_SAVE_TIMEOUT_MESSAGE =
  'The server did not confirm the save in time. Your settings are still here. Check Queue before retrying in case the job was created.';

export async function saveTrainingJob(request: TrainingJobSaveRequest): Promise<Pick<Job, 'id'>> {
  const controller = new AbortController();
  const timeoutMs = request.id ? TRAINING_JOB_UPDATE_TIMEOUT_MS : TRAINING_JOB_SAVE_TIMEOUT_MS;
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_resolve, reject) => {
    timeout = setTimeout(() => {
      reject(new Error(TRAINING_JOB_SAVE_TIMEOUT_MESSAGE));
      controller.abort();
    }, timeoutMs);
  });
  try {
    // Bound both connection setup and reading the response. Never retry a POST
    // automatically: the server may have saved it before the connection failed.
    const { data } = await Promise.race([
      apiClient.post<unknown>('/api/jobs', request, { signal: controller.signal }),
      deadline,
    ]);
    if (!data || typeof data !== 'object' || !('id' in data) || typeof data.id !== 'string' || !data.id.trim()) {
      throw new Error('The server returned an invalid save confirmation. Check Queue before trying again.');
    }
    return { id: data.id };
  } finally {
    clearTimeout(timeout);
  }
}

export function trainingJobSaveError(error: unknown): { message: string; conflict: boolean } {
  let message = error instanceof Error ? error.message : 'Failed to save job. Please try again.';
  let conflict = false;
  if (error && typeof error === 'object' && 'response' in error) {
    const response = error.response;
    if (response && typeof response === 'object') {
      conflict = 'status' in response && response.status === 409;
      if (
        'data' in response &&
        response.data &&
        typeof response.data === 'object' &&
        'error' in response.data &&
        typeof response.data.error === 'string'
      ) {
        message = response.data.error;
      }
    }
  }
  return { message, conflict };
}
