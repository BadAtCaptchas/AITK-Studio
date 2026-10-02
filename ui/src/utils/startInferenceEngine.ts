import type { Job } from '../types';

type EngineStartActions = {
  startJob: (jobID: string) => Promise<void>;
  startQueue: (gpuIds: string, workerID: string) => Promise<void>;
};

/** Local inference needs both the accepted job and its own GPU queue running. */
export async function startInferenceEngine(
  job: Pick<Job, 'id' | 'gpu_ids'>,
  actions: EngineStartActions,
): Promise<void> {
  await actions.startJob(job.id);
  await actions.startQueue(job.gpu_ids, 'local');
}
