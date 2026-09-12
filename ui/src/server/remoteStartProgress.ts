import { activeOperationForResource, getOperation, type Operation } from './operations';
import type { RemoteStartProgress } from '../types';

export function remoteStartProgress(operation: Operation): RemoteStartProgress {
  const progress = operation.progress;
  const ended = ['completed', 'failed', 'canceled', 'needs-keys'].includes(operation.state);
  const status = operation.state === 'completed' ? 'completed' : ended ? 'failed' :
    typeof progress.status === 'string' && ['queued', 'preparing', 'checking-datasets', 'zipping-dataset', 'uploading-dataset', 'importing-dataset', 'zipping-job', 'uploading-job', 'importing-job', 'starting'].includes(progress.status) ? progress.status as RemoteStartProgress['status'] : 'queued';
  return { startID: operation.id, jobID: operation.input.kind === 'remote-start' ? operation.input.jobID : '', status,
    operationState: operation.state, phase: operation.phase,
    message: operation.error || (typeof progress.message === 'string' ? progress.message : 'Queued remote start'),
    percent: ended ? 100 : typeof progress.percent === 'number' ? Math.min(100, Math.max(0, progress.percent)) : 0,
    datasetName: typeof progress.datasetName === 'string' ? progress.datasetName : null,
    bytesProcessed: typeof progress.bytesProcessed === 'number' ? progress.bytesProcessed : 0,
    bytesTotal: typeof progress.bytesTotal === 'number' ? progress.bytesTotal : 0,
    warnings: Array.isArray(progress.warnings) ? progress.warnings.filter((value: unknown): value is string => typeof value === 'string') : [],
    error: operation.error, remoteJobID: typeof progress.remoteJobID === 'string' ? progress.remoteJobID : null,
    createdAt: operation.createdAt, updatedAt: operation.updatedAt };
}
export async function getRemoteStartProgress(id: string) {
  const operation = await getOperation(id);
  return operation?.input.kind === 'remote-start' ? remoteStartProgress(operation) : null;
}
export async function hasActiveRemoteStartForJob(jobID: string) { return Boolean(await activeOperationForResource(`job:${jobID}`)); }
