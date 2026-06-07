import type { DatasetSummary } from '../types';
import type { WorkerNodeRecord } from './db';
import { remoteJson } from './remoteClient';
import {
  collectSameWorkerRemoteDatasetReferences,
  rewriteSameWorkerRemoteDatasetRefs,
} from './trainingJobTransfer';

export async function rewriteSameWorkerRemoteDatasetRefsForWorker(rawJobConfig: any, worker: WorkerNodeRecord) {
  const refs = collectSameWorkerRemoteDatasetReferences(rawJobConfig, worker.id);
  if (refs.length === 0) return rawJobConfig;

  const remoteDatasets = await remoteJson<DatasetSummary[]>(worker, '/api/datasets/list');
  return rewriteSameWorkerRemoteDatasetRefs(rawJobConfig, {
    workerID: worker.id,
    workerName: worker.name,
    datasets: Array.isArray(remoteDatasets) ? remoteDatasets : [],
  });
}
