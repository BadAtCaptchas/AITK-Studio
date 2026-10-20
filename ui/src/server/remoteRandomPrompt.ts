import { isRecord } from './commandInput';
import { DatasetScopeError } from './datasetScope';
import { getRemoteWorker, remoteJson, RemoteClientError, withoutRemoteRedirects } from './remoteClient';
import { parseRemoteDatasetRef, REMOTE_DATASET_REF_PREFIX } from '../utils/remoteDatasetRefs';
import {
  considerRandomPromptCandidate,
  type DatasetPromptRequest,
  type RandomPromptState,
} from './randomPromptCaptions';

function isCount(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

export async function scanRemoteRandomPromptDataset(
  dataset: DatasetPromptRequest,
  state: RandomPromptState,
): Promise<boolean> {
  if (!dataset.folderPath.startsWith(REMOTE_DATASET_REF_PREFIX)) return false;
  const ref = parseRemoteDatasetRef(dataset.folderPath);
  // Only send a single dataset name, resolved under the worker's own configured root.
  if (!ref || !ref.datasetName.trim() || ref.datasetName.startsWith('.') || /[\\/:\0]/.test(ref.datasetName)) {
    throw new DatasetScopeError('Invalid remote dataset selection.');
  }

  let result: unknown;
  try {
    const worker = await getRemoteWorker(ref.workerID);
    result = await remoteJson<unknown>(
      worker,
      '/api/datasets/randomPrompt',
      withoutRemoteRedirects({
        method: 'POST',
        body: JSON.stringify({
          datasets: [
            {
              folderPath: ref.datasetName,
              captionExt: dataset.captionExt,
              defaultCaption: dataset.defaultCaption,
            },
          ],
        }),
      }),
    );
  } catch (error) {
    if (error instanceof RemoteClientError && error.status === 404) {
      let empty: unknown;
      try {
        empty = JSON.parse(error.body);
      } catch {
        /* An unavailable route is not an empty dataset. */
      }
      if (isRecord(empty) && isCount(empty.scannedMediaCount) && Array.isArray(empty.skippedDatasets)) {
        state.scannedMediaCount += empty.scannedMediaCount;
        if (empty.skippedDatasets.length) state.skippedDatasets.push(dataset.folderPath);
        return true;
      }
    }
    if (error instanceof RemoteClientError && error.status === 410) {
      throw new DatasetScopeError(
        'Decrypt the legacy remote dataset to a plain copy before importing random prompts.',
        410,
      );
    }
    throw new DatasetScopeError(
      `Could not read captions from remote dataset "${ref.datasetName}". Check that its worker is available.`,
      502,
    );
  }

  if (
    !isRecord(result) ||
    typeof result.prompt !== 'string' ||
    !result.prompt.trim() ||
    !isCount(result.candidateCount) ||
    result.candidateCount === 0 ||
    !isCount(result.scannedMediaCount) ||
    result.scannedMediaCount < result.candidateCount ||
    typeof result.mediaPath !== 'string' ||
    (result.captionPath !== undefined && typeof result.captionPath !== 'string') ||
    (result.source !== 'caption' && result.source !== 'default_caption')
  ) {
    throw new DatasetScopeError('The remote worker returned an invalid random caption response.', 502);
  }
  state.scannedMediaCount += result.scannedMediaCount;
  considerRandomPromptCandidate(
    state,
    {
      prompt: result.prompt.trim(),
      datasetPath: dataset.folderPath,
      mediaPath: result.mediaPath,
      captionPath: typeof result.captionPath === 'string' ? result.captionPath : undefined,
      source: result.source,
    },
    result.candidateCount,
  );
  return true;
}
