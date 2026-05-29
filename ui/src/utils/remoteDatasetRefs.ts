export const REMOTE_DATASET_REF_PREFIX = 'aitk-dataset://remote/';
export const REMOTE_DATASET_ASSET_REF_PREFIX = 'aitk-dataset-asset://remote/';

export type RemoteDatasetRef = {
  workerID: string;
  datasetName: string;
};

export type RemoteDatasetAssetType = 'img' | 'file' | 'audio-art';

export type RemoteDatasetAssetRef = {
  workerID: string;
  type: RemoteDatasetAssetType;
  path: string;
  filename: string;
};

function basename(value: string) {
  return value.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || value || 'remote-file';
}

export function makeRemoteDatasetRef(workerID: string, datasetName: string) {
  return `${REMOTE_DATASET_REF_PREFIX}${encodeURIComponent(workerID)}/${encodeURIComponent(datasetName)}`;
}

export function parseRemoteDatasetRef(value: string | null | undefined): RemoteDatasetRef | null {
  if (!value || !value.startsWith(REMOTE_DATASET_REF_PREFIX)) return null;
  const raw = value.slice(REMOTE_DATASET_REF_PREFIX.length);
  const slashIndex = raw.indexOf('/');
  if (slashIndex < 0) return null;
  try {
    const workerID = decodeURIComponent(raw.slice(0, slashIndex));
    const datasetName = decodeURIComponent(raw.slice(slashIndex + 1));
    if (!workerID || !datasetName) return null;
    return { workerID, datasetName };
  } catch {
    return null;
  }
}

export function makeRemoteDatasetAssetRef(
  workerID: string,
  type: RemoteDatasetAssetType,
  remotePath: string,
  filename = basename(remotePath),
) {
  return `${REMOTE_DATASET_ASSET_REF_PREFIX}${encodeURIComponent(workerID)}/${encodeURIComponent(type)}/${encodeURIComponent(remotePath)}/${encodeURIComponent(filename)}`;
}

export function parseRemoteDatasetAssetRef(value: string | null | undefined): RemoteDatasetAssetRef | null {
  if (!value || !value.startsWith(REMOTE_DATASET_ASSET_REF_PREFIX)) return null;
  const raw = value.slice(REMOTE_DATASET_ASSET_REF_PREFIX.length);
  const parts = raw.split('/');
  if (parts.length < 3) return null;
  const [encodedWorkerID, encodedType, encodedPath, encodedFilename] = parts;
  try {
    const type = decodeURIComponent(encodedType || '') as RemoteDatasetAssetType;
    if (type !== 'img' && type !== 'file' && type !== 'audio-art') return null;
    const workerID = decodeURIComponent(encodedWorkerID || '');
    const path = decodeURIComponent(encodedPath || '');
    const filename = decodeURIComponent(encodedFilename || '') || basename(path);
    if (!workerID || !path) return null;
    return { workerID, type, path, filename };
  } catch {
    return null;
  }
}

export function remoteDatasetRememberKey(workerID: string, datasetName: string) {
  return `${workerID}:${datasetName}`;
}
