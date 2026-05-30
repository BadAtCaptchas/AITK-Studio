import { parseRemoteDatasetAssetRef } from './remoteDatasetRefs';

type RemoteAssetRef = {
  jobID: string;
  type: string;
  path: string;
  filename: string;
};

export function parseRemoteAssetRef(value: string): RemoteAssetRef | null {
  if (!value.startsWith('remote://')) return null;
  const raw = value.slice('remote://'.length);
  const parts = raw.split('/');
  if (parts.length < 4) return null;
  const [jobID, type, encodedPath, encodedFilename] = parts;
  if (!jobID || !type || !encodedPath) return null;
  try {
    return {
      jobID,
      type,
      path: decodeURIComponent(encodedPath),
      filename: decodeURIComponent(encodedFilename || 'remote-file'),
    };
  } catch {
    return null;
  }
}

export function getDisplayPath(value: string) {
  const remoteDataset = parseRemoteDatasetAssetRef(value);
  if (remoteDataset) return remoteDataset.filename;
  const remote = parseRemoteAssetRef(value);
  return remote?.filename || value;
}

export function getMediaUrl(value: string, overrideType?: 'img' | 'file' | 'audio-art') {
  if (value.startsWith('/api/') || /^https?:\/\//i.test(value)) return value;
  const remoteDataset = parseRemoteDatasetAssetRef(value);
  if (remoteDataset) {
    const type = overrideType || remoteDataset.type || 'img';
    const params = new URLSearchParams({
      worker_id: remoteDataset.workerID,
      type,
      path: remoteDataset.path,
    });
    if (remoteDataset.expires && remoteDataset.signature) {
      params.set('expires', String(remoteDataset.expires));
      params.set('sig', remoteDataset.signature);
    }
    return `/api/remote-datasets/assets?${params.toString()}`;
  }
  const remote = parseRemoteAssetRef(value);
  if (remote) {
    const type = overrideType || remote.type || 'img';
    return `/api/remote-assets?job_id=${encodeURIComponent(remote.jobID)}&type=${encodeURIComponent(type)}&path=${encodeURIComponent(remote.path)}`;
  }
  if (overrideType === 'audio-art') {
    return `/api/audio/art/${encodeURIComponent(value)}`;
  }
  if (overrideType === 'file') {
    return `/api/files/${encodeURIComponent(value)}`;
  }
  return `/api/img/${encodeURIComponent(value)}`;
}

export function getDownloadUrl(value: string) {
  if (value.startsWith('/api/') || /^https?:\/\//i.test(value)) return value;
  const remoteDataset = parseRemoteDatasetAssetRef(value);
  if (remoteDataset) {
    return getMediaUrl(value, 'file');
  }
  const remote = parseRemoteAssetRef(value);
  if (remote) {
    return getMediaUrl(value, 'file');
  }
  return `/api/files/${encodeURIComponent(value)}`;
}
