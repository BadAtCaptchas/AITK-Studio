import path from 'path';

export type RemoteAssetType = 'img' | 'file' | 'audio-art';

export function makeRemoteAssetRef(jobID: string, type: RemoteAssetType, remotePath: string) {
  const filename = path.basename(remotePath.replace(/\\/g, '/')) || 'remote-file';
  return `remote://${jobID}/${type}/${encodeURIComponent(remotePath)}/${encodeURIComponent(filename)}`;
}

export function remoteSampleAssetPath(remotePath: string, remoteJobID: string | null | undefined) {
  if (!remoteJobID || !remotePath.startsWith('/api/jobs/')) return null;
  if (remotePath.includes('?') || remotePath.includes('#') || remotePath.includes('\\')) return null;

  const segments = remotePath.split('/');
  if (
    segments.length !== 6 ||
    segments[0] !== '' ||
    segments[1] !== 'api' ||
    segments[2] !== 'jobs' ||
    segments[4] !== 'samples'
  ) {
    return null;
  }

  let decodedJobID: string;
  let decodedFilename: string;
  try {
    decodedJobID = decodeURIComponent(segments[3]);
    decodedFilename = decodeURIComponent(segments[5]);
  } catch {
    return null;
  }

  if (
    decodedJobID !== remoteJobID ||
    !decodedFilename ||
    decodedFilename === '.' ||
    decodedFilename === '..' ||
    decodedFilename.includes('/') ||
    decodedFilename.includes('\\')
  ) {
    return null;
  }

  return `/api/jobs/${encodeURIComponent(remoteJobID)}/samples/${encodeURIComponent(decodedFilename)}`;
}

export function remoteAssetProxyPath(type: RemoteAssetType, remotePath: string, remoteJobID?: string | null) {
  const remoteSamplePath = remoteSampleAssetPath(remotePath, remoteJobID);
  if (remoteSamplePath) return remoteSamplePath;

  const encoded = encodeURIComponent(remotePath);
  if (type === 'file') return `/api/files/${encoded}`;
  if (type === 'audio-art') return `/api/audio/art/${encoded}`;
  return `/api/img/${encoded}`;
}
