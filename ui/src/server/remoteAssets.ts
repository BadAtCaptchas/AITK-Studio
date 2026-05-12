import path from 'path';

export type RemoteAssetType = 'img' | 'file' | 'audio-art';

export function makeRemoteAssetRef(jobID: string, type: RemoteAssetType, remotePath: string) {
  const filename = path.basename(remotePath.replace(/\\/g, '/')) || 'remote-file';
  return `remote://${jobID}/${type}/${encodeURIComponent(remotePath)}/${encodeURIComponent(filename)}`;
}
