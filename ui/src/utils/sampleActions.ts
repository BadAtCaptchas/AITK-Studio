import { apiClient } from './api';
import { parseRemoteAssetRef } from './media';

export async function deleteSample(value: string): Promise<void> {
  const remote = parseRemoteAssetRef(value);
  const samplePath = remote?.path || value;
  const match = /^\/api\/jobs\/([^/]+)\/samples\/([^/?]+)$/.exec(samplePath);
  if (!match) throw new Error('Invalid sample reference');
  const jobID = remote?.jobID || decodeURIComponent(match[1]);
  const filename = decodeURIComponent(match[2]);
  await apiClient.delete(`/api/jobs/${encodeURIComponent(jobID)}/samples/${encodeURIComponent(filename)}`);
}
