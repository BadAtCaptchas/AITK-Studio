import path from 'path';
import type { Job } from '../types';
import { getJobTrainingRoot } from './trainingPaths';
import { normalizeStoragePathSetting } from './pathContainment';
import { jobStorageKey, validJobName } from '../utils/jobIdentity';

/** Resolve even a not-yet-created job folder, rejecting escaping symlink ancestors. */
export async function getSafeJobFolder(job: Job): Promise<string> {
  const key = jobStorageKey(job);
  if (!validJobName(key)) throw new Error('Invalid job storage key');
  const root = await getJobTrainingRoot(job);
  const folder = await normalizeStoragePathSetting(path.join(root, key), root);
  if (!folder) throw new Error('Job folder is outside the training root');
  return folder;
}
