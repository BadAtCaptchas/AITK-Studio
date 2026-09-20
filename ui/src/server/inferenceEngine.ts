import fs from 'fs/promises';
import path from 'path';
import { db } from './db';
import { getSafeJobFolder } from './jobFolder';
import { normalizeStoragePathSetting } from './pathContainment';
import { isRecord } from './commandInput';
import { inferenceToken } from './inferenceToken';

export async function getInferenceEndpoint(jobID: string) {
  const job = await db.jobs.findById(jobID);
  if (!job || job.job_type !== 'inference' || (job.worker_id && job.worker_id !== 'local')) return null;
  if (!['running', 'stopping'].includes(job.status) || !job.attempt_id) return null;
  const folder = await getSafeJobFolder(job);
  const filename = await normalizeStoragePathSetting(path.join(folder, 'engine.json'), folder);
  if (!filename) return null;
  const stat = await fs.stat(filename).catch(() => null);
  if (!stat?.isFile() || stat.size > 16384) return null;
  const endpoint: unknown = JSON.parse(await fs.readFile(filename, 'utf8'));
  if (
    !isRecord(endpoint) ||
    endpoint.job_id !== job.id ||
    endpoint.attempt_id !== job.attempt_id ||
    endpoint.pid !== job.pid ||
    endpoint.host !== '127.0.0.1' ||
    typeof endpoint.port !== 'number' ||
    !Number.isInteger(endpoint.port) ||
    endpoint.port < 1 ||
    endpoint.port > 65535
  )
    return null;
  return { job, url: `http://127.0.0.1:${endpoint.port}`, token: inferenceToken(job.id, job.attempt_id) };
}
