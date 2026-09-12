import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs/promises';
import { db } from '@/server/db';
import { getJobTrainingRoot } from '@/server/trainingPaths';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { clearDurableEncryptedDatasetKeys } from '@/server/encryptedDatasetSecrets';
import { claimJobMaintenance } from '@/server/jobAttempts';
import { isPathWithinRoot } from '@/server/pathContainment';
import { jobStorageKey } from '@/utils/jobIdentity';

export async function POST(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const candidate = await db.jobs.findById(jobID);
  if (!candidate) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  const job = await claimJobMaintenance(candidate, 'deleting');
  if (!job) return NextResponse.json({ error: 'Stop the job and wait for its process to exit before deleting it' }, { status: 409 });
  try {
    if (!isLocalWorker(job.worker_id)) {
      if (job.remote_job_id) {
        const worker = await getRemoteWorker(job.worker_id);
        await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/delete`, { method: 'POST' });
      }
    } else {
      const trainingRoot = await getJobTrainingRoot(job);
      const trainingFolder = path.resolve(trainingRoot, jobStorageKey(job));
      if (trainingFolder === path.resolve(trainingRoot) || !isPathWithinRoot(path.resolve(trainingRoot), trainingFolder)) throw new Error('Invalid job folder');
      const canonicalRoot = await fs.realpath(trainingRoot).catch(() => null);
      const canonicalFolder = await fs.realpath(trainingFolder).catch(() => null);
      if (canonicalFolder && (!canonicalRoot || canonicalFolder === canonicalRoot || !isPathWithinRoot(canonicalRoot, canonicalFolder))) throw new Error('Job folder escapes the training root');
      await fs.rm(trainingFolder, { recursive: true, force: true });
    }
    await clearDurableEncryptedDatasetKeys(jobID);
    await db.jobs.delete(jobID, { attempt_id: job.attempt_id ?? null, status: 'deleting' });
    return NextResponse.json(job);
  } catch (error) {
    await db.jobs.updateIf(jobID, { attempt_id: job.attempt_id ?? null, status: 'deleting' }, { status: 'error', info: 'Deletion did not complete. Review the error and retry.' });
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Deletion failed' }, { status: 409 });
  }
}

export function GET() {
  return NextResponse.json({ error: 'Use POST for this command' }, { status: 405, headers: { Allow: 'POST' } });
}
