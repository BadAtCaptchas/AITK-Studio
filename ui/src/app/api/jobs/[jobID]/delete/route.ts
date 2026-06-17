import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import { db } from '@/server/db';
import { getJobTrainingRoot } from '@/server/projects';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { clearDurableEncryptedDatasetKeys } from '@/server/encryptedDatasetSecrets';

function resolveWithinRoot(root: string, target: unknown) {
  if (typeof target !== 'string' || target.trim().length === 0) {
    return null;
  }

  const resolvedRoot = path.resolve(root);
  const resolvedPath = path.resolve(resolvedRoot, target);
  const relativePath = path.relative(resolvedRoot, resolvedPath);

  if (relativePath === '' || relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    return null;
  }

  return resolvedPath;
}

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (job.remote_job_id) {
      try {
        const worker = await getRemoteWorker(job.worker_id);
        await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/delete`);
      } catch (error) {
        console.error('Error deleting remote job before removing local mirror:', error);
      }
    }
    await clearDurableEncryptedDatasetKeys(jobID).catch(error =>
      console.error('Error clearing durable encrypted dataset keys:', error),
    );
    await db.jobs.delete(jobID);
    return NextResponse.json(job);
  }

  const trainingRoot = await getJobTrainingRoot(job);
  const trainingFolder = resolveWithinRoot(trainingRoot, job.name);

  if (!trainingFolder) {
    return NextResponse.json({ error: 'Invalid job path' }, { status: 400 });
  }

  if (fs.existsSync(trainingFolder)) {
    fs.rmSync(trainingFolder, { recursive: true, force: true });
  }

  await clearDurableEncryptedDatasetKeys(jobID).catch(error =>
    console.error('Error clearing durable encrypted dataset keys:', error),
  );
  await db.jobs.delete(jobID);

  return NextResponse.json(job);
}
