import { jobStorageKey } from '../../../../../utils/jobIdentity';
import path from 'path';
import fs from 'fs';
import { db } from '@/server/db';
import { getJobTrainingRoot } from '@/server/trainingPaths';
import {
  getRemoteWorker,
  isLocalWorker,
  isRemoteJobMissingError,
  markRemoteJobMissing,
  remoteJson,
} from '@/server/remoteClient';
import { makeRemoteAssetRef } from '@/server/remoteAssets';

export async function GET(request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return Response.json({ error: 'Job not found' }, { status: 404 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      return Response.json({ files: [] });
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      const data = await remoteJson<{ files: Array<{ path: string; size: number }> }>(
        worker,
        `/api/jobs/${encodeURIComponent(job.remote_job_id)}/files`,
      );
      return Response.json({
        files: (data.files || []).map(file => ({
          ...file,
          path: makeRemoteAssetRef(job.id, 'file', file.path),
        })),
      });
    } catch (error) {
      if (isRemoteJobMissingError(error)) {
        await markRemoteJobMissing(job);
        return Response.json({ files: [] });
      }
      console.error('Error reading remote files:', error);
      return Response.json({ error: 'Error reading remote files' }, { status: 502 });
    }
  }

  const trainingFolder = await getJobTrainingRoot(job);
  const jobFolder = path.join(trainingFolder, jobStorageKey(job));

  try {
    await fs.promises.access(jobFolder);
  } catch {
    return Response.json({ files: [] });
  }

  // find all safetensors files in the job folder
  const files = (await fs.promises.readdir(jobFolder))
    .filter(file => {
      return file.endsWith('.safetensors');
    })
    .map(file => {
      return path.join(jobFolder, file);
    })
    .sort();

  // get the file size for each file
  const fileObjects = await Promise.all(
    files.map(async file => {
      const stats = await fs.promises.stat(file);
      return {
        path: file,
        size: stats.size,
      };
    }),
  );

  // include the optimizer state if it exists
  const optimizerPath = path.join(jobFolder, 'optimizer.pt');
  try {
    const stats = await fs.promises.stat(optimizerPath);
    fileObjects.push({
      path: optimizerPath,
      size: stats.size,
    });
  } catch {
    // Optimizer state is optional.
  }

  return Response.json({ files: fileObjects });
}
