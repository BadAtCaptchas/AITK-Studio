import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';
import { db } from '@/server/db';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { makeRemoteAssetRef } from '@/server/remoteAssets';

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      return NextResponse.json({ files: [] });
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      const data = await remoteJson<{ files: Array<{ path: string; size: number }> }>(
        worker,
        `/api/jobs/${encodeURIComponent(job.remote_job_id)}/files`,
      );
      return NextResponse.json({
        files: (data.files || []).map(file => ({
          ...file,
          path: makeRemoteAssetRef(job.id, 'file', file.path),
        })),
      });
    } catch (error) {
      console.error('Error reading remote files:', error);
      return NextResponse.json({ error: 'Error reading remote files' }, { status: 502 });
    }
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);

  if (!fs.existsSync(jobFolder)) {
    return NextResponse.json({ files: [] });
  }

  // find all safetensors files in the job folder
  let files = fs
    .readdirSync(jobFolder)
    .filter(file => {
      return file.endsWith('.safetensors');
    })
    .map(file => {
      return path.join(jobFolder, file);
    })
    .sort();

  // get the file size for each file
  const fileObjects = files.map(file => {
    const stats = fs.statSync(file);
    return {
      path: file,
      size: stats.size,
    };
  });

  // include the optimizer state if it exists
  const optimizerPath = path.join(jobFolder, 'optimizer.pt');
  if (fs.existsSync(optimizerPath)) {
    const stats = fs.statSync(optimizerPath);
    fileObjects.push({
      path: optimizerPath,
      size: stats.size,
    });
  }

  return NextResponse.json({ files: fileObjects });
}
