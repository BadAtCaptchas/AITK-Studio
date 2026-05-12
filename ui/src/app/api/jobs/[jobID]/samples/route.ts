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
      return NextResponse.json({ samples: [] });
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      const data = await remoteJson<{ samples: string[] }>(
        worker,
        `/api/jobs/${encodeURIComponent(job.remote_job_id)}/samples`,
      );
      return NextResponse.json({
        samples: (data.samples || []).map(sample => makeRemoteAssetRef(job.id, 'img', sample)),
      });
    } catch (error) {
      console.error('Error reading remote samples:', error);
      return NextResponse.json({ error: 'Error reading remote samples' }, { status: 502 });
    }
  }

  // setup the training
  const trainingFolder = await getTrainingFolder();

  const samplesFolder = path.join(trainingFolder, job.name, 'samples');
  if (!fs.existsSync(samplesFolder)) {
    return NextResponse.json({ samples: [] });
  }

  // find all img (png, jpg, jpeg) files in the samples folder
  const samples = fs
    .readdirSync(samplesFolder)
    .filter(file => {
      return file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg') || file.endsWith('.webp') || file.endsWith('.mp4') || file.endsWith('mp3') || file.endsWith('wav') || file.endsWith('flac') || file.endsWith('ogg');
    })
    .map(file => {
      return path.join(samplesFolder, file);
    })
    .sort();

  return NextResponse.json({ samples });
}
