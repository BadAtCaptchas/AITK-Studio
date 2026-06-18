import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import { db } from '@/server/db';
import { assertProjectJobEnabled, getJobTrainingRoot } from '@/server/projects';
import {
  getRemoteWorker,
  isLocalWorker,
  isRemoteJobMissingError,
  markRemoteJobMissing,
  remoteJson,
} from '@/server/remoteClient';
import { makeRemoteAssetRef } from '@/server/remoteAssets';

function isPathInsideRoot(root: string, filepath: string) {
  const relativePath = path.relative(root, filepath);
  return relativePath === '' || (!relativePath.startsWith('..') && !path.isAbsolute(relativePath));
}

export async function GET(_request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }
  try {
    await assertProjectJobEnabled(job);
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Project spaces are disabled' }, { status: error?.status || 403 });
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
      if (isRemoteJobMissingError(error)) {
        await markRemoteJobMissing(job);
        return NextResponse.json({ samples: [] });
      }
      console.error('Error reading remote samples:', error);
      return NextResponse.json({ error: 'Error reading remote samples' }, { status: 502 });
    }
  }

  // setup the training
  const trainingFolder = await getJobTrainingRoot(job);

  const canonicalTrainingFolder = await fs.promises.realpath(path.resolve(trainingFolder)).catch(() => null);
  const samplesFolder = path.resolve(trainingFolder, job.name, 'samples');
  const canonicalSamplesFolder = await fs.promises.realpath(samplesFolder).catch(() => null);
  if (
    !canonicalTrainingFolder ||
    !canonicalSamplesFolder ||
    !isPathInsideRoot(canonicalTrainingFolder, canonicalSamplesFolder)
  ) {
    return NextResponse.json({ samples: [] });
  }

  const allowedSampleExtensions = new Set([
    '.png',
    '.jpg',
    '.jpeg',
    '.webp',
    '.jxl',
    '.mp4',
    '.mp3',
    '.wav',
    '.flac',
    '.ogg',
  ]);

  const samples = fs
    .readdirSync(canonicalSamplesFolder, { withFileTypes: true })
    .filter(entry => entry.isFile() && allowedSampleExtensions.has(path.extname(entry.name).toLowerCase()))
    .map(entry => `/api/jobs/${encodeURIComponent(job.id)}/samples/${encodeURIComponent(entry.name)}`)
    .sort();

  return NextResponse.json({ samples });
}
