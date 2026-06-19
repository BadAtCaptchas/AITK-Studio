import { NextResponse } from 'next/server';
import { db } from '@/server/db';
import { assertProjectJobEnabled } from '@/server/projects';
import {
  getRemoteWorker,
  isLocalWorker,
  isRemoteJobMissingError,
  markRemoteJobMissing,
  remoteJson,
} from '@/server/remoteClient';
import { makeRemoteAssetRef } from '@/server/remoteAssets';
import { listJobSampleUrls } from '@/server/jobSamples';

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

  const samples = await listJobSampleUrls(job);

  return NextResponse.json({ samples });
}
