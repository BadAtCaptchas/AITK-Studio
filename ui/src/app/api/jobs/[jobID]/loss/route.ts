import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import { getTrainingFolder } from '@/server/settings';
import { db } from '@/server/db';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';

export const runtime = 'nodejs';

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  // this must be awaited to avoid TS error
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const url = new URL(request.url);
  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) return NextResponse.json({ keys: [], key: 'loss', points: [] });
    try {
      const worker = await getRemoteWorker(job.worker_id);
      return NextResponse.json(
        await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/loss?${url.searchParams}`),
      );
    } catch (error) {
      console.error('Error reading remote loss log:', error);
      return NextResponse.json({ error: 'Error reading remote loss log' }, { status: 502 });
    }
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'loss_log.db');

  const key = url.searchParams.get('key') ?? 'loss';
  const limit = Math.min(Number(url.searchParams.get('limit') ?? 2000), 20000);
  const sinceStepParam = url.searchParams.get('since_step');
  const sinceStep = sinceStepParam != null ? Number(sinceStepParam) : null;
  const stride = Math.max(1, Number(url.searchParams.get('stride') ?? 1));

  return NextResponse.json(await db.metrics.getLossLog(jobID, logPath, { key, limit, sinceStep, stride }));
}
