import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import { db } from '@/server/db';
import { assertProjectJobEnabled, getJobTrainingRoot } from '@/server/projects';
import {
  getRemoteWorker,
  isLocalWorker,
  isRemoteJobMissingError,
  markRemoteJobMissing,
  remoteJson,
} from '@/server/remoteClient';

export const runtime = 'nodejs';

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  // this must be awaited to avoid TS error
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  try {
    await assertProjectJobEnabled(job);
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Project spaces are disabled' }, { status: error?.status || 403 });
  }

  const url = new URL(request.url);
  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) return NextResponse.json({ keys: [], key: 'loss', points: [] });
    try {
      const worker = await getRemoteWorker(job.worker_id);
      return NextResponse.json(
        await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/loss?${url.searchParams}`),
      );
    } catch (error) {
      if (isRemoteJobMissingError(error)) {
        await markRemoteJobMissing(job);
        return NextResponse.json({ keys: [], key: 'loss', points: [] });
      }
      console.error('Error reading remote loss log:', error);
      return NextResponse.json({ error: 'Error reading remote loss log' }, { status: 502 });
    }
  }

  const trainingFolder = await getJobTrainingRoot(job);
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'loss_log.db');

  const key = url.searchParams.get('key') ?? 'loss';
  const limit = Math.min(Number(url.searchParams.get('limit') ?? 2000), 20000);
  const sinceStepParam = url.searchParams.get('since_step');
  const sinceStep = sinceStepParam != null ? Number(sinceStepParam) : null;
  const stride = Math.max(1, Number(url.searchParams.get('stride') ?? 1));

  return NextResponse.json(await db.metrics.getLossLog(jobID, logPath, { key, limit, sinceStep, stride }));
}

// Delete every logged step in [min_step, max_step] (inclusive) across all
// metric keys. Used by the loss graph's "Delete Selected Range" action.
export async function DELETE(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  try {
    await assertProjectJobEnabled(job, 'write');
  } catch (error: unknown) {
    const projectError = error as { message?: string; status?: number };
    return NextResponse.json(
      { error: projectError.message || 'Project spaces are disabled' },
      { status: projectError.status || 403 },
    );
  }

  let body: { min_step?: unknown; max_step?: unknown } = {};
  try {
    body = await request.json();
  } catch {
    // fall through to validation below
  }
  const minStep = Number(body.min_step);
  const maxStep = Number(body.max_step);
  if (!Number.isFinite(minStep) || !Number.isFinite(maxStep) || minStep > maxStep) {
    return NextResponse.json({ error: 'min_step and max_step must be numbers with min_step <= max_step' }, { status: 400 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) return NextResponse.json({ error: 'Remote job is unavailable' }, { status: 404 });
    try {
      const worker = await getRemoteWorker(job.worker_id);
      return NextResponse.json(
        await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/loss`, {
          method: 'DELETE',
          body: JSON.stringify({ min_step: minStep, max_step: maxStep }),
        }),
      );
    } catch (error) {
      if (isRemoteJobMissingError(error)) {
        await markRemoteJobMissing(job);
        return NextResponse.json({ error: 'Remote job not found' }, { status: 404 });
      }
      console.error('Error deleting remote loss range:', error);
      return NextResponse.json({ error: 'Error deleting remote loss range' }, { status: 502 });
    }
  }

  const trainingFolder = await getJobTrainingRoot(job);
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'loss_log.db');

  try {
    await db.metrics.deleteLossRange(jobID, logPath, minStep, maxStep);
    return NextResponse.json({ ok: true, min_step: minStep, max_step: maxStep });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Failed to delete range';
    if (message === 'No loss log for this job') {
      return NextResponse.json({ error: message }, { status: 404 });
    }
    console.error('Error deleting loss range:', error);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
