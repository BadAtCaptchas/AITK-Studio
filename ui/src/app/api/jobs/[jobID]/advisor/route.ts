import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import { analyzeTrainingAdvisor } from '@/server/trainingAdvisor';
import { db } from '@/server/db';
import { getTrainingFolder } from '@/server/settings';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import type { JobConfig } from '@/types';

export const runtime = 'nodejs';

const ADVISOR_METRIC_KEYS = ['loss*', 'learning_rate*', 'lr*', 'phase/*', 'event/*', 'train/*'];

function ensureApiAccess(request: NextRequest): NextResponse | null {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) return null;

  const token = request.headers.get('authorization')?.split(' ')[1];
  if (token !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  return null;
}

function isValidJobId(jobID: string) {
  return /^[a-zA-Z0-9_-]+$/.test(jobID);
}

function parseJobConfig(raw: string): JobConfig | null {
  try {
    return JSON.parse(raw) as JobConfig;
  } catch {
    return null;
  }
}

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;

  const { jobID } = await params;
  if (!isValidJobId(jobID)) {
    return NextResponse.json({ error: 'Invalid job ID' }, { status: 400 });
  }

  const job = await db.jobs.findById(jobID);
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      const jobConfig = parseJobConfig(job.job_config);
      if (!jobConfig) return NextResponse.json({ error: 'Invalid job config' }, { status: 400 });
      return NextResponse.json(analyzeTrainingAdvisor(jobConfig, { gpuIds: job.gpu_ids, job }));
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      return NextResponse.json(await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/advisor`));
    } catch (error) {
      console.error('Error reading remote training advisor:', error);
      return NextResponse.json({ error: 'Error reading remote training advisor' }, { status: 502 });
    }
  }

  const jobConfig = parseJobConfig(job.job_config);
  if (!jobConfig) return NextResponse.json({ error: 'Invalid job config' }, { status: 400 });

  const trainingFolder = await getTrainingFolder();
  const logPath = path.join(trainingFolder, job.name, 'loss_log.db');
  const metrics = await db.metrics.getMetrics(jobID, logPath, {
    keys: ADVISOR_METRIC_KEYS,
    maxPoints: 5000,
    sinceStep: null,
  });

  return NextResponse.json(
    analyzeTrainingAdvisor(jobConfig, {
      gpuIds: job.gpu_ids,
      job,
      metrics,
      scanDatasets: true,
    }),
  );
}
