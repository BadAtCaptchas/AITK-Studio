import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import { getTrainingFolder } from '@/server/settings';
import { db } from '@/server/db';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';

export const runtime = 'nodejs';

const DEFAULT_KEYS = ['loss*', 'learning_rate*', 'lr*', 'phase/*', 'event/*', 'train/*'];

function parseKeys(value: string | null) {
  if (!value) return DEFAULT_KEYS;
  const keys = value
    .split(',')
    .map(key => key.trim())
    .filter(Boolean);
  return keys.length ? keys : DEFAULT_KEYS;
}

function parseSinceSteps(value: string | null): Record<string, number | null> | undefined {
  if (!value) return undefined;
  const out: Record<string, number | null> = {};
  for (const part of value.split(',')) {
    const [encodedKey, rawStep] = part.split(':');
    if (!encodedKey || rawStep == null) continue;
    const step = Number(rawStep);
    if (!Number.isFinite(step)) continue;
    out[decodeURIComponent(encodedKey)] = step;
  }
  return out;
}

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });

  const url = new URL(request.url);

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      return NextResponse.json({ keys: [], keyInfo: [], series: {} });
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      return NextResponse.json(
        await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/metrics?${url.searchParams}`),
      );
    } catch (error) {
      console.error('Error reading remote metrics:', error);
      return NextResponse.json({ error: 'Error reading remote metrics' }, { status: 502 });
    }
  }

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'loss_log.db');
  const sinceStepParam = url.searchParams.get('since_step');
  const sinceStep = sinceStepParam != null ? Number(sinceStepParam) : null;
  const maxPoints = Math.min(Math.max(Number(url.searchParams.get('max_points') ?? 4000), 2), 20000);

  return NextResponse.json(
    await db.metrics.getMetrics(jobID, logPath, {
      keys: parseKeys(url.searchParams.get('keys')),
      maxPoints,
      sinceStep: Number.isFinite(sinceStep) ? sinceStep : null,
      sinceSteps: parseSinceSteps(url.searchParams.get('since_steps')),
    }),
  );
}
