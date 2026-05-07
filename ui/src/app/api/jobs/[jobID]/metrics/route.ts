import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import { getTrainingFolder } from '@/server/settings';
import { db } from '@/server/db';

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

  const trainingFolder = await getTrainingFolder();
  const jobFolder = path.join(trainingFolder, job.name);
  const logPath = path.join(jobFolder, 'loss_log.db');

  const url = new URL(request.url);
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
