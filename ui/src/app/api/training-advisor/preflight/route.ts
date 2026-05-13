import { NextRequest, NextResponse } from 'next/server';
import { analyzeTrainingAdvisor } from '@/server/trainingAdvisor';
import type { JobConfig } from '@/types';

export const runtime = 'nodejs';

function ensureApiAccess(request: NextRequest): NextResponse | null {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) return null;

  const token = request.headers.get('authorization')?.split(' ')[1];
  if (token !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  return null;
}

function isSafeJobConfig(jobConfig: unknown): jobConfig is JobConfig {
  if (!jobConfig || typeof jobConfig !== 'object') return false;
  const config = (jobConfig as Record<string, unknown>).config;
  if (!config || typeof config !== 'object') return false;
  const processList = (config as Record<string, unknown>).process;
  return Array.isArray(processList) && processList.length > 0;
}

export async function POST(request: NextRequest) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;

  try {
    const body = await request.json();
    const jobConfig = body.job_config ?? body.jobConfig;
    if (!isSafeJobConfig(jobConfig)) {
      return NextResponse.json({ error: 'Invalid job config' }, { status: 400 });
    }

    return NextResponse.json(
      analyzeTrainingAdvisor(jobConfig, {
        gpuIds: body.gpu_ids ?? body.gpuIds ?? null,
        scanDatasets: true,
      }),
    );
  } catch (error) {
    console.error('Error running training advisor preflight:', error);
    return NextResponse.json({ error: 'Failed to run training advisor' }, { status: 500 });
  }
}
