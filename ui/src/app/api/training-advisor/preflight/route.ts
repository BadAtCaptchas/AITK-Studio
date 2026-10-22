import { NextRequest, NextResponse } from 'next/server';
import { isAdvisorJobConfig, runTrainingPreflight } from '@/server/trainingPreflight';
import { commandError, isRecord, readJsonCommand } from '@/server/commandInput';
import { isRequestAuthenticated } from '@/utils/authSession';

export const runtime = 'nodejs';

async function ensureApiAccess(request: NextRequest): Promise<NextResponse | null> {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) return null;

  if (!(await isRequestAuthenticated(request, tokenToUse))) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  return null;
}

export async function POST(request: NextRequest) {
  const accessResponse = await ensureApiAccess(request);
  if (accessResponse) return accessResponse;

  let body: unknown;
  try {
    body = await readJsonCommand(request);
  } catch (error) {
    const invalid = commandError(error);
    return NextResponse.json({ error: invalid?.error || 'Invalid JSON body' }, { status: invalid?.status || 400 });
  }

  if (!isRecord(body)) {
    return NextResponse.json({ error: 'Missing request body' }, { status: 400 });
  }

  try {
    const jobConfig = body.job_config ?? body.jobConfig;
    if (!isAdvisorJobConfig(jobConfig)) {
      return NextResponse.json({ error: 'Invalid job config' }, { status: 400 });
    }

    const gpuIds = body.gpu_ids ?? body.gpuIds ?? null;
    const workerId = body.worker_id ?? 'local';
    if ((gpuIds !== null && typeof gpuIds !== 'string') || typeof workerId !== 'string' || !workerId || workerId.length > 180)
      return NextResponse.json({ error: 'Invalid worker or GPU assignment' }, { status: 400 });
    return NextResponse.json(await runTrainingPreflight(jobConfig, gpuIds, workerId));
  } catch (error) {
    console.error('Error running training advisor preflight:', error);
    return NextResponse.json({ error: 'Failed to run training advisor' }, { status: 500 });
  }
}
