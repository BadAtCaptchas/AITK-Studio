import { NextRequest, NextResponse } from 'next/server';
import { JobStartError } from '@/server/jobStart';
import {
  restartTrainingJobFromScratch,
  TrainingJobRestartError,
} from '@/server/trainingJobRestart';
import type { JobStartRequest } from '@/types';
import { isRequestAuthenticated } from '@/utils/authSession';

async function ensureApiAccess(request: NextRequest): Promise<NextResponse | null> {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) {
    return null;
  }

  if (!(await isRequestAuthenticated(request, tokenToUse))) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  return null;
}

function handleRestartError(error: unknown) {
  if (error instanceof JobStartError || error instanceof TrainingJobRestartError) {
    return NextResponse.json(error.payload, { status: error.status });
  }
  throw error;
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const accessResponse = await ensureApiAccess(request);
  if (accessResponse) {
    return accessResponse;
  }

  let body: JobStartRequest = {};
  try {
    body = await request.json();
  } catch {
    body = {};
  }

  const { jobID } = await params;

  try {
    return NextResponse.json(
      await restartTrainingJobFromScratch(jobID, {
        encryptedDatasetKeys: body.encryptedDatasetKeys,
        durableEncryptedDatasetKeys: body.durableEncryptedDatasetKeys === true,
      }),
    );
  } catch (error) {
    return handleRestartError(error);
  }
}
