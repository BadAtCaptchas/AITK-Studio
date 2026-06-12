import { NextRequest, NextResponse } from 'next/server';
import {
  JobStartError,
  prepareJobStart,
  startJobFromRequest,
  startPreparedJob,
} from '@/server/jobStart';
import { isLocalWorker } from '@/server/remoteClient';
import {
  createRemoteStartProgress,
  hasActiveRemoteStartForJob,
  updateRemoteStartProgress,
} from '@/server/remoteStartProgress';
import type { JobStartRequest } from '@/types';

function ensureApiAccess(request: NextRequest): NextResponse | null {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) {
    return null;
  }

  const token = request.headers.get('authorization')?.split(' ')[1];
  if (token !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  return null;
}

function handleJobStartError(error: unknown) {
  if (error instanceof JobStartError) {
    return NextResponse.json(error.payload, { status: error.status });
  }
  throw error;
}

function runBackgroundRemoteStart(startID: string, prepared: Awaited<ReturnType<typeof prepareJobStart>>) {
  void startPreparedJob(prepared, {
    onRemoteStartProgress: progress => updateRemoteStartProgress(startID, progress),
  }).catch(error => {
    const message = error instanceof Error ? error.message : 'Failed to start remote job';
    updateRemoteStartProgress(startID, {
      status: 'failed',
      message: 'Remote start failed',
      percent: 100,
      error: message,
    });
  });
}

async function handleStart(
  request: NextRequest,
  { params }: { params: Promise<{ jobID: string }> },
  body: JobStartRequest = {},
) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) {
    return accessResponse;
  }

  const { jobID } = await params;

  try {
    if (body.background === true) {
      const prepared = await prepareJobStart(
        jobID,
        body.encryptedDatasetKeys,
        body.durableEncryptedDatasetKeys === true,
      );
      if (!isLocalWorker(prepared.job.worker_id) && prepared.job.job_type === 'train') {
        if (hasActiveRemoteStartForJob(jobID)) {
          return NextResponse.json({ error: 'Remote start already in progress' }, { status: 409 });
        }
        const progress = createRemoteStartProgress(jobID);
        runBackgroundRemoteStart(progress.startID, prepared);
        return NextResponse.json({
          startID: progress.startID,
          statusUrl: `/api/jobs/${jobID}/start-progress/${progress.startID}`,
          progress,
        });
      }

      return NextResponse.json(await startPreparedJob(prepared));
    }

    return NextResponse.json(
      await startJobFromRequest(jobID, body.encryptedDatasetKeys, body.durableEncryptedDatasetKeys === true),
    );
  } catch (error) {
    return handleJobStartError(error);
  }
}

export async function GET(request: NextRequest, context: { params: Promise<{ jobID: string }> }) {
  return handleStart(request, context);
}

export async function POST(request: NextRequest, context: { params: Promise<{ jobID: string }> }) {
  let body: JobStartRequest = {};
  try {
    body = await request.json();
  } catch {
    body = {};
  }
  return handleStart(request, context, body);
}
