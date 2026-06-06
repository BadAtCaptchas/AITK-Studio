import { NextRequest, NextResponse } from 'next/server';
import { JobStartError, startJobFromRequest } from '@/server/jobStart';
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
