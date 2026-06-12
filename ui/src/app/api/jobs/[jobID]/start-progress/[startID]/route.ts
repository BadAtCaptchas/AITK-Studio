import { NextRequest, NextResponse } from 'next/server';
import { getRemoteStartProgress } from '@/server/remoteStartProgress';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ jobID: string; startID: string }> },
) {
  const { jobID, startID } = await params;
  const progress = getRemoteStartProgress(startID);

  if (!progress || progress.jobID !== jobID) {
    return NextResponse.json({ error: 'Remote start progress not found' }, { status: 404 });
  }

  return NextResponse.json(progress);
}
