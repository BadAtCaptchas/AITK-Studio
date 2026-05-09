import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import { getHFDownloadProgress } from '@/server/hfDownloadProgress';

export async function GET(_request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  return NextResponse.json({ progress: await getHFDownloadProgress(job) });
}
