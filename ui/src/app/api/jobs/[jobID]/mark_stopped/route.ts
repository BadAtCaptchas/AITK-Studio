import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);

  // update job status to 'running'
  await db.jobs.update(jobID, {
    stop: true,
    status: 'stopped',
    info: 'Job stopped',
    pid: null,
  });

  console.log(`Job ${jobID} marked as stopped`);

  return NextResponse.json(job);
}
