import { NextResponse } from 'next/server';
import { getJobDashboard } from '@/server/jobDashboard';

export async function GET(_request: Request, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  const dashboard = await getJobDashboard(jobID);
  if (!dashboard) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }
  return NextResponse.json(dashboard);
}
