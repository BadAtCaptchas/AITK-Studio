import { NextResponse } from 'next/server';
import { getJobDashboard, isTransientDbError } from '@/server/jobDashboard';

export async function GET(_request: Request, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;
  try {
    const dashboard = await getJobDashboard(jobID);
    if (!dashboard) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }
    return NextResponse.json(dashboard);
  } catch (error) {
    if (isTransientDbError(error)) {
      return NextResponse.json({ error: 'Database busy, retry shortly' }, { status: 503 });
    }
    throw error;
  }
}
