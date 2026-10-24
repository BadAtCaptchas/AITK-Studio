import { getRemoteStartProgress } from '@/server/remoteStartProgress';


export async function GET(
  _request: Request,
  { params }: { params: Promise<{ jobID: string; startID: string }> },
) {
  const { jobID, startID } = await params;
  const progress = await getRemoteStartProgress(startID);

  if (!progress || progress.jobID !== jobID) {
    return Response.json({ error: 'Remote start progress not found' }, { status: 404 });
  }

  return Response.json(progress);
}
