import { db } from '@/server/db';
import { getHFDownloadProgress } from '@/server/hfDownloadProgress';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';

export async function GET(_request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const job = await db.jobs.findById(jobID);

  if (!job) {
    return Response.json({ error: 'Job not found' }, { status: 404 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) return Response.json({ progress: null });
    try {
      const worker = await getRemoteWorker(job.worker_id);
      return Response.json(
        await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/hf-download-progress`),
      );
    } catch {
      return Response.json({ progress: null });
    }
  }

  return Response.json({ progress: await getHFDownloadProgress(job) });
}
