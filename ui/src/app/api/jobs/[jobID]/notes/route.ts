import { db } from '@/server/db';
import { readJobNotes, writeJobNotes } from '@/server/jobNotes';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { isRequestAuthenticated } from '@/utils/authSession';
import { commandError, readJsonCommand, CommandInputError } from '@/server/commandInput';

async function notes(request: Request, { params }: { params: Promise<{ jobID: string }> }): Promise<Response> {
  if (!(await isRequestAuthenticated(request, process.env.AI_TOOLKIT_AUTH)))
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  try {
    const job = await db.jobs.findById((await params).jobID);
    if (!job) throw new CommandInputError('Job not found', 404);
    const body = request.method === 'POST' ? await readJsonCommand(request, { maxBytes: 1024 * 1024 }) : null;
    if (body && typeof body.notes !== 'string') throw new CommandInputError('Notes must be text');
    if (!isLocalWorker(job.worker_id)) {
      if (!job.remote_job_id) throw new CommandInputError('Remote job is not available', 409);
      const data = await remoteJson<unknown>(
        await getRemoteWorker(job.worker_id),
        `/api/jobs/${encodeURIComponent(job.remote_job_id)}/notes`,
        body ? { method: 'POST', body: JSON.stringify(body) } : {},
      );
      return Response.json(data);
    }
    if (body) {
      await writeJobNotes(job, String(body.notes));
      return Response.json({ saved: true });
    }
    return Response.json({ notes: await readJobNotes(job) });
  } catch (error) {
    const invalid = commandError(error);
    return Response.json({ error: invalid?.error || 'Could not access job notes' }, { status: invalid?.status || 500 });
  }
}
export const GET = notes;
export const POST = notes;
