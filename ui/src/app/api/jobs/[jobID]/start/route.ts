import { commandError, parseJobStartCommand, readJsonCommand } from '@/server/commandInput';
import { JobStartError, startJobFromRequest } from '@/server/jobStart';
import { db } from '@/server/db';
import { isLocalWorker } from '@/server/remoteClient';
import { acceptRemoteStart } from '@/server/remoteStartOperation';
import { remoteStartProgress } from '@/server/remoteStartProgress';
import { isRequestAuthenticated } from '@/utils/authSession';
import { idempotentStart } from '@/server/startCommand';

export function GET() { return Response.json({ error: 'Use POST to start a job' }, { status: 405, headers: { Allow: 'POST' } }); }
export async function POST(request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  if (!await isRequestAuthenticated(request, process.env.AI_TOOLKIT_AUTH)) return Response.json({ error: 'Unauthorized' }, { status: 401 });
  try {
    const body = parseJobStartCommand(await readJsonCommand(request, { allowEmpty: true }));
    const { jobID } = await params;
    const job = await db.jobs.findById(jobID);
    if (job && !isLocalWorker(job.worker_id)) {
      const operation = await acceptRemoteStart(jobID, body);
      return Response.json({ startID: operation.id, statusUrl: `/api/jobs/${jobID}/start-progress/${operation.id}`, progress: remoteStartProgress(operation) }, { status: 202 });
    }
    return Response.json(await idempotentStart(jobID, body.idempotencyKey, () => startJobFromRequest(jobID, body.encryptedDatasetKeys, body.durableEncryptedDatasetKeys)));
  } catch (error) {
    if (error instanceof JobStartError) return Response.json(error.payload, { status: error.status });
    const invalid = commandError(error);
    if (invalid) return Response.json({ error: invalid.error, code: invalid.code }, { status: invalid.status });
    throw error;
  }
}
