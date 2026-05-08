import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';

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

function isValidJobId(jobID: string) {
  return /^[a-zA-Z0-9_-]+$/.test(jobID);
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) {
    return accessResponse;
  }

  const { jobID } = await params;

  if (!isValidJobId(jobID)) {
    return NextResponse.json({ error: 'Invalid job ID' }, { status: 400 });
  }

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  // get highest queue position
  const newQueuePosition = (await db.jobs.maxQueuePosition()) + 1000;

  await db.jobs.update(jobID, { queue_position: newQueuePosition });

  // make sure the queue is running
  const queue = await db.queues.findByGpuIds(job.gpu_ids);

  // if queue doesn't exist, create it
  if (!queue) {
    await db.queues.create({
      gpu_ids: job.gpu_ids,
      is_running: false,
    });
  }

  await db.jobs.update(jobID, {
    status: 'queued',
    stop: false,
    return_to_queue: false,
    info: 'Job queued',
  });

  // Return the response immediately
  return NextResponse.json(job);
}
