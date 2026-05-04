import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

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
