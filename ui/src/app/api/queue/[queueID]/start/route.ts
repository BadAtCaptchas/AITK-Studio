import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';

export async function GET(request: NextRequest, { params }: { params: { queueID: string } }) {
  const { queueID } = await params;

  const queue = await db.queues.findByGpuIds(queueID);

  if (!queue) {
    // create it if it doesn't exist
    const newQueue = await db.queues.create({ gpu_ids: queueID, is_running: true });
    return NextResponse.json(newQueue);
  }

  await db.queues.update(queue.id, { is_running: true });

  return NextResponse.json(queue);
}
