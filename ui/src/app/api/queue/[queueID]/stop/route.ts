import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';

export async function GET(request: NextRequest, { params }: { params: { queueID: string } }) {
  const { queueID } = await params;

  const queue = await db.queues.findByGpuIds(queueID);

  if (!queue) {
    return NextResponse.json({ error: 'Queue not found' }, { status: 404 });
  }

  await db.queues.update(queue.id, { is_running: false });

  return NextResponse.json(queue);
}
