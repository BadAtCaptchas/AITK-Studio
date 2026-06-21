import { NextResponse } from 'next/server';
import { listQueuesForQueueApi } from '@/server/queueSync';

export async function GET() {
  try {
    const queues = await listQueuesForQueueApi();
    return NextResponse.json({ queues });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch queue' }, { status: 500 });
  }
}
