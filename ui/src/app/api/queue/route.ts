import { NextResponse } from 'next/server';
import { db } from '@/server/db';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);

  try {
    const queues = await db.queues.list('gpu_ids');
    return NextResponse.json({ queues: queues });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch queue' }, { status: 500 });
  }
}
