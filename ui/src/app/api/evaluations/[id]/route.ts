import { NextResponse } from 'next/server';
import { db } from '@/server/db';

export async function GET(_request: Request, { params }: { params: { id: string } }) {
  const { id } = await params;
  const runs = await db.evaluations.listRuns({ limit: 1000 });
  const run = runs.find(item => item.id === id);
  if (!run) return NextResponse.json({ error: 'Evaluation not found' }, { status: 404 });
  const items = await db.evaluations.listItems(id);
  return NextResponse.json({ run, items });
}
