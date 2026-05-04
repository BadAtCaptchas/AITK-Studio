import { NextResponse } from 'next/server';
import { db } from '@/server/db';

export async function GET(_request: Request, { params }: { params: { id: string } }) {
  const { id } = await params;
  const artifact = await db.modelArtifacts.findById(id);
  if (!artifact) return NextResponse.json({ error: 'Model artifact not found' }, { status: 404 });
  return NextResponse.json(artifact);
}
