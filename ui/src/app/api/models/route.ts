import { NextResponse } from 'next/server';
import { listModelArtifacts } from '@/server/modelArtifacts';

export async function GET() {
  const artifacts = await listModelArtifacts();
  return NextResponse.json({ artifacts });
}
