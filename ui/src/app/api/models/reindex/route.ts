import { NextResponse } from 'next/server';
import { reindexModelArtifacts } from '@/server/modelArtifacts';

export async function POST() {
  const artifacts = await reindexModelArtifacts();
  return NextResponse.json({ artifacts, count: artifacts.length });
}
