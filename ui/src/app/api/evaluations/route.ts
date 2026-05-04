import { NextResponse } from 'next/server';
import { db } from '@/server/db';
import { createEvaluationRun } from '@/server/evaluations';

export async function GET() {
  const runs = await db.evaluations.listRuns({ limit: 200 });
  return NextResponse.json({ runs });
}

export async function POST(request: Request) {
  const body = await request.json();
  const jobIds = Array.isArray(body.jobIds) ? body.jobIds.map(String).filter(Boolean) : [];
  const artifactIds = Array.isArray(body.artifactIds) ? body.artifactIds.map(String).filter(Boolean) : [];
  if (jobIds.length === 0 && artifactIds.length === 0) {
    return NextResponse.json({ error: 'At least one job or artifact is required' }, { status: 400 });
  }
  const run = await createEvaluationRun({
    name: typeof body.name === 'string' ? body.name : undefined,
    jobIds,
    artifactIds,
    referencePath: typeof body.referencePath === 'string' && body.referencePath.trim() ? body.referencePath : null,
  });
  return NextResponse.json(run);
}
