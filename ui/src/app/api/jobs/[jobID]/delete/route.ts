import { NextRequest, NextResponse } from 'next/server';
import { getTrainingFolder } from '@/server/settings';
import path from 'path';
import fs from 'fs';
import { db } from '@/server/db';

function resolveWithinRoot(root: string, target: string) {
  const resolvedRoot = path.resolve(root);
  const resolvedPath = path.resolve(resolvedRoot, target);
  const relativePath = path.relative(resolvedRoot, resolvedPath);

  if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    return null;
  }

  return resolvedPath;
}

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingRoot = await getTrainingFolder();
  const trainingFolder = resolveWithinRoot(trainingRoot, job.name);

  if (!trainingFolder) {
    return NextResponse.json({ error: 'Invalid job path' }, { status: 400 });
  }

  if (fs.existsSync(trainingFolder)) {
    fs.rmSync(trainingFolder, { recursive: true, force: true });
  }

  await db.jobs.delete(jobID);

  return NextResponse.json(job);
}
