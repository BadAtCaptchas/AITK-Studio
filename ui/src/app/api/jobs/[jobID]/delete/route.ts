import { NextRequest, NextResponse } from 'next/server';
import { getTrainingFolder } from '@/server/settings';
import path from 'path';
import fs from 'fs';
import { db } from '@/server/db';

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const trainingRoot = await getTrainingFolder();
  const trainingFolder = path.join(trainingRoot, job.name);

  if (fs.existsSync(trainingFolder)) {
    fs.rmSync(trainingFolder, { recursive: true, force: true });
  }

  await db.jobs.delete(jobID);

  return NextResponse.json(job);
}
