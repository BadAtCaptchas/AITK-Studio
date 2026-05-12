import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import { getTrainingFolder } from '@/server/settings';
import { db } from '@/server/db';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';

const MAX_LOG_BYTES = 200 * 1024;

export async function GET(request: NextRequest, { params }: { params: { jobID: string } }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      return NextResponse.json({ log: '' });
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      return NextResponse.json(await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/log`));
    } catch (error) {
      console.error('Error reading remote log file:', error);
      return NextResponse.json({ error: 'Error reading remote log file' }, { status: 502 });
    }
  }

  try {
    const trainingFolder = await getTrainingFolder();
    const trainingFolderRealPath = await fs.promises.realpath(trainingFolder);
    const logPath = path.resolve(trainingFolderRealPath, job.name, 'log.txt');
    const relativePath = path.relative(trainingFolderRealPath, logPath);
    const isPathOutsideTrainingFolder = relativePath.startsWith('..') || path.isAbsolute(relativePath);

    if (isPathOutsideTrainingFolder) {
      return NextResponse.json({ error: 'Invalid job path' }, { status: 400 });
    }

    if (!fs.existsSync(logPath)) {
      return NextResponse.json({ log: '' });
    }

    const { size } = await fs.promises.stat(logPath);
    const bytesToRead = Math.min(size, MAX_LOG_BYTES);
    const buffer = Buffer.alloc(bytesToRead);
    const start = Math.max(0, size - bytesToRead);
    let fileHandle: fs.promises.FileHandle | undefined;
    let bytesRead = 0;
    try {
      fileHandle = await fs.promises.open(logPath, 'r');
      const readResult = await fileHandle.read(buffer, 0, bytesToRead, start);
      bytesRead = readResult.bytesRead;
    } finally {
      await fileHandle?.close();
    }

    const log = buffer.subarray(0, bytesRead).toString('utf-8');
    return NextResponse.json({ log });
  } catch (error) {
    console.error('Error reading log file:', error);
    return NextResponse.json({ error: 'Error reading log file' }, { status: 500 });
  }
}
