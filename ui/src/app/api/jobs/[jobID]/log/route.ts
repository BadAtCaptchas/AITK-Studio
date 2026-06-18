import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import { db } from '@/server/db';
import { assertProjectJobEnabled, getJobTrainingRoot } from '@/server/projects';
import { getTrainingFolder } from '@/server/settings';
import {
  getRemoteWorker,
  isLocalWorker,
  isRemoteJobMissingError,
  markRemoteJobMissing,
  remoteJson,
} from '@/server/remoteClient';

const MAX_LOG_BYTES = 200 * 1024;
const LAUNCH_LOG_FILE = 'launch.log';

async function readTail(logPath: string) {
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

  return buffer.subarray(0, bytesRead).toString('utf-8');
}

async function launchLogBelongsToJob(launchLogPath: string, jobID: string) {
  if (!fs.existsSync(launchLogPath)) return false;
  const launchLog = await readTail(launchLogPath).catch(() => '');
  return launchLog.includes(`starting job ${jobID}`);
}

async function resolveReadableJobLogPath(
  trainingFolder: string,
  jobName: string,
  jobID: string,
  options: { requireLaunchJobMatch?: boolean } = {},
) {
  if (!fs.existsSync(trainingFolder)) return null;

  const trainingFolderRealPath = await fs.promises.realpath(trainingFolder);
  const jobFolder = path.resolve(trainingFolderRealPath, jobName);
  const relativePath = path.relative(trainingFolderRealPath, jobFolder);
  const isPathOutsideTrainingFolder = relativePath.startsWith('..') || path.isAbsolute(relativePath);

  if (isPathOutsideTrainingFolder) {
    throw new Error('Invalid job path');
  }

  const logPath = path.join(jobFolder, 'log.txt');
  const launchLogPath = path.join(jobFolder, LAUNCH_LOG_FILE);
  if (options.requireLaunchJobMatch && !(await launchLogBelongsToJob(launchLogPath, jobID))) {
    return null;
  }

  return fs.existsSync(logPath) ? logPath : fs.existsSync(launchLogPath) ? launchLogPath : null;
}

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }
  try {
    await assertProjectJobEnabled(job);
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Project spaces are disabled' }, { status: error?.status || 403 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      return NextResponse.json({ log: '' });
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      return NextResponse.json(await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/log`));
    } catch (error) {
      if (isRemoteJobMissingError(error)) {
        await markRemoteJobMissing(job);
        return NextResponse.json({ log: '' });
      }
      console.error('Error reading remote log file:', error);
      return NextResponse.json({ error: 'Error reading remote log file' }, { status: 502 });
    }
  }

  try {
    const trainingFolder = await getJobTrainingRoot(job);
    let readableLogPath = await resolveReadableJobLogPath(trainingFolder, job.name, jobID);
    if (!readableLogPath && job.project_id) {
      const globalTrainingFolder = await getTrainingFolder();
      if (path.resolve(globalTrainingFolder) !== path.resolve(trainingFolder)) {
        readableLogPath = await resolveReadableJobLogPath(globalTrainingFolder, job.name, jobID, {
          requireLaunchJobMatch: true,
        });
      }
    }
    if (!readableLogPath) {
      return NextResponse.json({ log: '' });
    }

    const log = await readTail(readableLogPath);
    return NextResponse.json({ log });
  } catch (error) {
    if (error instanceof Error && error.message === 'Invalid job path') {
      return NextResponse.json({ error: 'Invalid job path' }, { status: 400 });
    }
    console.error('Error reading log file:', error);
    return NextResponse.json({ error: 'Error reading log file' }, { status: 500 });
  }
}
