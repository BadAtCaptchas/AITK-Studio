import { jobStorageKey } from '../../../../../utils/jobIdentity';
import path from 'path';
import { analyzeTrainingAdvisor } from '@/server/trainingAdvisor';
import { isAdvisorJobConfig, runTrainingPreflight } from '@/server/trainingPreflight';
import { db } from '@/server/db';
import { getJobTrainingRoot } from '@/server/trainingPaths';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import type { JobConfig } from '@/types';
import { isRequestAuthenticated } from '@/utils/authSession';


const ADVISOR_METRIC_KEYS = ['loss*', 'learning_rate*', 'lr*', 'phase/*', 'event/*', 'train/*'];

async function ensureApiAccess(request: Request): Promise<Response | null> {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) return null;

  if (!(await isRequestAuthenticated(request, tokenToUse))) {
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  }
  return null;
}

function isValidJobId(jobID: string) {
  return /^[a-zA-Z0-9_-]+$/.test(jobID);
}

function parseJobConfig(raw: string): JobConfig | null {
  try {
    const value: unknown = JSON.parse(raw);
    return isAdvisorJobConfig(value) ? value : null;
  } catch {
    return null;
  }
}

export async function GET(request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  const accessResponse = await ensureApiAccess(request);
  if (accessResponse) return accessResponse;

  const { jobID } = await params;
  if (!isValidJobId(jobID)) {
    return Response.json({ error: 'Invalid job ID' }, { status: 400 });
  }

  const job = await db.jobs.findById(jobID);
  if (!job) {
    return Response.json({ error: 'Job not found' }, { status: 404 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      const jobConfig = parseJobConfig(job.job_config);
      if (!jobConfig) return Response.json({ error: 'Invalid job config' }, { status: 400 });
      return Response.json(await runTrainingPreflight(jobConfig, job.gpu_ids, job.worker_id));
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      return Response.json(await remoteJson(worker, `/api/jobs/${encodeURIComponent(job.remote_job_id)}/advisor`));
    } catch (error) {
      console.error('Error reading remote training advisor:', error);
      return Response.json({ error: 'Error reading remote training advisor' }, { status: 502 });
    }
  }

  const jobConfig = parseJobConfig(job.job_config);
  if (!jobConfig) return Response.json({ error: 'Invalid job config' }, { status: 400 });

  const trainingFolder = await getJobTrainingRoot(job);
  const logPath = path.join(trainingFolder, jobStorageKey(job), 'loss_log.db');
  const metrics = await db.metrics.getMetrics(jobID, logPath, {
    keys: ADVISOR_METRIC_KEYS,
    maxPoints: 5000,
    sinceStep: null,
  });

  return Response.json(
    analyzeTrainingAdvisor(jobConfig, {
      gpuIds: job.gpu_ids,
      job,
      metrics,
      scanDatasets: true,
    }),
  );
}
