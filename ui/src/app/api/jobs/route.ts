import { NextResponse } from 'next/server';
import { isMac } from '@/helpers/basic';
import { db } from '@/server/db';
import { withComfyInstallProgress } from '@/server/comfyInstallProgress';
import { withHFDownloadProgress } from '@/server/hfDownloadProgress';
import { reconcileLocalJobProcess } from '@/server/jobProcess';
import {
  getRemoteWorker,
  isLocalWorker,
  remoteJson,
  syncRemoteJob,
} from '@/server/remoteClient';
import { listJobsForJobsApi } from '@/server/jobsApiList';
import { rewriteSameWorkerRemoteDatasetRefsForWorker } from '@/server/remoteDatasetPaths';
import { syncRemoteCaptionResultForJob } from '@/server/remoteCaptionResults';
import { isDirectRemoteOllamaCaptionJob } from '@/server/secureRemoteCaptionJobs';
import { prepareJobConfigForProject, resolveOptionalProject } from '@/server/projects';
import {
  assertProjectsEnabled,
  isProjectSpacesDisabledError,
  PROJECT_SPACES_DISABLED_MESSAGE,
} from '@/server/settings';
import type { Job } from '@/types';


function ensureApiAccess(request: Request): NextResponse | null {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) {
    return null;
  }

  const token = request.headers.get('authorization')?.split(' ')[1];
  if (token !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  return null;
}

function hasForbiddenOpenRouterFields(value: unknown) {
  if (!value || typeof value !== 'object') {
    return false;
  }

  return 'api_key_env' in value || 'base_url' in value;
}

function isSafeJobConfig(jobConfig: unknown) {
  if (!jobConfig || typeof jobConfig !== 'object') {
    return false;
  }

  const config = (jobConfig as Record<string, unknown>).config;
  if (!config || typeof config !== 'object') {
    return false;
  }

  const processList = (config as Record<string, unknown>).process;
  if (!Array.isArray(processList) || processList.length === 0) {
    return false;
  }

  return processList.every(processConfig => {
    if (!processConfig || typeof processConfig !== 'object') {
      return false;
    }

    const processRecord = processConfig as Record<string, unknown>;
    if (processRecord.type !== 'OpenRouterCaptioner') {
      return true;
    }

    return (
      !hasForbiddenOpenRouterFields(processRecord) &&
      !hasForbiddenOpenRouterFields(processRecord.caption)
    );
  });
}

function isValidGpuIds(gpuIds: unknown) {
  if (typeof gpuIds !== 'string' || gpuIds.trim().length === 0) {
    return false;
  }

  if (gpuIds === 'mps') {
    return true;
  }

  return /^\d+(,\d+)*$/.test(gpuIds);
}

function normalizeWorkerId(value: unknown) {
  return typeof value === 'string' && value.trim() ? value.trim() : 'local';
}

function duplicateJobNameError(projectID: string | null) {
  return projectID
    ? 'A run with this name already exists in this project.'
    : 'A global run with this name already exists.';
}

function isValidJobName(name: unknown) {
  if (typeof name !== 'string' || name.trim().length === 0) {
    return false;
  }

  if (name === '.' || name.includes('..')) {
    return false;
  }

  return name === name.split('/').pop() && name === name.split('\\').pop();
}

async function withJobProgress(job: Job) {
  return withComfyInstallProgress(await withHFDownloadProgress(job));
}

async function assertProjectJobVisible(job: Job | null) {
  if (job?.project_id) {
    await assertProjectsEnabled();
  }
}

export async function GET(request: Request) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) {
    return accessResponse;
  }

  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');
  const job_ref = searchParams.get('job_ref');
  const job_type = searchParams.get('job_type');
  const projectParam = searchParams.get('project_id');
  const localOnly = searchParams.get('local_only') === '1';
  const includeProjectActive = searchParams.get('include_project_active') === '1';

  try {
    const project = await resolveOptionalProject(projectParam);
    if (id) {
      const job = await db.jobs.findById(id);
      await assertProjectJobVisible(job);
      if (job && !isLocalWorker(job.worker_id)) {
        const synced = await syncRemoteJob(job);
        const captionSynced = await syncRemoteCaptionResultForJob(synced);
        return NextResponse.json(await withJobProgress(captionSynced));
      }
      const reconciled = await reconcileLocalJobProcess(job);
      return NextResponse.json(reconciled ? await withJobProgress(reconciled) : reconciled);
    }
    if (job_ref) {
      const job = await db.jobs.findLatestByRef(job_ref);
      await assertProjectJobVisible(job);
      if (job && !isLocalWorker(job.worker_id)) {
        const synced = await syncRemoteJob(job);
        const captionSynced = await syncRemoteCaptionResultForJob(synced);
        return NextResponse.json(await withJobProgress(captionSynced));
      }
      const reconciled = await reconcileLocalJobProcess(job);
      return NextResponse.json(reconciled ? await withJobProgress(reconciled) : reconciled);
    }

    const jobs = await listJobsForJobsApi({
      jobType: job_type,
      localOnly,
      projectID: project?.id || null,
      includeProjectActive,
    });
    const reconciledJobs = (await Promise.all(jobs.map(job => reconcileLocalJobProcess(job)))).filter(
      (job): job is Job => job !== null,
    );
    const resultSyncedJobs = await Promise.all(reconciledJobs.map(job => syncRemoteCaptionResultForJob(job)));
    return NextResponse.json({
      jobs: await Promise.all(resultSyncedJobs.map(job => withJobProgress(job))),
    });
  } catch (error) {
    if (isProjectSpacesDisabledError(error)) {
      return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
    }
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch training data' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) {
    return accessResponse;
  }

  try {
    const body = await request.json();
    const { id, name, job_config } = body;
    const project = await resolveOptionalProject(body.project_id);
    const projectJobConfig = project ? await prepareJobConfigForProject(job_config, project) : job_config;
    let worker_id = normalizeWorkerId(body.worker_id);
    if (isDirectRemoteOllamaCaptionJob(projectJobConfig)) {
      worker_id = 'local';
    }

    if (!isValidJobName(name)) {
      return NextResponse.json({ error: 'Invalid job name' }, { status: 400 });
    }
    let gpu_ids: string = body.gpu_ids;

    if (isMac()) {
      gpu_ids = 'mps';
    }

    if (!isValidGpuIds(gpu_ids)) {
      return NextResponse.json({ error: 'Invalid gpu_ids value' }, { status: 400 });
    }

    if (!isLocalWorker(worker_id)) {
      const worker = await db.workerNodes.findById(worker_id);
      if (!worker) {
        return NextResponse.json({ error: 'Worker not found' }, { status: 400 });
      }
      if (!worker.enabled) {
        return NextResponse.json({ error: 'Worker is disabled' }, { status: 400 });
      }
    }

    if (!isSafeJobConfig(projectJobConfig)) {
      return NextResponse.json({ error: 'Invalid job config' }, { status: 400 });
    }

    const extra: any = {};
    if ("job_ref" in body) {
      extra["job_ref"] = body.job_ref;
    }

    if ("job_type" in body) {
      extra["job_type"] = body.job_type;
    }

    if (id && typeof id !== 'string') {
      return NextResponse.json({ error: 'Invalid job id' }, { status: 400 });
    }

    if (id) {
      // Update existing training
      const existing = await db.jobs.findById(id);
      if (!existing) {
        return NextResponse.json({ error: 'Job not found' }, { status: 404 });
      }
      await assertProjectJobVisible(existing);

      const targetProjectID = project?.id || existing.project_id || null;
      const duplicateJob = await db.jobs.findByNameInScope(name, targetProjectID);
      if (duplicateJob && duplicateJob.id !== id) {
        return NextResponse.json({ error: duplicateJobNameError(targetProjectID) }, { status: 409 });
      }

      const workerChanged = existing.worker_id !== worker_id;
      let remotePatch: any = {};
      if (!workerChanged && !isLocalWorker(worker_id) && existing.remote_job_id) {
        const worker = await getRemoteWorker(worker_id);
        const remoteJobConfig = await rewriteSameWorkerRemoteDatasetRefsForWorker(projectJobConfig, worker);
        const remoteJob = await remoteJson<any>(worker, '/api/jobs', {
          method: 'POST',
          body: JSON.stringify({
            id: existing.remote_job_id,
            name,
            gpu_ids,
            job_config: remoteJobConfig,
            ...extra,
          }),
        });
        remotePatch = {
          name: remoteJob.name,
          gpu_ids: remoteJob.gpu_ids,
          remote_sync_at: new Date(),
          remote_error: null,
        };
      }

      const training = await db.jobs.update(id, {
        name,
        project_id: targetProjectID,
        worker_id,
        remote_job_id: workerChanged ? null : existing.remote_job_id,
        remote_error: workerChanged ? null : existing.remote_error,
        gpu_ids,
        job_config: JSON.stringify(projectJobConfig),
        ...extra,
        ...remotePatch,
      });
      return NextResponse.json(training);
    } else {
      // find the highest queue position and add 1000
      const newQueuePosition = (await db.jobs.maxQueuePosition()) + 1000;
      const targetProjectID = project?.id || null;
      const duplicateJob = await db.jobs.findByNameInScope(name, targetProjectID);
      if (duplicateJob) {
        return NextResponse.json({ error: duplicateJobNameError(targetProjectID) }, { status: 409 });
      }

      // Create new training
      const training = await db.jobs.create({
        name,
        project_id: targetProjectID,
        worker_id,
        gpu_ids,
        job_config: JSON.stringify(projectJobConfig),
        queue_position: newQueuePosition,
        ...extra,
      });
      return NextResponse.json(training);
    }
  } catch (error: any) {
    if (isProjectSpacesDisabledError(error)) {
      return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
    }
    if (error.code === 'P2002') {
      // Handle unique constraint violation, 409=Conflict
      return NextResponse.json({ error: 'Job name already exists in this workspace' }, { status: 409 });
    }
    console.error(error);
    // Handle other errors
    return NextResponse.json({ error: 'Failed to save training data' }, { status: 500 });
  }
}
