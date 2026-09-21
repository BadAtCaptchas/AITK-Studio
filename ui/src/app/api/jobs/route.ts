import { configContractErrors } from '@/domain/configContract';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { readJsonCommand, commandError, CommandInputError, isRecord, withCommandBoundary } from '@/server/commandInput';
import { validJobName, deviceIds } from '@/utils/jobIdentity';
import type { JobUpdateInput } from '@/server/db';
import { NextResponse } from 'next/server';
import { isMac } from '@/helpers/basic';
import { db } from '@/server/db';
import { withComfyInstallProgress } from '@/server/comfyInstallProgress';
import { withHFDownloadProgress } from '@/server/hfDownloadProgress';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { listJobsForJobsApi } from '@/server/jobsApiList';
import { rewriteSameWorkerRemoteDatasetRefsForWorker } from '@/server/remoteDatasetPaths';

import type { Job } from '@/types';
import { isRequestAuthenticated } from '@/utils/authSession';
import { getJobValidationConfigErrors } from '@/utils/validationConfig';

async function ensureApiAccess(request: Request): Promise<NextResponse | null> {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) {
    return null;
  }

  if (!(await isRequestAuthenticated(request, tokenToUse))) {
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

    return !hasForbiddenOpenRouterFields(processRecord) && !hasForbiddenOpenRouterFields(processRecord.caption);
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

function duplicateJobNameError() {
  return 'A run with this name already exists.';
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

export async function GET(request: Request) {
  const accessResponse = await ensureApiAccess(request);
  if (accessResponse) {
    return accessResponse;
  }

  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');
  const job_ref = searchParams.get('job_ref');
  const job_type = searchParams.get('job_type');
  const localOnly = searchParams.get('local_only') === '1';

  try {
    if (id) {
      const job = await db.jobs.findById(id);
      return NextResponse.json(job ? await withJobProgress(job) : null);
    }
    if (job_ref) {
      const job = await db.jobs.findLatestByRef(job_ref, job_type);
      return NextResponse.json(job ? await withJobProgress(job) : null);
    }

    const page = await listJobsForJobsApi({ jobType: job_type, localOnly, cursor: searchParams.get('cursor'), limit: searchParams.get('limit'), view: searchParams.get('view'), summary: searchParams.get('summary') !== '0' });
    return NextResponse.json(page);
  } catch (error) {
    console.error(error);
    const detail = commandError(error);
    return NextResponse.json(detail || { error: 'Failed to fetch training data' }, { status: detail?.status || 500 });
  }
}

async function postCommand(request: Request) {
  const accessResponse = await ensureApiAccess(request);
  if (accessResponse) {
    return accessResponse;
  }

  try {
    const body = await readJsonCommand(request);
    const { id, name, job_config } = body;
    const resolvedJobConfig = isRecord(job_config) ? { ...job_config, capability_version: job_config.capability_version ?? 1 } : job_config;
    const worker_id = normalizeWorkerId(body.worker_id);

    if (!validJobName(name)) {
      return NextResponse.json({ error: 'Invalid job name' }, { status: 400 });
    }
    let gpu_ids: string;
    try { gpu_ids = deviceIds(body.gpu_ids).join(','); }
    catch { throw new CommandInputError('Invalid device selection'); }

    if (isMac() && isLocalWorker(worker_id)) {
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

    if (!isSafeJobConfig(resolvedJobConfig)) {
      return NextResponse.json({ error: 'Invalid job config' }, { status: 400 });
    }
    const validationConfigErrors = [...configContractErrors(resolvedJobConfig), ...getJobValidationConfigErrors(resolvedJobConfig)];
    if (validationConfigErrors.length > 0) {
      return NextResponse.json(
        { error: validationConfigErrors[0], validation_errors: validationConfigErrors },
        { status: 400 },
      );
    }

    const extra: Pick<JobUpdateInput, 'job_ref' | 'job_type'> = {};
    if ('job_ref' in body) {
      if (body.job_ref !== null && typeof body.job_ref !== 'string') throw new CommandInputError('Invalid job_ref');
      extra['job_ref'] = body.job_ref;
    }

    if ('job_type' in body) {
      if (typeof body.job_type !== 'string' || !['train', 'caption', 'generate', 'inference'].includes(body.job_type)) throw new CommandInputError('Invalid job_type');
      extra['job_type'] = body.job_type;
    }

    const processes = isRecord(resolvedJobConfig) && isRecord(resolvedJobConfig.config) && Array.isArray(resolvedJobConfig.config.process) ? resolvedJobConfig.config.process : [];
    const hasEngine = processes.some((process: unknown) => isRecord(process) && process.type === 'InferenceEngine');
    if (hasEngine) {
      if (processes.length !== 1) throw new CommandInputError('An inference engine must be its own job');
      const process = processes[0];
      if (isRecord(process) && isRecord(process.engine) && 'token' in process.engine) throw new CommandInputError('Studio manages engine credentials; remove engine.token from the config');
      extra.job_type = 'inference';
    } else if (extra.job_type === 'inference') throw new CommandInputError('Inference jobs require an InferenceEngine process');

    if (id !== undefined && id !== null && typeof id !== 'string') {
      return NextResponse.json({ error: 'Invalid job id' }, { status: 400 });
    }

    if (id) {
      // Update existing training
      const existing = await db.jobs.findById(id);
      if (!existing) {
        return NextResponse.json({ error: 'Job not found' }, { status: 404 });
      }
      if (!['queued', 'stopped', 'error', 'completed'].includes(existing.status)) {
        return NextResponse.json({ error: 'Stop the active job before changing its configuration', code: 'JOB_ACTIVE' }, { status: 409 });
      }

      const duplicateJob = await db.jobs.findByName(name);
      if (duplicateJob && duplicateJob.id !== id) {
        return NextResponse.json({ error: duplicateJobNameError() }, { status: 409 });
      }

      const editing = await claimJobMaintenance(existing, 'editing');
      if (!editing) return NextResponse.json({ error: 'Job changed or its process has not exited. Refresh and retry.' }, { status: 409 });
      try {
      const workerChanged = existing.worker_id !== worker_id;
      let remotePatch: JobUpdateInput = {};
      if (!workerChanged && !isLocalWorker(worker_id) && existing.remote_job_id) {
        const worker = await getRemoteWorker(worker_id);
        const remoteJobConfig = await rewriteSameWorkerRemoteDatasetRefsForWorker(resolvedJobConfig, worker);
        const remoteJob = await remoteJson<unknown>(worker, '/api/jobs', {
          method: 'POST',
          body: JSON.stringify({
            id: existing.remote_job_id,
            name,
            gpu_ids,
            job_config: remoteJobConfig,
            ...extra,
          }),
        });
        if (!isRecord(remoteJob) || typeof remoteJob.name !== 'string' || typeof remoteJob.gpu_ids !== 'string') throw new Error('Invalid remote job response');
        remotePatch = {
          name: remoteJob.name,
          gpu_ids: remoteJob.gpu_ids,
          remote_sync_at: new Date(),
          remote_error: null,
        };
      }

      const training = await db.jobs.updateIf(id, { attempt_id: editing.attempt_id ?? null, status: 'editing' }, {
        status: existing.status,
        name,
        storage_key: existing.storage_key || existing.name,
        worker_id,
        remote_job_id: workerChanged ? null : existing.remote_job_id,
        remote_error: workerChanged ? null : existing.remote_error,
        gpu_ids,
        job_config: JSON.stringify(resolvedJobConfig),
        ...extra,
        ...remotePatch,
      });
      if (!training) return NextResponse.json({ error: 'Job changed while saving. Refresh and retry.' }, { status: 409 });
      return NextResponse.json(training);
      } catch (error) {
        await db.jobs.updateIf(id, { attempt_id: editing.attempt_id ?? null, status: 'editing' }, { status: existing.status });
        throw error;
      }
    } else {
      // find the highest queue position and add 1000
      const newQueuePosition = (await db.jobs.maxQueuePosition()) + 1000;
      const duplicateJob = await db.jobs.findByName(name);
      if (duplicateJob) {
        return NextResponse.json({ error: duplicateJobNameError() }, { status: 409 });
      }

      // Create new training
      const training = await db.jobs.create({
        name,
        worker_id,
        gpu_ids,
        job_config: JSON.stringify(resolvedJobConfig),
        queue_position: newQueuePosition,
        ...extra,
      });
      return NextResponse.json(training);
    }
  } catch (error: any) {
    const invalid = commandError(error);
    if (invalid) return NextResponse.json({ error: invalid.error, code: invalid.code }, { status: invalid.status });
    if (error.code === 'P2002') {
      // Handle unique constraint violation, 409=Conflict
      return NextResponse.json({ error: 'Job name already exists in this workspace' }, { status: 409 });
    }
    console.error(error);
    // Handle other errors
    return NextResponse.json({ error: 'Failed to save training data' }, { status: 500 });
  }
}

export const POST = withCommandBoundary(postCommand);
import { claimJobMaintenance } from '@/server/jobAttempts';
