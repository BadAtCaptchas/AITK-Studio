import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import fsp from 'fs/promises';
import { createRemoteTrainingJobBundle } from '@/server/trainingJobBundle';
import {
  getRemoteWorker,
  isLocalWorker,
  remoteJson,
  syncRemoteJob,
  uploadBundleToWorker,
} from '@/server/remoteClient';
import {
  getEncryptedDatasetsForJobConfig,
} from '@/server/encryptedDatasets';
import {
  getEncryptedKeyCoverage,
  isDurableEncryptedDatasetKeySecretError,
  storeDurableEncryptedDatasetKeys,
} from '@/server/encryptedDatasetSecrets';
import { isSecureRemoteOllamaCaptionJob } from '@/server/secureRemoteCaptionJobs';
import { startJobNow } from '../../../../../../cron/actions/startJob';
import type { EncryptedDatasetStartKey, JobStartRequest } from '@/types';

function ensureApiAccess(request: NextRequest): NextResponse | null {
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

function isValidJobId(jobID: string) {
  return /^[a-zA-Z0-9_-]+$/.test(jobID);
}

function isSecureRemoteOllamaCaptionJobConfigJson(jobConfigJson: unknown) {
  if (typeof jobConfigJson !== 'string' || !jobConfigJson.trim()) return false;
  try {
    return isSecureRemoteOllamaCaptionJob(JSON.parse(jobConfigJson));
  } catch {
    return false;
  }
}

async function handleStart(
  request: NextRequest,
  { params }: { params: { jobID: string } },
  encryptedDatasetKeys?: EncryptedDatasetStartKey[],
  durableEncryptedDatasetKeys = false,
) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) {
    return accessResponse;
  }

  const { jobID } = await params;

  if (!isValidJobId(jobID)) {
    return NextResponse.json({ error: 'Invalid job ID' }, { status: 400 });
  }

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  let jobConfig: any;
  try {
    jobConfig = JSON.parse(job.job_config);
  } catch {
    return NextResponse.json({ error: 'Invalid job config' }, { status: 400 });
  }

  const requiredEncryptedDatasets = await getEncryptedDatasetsForJobConfig(jobConfig);
  let encryptedKeyCoverage = await getEncryptedKeyCoverage(jobID, requiredEncryptedDatasets, encryptedDatasetKeys);
  if (encryptedKeyCoverage.missingDatasets.length > 0) {
    return NextResponse.json(
      {
        error: 'decryption key required',
        encryptedDatasets: encryptedKeyCoverage.missingDatasets,
      },
      { status: 409 },
    );
  }
  let encryptedKeysForLaunch = encryptedKeyCoverage.combinedKeys;
  let useDurableEncryptedKeys = requiredEncryptedDatasets.length > 0 && encryptedKeyCoverage.durableKeys.length > 0;
  if (durableEncryptedDatasetKeys && requiredEncryptedDatasets.length > 0) {
    try {
      await storeDurableEncryptedDatasetKeys(jobID, encryptedKeysForLaunch);
    } catch (error) {
      if (isDurableEncryptedDatasetKeySecretError(error)) {
        return NextResponse.json({ error: error.message }, { status: 400 });
      }
      throw error;
    }
    encryptedKeyCoverage = await getEncryptedKeyCoverage(jobID, requiredEncryptedDatasets, encryptedKeysForLaunch);
    encryptedKeysForLaunch = encryptedKeyCoverage.combinedKeys;
    useDurableEncryptedKeys = true;
  }

  if (!isLocalWorker(job.worker_id)) {
    try {
      const worker = await getRemoteWorker(job.worker_id);
      if (
        requiredEncryptedDatasets.length > 0 &&
        !worker.base_url.toLowerCase().startsWith('https://') &&
        process.env.AITK_ALLOW_INSECURE_REMOTE_ENCRYPTED_DATASETS !== '1'
      ) {
        return NextResponse.json(
          { error: 'Remote encrypted training requires an HTTPS worker URL.' },
          { status: 400 },
        );
      }
      let remoteJobId = job.remote_job_id;

      if (!remoteJobId) {
        const bundle = await createRemoteTrainingJobBundle(jobID, { includeDatasets: true, checkpointMode: 'all' });
        try {
          const imported = await uploadBundleToWorker(worker, bundle.zipPath, job.gpu_ids);
          remoteJobId = imported.job.id;
          await db.jobs.update(jobID, {
            name: imported.job.name,
            gpu_ids: imported.job.gpu_ids,
            job_config: imported.job.job_config,
            remote_job_id: imported.job.id,
            remote_error: [...bundle.warnings, ...(imported.warnings || [])].join('\n') || null,
            remote_sync_at: new Date(),
          });
        } finally {
          await fsp.rm(bundle.zipPath, { force: true }).catch(() => undefined);
        }
      }

      await remoteJson(worker, `/api/jobs/${encodeURIComponent(remoteJobId)}/start`, {
        method: 'POST',
        body: JSON.stringify({
          encryptedDatasetKeys: requiredEncryptedDatasets.length > 0 ? encryptedKeysForLaunch : undefined,
          durableEncryptedDatasetKeys: useDurableEncryptedKeys,
        }),
      });
      await remoteJson(worker, `/api/queue/${encodeURIComponent(job.gpu_ids)}/start`);
      await db.queues
        .findByGpuIds(job.gpu_ids, job.worker_id)
        .then(queue =>
          queue
            ? db.queues.update(queue.id, { is_running: true })
            : db.queues.create({ worker_id: job.worker_id, gpu_ids: job.gpu_ids, is_running: true }),
        );
      const synced = await syncRemoteJob({
        ...(await db.jobs.findById(jobID))!,
        remote_job_id: remoteJobId,
      });
      return NextResponse.json(synced);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to start remote job';
      await db.jobs.update(jobID, { remote_error: message, remote_sync_at: new Date() }).catch(() => undefined);
      return NextResponse.json({ error: message }, { status: 502 });
    }
  }

  const queueLocalJob = async () => {
    const newQueuePosition = (await db.jobs.maxQueuePosition()) + 1000;

    await db.jobs.update(jobID, { queue_position: newQueuePosition });

    const queue = await db.queues.findByGpuIds(job.gpu_ids);

    if (!queue) {
      await db.queues.create({
        gpu_ids: job.gpu_ids,
        is_running: false,
      });
    }

    await db.jobs.update(jobID, {
      status: 'queued',
      stop: false,
      return_to_queue: false,
      info: 'Job queued',
    });

    return (await db.jobs.findById(jobID)) || job;
  };

  if (isSecureRemoteOllamaCaptionJob(jobConfig)) {
    await startJobNow(jobID, {
      encryptedDatasetKeys: requiredEncryptedDatasets.length > 0 ? encryptedKeysForLaunch : undefined,
    });
    return NextResponse.json((await db.jobs.findById(jobID)) || job);
  }

  if (requiredEncryptedDatasets.length > 0 && useDurableEncryptedKeys) {
    return NextResponse.json(await queueLocalJob());
  }

  if (requiredEncryptedDatasets.length > 0) {
    const runningJobs = await db.jobs.list({
      status: ['running', 'stopping'],
      gpu_ids: job.gpu_ids,
      worker_id: 'local',
    });
    const runningJob = runningJobs.find(
      candidate => candidate.id !== job.id && !isSecureRemoteOllamaCaptionJobConfigJson(candidate.job_config),
    );
    if (runningJob && runningJob.id !== job.id) {
      return NextResponse.json(
        { error: 'Encrypted jobs must start immediately; the selected local GPU is busy.' },
        { status: 409 },
      );
    }

    await startJobNow(jobID, { encryptedDatasetKeys: encryptedKeysForLaunch });
    return NextResponse.json((await db.jobs.findById(jobID)) || job);
  }

  return NextResponse.json(await queueLocalJob());
}

export async function GET(request: NextRequest, context: { params: { jobID: string } }) {
  return handleStart(request, context);
}

export async function POST(request: NextRequest, context: { params: { jobID: string } }) {
  let body: JobStartRequest = {};
  try {
    body = await request.json();
  } catch {
    body = {};
  }
  return handleStart(request, context, body.encryptedDatasetKeys, body.durableEncryptedDatasetKeys === true);
}
