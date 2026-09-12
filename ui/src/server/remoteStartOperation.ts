import { createHash } from 'crypto';
import { db } from './db';
import { prepareJobStart, startPreparedJob, JobStartError } from './jobStart';
import {
  activeOperationForResource,
  enqueueOperation,
  getOperation,
  updateOperation,
  type Operation,
} from './operations';
import { rememberOperationKeys, fetchOperationKeys } from './operationKeys';
import type { JobStartRequest } from '../types';

function configHash(value: string): string {
  return createHash('sha256').update(value).digest('hex');
}
export async function acceptRemoteStart(jobID: string, request: JobStartRequest): Promise<Operation> {
  const existing = await activeOperationForResource(`job:${jobID}`);
  if (existing && existing.state !== 'needs-keys') return existing;
  const prepared = await prepareJobStart(jobID, request.encryptedDatasetKeys, request.durableEncryptedDatasetKeys, {
    operationID: existing?.id,
  });
  const operation =
    existing ||
    (await enqueueOperation(
      {
        kind: 'remote-start',
        jobID,
        durableKeys: prepared.useDurableEncryptedKeys,
        needsEphemeralKeys: prepared.encryptedKeysForLaunch.length > 0 && !prepared.useDurableEncryptedKeys,
        configHash: configHash(prepared.job.job_config),
      },
      request.idempotencyKey,
    ));
  if (['completed', 'failed', 'canceled'].includes(operation.state)) return operation;
  if (prepared.encryptedKeysForLaunch.length > 0 && !prepared.useDurableEncryptedKeys)
    rememberOperationKeys(operation.id, prepared.encryptedKeysForLaunch);
  const claimed = await db.jobs.updateIf(
    jobID,
    {
      attempt_id: prepared.job.attempt_id ?? null,
      status: prepared.job.status,
      updated_at: new Date(prepared.job.updated_at),
    },
    {
      attempt_id: operation.id,
      status: 'remote-starting',
      info: 'Remote start accepted',
      remote_error: null,
    },
  );
  if (!claimed && (await db.jobs.findById(jobID))?.attempt_id !== operation.id) {
    await updateOperation(operation.id, { state: 'failed', error: 'Job changed before remote start acceptance' });
    throw new JobStartError({ error: 'Job changed before remote start acceptance' }, 409);
  }
  await db.runtime.compareAndSwap(`operation-pending:${operation.id}`, null, operation.id);
  return updateOperation(operation.id, { state: 'queued', nextAttemptAt: 0, error: null });
}

export async function executeRemoteStart(operation: Operation): Promise<unknown> {
  const input = operation.input;
  if (input.kind !== 'remote-start') throw new Error('Not a remote start');
  let job = await db.jobs.findById(input.jobID);
  if (!job) throw new Error('Job was deleted');
  if (job.attempt_id !== operation.id) {
    if (configHash(job.job_config) !== input.configHash)
      throw new Error('Job configuration changed before operation acceptance');
    if (!['queued', 'stopped', 'completed', 'error'].includes(job.status))
      throw new Error('Job is owned by another operation');
    job = await db.jobs.updateIf(
      job.id,
      { attempt_id: job.attempt_id ?? null, status: job.status, updated_at: new Date(job.updated_at) },
      { attempt_id: operation.id, status: 'remote-starting', info: 'Preparing remote start' },
    );
    if (!job) throw new Error('Job changed before operation acceptance');
  }
  const keys = input.needsEphemeralKeys ? await fetchOperationKeys(operation.id) : undefined;
  const prepared = await prepareJobStart(input.jobID, keys, input.durableKeys, { operationID: operation.id });
  let progressWrites = Promise.resolve();
  const result = await startPreparedJob(prepared, {
    operationID: operation.id,
    onRemoteStartProgress: progress => {
      progressWrites = progressWrites
        .then(async () => {
          await updateOperation(operation.id, { progress });
        })
        .catch(() => undefined);
    },
  });
  await progressWrites;
  return { jobID: result.id, remoteJobID: result.remote_job_id };
}
