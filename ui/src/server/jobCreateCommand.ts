import { createHash, randomUUID } from 'crypto';
import { db, type JobCreateInput } from './db';
import { CommandInputError, isRecord } from './commandInput';
import { LeaseBusyError, withProcessLease } from './processLease';
import type { Job } from '../types';

type CreateReceipt = {
  version: 1;
  inputDigest: string;
  jobID: string;
  resultDigest: string | null;
  state: 'prepared' | 'completed';
};

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (isRecord(value)) {
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonicalValue(value[key])]));
  }
  return value;
}

function digest(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(canonicalValue(value))).digest('hex');
}

function decodeReceipt(value: unknown): CreateReceipt {
  if (!isRecord(value) || value.version !== 1 || typeof value.inputDigest !== 'string' ||
    !/^[a-f0-9]{64}$/.test(value.inputDigest) || typeof value.jobID !== 'string' ||
    !/^[a-zA-Z0-9_-]{1,128}$/.test(value.jobID) ||
    !(value.resultDigest === null || typeof value.resultDigest === 'string' && /^[a-f0-9]{64}$/.test(value.resultDigest)) ||
    !(value.state === 'prepared' || value.state === 'completed')) {
    throw new CommandInputError('Job creation receipt is invalid. Review the saved job before retrying.', 500, 'CREATE_RECEIPT_INVALID');
  }
  return { version: 1, inputDigest: value.inputDigest, jobID: value.jobID, resultDigest: value.resultDigest, state: value.state };
}

function resultDigest(job: JobCreateInput): string {
  let config: unknown = job.job_config;
  try { config = JSON.parse(job.job_config); } catch { /* A changed invalid config must not match the saved request. */ }
  return digest({
    id: job.id, name: job.name, worker_id: job.worker_id ?? 'local', gpu_ids: job.gpu_ids,
    job_type: job.job_type ?? 'train', job_ref: job.job_ref ?? null,
    storage_key: job.storage_key ?? null, job_config: config,
  });
}

/** Durable command identity contains hashes and a preallocated id, never raw configuration. */
export async function createJobWithReceipt(
  key: string | undefined,
  normalizedInput: Record<string, unknown>,
  prepare: () => Promise<JobCreateInput>,
): Promise<Job> {
  if (key === undefined) return db.jobs.create(await prepare());
  if (!/^[a-zA-Z0-9_-]{8,128}$/.test(key)) throw new CommandInputError('Invalid idempotencyKey');
  const recordKey = `job-create-command:${digest(key)}`;
  const inputDigest = digest(normalizedInput);
  return withProcessLease(recordKey, async lease => {
    let row = await db.runtime.get(recordKey);
    let receipt: CreateReceipt = row ? decodeReceipt(row.value) : {
      version: 1, inputDigest, jobID: randomUUID(), resultDigest: null, state: 'prepared',
    };
    if (receipt.inputDigest !== inputDigest) {
      throw new CommandInputError('Creation key was already used with different input.', 409, 'CREATE_KEY_CONFLICT');
    }
    if (!row) {
      if (!await db.runtime.compareAndSwap(recordKey, null, receipt)) {
        throw new CommandInputError('Job creation is still in progress. Retry the same command.', 409, 'CREATE_BUSY');
      }
      row = await db.runtime.get(recordKey);
      if (!row) throw new CommandInputError('Could not preserve the creation receipt. Retry the same command.', 500, 'CREATE_RECEIPT_WRITE_FAILED');
    }
    let receiptVersion = row.version;
    const writeReceipt = async (next: CreateReceipt) => {
      await lease.assertOwned();
      if (!await db.runtime.compareAndSwap(recordKey, receiptVersion, next)) {
        throw new CommandInputError('Could not preserve the creation receipt. Retry the same command.', 500, 'CREATE_RECEIPT_WRITE_FAILED');
      }
      const updated = await db.runtime.get(recordKey);
      if (!updated) throw new CommandInputError('Could not read the creation receipt. Retry the same command.', 500, 'CREATE_RECEIPT_WRITE_FAILED');
      receiptVersion = updated.version;
      receipt = next;
    };
    const existing = await db.jobs.findById(receipt.jobID);
    if (existing) {
      if (!receipt.resultDigest || resultDigest(existing) !== receipt.resultDigest) {
        throw new CommandInputError('The saved job changed after this creation command. Review it before starting.', 409, 'CREATE_RESULT_CHANGED');
      }
      if (receipt.state !== 'completed') await writeReceipt({ ...receipt, state: 'completed' });
      return existing;
    }
    if (receipt.state === 'completed') {
      throw new CommandInputError('The job created by this command was deleted. Create a new job explicitly.', 410, 'CREATE_RESULT_DELETED');
    }
    const prepared = await prepare();
    const input = { ...prepared, id: receipt.jobID, storage_key: `job-${receipt.jobID}` };
    // Persist the expected row before insertion so an interrupted acknowledgement
    // can recover only this command's unchanged result, including its runtime paths.
    await writeReceipt({ ...receipt, resultDigest: resultDigest(input) });
    await lease.assertOwned();
    const created = await db.jobs.create(input);
    if (created.id !== receipt.jobID || resultDigest(created) !== receipt.resultDigest) {
      throw new CommandInputError('The saved job does not match this creation command. Review it before starting.', 409, 'CREATE_RESULT_CHANGED');
    }
    await writeReceipt({ ...receipt, state: 'completed' });
    return created;
  }, { waitMs: 5_000 }).catch((error: unknown) => {
    if (error instanceof LeaseBusyError) {
      throw new CommandInputError('Job creation is still in progress. Retry the same command.', 409, 'CREATE_BUSY');
    }
    throw error;
  });
}
