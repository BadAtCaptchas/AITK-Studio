import { assertGlobalPayload, isLegacyScopedRecord } from '../utils/obsoleteWorkspaceGuard';
import { db } from './db';
import { CommandInputError, isRecord } from './commandInput';
import type { Job } from '../types';

export const ACTIVE_JOB_STATUSES = ['queued', 'starting', 'remote-starting', 'running', 'stopping'];
export function encodeJobCursor(job: Job): string {
  return Buffer.from(JSON.stringify({ created_at: new Date(job.created_at).toISOString(), id: job.id })).toString('base64url');
}
export function decodeJobCursor(cursor?: string | null): { created_at: string; id: string } | undefined {
  if (!cursor) return undefined;
  try {
    if (cursor.length > 512) throw new Error();
    const value: unknown = JSON.parse(Buffer.from(cursor, 'base64url').toString());
    if (!isRecord(value) || typeof value.id !== 'string' || !value.id || value.id.length > 180 ||
        typeof value.created_at !== 'string' || !Number.isFinite(Date.parse(value.created_at))) throw new Error();
    return { id: value.id, created_at: new Date(value.created_at).toISOString() };
  } catch { throw new CommandInputError('Invalid job cursor'); }
}
/** List-only config projection. Complete configuration is available from the detail endpoint. */
export function summarizeJob(job: Job): Job & { is_summary: true } {
  let process: Record<string, unknown> = {};
  let remoteCaption: Record<string, unknown> | undefined;
  try {
    const value: unknown = JSON.parse(job.job_config);
    if (isRecord(value) && isRecord(value.config)) {
      const first = Array.isArray(value.config.process) ? value.config.process[0] : null;
      if (isRecord(first)) process = first;
      if (isRecord(value.config.remote_caption)) {
        remoteCaption = { downloadStatus: value.config.remote_caption.downloadStatus, lastError: value.config.remote_caption.lastError };
      }
    }
  } catch { /* Corrupt legacy configuration does not break the history list. */ }
  return { ...job, is_summary: true, job_config: JSON.stringify({ config: {
    process: [{ type: process.type, train: { steps: isRecord(process.train) ? process.train.steps : 0, auto_train: isRecord(process.train) ? process.train.auto_train : undefined } }],
    remote_caption: remoteCaption,
  } }) };
}
export async function listJobsForJobsApi(
  options: { jobType?: string | null; localOnly?: boolean; cursor?: string | null; limit?: string | null; view?: string | null; summary?: boolean },
  deps = { listJobs: db.jobs.list, runtimeGet: db.runtime.get },
) {
  assertGlobalPayload(options);
  const limit = options.limit ? Number(options.limit) : 50;
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > 100) throw new CommandInputError('Job page size must be between 1 and 100');
  if (options.view && !['all', 'active', 'history', 'failed'].includes(options.view)) throw new CommandInputError('Invalid job view');
  const status = options.view === 'active' ? ACTIVE_JOB_STATUSES : options.view === 'failed' ? ['error', 'failed'] :
    options.view === 'history' ? ['stopped', 'completed', 'error', 'failed'] : undefined;
  const rows = await deps.listJobs({ job_type: options.jobType, worker_id: options.localOnly ? 'local' : undefined,
    status, limit: limit + 1, before: decodeJobCursor(options.cursor) });
  let selected = rows.slice(0, limit);
  if (options.summary === false) {
    // Keep full-config worker pages below the remote JSON transport's 8 MiB cap.
    let bytes = 0, count = 0;
    for (const job of selected) {
      const size = Buffer.byteLength(JSON.stringify(job));
      if (bytes + size > 7 * 1024 ** 2) break;
      bytes += size; count++;
    }
    if (!count && selected.length) throw new CommandInputError('Job configuration exceeds the remote page size limit', 413);
    selected = selected.slice(0, count);
  }
  const jobs = selected.filter(job => !isLegacyScopedRecord(job));
  const freshness = await deps.runtimeGet('job-read-model:freshness');
  return { jobs: options.summary === false ? jobs : jobs.map(summarizeJob),
    nextCursor: rows.length > selected.length ? encodeJobCursor(selected[selected.length - 1]) : null,
    freshness: freshness?.value ?? null };
}
