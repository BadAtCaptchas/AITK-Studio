import { randomUUID } from 'crypto';
import { db } from './db';
import type { Job } from '../types';
import { isProcessRunning } from './jobProcess';
import { deviceIds } from '../utils/jobIdentity';
import { isAnyRemoteOllamaCaptionJob } from './secureRemoteCaptionJobs';

type Reservation = { attemptID: string; jobID: string; devices: string[] };
function reservations(value: unknown): Reservation[] {
  if (!Array.isArray(value)) throw new Error('Invalid device reservation record');
  return value.map((entry: unknown) => {
    if (
      !entry ||
      typeof entry !== 'object' ||
      !('attemptID' in entry) ||
      typeof entry.attemptID !== 'string' ||
      !('jobID' in entry) ||
      typeof entry.jobID !== 'string' ||
      !('devices' in entry) ||
      !Array.isArray(entry.devices) ||
      !entry.devices.every((id: unknown) => typeof id === 'string')
    )
      throw new Error('Invalid device reservation');
    return { attemptID: entry.attemptID, jobID: entry.jobID, devices: entry.devices as string[] };
  });
}
function needsLocalDevice(job: Job): boolean {
  const config: unknown = JSON.parse(job.job_config);
  return !isAnyRemoteOllamaCaptionJob(config);
}

export async function claimJobMaintenance(
  job: Job,
  status: 'editing' | 'deleting' | 'restarting',
  attemptID = randomUUID(),
): Promise<Job | null> {
  if (!['queued', 'stopped', 'error', 'completed'].includes(job.status) || isProcessRunning(job.pid)) return null;
  const resource = await db.runtime.get(`operation-resource:job:${job.id}`);
  if (resource && typeof resource.value === 'string') {
    const operation = await db.runtime.get(`operation:${resource.value}`);
    if (
      operation?.value &&
      typeof operation.value === 'object' &&
      'state' in operation.value &&
      !['completed', 'failed', 'canceled'].includes(String(operation.value.state))
    )
      return null;
  }
  return db.jobs.updateIf(
    job.id,
    { attempt_id: job.attempt_id ?? null, status: job.status, updated_at: new Date(job.updated_at) },
    {
      attempt_id: attemptID,
      status,
      pid: null,
      process_started_at: null,
    },
  );
}

export async function reserveAttemptDevices(job: Job): Promise<boolean> {
  if (!job.attempt_id) throw new Error('Cannot reserve devices without an attempt');
  if (!needsLocalDevice(job)) return true;
  const key = `devices:${job.worker_id || 'local'}`;
  const devices = deviceIds(job.gpu_ids).sort();
  for (let retry = 0; retry < 20; retry++) {
    const row = await db.runtime.get(key);
    const current = row ? reservations(row.value) : [];
    const live: Reservation[] = [];
    for (const reservation of current) {
      const owner = await db.jobs.findById(reservation.jobID);
      if (
        owner?.attempt_id === reservation.attemptID &&
        (['starting', 'running', 'stopping'].includes(owner.status) || isProcessRunning(owner.pid))
      )
        live.push(reservation);
    }
    // Include legacy processes and claims racing to acquire this record. A
    // reservation is never reclaimed by elapsed time alone.
    const active = await db.jobs.list({ worker_id: job.worker_id || 'local', status: ['running', 'stopping'] });
    if (
      active.some(
        other =>
          other.id !== job.id && needsLocalDevice(other) && deviceIds(other.gpu_ids).some(id => devices.includes(id)),
      )
    )
      return false;
    if (live.some(other => other.attemptID !== job.attempt_id && other.devices.some(id => devices.includes(id))))
      return false;
    const next = [
      ...live.filter(item => item.attemptID !== job.attempt_id),
      { attemptID: job.attempt_id, jobID: job.id, devices },
    ];
    if (await db.runtime.compareAndSwap(key, row?.version ?? null, next)) return true;
  }
  throw new Error('Device reservation is busy. Retry shortly.');
}

export async function claimJobAttempt(job: Job): Promise<Job | null> {
  if (!['queued', 'stopped', 'error', 'completed'].includes(job.status)) return null;
  if (isProcessRunning(job.pid)) return null;
  const attemptID = randomUUID();
  await db.runtime.compareAndSwap(`attempt:${attemptID}`, null, {
    jobID: job.id,
    config: job.job_config,
    workerID: job.worker_id,
    deviceOrder: deviceIds(job.gpu_ids),
    createdAt: new Date().toISOString(),
    status: 'starting',
  });
  const claimed = await db.jobs.updateIf(
    job.id,
    {
      attempt_id: job.attempt_id ?? null,
      status: job.status,
      updated_at: new Date(job.updated_at),
    },
    {
      attempt_id: attemptID,
      status: 'starting',
      pid: null,
      process_started_at: null,
      stop: false,
      return_to_queue: false,
      sample_now: false,
      info: 'Reserving execution resources',
    },
  );
  if (!claimed) {
    await db.runtime.delete(`attempt:${attemptID}`);
    return null;
  }
  try {
    if (await reserveAttemptDevices(claimed)) return claimed;
  } catch (error) {
    await db.jobs.updateIf(
      job.id,
      { attempt_id: attemptID, status: 'starting' },
      { status: job.status, attempt_id: job.attempt_id ?? null },
    );
    await db.runtime.delete(`attempt:${attemptID}`);
    throw error;
  }
  await db.jobs.updateIf(
    job.id,
    { attempt_id: attemptID, status: 'starting' },
    { status: job.status, attempt_id: job.attempt_id ?? null, info: 'Waiting for selected devices' },
  );
  await db.runtime.delete(`attempt:${attemptID}`);
  return null;
}
