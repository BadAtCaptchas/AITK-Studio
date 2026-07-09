import { db } from './db';
import type { Job, ProjectSyncOperation } from '@/types';

const ACTIVE_SYNC_STATUSES: ProjectSyncOperation['status'][] = [
  'queued',
  'running',
  'waiting_for_job',
  'waiting_for_worker',
  'conflict',
];

export async function queueProjectResultsSync(job: Job) {
  if (
    job.status !== 'completed' ||
    !job.project_id ||
    !job.worker_id ||
    job.worker_id === 'local' ||
    !job.remote_job_id
  ) {
    return null;
  }
  const replica = await db.projectReplicas.findByProjectAndWorker(job.project_id, job.worker_id);
  if (!replica || !replica.auto_pull_results || replica.state === 'incompatible' || replica.state === 'detached') {
    return null;
  }
  const existingJobReplica = await db.jobReplicas.findByJobAndWorker(job.id, job.worker_id);
  if (existingJobReplica?.last_synced_at) return null;
  await db.jobReplicas.upsert({
    job_id: job.id,
    worker_id: job.worker_id,
    remote_job_id: job.remote_job_id,
    remote_project_id: job.project_id,
    role: 'execution',
    last_synced_at: new Date(),
    last_error: null,
  });
  const operations = await db.projectSyncOperations.list({
    project_id: job.project_id,
    worker_id: job.worker_id,
    status: ACTIVE_SYNC_STATUSES,
  });
  const existing = operations.find(operation => operation.profile === 'results');
  if (existing) return existing;
  return db.projectSyncOperations.create({
    project_id: job.project_id,
    worker_id: job.worker_id,
    profile: 'results',
    status: 'queued',
    phase: 'queued',
  });
}
