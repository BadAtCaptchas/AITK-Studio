import { db } from './db';
import {
  discoverRemoteJobs,
  isLocalWorker,
  syncRemoteJobs,
} from './remoteClient';
import type { Job } from '../types';

type JobsListOptions = Parameters<typeof db.jobs.list>[0];

type ListJobsForJobsApiDeps = {
  listJobs: (options: JobsListOptions) => Promise<Job[]>;
  discoverRemoteJobs: (jobType?: string | null) => Promise<Set<string>>;
  syncRemoteJobs: (jobs: Job[], alreadySyncedJobIds?: Set<string>) => Promise<Job[]>;
};

const defaultDeps: ListJobsForJobsApiDeps = {
  listJobs: options => db.jobs.list(options),
  discoverRemoteJobs,
  syncRemoteJobs,
};

export async function listJobsForJobsApi(
  options: { jobType?: string | null; localOnly?: boolean },
  deps: ListJobsForJobsApiDeps = defaultDeps,
) {
  const listOptions: JobsListOptions = { job_type: options.jobType };

  if (options.localOnly) {
    const jobs = await deps.listJobs(listOptions);
    return jobs.filter(job => isLocalWorker(job.worker_id));
  }

  const discoveredJobIds = await deps.discoverRemoteJobs(options.jobType);
  return deps.syncRemoteJobs(await deps.listJobs(listOptions), discoveredJobIds);
}
