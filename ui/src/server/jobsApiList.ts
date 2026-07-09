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

const activeProjectJobStatuses = ['queued', 'starting', 'running', 'stopping'];

const defaultDeps: ListJobsForJobsApiDeps = {
  listJobs: options => db.jobs.list(options),
  discoverRemoteJobs,
  syncRemoteJobs,
};

function mergeJobs(...jobLists: Job[][]) {
  const seen = new Set<string>();
  const merged: Job[] = [];

  for (const jobs of jobLists) {
    for (const job of jobs) {
      if (seen.has(job.id)) continue;
      seen.add(job.id);
      merged.push(job);
    }
  }

  return merged;
}

export async function listJobsForJobsApi(
  options: {
    jobType?: string | null;
    localOnly?: boolean;
    projectID?: string | null;
    scope?: 'global' | 'all' | 'project';
    includeProjectActive?: boolean;
  },
  deps: ListJobsForJobsApiDeps = defaultDeps,
) {
  const projectID = options.projectID ?? null;
  const scope = options.scope || (projectID ? 'project' : 'global');
  const listOptions: JobsListOptions =
    scope === 'all'
      ? { job_type: options.jobType }
      : { job_type: options.jobType, project_id: scope === 'project' ? projectID : null };
  const loadJobs = async () => {
    if (options.includeProjectActive && scope === 'global' && !projectID) {
      const [globalJobs, activeJobs] = await Promise.all([
        deps.listJobs(listOptions),
        deps.listJobs({ job_type: options.jobType, status: activeProjectJobStatuses }),
      ]);
      return mergeJobs(globalJobs, activeJobs.filter(job => Boolean(job.project_id)));
    }

    return deps.listJobs(listOptions);
  };

  if (options.localOnly) {
    const jobs = await loadJobs();
    return jobs.filter(job => isLocalWorker(job.worker_id));
  }

  const discoveredJobIds = await deps.discoverRemoteJobs(options.jobType);
  return deps.syncRemoteJobs(await loadJobs(), discoveredJobIds);
}
