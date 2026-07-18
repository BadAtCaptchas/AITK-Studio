import fsp from 'fs/promises';
import path from 'path';
import { NextResponse } from 'next/server';
import { db } from '@/server/db';
import { listDatasetSummaries } from '@/server/encryptedDatasets';
import { getDatasetWatcherStatuses, listDatasetWatchers } from '@/server/datasetWatchers';
import { indexProjectArtifacts } from '@/server/projectArtifactIndex';
import { getProjectRoots, resolveProject } from '@/server/projects';
import { areProjectsEnabled, PROJECT_SPACES_DISABLED_MESSAGE } from '@/server/settings';
import type { Job } from '@/types';
import { ensureProjectApiAccess, projectApiError } from '../../projectApi';

const ACTIVE_STATUSES = new Set(['queued', 'starting', 'running', 'stopping']);

function jobTotalSteps(job: Job) {
  try {
    const parsed: unknown = JSON.parse(job.job_config);
    if (!parsed || typeof parsed !== 'object') return null;
    const process = (parsed as { config?: { process?: unknown[] } }).config?.process?.[0];
    if (!process || typeof process !== 'object') return null;
    const train = (process as { train?: { auto_train?: unknown; steps?: unknown } }).train;
    if (train?.auto_train) return null;
    const steps = Number(train?.steps ?? 0);
    return Number.isFinite(steps) && steps > 0 ? steps : null;
  } catch {
    return null;
  }
}

function summarizeJob(job: Job) {
  return {
    id: job.id,
    name: job.name,
    project_id: job.project_id,
    worker_id: job.worker_id,
    remote_job_id: job.remote_job_id,
    remote_sync_at: job.remote_sync_at,
    remote_error: job.remote_error,
    gpu_ids: job.gpu_ids,
    created_at: job.created_at,
    updated_at: job.updated_at,
    status: job.status,
    stop: job.stop,
    return_to_queue: job.return_to_queue,
    step: job.step,
    info: job.info,
    speed_string: job.speed_string,
    queue_position: job.queue_position,
    pid: job.pid,
    job_type: job.job_type,
    job_ref: job.job_ref,
    save_now: job.save_now,
    sample_now: job.sample_now,
    total_steps: jobTotalSteps(job),
  };
}

async function shallowZoneSummary(folder: string) {
  const entries = await fsp.readdir(folder, { withFileTypes: true }).catch(() => []);
  const rows = await Promise.all(
    entries
      .filter(entry => !entry.name.startsWith('.') && !entry.isSymbolicLink())
      .map(async entry => {
        const filePath = path.join(folder, entry.name);
        const stat = await fsp.lstat(filePath).catch(() => null);
        if (!stat) return null;
        return {
          name: entry.name,
          path: filePath,
          kind: entry.isDirectory() ? ('folder' as const) : ('file' as const),
          size: stat.isFile() ? stat.size : 0,
          updatedAt: stat.mtime.toISOString(),
        };
      }),
  );
  const present = rows.filter((entry): entry is NonNullable<(typeof rows)[number]> => entry !== null);
  present.sort((left, right) => new Date(right.updatedAt).getTime() - new Date(left.updatedAt).getTime());
  return {
    fileCount: present.filter(entry => entry.kind === 'file').length,
    folderCount: present.filter(entry => entry.kind === 'folder').length,
    mediaCount: 0,
    totalBytes: present.reduce((total, entry) => total + entry.size, 0),
    recent: present.slice(0, 8),
  };
}

export async function GET(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = await ensureProjectApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const { projectID } = await params;
    const project = await resolveProject(decodeURIComponent(projectID), { intent: 'read' });
    const roots = await getProjectRoots(project);
    const [datasets, jobs, artifacts, replicas, syncOperations, watchers, inputSummary, runSummary, outputSummary, modelSummary] =
      await Promise.all([
        listDatasetSummaries(roots.datasets, { createIfMissing: false }),
        db.jobs.list({ project_id: project.id }),
        indexProjectArtifacts(project, { kind: 'all', maxEntries: 6_000 }),
        db.projectReplicas.listByProject(project.id),
        db.projectSyncOperations.list({ project_id: project.id }),
        listDatasetWatchers({ projectID: project.id }, { intent: 'read' }),
        shallowZoneSummary(roots.datasets),
        shallowZoneSummary(roots.runs),
        shallowZoneSummary(roots.outputs),
        shallowZoneSummary(roots.models),
      ]);
    const watcherStatuses = await getDatasetWatcherStatuses(watchers.map(watcher => watcher.id));
    const watcherErrors = Object.values(watcherStatuses).filter(status => status.state === 'error' || status.lastError).length;
    const jobsByRecentUpdate = [...jobs].sort(
      (left, right) => new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime(),
    );
    const activeJobs = jobsByRecentUpdate.filter(job => ACTIVE_STATUSES.has(job.status));
    const outputArtifacts = artifacts.filter(artifact => ['image', 'video', 'audio'].includes(artifact.kind));
    const modelArtifacts = artifacts.filter(artifact => artifact.kind === 'model');
    const totalItems = datasets.reduce((total, dataset) => total + (dataset.itemCount || 0), 0);
    const captionedItems = datasets.reduce((total, dataset) => total + (dataset.captionedItemCount || 0), 0);
    const missingCaptions = datasets.reduce((total, dataset) => total + (dataset.missingCaptionCount || 0), 0);
    const serializedJobs = jobsByRecentUpdate.map(summarizeJob);
    const recentActivity = [
      ...jobsByRecentUpdate.slice(0, 8).map(job => ({
        id: `job:${job.id}`,
        label: `${job.job_type === 'generate' ? 'Generate' : job.job_type === 'caption' ? 'Caption' : 'Train'} ${job.name}`,
        detail: job.info || job.status,
        kind: 'job',
        updatedAt: new Date(job.updated_at).toISOString(),
      })),
      ...outputArtifacts.slice(0, 4).map(artifact => ({
        id: `output:${artifact.id}`,
        label: artifact.name,
        detail: 'Output file',
        kind: 'output',
        updatedAt: artifact.updatedAt,
      })),
    ]
      .sort((left, right) => new Date(right.updatedAt).getTime() - new Date(left.updatedAt).getTime())
      .slice(0, 10);

    return NextResponse.json({
      project,
      roots,
      canonical_project_id: project.id,
      datasets,
      jobs: serializedJobs,
      activeJob: activeJobs[0] ? summarizeJob(activeJobs[0]) : null,
      counts: {
        datasets: datasets.length,
        jobs: jobs.length,
        activeJobs: activeJobs.length,
        outputs: outputArtifacts.length,
        models: modelArtifacts.length,
      },
      zones: {
        inputs: inputSummary,
        runs: runSummary,
        outputs: { ...outputSummary, mediaCount: outputArtifacts.length },
        models: { ...modelSummary, fileCount: modelArtifacts.length },
      },
      recentActivity,
      recentOutputs: outputArtifacts.slice(0, 8),
      recentModels: modelArtifacts.slice(0, 8),
      replicas,
      syncOperations: syncOperations.slice(0, 12),
      dataset: {
        count: datasets.length,
        item_count: totalItems,
        captioned_count: captionedItems,
        missing_caption_count: missingCaptions,
        watcher_errors: watcherErrors,
      },
      training: {
        active_jobs: activeJobs.map(summarizeJob),
        latest_job: serializedJobs[0] || null,
      },
      review: {
        recent_outputs: outputArtifacts.slice(0, 8),
        model_count: modelArtifacts.length,
      },
    });
  } catch (error) {
    return projectApiError(error, 'Failed to load project overview');
  }
}
