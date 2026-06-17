import fsp from 'fs/promises';
import path from 'path';
import { NextResponse } from 'next/server';
import { db } from '@/server/db';
import { listDatasetSummaries } from '@/server/encryptedDatasets';
import { ensureProjectFolders, isPathInside, resolveProject } from '@/server/projects';
import type { Job } from '@/types';

const ACTIVE_STATUSES = new Set(['queued', 'running', 'stopping']);
const MEDIA_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.jxl', '.mp4', '.mp3', '.wav', '.flac', '.ogg']);

function ensureApiAccess(request: Request): NextResponse | null {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) return null;

  const token = request.headers.get('authorization')?.split(' ')[1];
  if (token !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  return null;
}

function jobTotalSteps(job: Job) {
  try {
    const config = JSON.parse(job.job_config);
    const processConfig = Array.isArray(config?.config?.process) ? config.config.process[0] : null;
    if (processConfig?.train?.auto_train) return null;
    const steps = Number(processConfig?.train?.steps ?? 0);
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
  };
}

async function directorySummary(folder: string) {
  const entries = await fsp.readdir(folder, { withFileTypes: true }).catch(() => []);
  let fileCount = 0;
  let folderCount = 0;
  let mediaCount = 0;
  let totalBytes = 0;
  const recent: Array<{ name: string; path: string; kind: 'file' | 'folder'; updatedAt: string; size: number }> = [];

  for (const entry of entries) {
    const entryPath = path.join(folder, entry.name);
    const stat = await fsp.stat(entryPath).catch(() => null);
    if (!stat) continue;
    if (entry.isDirectory()) folderCount += 1;
    if (entry.isFile()) {
      fileCount += 1;
      totalBytes += stat.size;
      if (MEDIA_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) mediaCount += 1;
    }
    recent.push({
      name: entry.name,
      path: entryPath,
      kind: entry.isDirectory() ? 'folder' : 'file',
      updatedAt: stat.mtime.toISOString(),
      size: stat.size,
    });
  }

  recent.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
  return { fileCount, folderCount, mediaCount, totalBytes, recent: recent.slice(0, 8) };
}

async function listTree(root: string, maxEntries = 80) {
  const rootReal = await fsp.realpath(root).catch(() => path.resolve(root));
  const out: Array<{ name: string; path: string; relativePath: string; kind: 'file' | 'folder'; size: number; updatedAt: string }> = [];
  const stack = [''];

  while (stack.length > 0 && out.length < maxEntries) {
    const relativeDir = stack.shift() as string;
    const dir = path.join(rootReal, relativeDir);
    const entries = await fsp.readdir(dir, { withFileTypes: true }).catch(() => []);
    entries.sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name));
    for (const entry of entries) {
      if (out.length >= maxEntries || entry.name.startsWith('.')) break;
      const absolutePath = path.join(dir, entry.name);
      if (!isPathInside(rootReal, absolutePath)) continue;
      const stat = await fsp.stat(absolutePath).catch(() => null);
      if (!stat) continue;
      const relativePath = path.relative(rootReal, absolutePath);
      out.push({
        name: entry.name,
        path: absolutePath,
        relativePath,
        kind: entry.isDirectory() ? 'folder' : 'file',
        size: stat.size,
        updatedAt: stat.mtime.toISOString(),
      });
      if (entry.isDirectory() && relativePath.split(path.sep).length < 3) {
        stack.push(relativePath);
      }
    }
  }
  return out;
}

export async function GET(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;

  try {
    const { projectID } = await params;
    const project = await resolveProject(decodeURIComponent(projectID));
    const roots = await ensureProjectFolders(project);
    const [datasets, jobs, inputSummary, runSummary, outputSummary, modelSummary, fileTree] = await Promise.all([
      listDatasetSummaries(roots.datasets),
      db.jobs.list({ project_id: project.id }),
      directorySummary(roots.datasets),
      directorySummary(roots.runs),
      directorySummary(roots.outputs),
      directorySummary(roots.models),
      listTree(roots.root),
    ]);

    const activeJobs = jobs.filter(job => ACTIVE_STATUSES.has(job.status));
    const activeJob = activeJobs[0] || null;
    const recentActivity = [
      ...jobs.slice(0, 8).map(job => ({
        id: `job:${job.id}`,
        label: `${job.job_type === 'generate' ? 'Generate' : job.job_type === 'caption' ? 'Caption' : 'Train'} ${job.name}`,
        detail: job.info || job.status,
        kind: 'job',
        updatedAt: new Date(job.updated_at).toISOString(),
      })),
      ...outputSummary.recent.slice(0, 4).map(item => ({
        id: `output:${item.path}`,
        label: item.name,
        detail: item.kind === 'folder' ? 'Output folder' : 'Output file',
        kind: 'output',
        updatedAt: item.updatedAt,
      })),
    ].sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()).slice(0, 10);

    return NextResponse.json({
      project,
      roots,
      datasets,
      jobs: jobs.map(summarizeJob),
      activeJob: activeJob
        ? {
            ...summarizeJob(activeJob),
            total_steps: jobTotalSteps(activeJob),
          }
        : null,
      counts: {
        datasets: datasets.length,
        jobs: jobs.length,
        activeJobs: activeJobs.length,
        outputs: outputSummary.fileCount + outputSummary.folderCount,
        models: modelSummary.fileCount + modelSummary.folderCount,
      },
      zones: {
        inputs: inputSummary,
        runs: runSummary,
        outputs: outputSummary,
        models: modelSummary,
      },
      recentActivity,
      fileTree,
    });
  } catch (error: any) {
    console.error('Failed to load project summary:', error);
    return NextResponse.json({ error: error?.message || 'Failed to load project summary' }, { status: 500 });
  }
}
