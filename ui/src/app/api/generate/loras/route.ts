import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { db } from '@/server/db';
import { getTrainingFolder } from '@/server/settings';
import { getProjectRoots, ProjectError, resolveProject } from '@/server/projects';
import type { Job, Project } from '@/types';
import {
  extractTriggerWordsFromMetadata,
  listUploadedLoras,
  mergeTriggerWords,
  readSafetensorsMetadata,
  splitTriggerWords,
} from '@/server/loraLibrary';

const LORA_JOB_TYPES = new Set(['lora', 'locon', 'lokr', 'lorm']);

type LoraModelSummary = {
  name_or_path?: string;
  arch?: string;
  quantize?: boolean;
  quantize_te?: boolean;
  qtype?: string;
  qtype_te?: string;
  low_vram?: boolean;
  layer_offloading?: boolean;
  layer_offloading_transformer_percent?: number;
  layer_offloading_text_encoder_percent?: number;
  layer_offloading_backend?: 'block' | 'legacy';
  model_kwargs?: Record<string, unknown>;
  extras_name_or_path?: string;
  vae_path?: string;
  refiner_name_or_path?: string;
  te_name_or_path?: string;
  quantize_kwargs?: Record<string, unknown>;
};

type LoraLibraryItem = {
  id: string;
  label: string;
  path: string;
  portableRef?: string;
  filename: string;
  source: 'job' | 'uploaded' | 'project';
  scope: 'global' | 'project';
  projectId?: string;
  projectName?: string;
  jobId?: string;
  jobName?: string;
  jobStatus?: string;
  updatedAt: string;
  sizeBytes: number;
  triggerWords: string[];
  triggerWordSource: 'metadata' | 'user' | 'none';
  model?: LoraModelSummary;
};

function parseJobConfig(jobConfig: string) {
  try {
    return JSON.parse(jobConfig);
  } catch {
    return null;
  }
}

function isLoraTrainingJob(jobConfig: any) {
  const networkType = String(jobConfig?.config?.process?.[0]?.network?.type || '').toLowerCase();
  return LORA_JOB_TYPES.has(networkType);
}

function getModelSummary(jobConfig: any): LoraModelSummary {
  const model = jobConfig?.config?.process?.[0]?.model || {};
  return {
    name_or_path: model.name_or_path,
    arch: model.arch,
    quantize: model.quantize,
    quantize_te: model.quantize_te,
    qtype: model.qtype,
    qtype_te: model.qtype_te,
    low_vram: model.low_vram,
    layer_offloading: model.layer_offloading,
    layer_offloading_transformer_percent: model.layer_offloading_transformer_percent,
    layer_offloading_text_encoder_percent: model.layer_offloading_text_encoder_percent,
    layer_offloading_backend: model.layer_offloading_backend,
    model_kwargs: model.model_kwargs,
    extras_name_or_path: model.extras_name_or_path,
    vae_path: model.vae_path,
    refiner_name_or_path: model.refiner_name_or_path,
    te_name_or_path: model.te_name_or_path,
    quantize_kwargs: model.quantize_kwargs,
  };
}

async function getSafeJobFolder(trainingRoot: string, jobName: string) {
  const root = await fs.promises.realpath(trainingRoot).catch(() => null);
  if (!root) return null;

  const folder = path.resolve(root, jobName);
  const relativePath = path.relative(root, folder);
  if (relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    return null;
  }

  const stat = await fs.promises.stat(folder).catch(() => null);
  if (!stat?.isDirectory()) {
    return null;
  }

  return folder;
}

function portableProjectRef(projectID: string, zone: 'models' | 'runs', relativePath: string) {
  const encodedPath = relativePath
    .split(/[\\/]+/)
    .filter(Boolean)
    .map(segment => encodeURIComponent(segment))
    .join('/');
  return `aitk-project://${projectID}/${zone}/${encodedPath}`;
}

async function collectSafetensorsFiles(root: string, limit = 2_000) {
  const rootStat = await fs.promises.stat(root).catch(() => null);
  if (!rootStat?.isDirectory()) return [] as string[];

  const files: string[] = [];
  const pending = [root];
  while (pending.length > 0 && files.length < limit) {
    const current = pending.shift() as string;
    const entries = await fs.promises.readdir(current, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (entry.isSymbolicLink()) continue;
      const child = path.join(current, entry.name);
      if (entry.isDirectory()) pending.push(child);
      else if (entry.isFile() && entry.name.toLowerCase().endsWith('.safetensors')) files.push(child);
      if (files.length >= limit) break;
    }
  }
  return files;
}

function matchingProjectJob(filePath: string, runsRoot: string, jobs: Job[]) {
  const relative = path.relative(runsRoot, filePath);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) return null;
  const firstSegment = relative.split(path.sep)[0];
  return jobs.find(job => job.name === firstSegment) || null;
}

async function listProjectLoras(project: Project): Promise<LoraLibraryItem[]> {
  const roots = await getProjectRoots(project);
  const jobs = await db.jobs.list({ job_type: 'train', project_id: project.id });
  const candidates = await Promise.all(
    (['runs', 'models'] as const).map(async zone => ({
      zone,
      root: roots[zone],
      files: await collectSafetensorsFiles(roots[zone]),
    })),
  );
  const items: LoraLibraryItem[] = [];

  for (const candidate of candidates) {
    for (const filePath of candidate.files) {
      const relativePath = path.relative(candidate.root, filePath);
      if (!relativePath || relativePath.startsWith('..') || path.isAbsolute(relativePath)) continue;
      const stat = await fs.promises.stat(filePath).catch(() => null);
      if (!stat?.isFile()) continue;
      const sourceJob = candidate.zone === 'runs' ? matchingProjectJob(filePath, roots.runs, jobs) : null;
      const jobConfig = sourceJob ? parseJobConfig(sourceJob.job_config) : null;
      const metadata = await readSafetensorsMetadata(filePath);
      const triggerWords = mergeTriggerWords(
        extractTriggerWordsFromMetadata(metadata),
        splitTriggerWords(jobConfig?.config?.process?.[0]?.trigger_word),
      );
      const portableRef = portableProjectRef(project.id, candidate.zone, relativePath);

      items.push({
        id: portableRef,
        label: `${project.name} / ${relativePath.replaceAll('\\', ' / ')}`,
        path: filePath,
        portableRef,
        filename: path.basename(filePath),
        source: sourceJob ? 'job' : 'project',
        scope: 'project',
        projectId: project.id,
        projectName: project.name,
        jobId: sourceJob?.id,
        jobName: sourceJob?.name,
        jobStatus: sourceJob?.status,
        updatedAt: stat.mtime.toISOString(),
        sizeBytes: stat.size,
        triggerWords,
        triggerWordSource: triggerWords.length > 0 ? 'metadata' : 'none',
        model: jobConfig ? getModelSummary(jobConfig) : undefined,
      });
    }
  }

  return items;
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const requestedScope = searchParams.get('scope');
    const rawProjectIdentifier = searchParams.get('project_id');
    if (searchParams.has('project_id') && !rawProjectIdentifier?.trim()) {
      return NextResponse.json(
        { error: 'project_id cannot be blank', code: 'PROJECT_INVALID_SCOPE' },
        { status: 400 },
      );
    }
    if (requestedScope && !['global', 'all', 'project'].includes(requestedScope)) {
      return NextResponse.json(
        { error: 'scope must be global, all, or project', code: 'INVALID_SCOPE' },
        { status: 400 },
      );
    }
    const projectIdentifier = rawProjectIdentifier?.trim() || '';
    const scope = (requestedScope || (projectIdentifier ? 'project' : 'global')) as 'global' | 'all' | 'project';
    if (scope === 'project' && !projectIdentifier) {
      return NextResponse.json(
        { error: 'project_id is required for project scope', code: 'PROJECT_ID_REQUIRED' },
        { status: 400 },
      );
    }
    if (requestedScope && scope !== 'project' && projectIdentifier) {
      return NextResponse.json(
        { error: 'project_id is only valid for project scope', code: 'INVALID_SCOPE' },
        { status: 400 },
      );
    }
    const loras: LoraLibraryItem[] = [];

    if (scope === 'global' || scope === 'all') {
      const trainingRoot = await getTrainingFolder();
      const jobs = await db.jobs.list({ job_type: 'train', project_id: null });

      for (const job of jobs) {
        if (job.worker_id && job.worker_id !== 'local') continue;

        const jobConfig = parseJobConfig(job.job_config);
        if (!jobConfig || !isLoraTrainingJob(jobConfig)) continue;

        const jobFolder = await getSafeJobFolder(trainingRoot, job.name);
        if (!jobFolder) continue;

        const entries = await fs.promises.readdir(jobFolder, { withFileTypes: true }).catch(() => []);
        for (const entry of entries) {
          if (!entry.isFile() || !entry.name.toLowerCase().endsWith('.safetensors')) continue;

          const filePath = path.join(jobFolder, entry.name);
          const stat = await fs.promises.stat(filePath).catch(() => null);
          if (!stat) continue;
          const metadata = await readSafetensorsMetadata(filePath);
          const triggerWords = mergeTriggerWords(
            extractTriggerWordsFromMetadata(metadata),
            splitTriggerWords(jobConfig?.config?.process?.[0]?.trigger_word),
          );

          loras.push({
            id: `${job.id}:${entry.name}`,
            label: `Global / ${job.name} / ${entry.name}`,
            path: filePath,
            filename: entry.name,
            source: 'job',
            scope: 'global',
            jobId: job.id,
            jobName: job.name,
            jobStatus: job.status,
            updatedAt: stat.mtime.toISOString(),
            sizeBytes: stat.size,
            triggerWords,
            triggerWordSource: triggerWords.length > 0 ? 'metadata' : 'none',
            model: getModelSummary(jobConfig),
          });
        }
      }

      loras.push(
        ...(await listUploadedLoras()).map(item => ({ ...item, label: `Global / ${item.label}`, scope: 'global' as const })),
      );
    }

    if (scope === 'project' || scope === 'all') {
      const projects = projectIdentifier
        ? [await resolveProject(projectIdentifier, { intent: 'read' })]
        : scope === 'all'
          ? await db.projects.list({ lifecycle_state: ['active', 'archived'] })
          : [];
      for (const project of projects) loras.push(...(await listProjectLoras(project)));
    }

    loras.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
    return NextResponse.json({ loras, scope, project_id: projectIdentifier || null });
  } catch (error) {
    if (error instanceof ProjectError) {
      return NextResponse.json(
        { error: error.message, code: error.code, details: error.details },
        { status: error.status },
      );
    }
    console.error('Error listing generated LoRAs:', error);
    return NextResponse.json({ error: 'Failed to list generated LoRAs' }, { status: 500 });
  }
}
