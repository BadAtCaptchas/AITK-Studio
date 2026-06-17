import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { db } from './db';
import { getDatasetsRoot, getProjectsRoot, getTrainingFolder } from './settings';
import type { Job, Project } from '@/types';

export const PROJECT_FOLDERS = ['datasets', 'configs', 'runs', 'outputs', 'models', 'assets', 'notes', 'cache'] as const;

export type ProjectFolderName = (typeof PROJECT_FOLDERS)[number];

export type ProjectRoots = Record<ProjectFolderName, string> & {
  root: string;
};

const DATASET_PATH_FIELDS = [
  'folder_path',
  'dataset_path',
  'control_path',
  'control_path_1',
  'control_path_2',
  'control_path_3',
  'mask_path',
  'unconditional_path',
  'inpaint_path',
  'clip_image_path',
];

export class ProjectNotFoundError extends Error {
  constructor(identifier: string) {
    super(`Project not found: ${identifier}`);
    this.name = 'ProjectNotFoundError';
  }
}

export function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

export function cleanProjectSlug(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 72);
}

export function safeProjectName(value: unknown) {
  const name = typeof value === 'string' ? value.trim() : '';
  return name.slice(0, 120);
}

export async function uniqueProjectSlug(preferred: string) {
  const base = cleanProjectSlug(preferred) || 'project';
  let candidate = base;
  let suffix = 2;
  while (await db.projects.findBySlug(candidate)) {
    candidate = `${base}-${suffix}`;
    suffix += 1;
  }
  return candidate;
}

export async function resolveProject(identifier: string): Promise<Project> {
  const normalized = identifier.trim();
  if (!normalized) throw new ProjectNotFoundError(identifier);
  const project = (await db.projects.findById(normalized)) || (await db.projects.findBySlug(cleanProjectSlug(normalized)));
  if (!project) throw new ProjectNotFoundError(identifier);
  return project;
}

export async function resolveOptionalProject(identifier: unknown): Promise<Project | null> {
  if (typeof identifier !== 'string' || !identifier.trim()) return null;
  return resolveProject(identifier);
}

export async function getProjectRoots(project: Project): Promise<ProjectRoots> {
  const projectsRoot = path.resolve(await getProjectsRoot());
  const root = path.resolve(project.root_path?.trim() || path.join(projectsRoot, project.slug));
  if (!isPathInside(projectsRoot, root)) {
    throw new Error('Project root must be inside PROJECTS_FOLDER');
  }

  return PROJECT_FOLDERS.reduce(
    (acc, folder) => {
      acc[folder] = path.join(root, folder);
      return acc;
    },
    { root } as ProjectRoots,
  );
}

export async function ensureProjectFolders(project: Project) {
  const roots = await getProjectRoots(project);
  await Promise.all(Object.values(roots).map(folder => fsp.mkdir(folder, { recursive: true })));
  return roots;
}

export async function createProject(input: { name: unknown; slug?: unknown; description?: unknown; badge_asset?: unknown }) {
  const name = safeProjectName(input.name);
  if (!name) {
    const error = new Error('Project name is required');
    (error as any).status = 400;
    throw error;
  }
  const requestedSlug = typeof input.slug === 'string' && input.slug.trim() ? input.slug : name;
  const slug = await uniqueProjectSlug(requestedSlug);
  const projectsRoot = path.resolve(await getProjectsRoot());
  const project = await db.projects.create({
    slug,
    name,
    description: typeof input.description === 'string' ? input.description.trim().slice(0, 500) : '',
    badge_asset:
      typeof input.badge_asset === 'string' && input.badge_asset.trim()
        ? input.badge_asset.trim()
        : '/assets/projects/project-badge-default.png',
    root_path: path.join(projectsRoot, slug),
  });
  await ensureProjectFolders(project);
  return project;
}

function isProtocolPath(value: string) {
  return /^[a-z][a-z0-9+.-]*:\/\//i.test(value) || value.startsWith('aitk-');
}

function safeCopyName(sourcePath: string) {
  const base = path.basename(sourcePath.replace(/[\\/]+$/, ''));
  return (
    base
      .trim()
      .replace(/[<>:"/\\|?*\x00-\x1f]/g, '_')
      .replace(/^\.+/, '')
      .slice(0, 100) || 'asset'
  );
}

async function copyPathIntoProject(value: string, roots: ProjectRoots, globalDatasetsRoot: string) {
  if (!value.trim() || isProtocolPath(value)) return value;

  const source = path.resolve(value);
  if (isPathInside(roots.root, source)) return source;
  if (!fs.existsSync(source)) return value;

  const globalRoot = path.resolve(globalDatasetsRoot);
  if (!isPathInside(globalRoot, source)) return value;

  const target = path.join(roots.datasets, safeCopyName(source));
  if (!fs.existsSync(target)) {
    await fsp.cp(source, target, { recursive: true, force: false, errorOnExist: false });
  }
  return target;
}

async function rewriteDatasetPathValue(value: unknown, roots: ProjectRoots, globalDatasetsRoot: string): Promise<unknown> {
  if (typeof value === 'string') {
    return copyPathIntoProject(value, roots, globalDatasetsRoot);
  }
  if (Array.isArray(value)) {
    return Promise.all(value.map(item => rewriteDatasetPathValue(item, roots, globalDatasetsRoot)));
  }
  return value;
}

export async function prepareJobConfigForProject(rawJobConfig: any, project: Project) {
  const jobConfig = JSON.parse(JSON.stringify(rawJobConfig || null));
  const roots = await ensureProjectFolders(project);
  const globalDatasetsRoot = await getDatasetsRoot();
  const processes = Array.isArray(jobConfig?.config?.process) ? jobConfig.config.process : [];
  const jobName = typeof jobConfig?.config?.name === 'string' && jobConfig.config.name.trim() ? jobConfig.config.name : 'job';

  for (const processConfig of processes) {
    if (!processConfig || typeof processConfig !== 'object') continue;
    processConfig.training_folder = roots.runs;
    if (typeof processConfig.output_folder === 'string' && processConfig.output_folder.trim()) {
      processConfig.output_folder = path.join(roots.outputs, safeCopyName(jobName));
    }

    const datasets = Array.isArray(processConfig.datasets) ? processConfig.datasets : [];
    for (const dataset of datasets) {
      if (!dataset || typeof dataset !== 'object') continue;
      for (const field of DATASET_PATH_FIELDS) {
        if (field in dataset) {
          dataset[field] = await rewriteDatasetPathValue(dataset[field], roots, globalDatasetsRoot);
        }
      }
    }

    const captionPath = processConfig.caption?.path_to_caption;
    if (typeof captionPath === 'string') {
      processConfig.caption.path_to_caption = await copyPathIntoProject(captionPath, roots, globalDatasetsRoot);
    }
  }

  return jobConfig;
}

export async function getJobTrainingRoot(job: Job) {
  if (job.project_id) {
    const project = await db.projects.findById(job.project_id);
    if (project) {
      return (await getProjectRoots(project)).runs;
    }
  }

  try {
    const jobConfig = JSON.parse(job.job_config);
    const processConfig = Array.isArray(jobConfig?.config?.process) ? jobConfig.config.process[0] : null;
    if (typeof processConfig?.training_folder === 'string' && processConfig.training_folder.trim()) {
      return processConfig.training_folder;
    }
  } catch {
    // Fall back to the current global training folder below.
  }

  return getTrainingFolder();
}

export async function getAllowedProjectRootIfExists() {
  const root = path.resolve(await getProjectsRoot());
  return fs.existsSync(root) ? fs.promises.realpath(root).catch(() => root) : null;
}
