import { createHash, randomUUID } from 'crypto';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { db } from './db';
import { areProjectsEnabled, assertProjectsEnabled, getDatasetsRoot, getProjectsRoot, getTrainingFolder } from './settings';
import { clearDurableEncryptedDatasetKeys } from './encryptedDatasetSecrets';
import type { Job, Project, ProjectErrorCode, ProjectLifecycleState, ProjectScopeIntent } from '@/types';

export const PROJECT_FOLDERS = ['datasets', 'configs', 'runs', 'outputs', 'models', 'assets', 'notes', 'cache'] as const;

export type ProjectFolderName = (typeof PROJECT_FOLDERS)[number];

export type ProjectRoots = Record<ProjectFolderName, string> & {
  root: string;
};

export const PROJECT_MANIFEST_FILE = '.aitk-project.json';
export const AITK_INSTANCE_ID_KEY = 'AITK_INSTANCE_ID';
export const ACTIVE_PROJECT_JOB_STATUSES = ['queued', 'starting', 'running', 'stopping'] as const;

type ProjectManifest = {
  format: 'aitk-project';
  version: 1;
  project_id: string;
  slug: string;
  created_at: string;
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

export class ProjectError extends Error {
  status: number;
  code: ProjectErrorCode | 'PROJECT_NOT_FOUND' | 'PROJECTS_DISABLED' | 'PROJECT_INVALID_INPUT';
  details?: unknown;

  constructor(
    message: string,
    options: {
      status?: number;
      code?: ProjectError['code'];
      details?: unknown;
    } = {},
  ) {
    super(message);
    this.name = 'ProjectError';
    this.status = options.status ?? 400;
    this.code = options.code ?? 'PROJECT_INVALID_INPUT';
    this.details = options.details;
  }
}

export class ProjectNotFoundError extends ProjectError {
  constructor(identifier: string) {
    super(`Project not found: ${identifier}`, { status: 404, code: 'PROJECT_NOT_FOUND' });
    this.name = 'ProjectNotFoundError';
  }
}

export function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

export function isPathStrictlyInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return !!relative && !relative.startsWith('..') && !path.isAbsolute(relative);
}

async function resolvePathWithExistingAncestors(value: string) {
  const resolved = path.resolve(value);
  const unresolved: string[] = [];
  let current = resolved;

  while (true) {
    try {
      const real = await fsp.realpath(current);
      return path.resolve(real, ...unresolved.reverse());
    } catch {
      const parent = path.dirname(current);
      if (parent === current) return resolved;
      unresolved.push(path.basename(current));
      current = parent;
    }
  }
}

async function normalizeStorageRoot(value: string) {
  const resolved = await resolvePathWithExistingAncestors(value);
  if (resolved === path.parse(resolved).root) {
    throw new ProjectError('A project storage root cannot be a filesystem root', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }
  return resolved;
}

async function lstatIfExists(value: string): Promise<fs.Stats | null> {
  try {
    return await fsp.lstat(value);
  } catch (error: unknown) {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') return null;
    throw error;
  }
}

function pathsAreEqual(first: string, second: string) {
  return isPathInside(first, second) && isPathInside(second, first);
}

function pathsOverlap(first: string, second: string) {
  return isPathInside(first, second) || isPathInside(second, first);
}

type ProjectManagementDirectory = '.aitk-staging' | '.aitk-trash';

/**
 * Management directories are destructive-operation boundaries. Never follow a
 * symlink/junction placed at one of these names, even when it points back into
 * the registered storage root.
 */
async function getSafeManagementDirectory(
  storageRootValue: string,
  name: ProjectManagementDirectory,
  options: { create: boolean },
) {
  const storageRoot = await normalizeStorageRoot(storageRootValue);
  let storageStat = await lstatIfExists(storageRoot);
  if (!storageStat && options.create) {
    await fsp.mkdir(storageRoot, { recursive: true });
    storageStat = await lstatIfExists(storageRoot);
  }
  if (storageStat && (!storageStat.isDirectory() || storageStat.isSymbolicLink())) {
    throw new ProjectError('The registered project storage boundary is invalid', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }

  const canonicalStorageRoot = storageStat ? await fsp.realpath(storageRoot) : storageRoot;
  if (!pathsAreEqual(storageRoot, canonicalStorageRoot)) {
    throw new ProjectError('The registered project storage boundary redirects through a filesystem link', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }

  const managedDirectory = path.join(canonicalStorageRoot, name);
  if (!isPathStrictlyInside(canonicalStorageRoot, managedDirectory)) {
    throw new ProjectError('Invalid project management directory', { status: 409, code: 'PROJECT_ROOT_INVALID' });
  }

  let managedStat = await lstatIfExists(managedDirectory);
  if (!managedStat && options.create) {
    try {
      await fsp.mkdir(managedDirectory, { recursive: false });
    } catch (error: unknown) {
      if (!error || typeof error !== 'object' || !('code' in error) || error.code !== 'EEXIST') throw error;
    }
    managedStat = await lstatIfExists(managedDirectory);
  }
  if (managedStat) {
    if (!managedStat.isDirectory() || managedStat.isSymbolicLink()) {
      throw new ProjectError(`${name} must be a real directory, not a filesystem link`, {
        status: 409,
        code: 'PROJECT_ROOT_INVALID',
      });
    }
    const canonicalManagedDirectory = await fsp.realpath(managedDirectory);
    if (
      !pathsAreEqual(managedDirectory, canonicalManagedDirectory) ||
      !isPathStrictlyInside(canonicalStorageRoot, canonicalManagedDirectory)
    ) {
      throw new ProjectError(`${name} redirects outside its registered storage boundary`, {
        status: 409,
        code: 'PROJECT_ROOT_INVALID',
      });
    }
    return canonicalManagedDirectory;
  }

  const canonicalPlannedDirectory = await resolvePathWithExistingAncestors(managedDirectory);
  if (
    !pathsAreEqual(managedDirectory, canonicalPlannedDirectory) ||
    !isPathStrictlyInside(canonicalStorageRoot, canonicalPlannedDirectory)
  ) {
    throw new ProjectError(`Invalid ${name} path`, { status: 409, code: 'PROJECT_ROOT_INVALID' });
  }
  return canonicalPlannedDirectory;
}

async function getSafeManagementChild(
  managementDirectory: string,
  childName: string,
  options: { mustExist?: boolean } = {},
) {
  if (!childName || path.basename(childName) !== childName || childName === '.' || childName === '..') {
    throw new ProjectError('Invalid project management path', { status: 409, code: 'PROJECT_ROOT_INVALID' });
  }
  const child = path.join(managementDirectory, childName);
  if (!isPathStrictlyInside(managementDirectory, child)) {
    throw new ProjectError('Invalid project management path', { status: 409, code: 'PROJECT_ROOT_INVALID' });
  }
  const stat = await lstatIfExists(child);
  if (!stat) {
    if (options.mustExist) {
      throw new ProjectError('Project management path is missing', { status: 409, code: 'PROJECT_ROOT_MISSING' });
    }
    const canonicalPlannedChild = await resolvePathWithExistingAncestors(child);
    if (!pathsAreEqual(child, canonicalPlannedChild) || !isPathStrictlyInside(managementDirectory, canonicalPlannedChild)) {
      throw new ProjectError('Project management path redirects outside its boundary', {
        status: 409,
        code: 'PROJECT_ROOT_INVALID',
      });
    }
    return canonicalPlannedChild;
  }
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    throw new ProjectError('Project management path must be a real directory', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }
  const canonicalChild = await fsp.realpath(child);
  if (!pathsAreEqual(child, canonicalChild) || !isPathStrictlyInside(managementDirectory, canonicalChild)) {
    throw new ProjectError('Project management path redirects outside its boundary', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }
  return canonicalChild;
}

async function assertSafeTreeWithinBoundary(tree: string, boundary: string) {
  const treeStat = await lstatIfExists(tree);
  if (!treeStat) return false;
  if (!treeStat.isDirectory() || treeStat.isSymbolicLink()) {
    throw new ProjectError('Project filesystem entry must be a real directory', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }
  const [canonicalTree, canonicalBoundary] = await Promise.all([
    fsp.realpath(tree),
    resolvePathWithExistingAncestors(boundary),
  ]);
  if (!pathsAreEqual(tree, canonicalTree) || !isPathStrictlyInside(canonicalBoundary, canonicalTree)) {
    throw new ProjectError('Project filesystem entry redirects outside its registered boundary', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }
  return true;
}

async function resolveRegisteredPaths(project: Project) {
  const configuredRoot = await normalizeStorageRoot(await getProjectsRoot());
  const lexicalRoot = path.resolve(project.root_path?.trim() || path.join(configuredRoot, project.slug));
  const lexicalStorageRoot = path.resolve(
    project.storage_root_path?.trim() || (isPathInside(configuredRoot, lexicalRoot) ? configuredRoot : path.dirname(lexicalRoot)),
  );
  const storageRoot = await normalizeStorageRoot(lexicalStorageRoot);
  const root = await resolvePathWithExistingAncestors(lexicalRoot);

  if (!isPathStrictlyInside(storageRoot, root)) {
    throw new ProjectError('Project root is outside its registered storage boundary', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
      details: { project_id: project.id },
    });
  }

  return { root, storageRoot };
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
  const storageRoot = await normalizeStorageRoot(await getProjectsRoot());
  let candidate = base;
  let suffix = 2;
  while ((await db.projects.findBySlug(candidate)) || fs.existsSync(path.join(storageRoot, candidate))) {
    candidate = `${base}-${suffix}`;
    suffix += 1;
  }
  return candidate;
}

async function assertLocalAuthoritativeHome(project: Project) {
  const localInstanceID = await getAITKInstanceID();
  if (project.home_instance_id && project.home_instance_id !== localInstanceID) {
    throw new ProjectError('This project is owned by another AITK instance and cannot be managed here', {
      status: 409,
      code: 'PROJECT_REPLICA_READ_ONLY',
      details: {
        project_id: project.id,
        home_worker_id: project.home_worker_id,
        home_instance_id: project.home_instance_id,
        local_instance_id: localInstanceID,
      },
    });
  }
}

async function assertProjectIntent(project: Project, intent: ProjectScopeIntent) {
  if (intent === 'lifecycle') return;
  if (intent === 'read' && (project.lifecycle_state === 'active' || project.lifecycle_state === 'archived')) return;
  if ((intent === 'write' || intent === 'execute') && project.lifecycle_state === 'active') {
    await assertLocalAuthoritativeHome(project);
    return;
  }
  if (project.lifecycle_state === 'archived') {
    throw new ProjectError('Archived projects are read-only', {
      status: 409,
      code: 'PROJECT_ARCHIVED',
      details: { project_id: project.id },
    });
  }
  throw new ProjectError(`Project operation is unavailable while the project is ${project.lifecycle_state}`, {
    status: 409,
    code: 'PROJECT_OPERATION_IN_PROGRESS',
    details: { project_id: project.id, lifecycle_state: project.lifecycle_state },
  });
}

export async function resolveProjectScope(
  identifier: string,
  options: { intent: ProjectScopeIntent } = { intent: 'read' },
): Promise<Project> {
  await assertProjectsEnabled();
  const normalized = identifier.trim();
  if (!normalized) throw new ProjectNotFoundError(identifier);
  const project = (await db.projects.findById(normalized)) || (await db.projects.findBySlug(cleanProjectSlug(normalized)));
  if (!project) throw new ProjectNotFoundError(identifier);
  await assertProjectIntent(project, options.intent);
  return project;
}

export async function resolveProject(
  identifier: string,
  options: { intent?: ProjectScopeIntent } = {},
): Promise<Project> {
  return resolveProjectScope(identifier, { intent: options.intent ?? 'read' });
}

export async function resolveOptionalProject(
  identifier: unknown,
  options: { intent?: ProjectScopeIntent } = {},
): Promise<Project | null> {
  if (identifier == null) return null;
  if (typeof identifier !== 'string' || !identifier.trim()) {
    throw new ProjectError('project_id must be a project UUID or slug', { status: 400, code: 'PROJECT_INVALID_INPUT' });
  }
  return resolveProject(identifier, { intent: options.intent ?? 'read' });
}

export async function getProjectRoots(project: Project): Promise<ProjectRoots> {
  const { root } = await resolveRegisteredPaths(project);

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

let instanceIDPromise: Promise<string> | null = null;

export async function getAITKInstanceID() {
  if (!instanceIDPromise) {
    instanceIDPromise = (async () => {
      const configured = process.env.AITK_INSTANCE_ID?.trim();
      if (configured) {
        await db.settings.upsert(AITK_INSTANCE_ID_KEY, configured);
        return configured;
      }
      const existing = await db.settings.get(AITK_INSTANCE_ID_KEY);
      if (existing?.value.trim()) return existing.value.trim();
      const generated = randomUUID();
      await db.settings.upsert(AITK_INSTANCE_ID_KEY, generated);
      return (await db.settings.get(AITK_INSTANCE_ID_KEY))?.value.trim() || generated;
    })().catch(error => {
      instanceIDPromise = null;
      throw error;
    });
  }
  return instanceIDPromise;
}

async function readProjectManifest(root: string): Promise<ProjectManifest | null> {
  try {
    const raw: unknown = JSON.parse(await fsp.readFile(path.join(root, PROJECT_MANIFEST_FILE), 'utf8'));
    if (!raw || typeof raw !== 'object') return null;
    const manifest = raw as Partial<ProjectManifest>;
    if (manifest.format !== 'aitk-project' || manifest.version !== 1 || typeof manifest.project_id !== 'string') return null;
    return manifest as ProjectManifest;
  } catch {
    return null;
  }
}

async function writeProjectManifest(root: string, projectID: string, slug: string) {
  const manifest: ProjectManifest = {
    format: 'aitk-project',
    version: 1,
    project_id: projectID,
    slug,
    created_at: new Date().toISOString(),
  };
  await fsp.writeFile(path.join(root, PROJECT_MANIFEST_FILE), `${JSON.stringify(manifest, null, 2)}\n`, {
    encoding: 'utf8',
    flag: 'wx',
  });
}

export async function createProject(input: { name: unknown; slug?: unknown; description?: unknown; badge_asset?: unknown }) {
  await assertProjectsEnabled();
  const name = safeProjectName(input.name);
  if (!name) {
    throw new ProjectError('Project name is required', { status: 400, code: 'PROJECT_INVALID_INPUT' });
  }
  await recoverIncompleteProjectCreations();
  const explicitSlug = typeof input.slug === 'string' && input.slug.trim().length > 0;
  const cleanedRequestedSlug = cleanProjectSlug(explicitSlug ? String(input.slug) : name);
  if (explicitSlug && !cleanedRequestedSlug) {
    throw new ProjectError('Invalid project slug', { status: 400, code: 'PROJECT_INVALID_INPUT' });
  }
  const storageRoot = await normalizeStorageRoot(await getProjectsRoot());
  await fsp.mkdir(storageRoot, { recursive: true });
  const slug = explicitSlug ? cleanedRequestedSlug : await uniqueProjectSlug(cleanedRequestedSlug);
  const finalRoot = path.join(storageRoot, slug);
  if ((await db.projects.findBySlug(slug)) || fs.existsSync(finalRoot)) {
    throw new ProjectError('Project slug already exists', { status: 409, code: 'PROJECT_PATH_COLLISION' });
  }

  const id = randomUUID();
  const stagingRoot = path.join(storageRoot, '.aitk-staging', id);
  const homeInstanceID = await getAITKInstanceID();
  let inserted = false;
  let renamed = false;
  await fsp.mkdir(path.dirname(stagingRoot), { recursive: true });
  await fsp.mkdir(stagingRoot, { recursive: false });
  try {
    await Promise.all(PROJECT_FOLDERS.map(folder => fsp.mkdir(path.join(stagingRoot, folder), { recursive: false })));
    await writeProjectManifest(stagingRoot, id, slug);
    await db.projects.create({
      id,
      slug,
      name,
      description: typeof input.description === 'string' ? input.description.trim().slice(0, 500) : '',
      badge_asset:
        typeof input.badge_asset === 'string' && input.badge_asset.trim()
          ? input.badge_asset.trim()
          : '/assets/projects/project-badge-default.png',
      root_path: finalRoot,
      storage_root_path: storageRoot,
      lifecycle_state: 'creating',
      revision: 0,
      operation_started_at: new Date(),
      home_worker_id: 'local',
      home_instance_id: homeInstanceID,
    });
    inserted = true;
    await fsp.rename(stagingRoot, finalRoot);
    renamed = true;
    const project = await db.projects.compareAndSet(
      id,
      { revision: 0, lifecycle_state: 'creating' },
      { lifecycle_state: 'active', revision: 1, operation_started_at: null, operation_error: null },
    );
    if (!project) throw new Error('Project activation compare-and-set failed');
    return project;
  } catch (error) {
    if (!renamed) {
      await fsp.rm(stagingRoot, { recursive: true, force: true }).catch(() => undefined);
      if (inserted) await db.projects.delete(id).catch(() => undefined);
    } else {
      await db.projects
        .update(id, { operation_error: error instanceof Error ? error.message.slice(0, 500) : 'Project activation failed' })
        .catch(() => undefined);
    }
    throw error;
  }
}

let recoveryPromise: Promise<void> | null = null;

export async function recoverIncompleteProjectCreations() {
  if (recoveryPromise) return recoveryPromise;
  recoveryPromise = (async () => {
    const creating = await db.projects.list({ lifecycle_state: 'creating' });
    for (const project of creating) {
      try {
        const { root, storageRoot } = await resolveRegisteredPaths(project);
        const stagingRoot = path.join(storageRoot, '.aitk-staging', project.id);
        const [finalManifest, stagingManifest] = await Promise.all([
          readProjectManifest(root),
          readProjectManifest(stagingRoot),
        ]);

        if (finalManifest?.project_id === project.id) {
          await db.projects.compareAndSet(
            project.id,
            { revision: project.revision, lifecycle_state: 'creating' },
            {
              lifecycle_state: 'active',
              revision: Math.max(1, project.revision + 1),
              operation_started_at: null,
              operation_error: null,
            },
          );
          await fsp.rm(stagingRoot, { recursive: true, force: true }).catch(() => undefined);
          continue;
        }

        if (stagingManifest?.project_id === project.id && !fs.existsSync(root)) {
          await fsp.mkdir(path.dirname(root), { recursive: true });
          await fsp.rename(stagingRoot, root);
          await db.projects.compareAndSet(
            project.id,
            { revision: project.revision, lifecycle_state: 'creating' },
            {
              lifecycle_state: 'active',
              revision: Math.max(1, project.revision + 1),
              operation_started_at: null,
              operation_error: null,
            },
          );
          continue;
        }

        if (!fs.existsSync(root) && !fs.existsSync(stagingRoot)) {
          await db.projects.delete(project.id);
          continue;
        }

        await db.projects.update(project.id, {
          operation_error: 'Project creation recovery found files with a missing or mismatched manifest',
        });
      } catch (error) {
        await db.projects
          .update(project.id, {
            operation_error: error instanceof Error ? error.message.slice(0, 500) : 'Project creation recovery failed',
          })
          .catch(() => undefined);
      }
    }
  })().finally(() => {
    recoveryPromise = null;
  });
  return recoveryPromise;
}

function assertExpectedRevision(project: Project, expectedRevision: unknown) {
  if (!Number.isInteger(expectedRevision) || Number(expectedRevision) < 0) {
    throw new ProjectError('expected_revision must be a non-negative integer', {
      status: 400,
      code: 'PROJECT_INVALID_INPUT',
    });
  }
  if (project.revision !== Number(expectedRevision)) {
    throw new ProjectError('Project was modified by another operation', {
      status: 409,
      code: 'PROJECT_REVISION_CONFLICT',
      details: { expected_revision: Number(expectedRevision), current_revision: project.revision },
    });
  }
}

async function activeProjectJobs(projectID: string) {
  return db.jobs.list({ project_id: projectID, status: [...ACTIVE_PROJECT_JOB_STATUSES] });
}

function busyJobDetails(jobs: Job[]) {
  return jobs.map(job => ({ id: job.id, name: job.name, status: job.status, worker_id: job.worker_id }));
}

async function assertNoActiveProjectJobs(projectID: string) {
  const jobs = await activeProjectJobs(projectID);
  if (jobs.length > 0) {
    throw new ProjectError('Project has active or queued jobs', {
      status: 409,
      code: 'PROJECT_BUSY',
      details: { jobs: busyJobDetails(jobs) },
    });
  }
}

function revisionConflict(projectID: string) {
  return new ProjectError('Project was modified by another operation', {
    status: 409,
    code: 'PROJECT_REVISION_CONFLICT',
    details: { project_id: projectID },
  });
}

export async function archiveProject(identifier: string, expectedRevision: unknown) {
  const project = await resolveProject(identifier, { intent: 'lifecycle' });
  await assertLocalAuthoritativeHome(project);
  assertExpectedRevision(project, expectedRevision);
  if (project.lifecycle_state === 'archived') return project;
  if (project.lifecycle_state !== 'active') {
    throw new ProjectError(`Project cannot be archived while it is ${project.lifecycle_state}`, {
      status: 409,
      code: 'PROJECT_OPERATION_IN_PROGRESS',
    });
  }
  await assertNoActiveProjectJobs(project.id);
  const archived = await db.projects.compareAndSet(
    project.id,
    { revision: project.revision, lifecycle_state: 'active' },
    {
      lifecycle_state: 'archived',
      archived_at: new Date(),
      revision: project.revision + 1,
      operation_started_at: null,
      operation_error: null,
    },
  );
  if (!archived) throw revisionConflict(project.id);
  return archived;
}

export async function restoreProject(identifier: string, expectedRevision: unknown) {
  const project = await resolveProject(identifier, { intent: 'lifecycle' });
  await assertLocalAuthoritativeHome(project);
  assertExpectedRevision(project, expectedRevision);
  if (project.lifecycle_state === 'active') return project;
  if (project.lifecycle_state !== 'archived') {
    throw new ProjectError(`Project cannot be restored while it is ${project.lifecycle_state}`, {
      status: 409,
      code: 'PROJECT_OPERATION_IN_PROGRESS',
    });
  }

  const roots = await getProjectRoots(project);
  let stat: fs.Stats;
  try {
    stat = await fsp.lstat(roots.root);
  } catch (error: any) {
    if (error?.code === 'ENOENT') {
      throw new ProjectError('The registered project root is missing', {
        status: 409,
        code: 'PROJECT_ROOT_MISSING',
      });
    }
    throw error;
  }
  if (!stat.isDirectory() || stat.isSymbolicLink()) {
    throw new ProjectError('The registered project root is invalid', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }
  await Promise.all(PROJECT_FOLDERS.map(folder => fsp.mkdir(roots[folder], { recursive: true })));
  const restored = await db.projects.compareAndSet(
    project.id,
    { revision: project.revision, lifecycle_state: 'archived' },
    {
      lifecycle_state: 'active',
      archived_at: null,
      revision: project.revision + 1,
      operation_started_at: null,
      operation_error: null,
    },
  );
  if (!restored) throw revisionConflict(project.id);
  return restored;
}

type ProjectInventoryFile = {
  relative_path: string;
  size: number;
  sha256?: string;
};

type ProjectInventory = {
  files: ProjectInventoryFile[];
  directories: string[];
  file_count: number;
  total_bytes: number;
};

async function hashFile(filePath: string) {
  return new Promise<string>((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = fs.createReadStream(filePath);
    stream.on('error', reject);
    stream.on('data', chunk => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

async function scanProjectInventory(root: string, includeHashes = false): Promise<ProjectInventory> {
  const rootStat = await fsp.lstat(root);
  if (!rootStat.isDirectory() || rootStat.isSymbolicLink()) {
    throw new ProjectError('The registered project root is invalid', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }

  const files: ProjectInventoryFile[] = [];
  const directories: string[] = [];
  let totalBytes = 0;
  const walk = async (current: string, relative: string) => {
    const entries = await fsp.readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      const absolutePath = path.join(current, entry.name);
      const relativePath = relative ? path.join(relative, entry.name) : entry.name;
      const stat = await fsp.lstat(absolutePath);
      if (stat.isSymbolicLink()) {
        throw new ProjectError(`Project contains a symbolic link that cannot be managed: ${relativePath}`, {
          status: 409,
          code: 'PROJECT_ROOT_INVALID',
        });
      }
      if (stat.isDirectory()) {
        directories.push(relativePath);
        await walk(absolutePath, relativePath);
      } else if (stat.isFile()) {
        totalBytes += stat.size;
        files.push({
          relative_path: relativePath,
          size: stat.size,
          ...(includeHashes ? { sha256: await hashFile(absolutePath) } : {}),
        });
      } else {
        throw new ProjectError(`Project contains an unsupported filesystem entry: ${relativePath}`, {
          status: 409,
          code: 'PROJECT_ROOT_INVALID',
        });
      }
    }
  };
  await walk(root, '');
  files.sort((a, b) => a.relative_path.localeCompare(b.relative_path));
  directories.sort((a, b) => a.localeCompare(b));
  return { files, directories, file_count: files.length, total_bytes: totalBytes };
}

export type ProjectPurgePreview = {
  project_id: string;
  slug: string;
  revision: number;
  root_path: string;
  job_count: number;
  file_count: number;
  total_bytes: number;
  blockers: Array<{ code: string; message: string; job_ids?: string[] }>;
  warnings: Array<{ code: string; message: string }>;
  confirmation_text: string;
  can_purge: boolean;
};

export async function getProjectPurgePreview(identifier: string): Promise<ProjectPurgePreview> {
  const project = await resolveProject(identifier, { intent: 'lifecycle' });
  await assertLocalAuthoritativeHome(project);
  const roots = await getProjectRoots(project);
  const { storageRoot } = await resolveRegisteredPaths(project);
  const trashRoot = await getSafeManagementDirectory(storageRoot, '.aitk-trash', { create: false });
  const quarantineRoot = await getSafeManagementChild(trashRoot, project.id);
  const quarantineExists = (await lstatIfExists(quarantineRoot)) !== null;
  const jobs = await db.jobs.list({ project_id: project.id });
  const activeJobs = jobs.filter(job => (ACTIVE_PROJECT_JOB_STATUSES as readonly string[]).includes(job.status));
  const remoteJobs = jobs.filter(job => job.worker_id !== 'local' || !!job.remote_job_id);
  const replicas = (await db.projectReplicas.listByProject(project.id)).filter(
    replica => !(replica.worker_id === 'local' && replica.role === 'home') && replica.state !== 'detached',
  );
  const activeSyncOperations = (
    await db.projectSyncOperations.list({ project_id: project.id })
  ).filter(operation => !['completed', 'failed', 'cancelled'].includes(operation.status));
  const blockers: ProjectPurgePreview['blockers'] = [];
  const warnings: ProjectPurgePreview['warnings'] = [];

  if (project.lifecycle_state !== 'archived') {
    blockers.push({ code: 'PROJECT_NOT_ARCHIVED', message: 'Archive the project before permanently purging it.' });
  }
  if (quarantineExists) {
    blockers.push({ code: 'PROJECT_PATH_COLLISION', message: 'A purge quarantine path already exists.' });
  }
  if (activeJobs.length > 0) {
    blockers.push({
      code: 'PROJECT_BUSY',
      message: 'Active or queued jobs must be stopped and removed first.',
      job_ids: activeJobs.map(job => job.id),
    });
  }
  if (remoteJobs.length > 0) {
    blockers.push({
      code: 'PROJECT_REMOTE_JOBS_REMAIN',
      message: 'Remote-backed jobs must be explicitly removed while their workers are reachable.',
      job_ids: remoteJobs.map(job => job.id),
    });
  }
  if (replicas.length > 0) {
    blockers.push({
      code: 'PROJECT_REPLICAS_REMAIN',
      message: 'All project replicas must be detached or purged before the home project can be purged.',
    });
  }
  if (activeSyncOperations.length > 0) {
    blockers.push({
      code: 'PROJECT_SYNC_ACTIVE',
      message: 'Project sync operations must finish or be cancelled before purge.',
    });
  }

  let inventory: ProjectInventory = { files: [], directories: [], file_count: 0, total_bytes: 0 };
  try {
    inventory = await scanProjectInventory(roots.root);
  } catch (error: any) {
    if (error?.code === 'ENOENT') {
      warnings.push({ code: 'PROJECT_ROOT_MISSING', message: 'The project root is already missing; only metadata can be purged.' });
    } else {
      throw error;
    }
  }

  return {
    project_id: project.id,
    slug: project.slug,
    revision: project.revision,
    root_path: roots.root,
    job_count: jobs.length,
    file_count: inventory.file_count,
    total_bytes: inventory.total_bytes,
    blockers,
    warnings,
    confirmation_text: project.slug,
    can_purge: blockers.length === 0,
  };
}

async function removeProjectDatasetWatchers(projectID: string) {
  const watchersKey = 'DATASET_WATCHERS_V1';
  const statusKey = 'DATASET_WATCHER_STATUS_V1';
  const row = await db.settings.get(watchersKey);
  if (!row?.value) return;
  try {
    const parsed = JSON.parse(row.value);
    const watchers = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.watchers) ? parsed.watchers : [];
    const removedIDs = watchers
      .filter((watcher: any) => watcher && watcher.projectID === projectID)
      .map((watcher: any) => String(watcher.id));
    const retained = watchers.filter((watcher: any) => !watcher || watcher.projectID !== projectID);
    await db.settings.upsert(watchersKey, JSON.stringify({ version: 1, watchers: retained }));
    if (removedIDs.length === 0) return;
    const statusRow = await db.settings.get(statusKey);
    if (!statusRow?.value) return;
    const statusStore = JSON.parse(statusRow.value);
    if (statusStore?.statuses && typeof statusStore.statuses === 'object') {
      for (const id of removedIDs) delete statusStore.statuses[id];
      await db.settings.upsert(statusKey, JSON.stringify(statusStore));
    }
  } catch {
    // Invalid legacy watcher state is left intact rather than risking unrelated watcher deletion.
  }
}

export async function purgeProject(
  identifier: string,
  input: { expected_revision: unknown; confirmation: unknown; scope: unknown },
) {
  let project = await resolveProject(identifier, { intent: 'lifecycle' });
  await assertLocalAuthoritativeHome(project);
  if (input.scope !== 'project_and_all_data') {
    throw new ProjectError('scope must be project_and_all_data', { status: 400, code: 'PROJECT_INVALID_INPUT' });
  }
  if (input.confirmation !== project.slug) {
    throw new ProjectError('Project purge confirmation does not match the current slug', {
      status: 409,
      code: 'PROJECT_PURGE_CONFIRMATION_MISMATCH',
    });
  }

  let preview: ProjectPurgePreview;
  if (project.lifecycle_state === 'purging') {
    preview = {
      project_id: project.id,
      slug: project.slug,
      revision: project.revision,
      root_path: project.root_path,
      job_count: (await db.jobs.list({ project_id: project.id })).length,
      file_count: 0,
      total_bytes: 0,
      blockers: [],
      warnings: [],
      confirmation_text: project.slug,
      can_purge: true,
    };
  } else {
    assertExpectedRevision(project, input.expected_revision);
    preview = await getProjectPurgePreview(project.id);
    if (!preview.can_purge) {
      throw new ProjectError('Project cannot be purged until all blockers are resolved', {
        status: 409,
        code: preview.blockers.some(blocker => blocker.code === 'PROJECT_BUSY') ? 'PROJECT_BUSY' : 'PROJECT_OPERATION_IN_PROGRESS',
        details: { blockers: preview.blockers },
      });
    }
    const marked = await db.projects.compareAndSet(
      project.id,
      { revision: project.revision, lifecycle_state: 'archived' },
      {
        lifecycle_state: 'purging',
        revision: project.revision + 1,
        operation_started_at: new Date(),
        operation_error: null,
      },
    );
    if (!marked) throw revisionConflict(project.id);
    project = marked;
  }

  let destructivePhaseStarted = false;
  try {
    const { root, storageRoot } = await resolveRegisteredPaths(project);
    const trashRoot = await getSafeManagementDirectory(storageRoot, '.aitk-trash', { create: true });
    const quarantineRoot = await getSafeManagementChild(trashRoot, project.id);
    const sourceExists = await assertSafeTreeWithinBoundary(root, storageRoot);
    const quarantineExists = (await lstatIfExists(quarantineRoot)) !== null;
    if (sourceExists && quarantineExists) {
      throw new ProjectError('Purge quarantine path already exists', { status: 409, code: 'PROJECT_PATH_COLLISION' });
    }
    if (sourceExists) await fsp.rename(root, quarantineRoot);
    destructivePhaseStarted = true;
    if (sourceExists || quarantineExists) {
      const verifiedTrashRoot = await getSafeManagementDirectory(storageRoot, '.aitk-trash', { create: false });
      const verifiedQuarantineRoot = await getSafeManagementChild(verifiedTrashRoot, project.id, { mustExist: true });
      await fsp.rm(verifiedQuarantineRoot, { recursive: true, force: true });
    }

    const jobs = await db.jobs.list({ project_id: project.id });
    await removeProjectDatasetWatchers(project.id);
    for (const job of jobs) {
      await clearDurableEncryptedDatasetKeys(job.id);
      await db.metrics.deleteForJob(job.id);
      await db.jobReplicas.deleteByJob(job.id);
      await db.jobs.delete(job.id);
    }
    await db.projectSyncOperations.deleteByProject(project.id);
    await db.projectReplicas.deleteByProject(project.id);
    await db.projects.delete(project.id);

    return {
      success: true as const,
      purged: {
        id: project.id,
        slug: project.slug,
        deleted_jobs: jobs.length,
        deleted_files: preview.file_count,
        deleted_bytes: preview.total_bytes,
      },
    };
  } catch (error) {
    const operationError = error instanceof Error ? error.message.slice(0, 500) : 'Project purge failed';
    if (!destructivePhaseStarted) {
      await db.projects
        .compareAndSet(
          project.id,
          { revision: project.revision, lifecycle_state: 'purging' },
          {
            lifecycle_state: 'archived',
            revision: project.revision + 1,
            operation_started_at: null,
            operation_error: operationError,
          },
        )
        .catch(() => undefined);
    } else {
      await db.projects.update(project.id, { operation_error: operationError }).catch(() => undefined);
    }
    throw error;
  }
}

async function nearestExistingDirectory(value: string) {
  let current = path.resolve(value);
  while (!fs.existsSync(current)) {
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return current;
}

export type ProjectRelocatePreview = {
  project_id: string;
  revision: number;
  source_root: string;
  source_storage_root: string;
  destination_storage_root: string;
  target_root: string;
  mode: 'copy' | 'move';
  file_count: number;
  total_bytes: number;
  available_bytes: number | null;
  blockers: Array<{ code: string; message: string }>;
  can_relocate: boolean;
};

async function getRelocationPaths(project: Project, sourceRootValue: string, destinationStorageRootValue: string) {
  const sourceRoot = await resolvePathWithExistingAncestors(sourceRootValue);
  const destinationStorageRoot = await normalizeStorageRoot(destinationStorageRootValue);
  const targetRoot = await resolvePathWithExistingAncestors(path.join(destinationStorageRoot, project.slug));
  const stagingParent = await getSafeManagementDirectory(destinationStorageRoot, '.aitk-staging', { create: false });
  const stagingRoot = await getSafeManagementChild(
    stagingParent,
    `relocate-${project.id}-${project.revision + 1}`,
  );
  return { sourceRoot, destinationStorageRoot, targetRoot, stagingParent, stagingRoot };
}

export async function getProjectRelocatePreview(
  identifier: string,
  input: { destination_storage_root?: unknown; mode: unknown },
): Promise<ProjectRelocatePreview> {
  const project = await resolveProject(identifier, { intent: 'lifecycle' });
  await assertLocalAuthoritativeHome(project);
  if (input.mode !== 'copy' && input.mode !== 'move') {
    throw new ProjectError('mode must be copy or move', { status: 400, code: 'PROJECT_INVALID_INPUT' });
  }
  const source = await getProjectRoots(project);
  const registeredSource = await resolveRegisteredPaths(project);
  const requestedDestinationStorageRoot = await normalizeStorageRoot(
    typeof input.destination_storage_root === 'string' && input.destination_storage_root.trim()
      ? input.destination_storage_root.trim()
      : await getProjectsRoot(),
  );
  const relocationPaths = await getRelocationPaths(project, source.root, requestedDestinationStorageRoot);
  const { destinationStorageRoot, targetRoot, stagingParent, stagingRoot } = relocationPaths;
  const blockers: ProjectRelocatePreview['blockers'] = [];
  if (project.lifecycle_state !== 'archived') {
    blockers.push({ code: 'PROJECT_NOT_ARCHIVED', message: 'Archive the project before relocating it.' });
  }
  const activeJobs = await activeProjectJobs(project.id);
  if (activeJobs.length > 0) blockers.push({ code: 'PROJECT_BUSY', message: 'Active or queued jobs block relocation.' });
  const activeSyncOperations = (
    await db.projectSyncOperations.list({ project_id: project.id })
  ).filter(operation => !['completed', 'failed', 'cancelled'].includes(operation.status));
  if (activeSyncOperations.length > 0) {
    blockers.push({ code: 'PROJECT_SYNC_ACTIVE', message: 'Project sync operations must finish or be cancelled first.' });
  }
  const targetStat = await lstatIfExists(path.join(destinationStorageRoot, project.slug));
  if (targetStat) blockers.push({ code: 'PROJECT_PATH_COLLISION', message: 'The destination project path already exists.' });
  if (await lstatIfExists(stagingRoot)) {
    blockers.push({ code: 'PROJECT_PATH_COLLISION', message: 'A relocation staging path already exists.' });
  }

  // targetRoot and stagingRoot intentionally live under destinationStorageRoot.
  // Every other ancestor/descendant relationship can make a copy recurse into
  // itself or make move cleanup delete the newly committed target.
  if (
    pathsOverlap(relocationPaths.sourceRoot, destinationStorageRoot) ||
    pathsOverlap(relocationPaths.sourceRoot, targetRoot) ||
    pathsOverlap(relocationPaths.sourceRoot, stagingParent) ||
    pathsOverlap(relocationPaths.sourceRoot, stagingRoot) ||
    pathsOverlap(targetRoot, stagingParent) ||
    pathsOverlap(targetRoot, stagingRoot)
  ) {
    blockers.push({
      code: 'PROJECT_PATH_OVERLAP',
      message: 'Source, destination, target, and staging paths must not overlap.',
    });
  }
  if (
    !isPathStrictlyInside(destinationStorageRoot, targetRoot) ||
    !isPathStrictlyInside(destinationStorageRoot, stagingParent) ||
    !isPathStrictlyInside(stagingParent, stagingRoot)
  ) {
    blockers.push({
      code: 'PROJECT_ROOT_INVALID',
      message: 'Relocation paths escape the registered destination storage boundary.',
    });
  }
  if (input.mode === 'move') {
    await getSafeManagementDirectory(registeredSource.storageRoot, '.aitk-trash', { create: false });
  }

  const inventory = await scanProjectInventory(source.root);
  let availableBytes: number | null = null;
  try {
    const statfs = await fsp.statfs(await nearestExistingDirectory(destinationStorageRoot));
    availableBytes = Number(statfs.bavail) * Number(statfs.bsize);
    if (availableBytes < inventory.total_bytes) {
      blockers.push({ code: 'PROJECT_INSUFFICIENT_STORAGE', message: 'The destination does not have enough free space.' });
    }
  } catch {
    availableBytes = null;
  }

  return {
    project_id: project.id,
    revision: project.revision,
    source_root: source.root,
    source_storage_root: registeredSource.storageRoot,
    destination_storage_root: destinationStorageRoot,
    target_root: targetRoot,
    mode: input.mode,
    file_count: inventory.file_count,
    total_bytes: inventory.total_bytes,
    available_bytes: availableBytes,
    blockers,
    can_relocate: blockers.length === 0,
  };
}

async function copyAndVerifyProject(sourceRoot: string, stagingRoot: string) {
  const source = await scanProjectInventory(sourceRoot, true);
  await fsp.mkdir(stagingRoot, { recursive: false });
  for (const directory of source.directories) await fsp.mkdir(path.join(stagingRoot, directory), { recursive: true });
  for (const file of source.files) {
    const destination = path.join(stagingRoot, file.relative_path);
    await fsp.mkdir(path.dirname(destination), { recursive: true });
    await fsp.copyFile(path.join(sourceRoot, file.relative_path), destination, fs.constants.COPYFILE_EXCL);
  }
  const copied = await scanProjectInventory(stagingRoot, true);
  if (source.files.length !== copied.files.length || source.total_bytes !== copied.total_bytes) {
    throw new Error('Relocated project inventory does not match the source');
  }
  for (let index = 0; index < source.files.length; index += 1) {
    const sourceFile = source.files[index];
    const copiedFile = copied.files[index];
    if (
      sourceFile.relative_path !== copiedFile.relative_path ||
      sourceFile.size !== copiedFile.size ||
      sourceFile.sha256 !== copiedFile.sha256
    ) {
      throw new Error(`Relocated project checksum mismatch: ${sourceFile.relative_path}`);
    }
  }
  return source;
}

export async function relocateProject(
  identifier: string,
  input: {
    destination_storage_root?: unknown;
    mode: unknown;
    expected_revision: unknown;
    confirmation?: unknown;
  },
) {
  let project = await resolveProject(identifier, { intent: 'lifecycle' });
  await assertLocalAuthoritativeHome(project);
  assertExpectedRevision(project, input.expected_revision);
  const preview = await getProjectRelocatePreview(project.id, input);
  if (!preview.can_relocate) {
    throw new ProjectError('Project cannot be relocated until all blockers are resolved', {
      status: 409,
      code: preview.blockers.some(blocker => blocker.code === 'PROJECT_PATH_COLLISION')
        ? 'PROJECT_PATH_COLLISION'
        : preview.blockers.some(blocker => blocker.code === 'PROJECT_BUSY')
          ? 'PROJECT_BUSY'
          : preview.blockers.some(blocker =>
                ['PROJECT_PATH_OVERLAP', 'PROJECT_ROOT_INVALID'].includes(blocker.code),
            )
            ? 'PROJECT_ROOT_INVALID'
            : 'PROJECT_OPERATION_IN_PROGRESS',
      details: { blockers: preview.blockers },
    });
  }
  if (input.mode === 'move' && input.confirmation !== project.slug) {
    throw new ProjectError('Project relocation confirmation does not match the current slug', {
      status: 409,
      code: 'PROJECT_PURGE_CONFIRMATION_MISMATCH',
    });
  }

  const marked = await db.projects.compareAndSet(
    project.id,
    { revision: project.revision, lifecycle_state: 'archived' },
    {
      lifecycle_state: 'relocating',
      revision: project.revision + 1,
      operation_started_at: new Date(),
      operation_error: null,
    },
  );
  if (!marked) throw revisionConflict(project.id);
  project = marked;

  let stagingParent = path.join(preview.destination_storage_root, '.aitk-staging');
  let stagingRoot = path.join(stagingParent, `relocate-${project.id}-${project.revision}`);
  let targetCommitted = false;
  let switched: Project | null = null;
  try {
    stagingParent = await getSafeManagementDirectory(preview.destination_storage_root, '.aitk-staging', { create: true });
    stagingRoot = await getSafeManagementChild(stagingParent, `relocate-${project.id}-${project.revision}`);
    if (await lstatIfExists(stagingRoot)) {
      throw new ProjectError('A relocation staging path already exists', {
        status: 409,
        code: 'PROJECT_PATH_COLLISION',
      });
    }
    await copyAndVerifyProject(preview.source_root, stagingRoot);
    const verifiedStagingParent = await getSafeManagementDirectory(
      preview.destination_storage_root,
      '.aitk-staging',
      { create: false },
    );
    const verifiedStagingRoot = await getSafeManagementChild(
      verifiedStagingParent,
      `relocate-${project.id}-${project.revision}`,
      { mustExist: true },
    );
    if (await lstatIfExists(preview.target_root)) {
      throw new ProjectError('The destination project path already exists', {
        status: 409,
        code: 'PROJECT_PATH_COLLISION',
      });
    }
    await fsp.rename(verifiedStagingRoot, preview.target_root);
    targetCommitted = true;
    await assertSafeTreeWithinBoundary(preview.target_root, preview.destination_storage_root);
    switched = await db.projects.compareAndSet(
      project.id,
      { revision: project.revision, lifecycle_state: 'relocating' },
      {
        root_path: preview.target_root,
        storage_root_path: preview.destination_storage_root,
        lifecycle_state: 'archived',
        revision: project.revision + 1,
        operation_started_at: null,
        operation_error: null,
      },
    );
    if (!switched) throw revisionConflict(project.id);
  } catch (error) {
    await (async () => {
      const verifiedStagingParent = await getSafeManagementDirectory(
        preview.destination_storage_root,
        '.aitk-staging',
        { create: false },
      );
      const existingStagingRoot = await lstatIfExists(stagingRoot);
      if (!existingStagingRoot) return;
      const verifiedStagingRoot = await getSafeManagementChild(
        verifiedStagingParent,
        `relocate-${project.id}-${project.revision}`,
        { mustExist: true },
      );
      await fsp.rm(verifiedStagingRoot, { recursive: true, force: true });
    })().catch(() => undefined);
    if (targetCommitted && !switched) {
      await (async () => {
        if (await assertSafeTreeWithinBoundary(preview.target_root, preview.destination_storage_root)) {
          await fsp.rm(preview.target_root, { recursive: true, force: true });
        }
      })().catch(() => undefined);
    }
    if (!switched) {
      await db.projects
        .compareAndSet(
          project.id,
          { revision: project.revision, lifecycle_state: 'relocating' },
          {
            lifecycle_state: 'archived',
            revision: project.revision + 1,
            operation_started_at: null,
            operation_error: error instanceof Error ? error.message.slice(0, 500) : 'Project relocation failed',
          },
        )
        .catch(() => undefined);
    }
    throw error;
  }

  let cleanupWarning: string | null = null;
  let oldQuarantinePath: string | null = null;
  if (input.mode === 'move') {
    try {
      const oldStorageRoot = preview.source_storage_root;
      const oldTrashRoot = await getSafeManagementDirectory(oldStorageRoot, '.aitk-trash', { create: true });
      oldQuarantinePath = await getSafeManagementChild(oldTrashRoot, `relocate-${project.id}-${project.revision}`);
      if (await lstatIfExists(oldQuarantinePath)) {
        throw new ProjectError('Relocation cleanup quarantine path already exists', {
          status: 409,
          code: 'PROJECT_PATH_COLLISION',
        });
      }
      await assertSafeTreeWithinBoundary(preview.source_root, oldStorageRoot);
      await fsp.rename(preview.source_root, oldQuarantinePath);
      const verifiedOldTrashRoot = await getSafeManagementDirectory(oldStorageRoot, '.aitk-trash', { create: false });
      const verifiedOldQuarantinePath = await getSafeManagementChild(
        verifiedOldTrashRoot,
        `relocate-${project.id}-${project.revision}`,
        { mustExist: true },
      );
      await fsp.rm(verifiedOldQuarantinePath, { recursive: true, force: true });
      oldQuarantinePath = null;
    } catch (error) {
      cleanupWarning = error instanceof Error ? error.message : 'The old project root could not be removed';
      await db.projects.update(project.id, { operation_error: cleanupWarning.slice(0, 500) });
    }
  }

  return {
    project: switched,
    copied_files: preview.file_count,
    copied_bytes: preview.total_bytes,
    source_backup_path: input.mode === 'copy' ? preview.source_root : oldQuarantinePath,
    warning: cleanupWarning,
    target_committed: targetCommitted,
  };
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

type ScopedCopyInventory = {
  canonicalSource: string;
  kind: 'file' | 'directory';
  directories: string[];
  files: string[];
};

async function scanScopedCopySource(sourceValue: string, allowedRootValue: string): Promise<ScopedCopyInventory> {
  const source = path.resolve(sourceValue);
  const allowedRoot = await resolvePathWithExistingAncestors(allowedRootValue);
  const allowedRootStat = await lstatIfExists(allowedRoot);
  if (!allowedRootStat?.isDirectory() || allowedRootStat.isSymbolicLink()) {
    throw new ProjectError('The configured datasets root is invalid', { status: 409, code: 'PROJECT_ROOT_INVALID' });
  }
  const canonicalAllowedRoot = await fsp.realpath(allowedRoot);
  const sourceStat = await lstatIfExists(source);
  if (!sourceStat) {
    throw new ProjectError('Dataset source is missing', { status: 409, code: 'PROJECT_ROOT_MISSING' });
  }
  if (sourceStat.isSymbolicLink()) {
    throw new ProjectError('Dataset sources containing filesystem links cannot be copied into a project', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }
  const canonicalSource = await fsp.realpath(source);
  if (!pathsAreEqual(source, canonicalSource) || !isPathInside(canonicalAllowedRoot, canonicalSource)) {
    throw new ProjectError('Dataset source escapes the configured datasets root', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }
  if (sourceStat.isFile()) {
    return { canonicalSource, kind: 'file', directories: [], files: [''] };
  }
  if (!sourceStat.isDirectory()) {
    throw new ProjectError('Dataset source contains an unsupported filesystem entry', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }

  const directories: string[] = [];
  const files: string[] = [];
  const walk = async (current: string, relative: string) => {
    const entries = await fsp.readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      const absoluteEntry = path.join(current, entry.name);
      const relativeEntry = relative ? path.join(relative, entry.name) : entry.name;
      const entryStat = await fsp.lstat(absoluteEntry);
      if (entryStat.isSymbolicLink()) {
        throw new ProjectError(`Dataset source contains a filesystem link: ${relativeEntry}`, {
          status: 409,
          code: 'PROJECT_ROOT_INVALID',
        });
      }
      const canonicalEntry = await fsp.realpath(absoluteEntry);
      if (
        !pathsAreEqual(absoluteEntry, canonicalEntry) ||
        !isPathStrictlyInside(canonicalSource, canonicalEntry) ||
        !isPathInside(canonicalAllowedRoot, canonicalEntry)
      ) {
        throw new ProjectError(`Dataset source entry escapes its allowed boundary: ${relativeEntry}`, {
          status: 409,
          code: 'PROJECT_ROOT_INVALID',
        });
      }
      if (entryStat.isDirectory()) {
        directories.push(relativeEntry);
        await walk(absoluteEntry, relativeEntry);
      } else if (entryStat.isFile()) {
        files.push(relativeEntry);
      } else {
        throw new ProjectError(`Dataset source contains an unsupported filesystem entry: ${relativeEntry}`, {
          status: 409,
          code: 'PROJECT_ROOT_INVALID',
        });
      }
    }
  };
  await walk(canonicalSource, '');
  directories.sort((first, second) => first.localeCompare(second));
  files.sort((first, second) => first.localeCompare(second));
  return { canonicalSource, kind: 'directory', directories, files };
}

async function getSafeProjectDatasetRoot(roots: ProjectRoots) {
  const [projectStat, datasetsStat] = await Promise.all([
    lstatIfExists(roots.root),
    lstatIfExists(roots.datasets),
  ]);
  if (
    !projectStat?.isDirectory() ||
    projectStat.isSymbolicLink() ||
    !datasetsStat?.isDirectory() ||
    datasetsStat.isSymbolicLink()
  ) {
    throw new ProjectError('The project dataset boundary is invalid', { status: 409, code: 'PROJECT_ROOT_INVALID' });
  }
  const [canonicalProjectRoot, canonicalDatasetsRoot] = await Promise.all([
    fsp.realpath(roots.root),
    fsp.realpath(roots.datasets),
  ]);
  if (
    !pathsAreEqual(roots.root, canonicalProjectRoot) ||
    !pathsAreEqual(roots.datasets, canonicalDatasetsRoot) ||
    !isPathStrictlyInside(canonicalProjectRoot, canonicalDatasetsRoot)
  ) {
    throw new ProjectError('The project dataset boundary redirects outside its project', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }
  return canonicalDatasetsRoot;
}

async function copyScopedSourceIntoProject(
  source: string,
  target: string,
  allowedSourceRoot: string,
  allowedTargetRoot: string,
) {
  const inventory = await scanScopedCopySource(source, allowedSourceRoot);
  const canonicalTarget = await resolvePathWithExistingAncestors(target);
  if (!pathsAreEqual(target, canonicalTarget) || !isPathStrictlyInside(allowedTargetRoot, canonicalTarget)) {
    throw new ProjectError('Dataset copy target escapes the project dataset boundary', {
      status: 409,
      code: 'PROJECT_ROOT_INVALID',
    });
  }

  const existingTarget = await lstatIfExists(canonicalTarget);
  if (existingTarget) {
    if (existingTarget.isSymbolicLink()) {
      throw new ProjectError('Dataset copy target is a filesystem link', {
        status: 409,
        code: 'PROJECT_ROOT_INVALID',
      });
    }
    const realTarget = await fsp.realpath(canonicalTarget);
    if (!pathsAreEqual(canonicalTarget, realTarget) || !isPathStrictlyInside(allowedTargetRoot, realTarget)) {
      throw new ProjectError('Dataset copy target redirects outside the project dataset boundary', {
        status: 409,
        code: 'PROJECT_ROOT_INVALID',
      });
    }
    return canonicalTarget;
  }

  let targetCreated = false;
  try {
    if (inventory.kind === 'file') {
      const currentSourceStat = await fsp.lstat(inventory.canonicalSource);
      if (!currentSourceStat.isFile() || currentSourceStat.isSymbolicLink()) {
        throw new ProjectError('Dataset source changed before it could be copied', {
          status: 409,
          code: 'PROJECT_ROOT_INVALID',
        });
      }
      targetCreated = true;
      await fsp.copyFile(inventory.canonicalSource, canonicalTarget, fs.constants.COPYFILE_EXCL);
      return canonicalTarget;
    }

    await fsp.mkdir(canonicalTarget, { recursive: false });
    targetCreated = true;
    for (const directory of inventory.directories) {
      const destinationDirectory = path.join(canonicalTarget, directory);
      if (!isPathStrictlyInside(canonicalTarget, destinationDirectory)) {
        throw new ProjectError('Dataset directory escapes the copy target', {
          status: 409,
          code: 'PROJECT_ROOT_INVALID',
        });
      }
      await fsp.mkdir(destinationDirectory, { recursive: false });
    }
    for (const relativeFile of inventory.files) {
      const sourceFile = path.join(inventory.canonicalSource, relativeFile);
      const destinationFile = path.join(canonicalTarget, relativeFile);
      if (
        !isPathStrictlyInside(inventory.canonicalSource, sourceFile) ||
        !isPathStrictlyInside(canonicalTarget, destinationFile)
      ) {
        throw new ProjectError('Dataset file escapes its copy boundary', {
          status: 409,
          code: 'PROJECT_ROOT_INVALID',
        });
      }
      const currentSourceStat = await fsp.lstat(sourceFile);
      if (!currentSourceStat.isFile() || currentSourceStat.isSymbolicLink()) {
        throw new ProjectError(`Dataset source changed before it could be copied: ${relativeFile}`, {
          status: 409,
          code: 'PROJECT_ROOT_INVALID',
        });
      }
      const canonicalSourceFile = await fsp.realpath(sourceFile);
      if (!pathsAreEqual(sourceFile, canonicalSourceFile) || !isPathStrictlyInside(inventory.canonicalSource, canonicalSourceFile)) {
        throw new ProjectError(`Dataset source file redirects outside its boundary: ${relativeFile}`, {
          status: 409,
          code: 'PROJECT_ROOT_INVALID',
        });
      }
      await fsp.copyFile(canonicalSourceFile, destinationFile, fs.constants.COPYFILE_EXCL);
    }
    return canonicalTarget;
  } catch (error) {
    if (targetCreated) {
      const targetStat = await lstatIfExists(canonicalTarget).catch(() => null);
      if (targetStat?.isDirectory() && !targetStat.isSymbolicLink()) {
        const realTarget = await fsp.realpath(canonicalTarget).catch(() => null);
        if (realTarget && pathsAreEqual(canonicalTarget, realTarget) && isPathStrictlyInside(allowedTargetRoot, realTarget)) {
          await fsp.rm(realTarget, { recursive: true, force: true }).catch(() => undefined);
        }
      } else if (targetStat?.isFile() && !targetStat.isSymbolicLink()) {
        const realTarget = await fsp.realpath(canonicalTarget).catch(() => null);
        if (realTarget && pathsAreEqual(canonicalTarget, realTarget) && isPathStrictlyInside(allowedTargetRoot, realTarget)) {
          await fsp.rm(realTarget, { force: true }).catch(() => undefined);
        }
      }
    }
    throw error;
  }
}

async function copyPathIntoProject(value: string, roots: ProjectRoots, globalDatasetsRoot: string) {
  if (!value.trim() || isProtocolPath(value)) return value;

  const source = path.resolve(value);
  if (isPathInside(roots.root, source)) return source;
  if (!fs.existsSync(source)) return value;

  const lexicalGlobalRoot = path.resolve(globalDatasetsRoot);
  const globalRoot = await resolvePathWithExistingAncestors(lexicalGlobalRoot);
  const canonicalSource = await resolvePathWithExistingAncestors(source);
  if (!isPathInside(lexicalGlobalRoot, source) && !isPathInside(globalRoot, canonicalSource)) return value;

  const projectDatasetsRoot = await getSafeProjectDatasetRoot(roots);
  const target = path.join(projectDatasetsRoot, safeCopyName(source));
  return copyScopedSourceIntoProject(source, target, globalRoot, projectDatasetsRoot);
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
  await assertProjectsEnabled();
  await assertProjectIntent(project, 'write');
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

export async function assertProjectJobEnabled(
  job: Pick<Job, 'project_id'> | null | undefined,
  intent: ProjectScopeIntent = 'read',
) {
  if (job?.project_id) {
    await assertProjectsEnabled();
    await resolveProject(job.project_id, { intent });
  }
}

export async function getAllowedProjectRootIfExists() {
  if (!(await areProjectsEnabled())) return null;
  const root = path.resolve(await getProjectsRoot());
  return fs.existsSync(root) ? fs.promises.realpath(root).catch(() => root) : null;
}
