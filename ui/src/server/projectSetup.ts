import { createHash } from 'crypto';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { TOOLKIT_ROOT } from '../paths';
import type { Job, Project } from '@/types';
import { db, type JobCreateInput } from './db';
import {
  getProjectRoots,
  isPathInside,
  PROJECT_FOLDERS,
  ProjectError,
  resolveProject,
  type ProjectRoots,
} from './projects';

export type ProjectSetupMode = 'blank' | 'import' | 'clone';

export type ProjectSetupRequest = {
  mode: ProjectSetupMode;
  importRoot?: string;
  cloneFromProjectID?: string;
};

export type ProjectSetupResult = {
  mode: ProjectSetupMode;
  status: 'completed' | 'failed';
  copiedFiles: number;
  copiedBytes: number;
  clonedJobs: number;
  error: string | null;
};

export type ProjectSetupDeps = {
  getProjectRoots(project: Project): Promise<ProjectRoots>;
  resolveProject(identifier: string): Promise<Project>;
  listProjectJobs(projectID: string): Promise<Job[]>;
  createJob(input: JobCreateInput): Promise<Job>;
  deleteJob(jobID: string): Promise<unknown>;
  updateProjectOperationError(projectID: string, error: string | null): Promise<unknown>;
};

const defaultProjectSetupDeps: ProjectSetupDeps = {
  getProjectRoots,
  resolveProject: identifier => resolveProject(identifier, { intent: 'read' }),
  listProjectJobs: projectID => db.jobs.list({ project_id: projectID }),
  createJob: input => db.jobs.create(input),
  deleteJob: jobID => db.jobs.delete(jobID),
  updateProjectOperationError: (projectID, error) => db.projects.update(projectID, { operation_error: error }),
};

type ManifestEntry = { relativePath: string; size: number; sha256: string };

const COPY_ZONES = PROJECT_FOLDERS.filter(zone => zone !== 'cache');
const RUNTIME_FILE_NAMES = new Set([
  '.aitk-project.json',
  '.hf_download_progress.json',
  '.comfy_install_progress.json',
  '.pid',
  'pid',
]);

function skipRuntimeEntry(name: string) {
  const lower = name.toLowerCase();
  return (
    RUNTIME_FILE_NAMES.has(lower) ||
    lower.endsWith('.lock') ||
    lower.endsWith('.pid') ||
    lower.endsWith('.tmp') ||
    lower.endsWith('.temp') ||
    lower.endsWith('.part') ||
    lower.endsWith('.partial') ||
    lower === '__pycache__'
  );
}

async function hashFile(filePath: string) {
  return new Promise<string>((resolve, reject) => {
    const hash = createHash('sha256');
    const stream = fs.createReadStream(filePath);
    stream.on('error', reject);
    stream.on('data', chunk => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
  });
}

async function sourceDirectory(value: string) {
  const resolved = path.resolve(value);
  const stat = await fsp.lstat(resolved).catch(() => null);
  if (!stat?.isDirectory() || stat.isSymbolicLink()) {
    throw new ProjectError('The workspace import folder must be an existing directory, not a symbolic link', {
      status: 400,
      code: 'PROJECT_ROOT_INVALID',
    });
  }
  const real = await fsp.realpath(resolved);
  if (!process.env.AI_TOOLKIT_AUTH && !isPathInside(TOOLKIT_ROOT, real)) {
    throw new ProjectError('Importing a workspace outside the toolkit folder requires AI_TOOLKIT_AUTH', {
      status: 403,
      code: 'PROJECT_EXTERNAL_ROOT_REQUIRES_AUTH',
    });
  }
  return real;
}

async function collectFiles(root: string, relative = ''): Promise<ManifestEntry[]> {
  const current = relative ? path.join(root, relative) : root;
  const entries = await fsp.readdir(current, { withFileTypes: true }).catch(() => []);
  const files: ManifestEntry[] = [];
  for (const entry of entries) {
    if (entry.name.startsWith('.') || skipRuntimeEntry(entry.name)) continue;
    const relativePath = relative ? path.join(relative, entry.name) : entry.name;
    const absolutePath = path.join(root, relativePath);
    const stat = await fsp.lstat(absolutePath);
    if (stat.isSymbolicLink()) {
      throw new ProjectError(`Workspace setup cannot copy symbolic link: ${relativePath}`, {
        status: 409,
        code: 'PROJECT_ROOT_INVALID',
      });
    }
    if (stat.isDirectory()) files.push(...(await collectFiles(root, relativePath)));
    else if (stat.isFile()) {
      files.push({ relativePath, size: stat.size, sha256: await hashFile(absolutePath) });
    }
  }
  return files.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
}

async function copyManifest(sourceRoot: string, destinationRoot: string, manifest: ManifestEntry[]) {
  for (const entry of manifest) {
    const source = path.join(sourceRoot, entry.relativePath);
    const destination = path.join(destinationRoot, entry.relativePath);
    if (!isPathInside(sourceRoot, source) || !isPathInside(destinationRoot, destination)) {
      throw new ProjectError('Workspace setup path escaped its registered root', {
        status: 409,
        code: 'PROJECT_ROOT_INVALID',
      });
    }
    await fsp.mkdir(path.dirname(destination), { recursive: true });
    await fsp.copyFile(source, destination, fs.constants.COPYFILE_EXCL);
  }
}

async function verifyManifest(destinationRoot: string, expected: ManifestEntry[]) {
  const actual = await collectFiles(destinationRoot);
  if (actual.length !== expected.length) throw new Error('Workspace setup inventory count did not verify');
  for (let index = 0; index < expected.length; index += 1) {
    const source = expected[index];
    const copied = actual[index];
    if (
      source.relativePath !== copied.relativePath ||
      source.size !== copied.size ||
      source.sha256 !== copied.sha256
    ) {
      throw new Error(`Workspace setup checksum mismatch: ${source.relativePath}`);
    }
  }
}

function rewriteProjectPaths(value: unknown, sourceRoot: string, destinationRoot: string): unknown {
  if (typeof value === 'string') {
    if (/^[a-z][a-z0-9+.-]*:\/\//i.test(value) || value.startsWith('aitk-')) return value;
    if (!path.isAbsolute(value)) return value;
    const resolved = path.resolve(value);
    if (!isPathInside(sourceRoot, resolved)) return value;
    const relative = path.relative(sourceRoot, resolved);
    return relative ? path.join(destinationRoot, relative) : destinationRoot;
  }
  if (Array.isArray(value)) return value.map(item => rewriteProjectPaths(item, sourceRoot, destinationRoot));
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, child]) => [
        key,
        rewriteProjectPaths(child, sourceRoot, destinationRoot),
      ]),
    );
  }
  return value;
}

async function cloneJobs(sourceProject: Project, destinationProject: Project, deps: ProjectSetupDeps) {
  const [sourceRoots, destinationRoots, sourceJobs] = await Promise.all([
    deps.getProjectRoots(sourceProject),
    deps.getProjectRoots(destinationProject),
    deps.listProjectJobs(sourceProject.id),
  ]);
  const created: Job[] = [];
  try {
    for (const sourceJob of sourceJobs) {
      let jobConfig: unknown;
      try {
        jobConfig = JSON.parse(sourceJob.job_config);
      } catch {
        jobConfig = sourceJob.job_config;
      }
      const rewritten = rewriteProjectPaths(jobConfig, sourceRoots.root, destinationRoots.root);
      const job = await deps.createJob({
        name: sourceJob.name,
        project_id: destinationProject.id,
        worker_id: 'local',
        remote_job_id: null,
        remote_sync_at: null,
        remote_error: null,
        gpu_ids: sourceJob.gpu_ids,
        job_config: typeof rewritten === 'string' ? rewritten : JSON.stringify(rewritten),
        status: 'stopped',
        stop: false,
        return_to_queue: false,
        step: sourceJob.step,
        info: sourceJob.status === 'completed' ? 'Cloned from completed run' : 'Cloned run; resume when ready',
        speed_string: '',
        queue_position: 0,
        pid: null,
        job_type: sourceJob.job_type,
        job_ref:
          typeof sourceJob.job_ref === 'string'
            ? (rewriteProjectPaths(sourceJob.job_ref, sourceRoots.root, destinationRoots.root) as string)
            : sourceJob.job_ref,
        save_now: false,
      });
      created.push(job);
    }
    return created;
  } catch (error) {
    await Promise.all(created.map(job => deps.deleteJob(job.id).catch(() => null)));
    throw error;
  }
}

async function resetDestinationZones(project: Project, deps: ProjectSetupDeps) {
  const roots = await deps.getProjectRoots(project);
  await Promise.all(
    COPY_ZONES.map(async zone => {
      await fsp.rm(roots[zone], { recursive: true, force: true });
      await fsp.mkdir(roots[zone], { recursive: true });
    }),
  );
}

async function copyZones(
  sourceRoot: string,
  destinationProject: Project,
  importMode: boolean,
  deps: ProjectSetupDeps,
) {
  const destinationRoots = await deps.getProjectRoots(destinationProject);
  let copiedFiles = 0;
  let copiedBytes = 0;
  for (const zone of COPY_ZONES) {
    const sourceZoneName = importMode && zone === 'outputs' && !fs.existsSync(path.join(sourceRoot, zone)) ? 'output' : zone;
    const sourceZone = path.join(sourceRoot, sourceZoneName);
    const stat = await fsp.lstat(sourceZone).catch(() => null);
    if (!stat) continue;
    if (!stat.isDirectory() || stat.isSymbolicLink()) {
      throw new ProjectError(`Workspace zone is invalid: ${sourceZoneName}`, {
        status: 409,
        code: 'PROJECT_ROOT_INVALID',
      });
    }
    const manifest = await collectFiles(sourceZone);
    await copyManifest(sourceZone, destinationRoots[zone], manifest);
    await verifyManifest(destinationRoots[zone], manifest);
    copiedFiles += manifest.length;
    copiedBytes += manifest.reduce((total, entry) => total + entry.size, 0);
  }
  return { copiedFiles, copiedBytes };
}

export async function validateProjectSetupRequest(
  request: ProjectSetupRequest,
  deps: ProjectSetupDeps = defaultProjectSetupDeps,
) {
  if (request.mode === 'blank') return;
  if (request.mode === 'clone') {
    if (!request.cloneFromProjectID?.trim()) {
      throw new ProjectError('clone_from_project_id is required for clone setup', {
        status: 400,
        code: 'PROJECT_INVALID_INPUT',
      });
    }
    await deps.resolveProject(request.cloneFromProjectID.trim());
    return;
  }
  if (!request.importRoot?.trim()) {
    throw new ProjectError('import_root is required for workspace import', {
      status: 400,
      code: 'PROJECT_INVALID_INPUT',
    });
  }
  await sourceDirectory(request.importRoot);
}

export async function setupProjectWorkspace(
  project: Project,
  request: ProjectSetupRequest,
  deps: ProjectSetupDeps = defaultProjectSetupDeps,
): Promise<ProjectSetupResult> {
  if (request.mode === 'blank') {
    return { mode: 'blank', status: 'completed', copiedFiles: 0, copiedBytes: 0, clonedJobs: 0, error: null };
  }

  let createdJobs: Job[] = [];
  try {
    let sourceRoot: string;
    let sourceProject: Project | null = null;
    if (request.mode === 'clone') {
      if (!request.cloneFromProjectID) {
        throw new ProjectError('clone_from_project_id is required for clone setup', {
          status: 400,
          code: 'PROJECT_INVALID_INPUT',
        });
      }
      sourceProject = await deps.resolveProject(request.cloneFromProjectID);
      if (sourceProject.id === project.id) {
        throw new ProjectError('A project cannot be cloned from itself', {
          status: 400,
          code: 'PROJECT_INVALID_INPUT',
        });
      }
      sourceRoot = (await deps.getProjectRoots(sourceProject)).root;
    } else {
      if (!request.importRoot?.trim()) {
        throw new ProjectError('import_root is required for workspace import', {
          status: 400,
          code: 'PROJECT_INVALID_INPUT',
        });
      }
      sourceRoot = await sourceDirectory(request.importRoot);
    }

    const destinationRoot = (await deps.getProjectRoots(project)).root;
    if (isPathInside(sourceRoot, destinationRoot) || isPathInside(destinationRoot, sourceRoot)) {
      throw new ProjectError('Workspace setup source and destination cannot contain one another', {
        status: 409,
        code: 'PROJECT_ROOT_INVALID',
      });
    }
    const copied = await copyZones(sourceRoot, project, request.mode === 'import', deps);
    if (sourceProject) createdJobs = await cloneJobs(sourceProject, project, deps);
    await deps.updateProjectOperationError(project.id, null);
    return {
      mode: request.mode,
      status: 'completed',
      copiedFiles: copied.copiedFiles,
      copiedBytes: copied.copiedBytes,
      clonedJobs: createdJobs.length,
      error: null,
    };
  } catch (error) {
    await Promise.all(createdJobs.map(job => deps.deleteJob(job.id).catch(() => null)));
    await resetDestinationZones(project, deps).catch(() => undefined);
    const message = error instanceof Error ? error.message.slice(0, 500) : 'Project workspace setup failed';
    await deps.updateProjectOperationError(project.id, message).catch(() => undefined);
    return {
      mode: request.mode,
      status: 'failed',
      copiedFiles: 0,
      copiedBytes: 0,
      clonedJobs: 0,
      error: message,
    };
  }
}
