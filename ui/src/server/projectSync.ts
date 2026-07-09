import fs from 'fs/promises';
import { db, type WorkerNodeRecord } from './db';
import { getRemoteWorker, RemoteClientError, remoteFetch, remoteJson } from './remoteClient';
import { getAITKInstanceID, getProjectRoots, resolveProject } from './projects';
import {
  appendProjectSyncChunk,
  buildProjectSyncManifest,
  commitProjectSyncPlan,
  collectNonPortableAbsolutePaths,
  collectCredentialConfigKeys,
  detectProjectSyncConflicts,
  deterministicKeepBothPath,
  diffProjectSyncManifests,
  getProjectSyncChunkReceipt,
  PROJECT_SYNC_CHUNK_BYTES,
  PROJECT_SYNC_LIFECYCLE_BLOCKING_JOB_STATUSES,
  PROJECT_SYNC_PROFILES,
  PROJECT_SYNC_PROTOCOL,
  PROJECT_SYNC_TRANSFER_BLOCKING_JOB_STATUSES,
  ProjectSyncProtocolError,
  portableizeProjectConfig,
  parsePortableProjectRef,
  hashPortableProjectJobs,
  parseProjectSyncManifest,
  readProjectSyncBaseManifest,
  resolveProjectSyncPath,
  writeProjectSyncBaseManifest,
  type ProjectManifestEntry,
  type ProjectSyncConflict,
  type ProjectSyncConflictResolution,
  type ProjectSyncManifest,
  type ProjectSyncProfileName,
  type PortableProjectJob,
  type PortableProjectJobSnapshot,
} from './projectSyncProtocol';
import type { Job, Project, ProjectReplicaState, ProjectSyncOperation } from '@/types';

const runningOperations = new Set<string>();

export type CompatibleProjectSyncCapabilities = {
  protocol: typeof PROJECT_SYNC_PROTOCOL;
  instance_id: string;
  chunk_bytes: number;
  profiles: ProjectSyncProfileName[];
  features: Record<string, boolean>;
};

export class ProjectSyncError extends Error {
  status: number;
  code: string;
  details?: unknown;

  constructor(message: string, options: { status?: number; code?: string; details?: unknown } = {}) {
    super(message);
    this.name = 'ProjectSyncError';
    this.status = options.status ?? 400;
    this.code = options.code ?? 'PROJECT_SYNC_INVALID_REQUEST';
    this.details = options.details;
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function parseCapabilities(value: unknown): CompatibleProjectSyncCapabilities {
  if (!isObject(value)) throw new ProjectSyncError('Worker returned malformed capabilities', { status: 502 });
  if (
    value.protocol !== PROJECT_SYNC_PROTOCOL ||
    typeof value.instance_id !== 'string' ||
    !value.instance_id.trim() ||
    value.chunk_bytes !== PROJECT_SYNC_CHUNK_BYTES ||
    !Array.isArray(value.profiles) ||
    !value.profiles.every(profile => (PROJECT_SYNC_PROFILES as readonly unknown[]).includes(profile)) ||
    !isObject(value.features)
  ) {
    throw new ProjectSyncError('Worker is incompatible with project-sync-v1', {
      status: 409,
      code: 'PROJECT_SYNC_WORKER_INCOMPATIBLE',
    });
  }
  return {
    protocol: PROJECT_SYNC_PROTOCOL,
    instance_id: value.instance_id,
    chunk_bytes: PROJECT_SYNC_CHUNK_BYTES,
    profiles: value.profiles as ProjectSyncProfileName[],
    features: Object.fromEntries(Object.entries(value.features).filter((entry): entry is [string, boolean] => typeof entry[1] === 'boolean')),
  };
}

export async function fetchProjectSyncCapabilities(worker: WorkerNodeRecord) {
  return parseCapabilities(
    await remoteJson<unknown>(worker, '/api/project-sync/capabilities', {
      method: 'GET',
      timeoutMs: 15_000,
    }),
  );
}

async function assertLocalProjectHome(project: Project) {
  const localInstanceID = await getAITKInstanceID();
  if (project.home_instance_id && project.home_instance_id !== localInstanceID) {
    throw new ProjectSyncError('Project writes must be initiated by its authoritative home', {
      status: 409,
      code: 'PROJECT_SYNC_NOT_HOME',
      details: { home_instance_id: project.home_instance_id },
    });
  }
  return localInstanceID;
}

function remoteProjectBase(projectID: string) {
  return `/api/project-sync/projects/${encodeURIComponent(projectID)}`;
}

function remoteBlobPath(projectID: string, relativePath: string) {
  return `${remoteProjectBase(projectID)}/blobs/${relativePath.split('/').map(encodeURIComponent).join('/')}`;
}

async function assertReplicaInstance(replica: ProjectReplicaState | null, capabilities: CompatibleProjectSyncCapabilities) {
  if (replica?.remote_instance_id && replica.remote_instance_id !== capabilities.instance_id) {
    await db.projectReplicas.update(replica.id, {
      state: 'incompatible',
      last_error: 'Worker instance identity changed; relink is required before project writes.',
    });
    throw new ProjectSyncError('Worker instance identity changed; replica writes are blocked', {
      status: 409,
      code: 'PROJECT_SYNC_INSTANCE_CHANGED',
    });
  }
}

export async function linkProjectWorker(projectIdentifier: string, workerID: string) {
  if (!workerID || workerID === 'local') {
    throw new ProjectSyncError('A remote worker is required', { code: 'PROJECT_SYNC_WORKER_REQUIRED' });
  }
  const project = await resolveProject(projectIdentifier, { intent: 'write' });
  const homeInstanceID = await assertLocalProjectHome(project);
  const worker = await getRemoteWorker(workerID);
  const capabilities = await fetchProjectSyncCapabilities(worker);
  const existing = await db.projectReplicas.findByProjectAndWorker(project.id, worker.id);
  await assertReplicaInstance(existing, capabilities);
  const result = await remoteJson<unknown>(worker, '/api/project-sync/projects/link', {
    method: 'POST',
    body: JSON.stringify({
      project_id: project.id,
      slug: project.slug,
      name: project.name,
      description: project.description,
      badge_asset: project.badge_asset,
      revision: project.revision,
      home_worker_id: 'remote-home',
      home_instance_id: homeInstanceID,
    }),
    timeoutMs: 30_000,
  });
  if (!isObject(result) || !isObject(result.project) || result.project.id !== project.id) {
    throw new ProjectSyncError('Worker returned a malformed project link response', { status: 502 });
  }
  return db.projectReplicas.upsert({
    project_id: project.id,
    worker_id: worker.id,
    remote_project_id: project.id,
    remote_instance_id: capabilities.instance_id,
    role: 'execution',
    state: existing?.state === 'in_sync' ? 'in_sync' : 'dirty',
    base_manifest_hash: existing?.base_manifest_hash,
    local_manifest_hash: existing?.local_manifest_hash,
    remote_manifest_hash: existing?.remote_manifest_hash,
    last_synced_at: existing?.last_synced_at,
    last_error: null,
    auto_pull_results: existing?.auto_pull_results ?? true,
  });
}

export async function listProjectReplicas(projectIdentifier: string) {
  const project = await resolveProject(projectIdentifier, { intent: 'read' });
  return { project, replicas: await db.projectReplicas.listByProject(project.id) };
}

export async function removeProjectWorkerReplica(projectIdentifier: string, workerID: string) {
  const project = await resolveProject(projectIdentifier, { intent: 'write' });
  const homeInstanceID = await assertLocalProjectHome(project);
  const replica = await db.projectReplicas.findByProjectAndWorker(project.id, workerID);
  if (!replica) throw new ProjectSyncError('Project replica not found', { status: 404, code: 'PROJECT_SYNC_REPLICA_NOT_FOUND' });
  const activeJobs = await db.jobs.list({
    project_id: project.id,
    worker_id: workerID,
    status: [...PROJECT_SYNC_LIFECYCLE_BLOCKING_JOB_STATUSES],
  });
  if (activeJobs.length > 0) {
    throw new ProjectSyncError('Replica cannot be removed while project jobs are active on it', {
      status: 409,
      code: 'PROJECT_SYNC_ACTIVE_JOBS',
      details: { job_ids: activeJobs.map(job => job.id) },
    });
  }
  const worker = await getRemoteWorker(workerID);
  const capabilities = await fetchProjectSyncCapabilities(worker);
  await assertReplicaInstance(replica, capabilities);
  await remoteJson(worker, remoteProjectBase(project.id), {
    method: 'DELETE',
    body: JSON.stringify({ home_instance_id: homeInstanceID }),
    timeoutMs: 60_000,
  });
  await db.projectReplicas.delete(replica.id);
  return { removed: true, worker_id: workerID, project_id: project.id };
}

export async function createProjectSyncOperation(
  projectIdentifier: string,
  workerID: string,
  profile: ProjectSyncProfileName,
) {
  if (!(PROJECT_SYNC_PROFILES as readonly string[]).includes(profile)) {
    throw new ProjectSyncError('Unsupported project sync profile');
  }
  const project = await resolveProject(projectIdentifier, { intent: profile === 'results' ? 'write' : 'execute' });
  await assertLocalProjectHome(project);
  let replica = await db.projectReplicas.findByProjectAndWorker(project.id, workerID);
  if (!replica) replica = await linkProjectWorker(project.id, workerID);
  if (replica.state === 'incompatible') {
    throw new ProjectSyncError('Replica is incompatible and must be relinked', {
      status: 409,
      code: 'PROJECT_SYNC_WORKER_INCOMPATIBLE',
    });
  }
  return db.projectSyncOperations.create({
    project_id: project.id,
    worker_id: workerID,
    profile,
    status: 'queued',
    phase: 'queued',
  });
}

export async function prepareProjectJobReplica(job: Job) {
  if (!job.project_id || !job.worker_id || job.worker_id === 'local') {
    throw new ProjectSyncError('A project-scoped remote job is required', {
      code: 'PROJECT_SYNC_PROJECT_JOB_REQUIRED',
    });
  }
  const project = await resolveProject(job.project_id, { intent: 'execute' });
  const homeInstanceID = await assertLocalProjectHome(project);
  const roots = await getProjectRoots(project);
  let config: unknown;
  try {
    config = JSON.parse(job.job_config);
  } catch {
    throw new ProjectSyncError('Job config is invalid JSON');
  }
  const portableConfig = portableizeProjectConfig(config, roots.root, project.id);
  const credentialKeys = collectCredentialConfigKeys(portableConfig);
  if (credentialKeys.length > 0) {
    throw new ProjectSyncError('Project job config contains credentials that cannot be transferred', {
      status: 409,
      code: 'PROJECT_SYNC_SECRET_BLOCKED',
      details: { keys: credentialKeys },
    });
  }
  const nonPortablePaths = collectNonPortableAbsolutePaths(portableConfig);
  if (nonPortablePaths.length > 0) {
    throw new ProjectSyncError('Project job contains absolute host paths outside the project boundary', {
      status: 409,
      code: 'PROJECT_SYNC_NON_PORTABLE_PATH',
      details: { paths: nonPortablePaths.slice(0, 25) },
    });
  }
  const operation = await createProjectSyncOperation(project.id, job.worker_id, 'launch');
  const completed = await runProjectSyncOperation(operation.id);
  if (completed.status !== 'completed') {
    throw new ProjectSyncError('Project launch sync did not complete', {
      status: 409,
      code: completed.status === 'conflict' ? 'PROJECT_SYNC_CONFLICT' : 'PROJECT_SYNC_LAUNCH_INCOMPLETE',
      details: { operation: completed },
    });
  }
  const worker = await getRemoteWorker(job.worker_id);
  const result = await remoteJson<unknown>(worker, `${remoteProjectBase(project.id)}/jobs/link`, {
    method: 'POST',
    body: JSON.stringify({
      home_instance_id: homeInstanceID,
      job: {
        id: job.id,
        project_id: project.id,
        name: job.name,
        gpu_ids: job.gpu_ids,
        step: job.step,
        queue_position: job.queue_position,
        job_type: job.job_type,
        job_ref: job.job_ref,
      },
      job_config: portableConfig,
    }),
    timeoutMs: 60_000,
  });
  if (
    !isObject(result) ||
    !isObject(result.job) ||
    result.job.id !== job.id ||
    result.job.project_id !== project.id ||
    typeof result.job.job_config !== 'string' ||
    typeof result.job.gpu_ids !== 'string' ||
    typeof result.job.name !== 'string'
  ) {
    throw new ProjectSyncError('Worker returned a malformed linked job', { status: 502 });
  }
  await db.jobReplicas.upsert({
    job_id: job.id,
    worker_id: job.worker_id,
    remote_job_id: job.id,
    remote_project_id: project.id,
    role: 'execution',
    last_synced_at: null,
    last_error: null,
  });
  return { remoteJob: result.job as unknown as Job, operation: completed };
}

function portableDate(value: Job['created_at']) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) {
    throw new ProjectSyncError('Project job contains an invalid database date', {
      code: 'PROJECT_SYNC_JOB_SNAPSHOT_INVALID',
    });
  }
  return date.toISOString();
}

async function buildPortableProjectJobSnapshot(
  project: Project,
  projectRoot: string,
  homeInstanceID: string,
): Promise<PortableProjectJobSnapshot> {
  const projectJobs = await db.jobs.list({ project_id: project.id });
  const jobs: PortableProjectJob[] = projectJobs
    .map(job => {
      let config: unknown;
      try {
        config = JSON.parse(job.job_config);
      } catch {
        throw new ProjectSyncError(`Project job ${job.id} has invalid config JSON`, {
          status: 409,
          code: 'PROJECT_SYNC_JOB_SNAPSHOT_INVALID',
        });
      }
      const portableConfig = portableizeProjectConfig(config, projectRoot, project.id);
      const credentialKeys = collectCredentialConfigKeys(portableConfig);
      if (credentialKeys.length > 0) {
        throw new ProjectSyncError(`Project job ${job.id} contains credentials that cannot be transferred`, {
          status: 409,
          code: 'PROJECT_SYNC_SECRET_BLOCKED',
          details: { job_id: job.id, keys: credentialKeys },
        });
      }
      const nonPortablePaths = collectNonPortableAbsolutePaths(portableConfig);
      if (nonPortablePaths.length > 0) {
        throw new ProjectSyncError(`Project job ${job.id} contains non-portable absolute paths`, {
          status: 409,
          code: 'PROJECT_SYNC_NON_PORTABLE_PATH',
          details: { job_id: job.id, paths: nonPortablePaths.slice(0, 25) },
        });
      }
      return {
        id: job.id,
        name: job.name,
        source_worker_id: job.worker_id,
        remote_job_id: job.remote_job_id,
        remote_sync_at: job.remote_sync_at == null ? null : portableDate(job.remote_sync_at),
        remote_error: job.remote_error,
        gpu_ids: job.gpu_ids,
        job_config: portableConfig,
        status: job.status,
        stop: job.stop,
        return_to_queue: job.return_to_queue,
        step: job.step,
        info: job.info,
        speed_string: job.speed_string,
        queue_position: job.queue_position,
        job_type: job.job_type,
        job_ref: job.job_ref,
        save_now: job.save_now,
        created_at: portableDate(job.created_at),
        updated_at: portableDate(job.updated_at),
      } satisfies PortableProjectJob;
    })
    .sort((left, right) => left.id.localeCompare(right.id));
  return {
    protocol: PROJECT_SYNC_PROTOCOL,
    project_id: project.id,
    home_instance_id: homeInstanceID,
    generated_at: new Date().toISOString(),
    hash: hashPortableProjectJobs(jobs),
    jobs,
  };
}

async function syncProjectJobSnapshot(
  worker: WorkerNodeRecord,
  project: Project,
  projectRoot: string,
  homeInstanceID: string,
) {
  const snapshot = await buildPortableProjectJobSnapshot(project, projectRoot, homeInstanceID);
  const result = await remoteJson<unknown>(worker, `${remoteProjectBase(project.id)}/jobs/snapshot`, {
    method: 'POST',
    body: JSON.stringify(snapshot),
    timeoutMs: 120_000,
  });
  if (
    !isObject(result) ||
    result.hash !== snapshot.hash ||
    result.job_count !== snapshot.jobs.length ||
    result.imported !== true
  ) {
    throw new ProjectSyncError('Worker did not verify the complete project job snapshot', {
      status: 502,
      code: 'PROJECT_SYNC_JOB_SNAPSHOT_INVALID',
    });
  }
  return snapshot;
}

export async function generateOnProjectReplica(options: {
  projectIdentifier: string;
  workerID: string;
  jobConfig: unknown;
  gpuIDs: string;
  signal?: AbortSignal;
}) {
  const project = await resolveProject(options.projectIdentifier, { intent: 'execute' });
  const homeInstanceID = await assertLocalProjectHome(project);
  const roots = await getProjectRoots(project);
  const portableConfig = portableizeProjectConfig(options.jobConfig, roots.root, project.id);
  const credentialKeys = collectCredentialConfigKeys(portableConfig);
  if (credentialKeys.length > 0) {
    throw new ProjectSyncError('Project generation config contains credentials that cannot be transferred', {
      status: 409,
      code: 'PROJECT_SYNC_SECRET_BLOCKED',
      details: { keys: credentialKeys },
    });
  }
  const nonPortablePaths = collectNonPortableAbsolutePaths(portableConfig);
  if (nonPortablePaths.length > 0) {
    throw new ProjectSyncError('Project generation config contains absolute host paths outside the project boundary', {
      status: 409,
      code: 'PROJECT_SYNC_NON_PORTABLE_PATH',
      details: { paths: nonPortablePaths.slice(0, 25) },
    });
  }
  const launch = await createProjectSyncOperation(project.id, options.workerID, 'launch');
  const launched = await runProjectSyncOperation(launch.id);
  if (launched.status !== 'completed') {
    throw new ProjectSyncError('Project launch sync did not complete before generation', {
      status: 409,
      code: launched.status === 'conflict' ? 'PROJECT_SYNC_CONFLICT' : 'PROJECT_SYNC_LAUNCH_INCOMPLETE',
      details: { operation: launched },
    });
  }
  const worker = await getRemoteWorker(options.workerID);
  const generated = await remoteJson<unknown>(worker, '/api/generate/inline', {
    method: 'POST',
    headers: {
      'X-AITK-Project-Sync': PROJECT_SYNC_PROTOCOL,
      'X-AITK-Home-Instance': homeInstanceID,
    },
    body: JSON.stringify({
      project_id: project.id,
      worker_id: 'local',
      gpu_ids: options.gpuIDs,
      job_config: portableConfig,
    }),
    signal: options.signal,
  });
  if (!isObject(generated) || typeof generated.portable_image_ref !== 'string') {
    throw new ProjectSyncError('Worker generation response did not include a portable result reference', { status: 502 });
  }
  const reference = parsePortableProjectRef(generated.portable_image_ref);
  if (reference.project_id !== project.id) {
    throw new ProjectSyncError('Worker returned a cross-project generation result', {
      status: 502,
      code: 'PROJECT_SYNC_CROSS_PROJECT_REF',
    });
  }
  const results = await createProjectSyncOperation(project.id, options.workerID, 'results');
  const pulled = await runProjectSyncOperation(results.id);
  if (pulled.status !== 'completed') {
    throw new ProjectSyncError('Generation completed remotely, but its results have not reached the project home', {
      status: 409,
      code: pulled.status === 'conflict' ? 'PROJECT_SYNC_CONFLICT' : 'PROJECT_SYNC_RESULTS_INCOMPLETE',
      details: { operation: pulled, portable_image_ref: generated.portable_image_ref },
    });
  }
  const imagePath = resolveProjectSyncPath(roots.root, reference.relative_path, 'results');
  const stat = await fs.stat(imagePath).catch(() => null);
  if (!stat?.isFile()) {
    throw new ProjectSyncError('Result sync completed without the generated image', { status: 502 });
  }
  return {
    image_path: imagePath,
    imagePath,
    output_folder: resolveProjectSyncPath(roots.root, reference.relative_path.split('/').slice(0, -1).join('/'), 'results'),
    portable_image_ref: generated.portable_image_ref,
    worker_id: options.workerID,
    launch_operation_id: launched.id,
    results_operation_id: pulled.id,
  };
}

async function fetchRemoteManifest(
  worker: WorkerNodeRecord,
  project: Project,
  profile: ProjectSyncProfileName,
  homeInstanceID: string,
): Promise<ProjectSyncManifest> {
  const query = new URLSearchParams({ profile, home_instance_id: homeInstanceID });
  const value = await remoteJson<unknown>(worker, `${remoteProjectBase(project.id)}/manifest?${query.toString()}`, {
    timeoutMs: 120_000,
  });
  try {
    return parseProjectSyncManifest(value, project.id, profile);
  } catch (error) {
    throw new ProjectSyncError('Worker returned a malformed project manifest', {
      status: 502,
      code: 'PROJECT_SYNC_MANIFEST_INVALID',
      details: error instanceof Error ? error.message : undefined,
    });
  }
}

async function uploadEntry(
  worker: WorkerNodeRecord,
  project: Project,
  projectRoot: string,
  profile: ProjectSyncProfileName,
  homeInstanceID: string,
  entry: ProjectManifestEntry,
  progress: (bytes: number) => Promise<void>,
) {
  const source = resolveProjectSyncPath(projectRoot, entry.path, profile);
  const query = new URLSearchParams({ total: String(entry.size), home_instance_id: homeInstanceID });
  const chunkRoute = `${remoteProjectBase(project.id)}/chunks/${entry.sha256}`;
  const receiptValue = await remoteJson<unknown>(worker, `${chunkRoute}?${query.toString()}`, { timeoutMs: 30_000 });
  if (!isObject(receiptValue) || !Number.isSafeInteger(receiptValue.received)) {
    throw new ProjectSyncError('Worker returned malformed chunk status', { status: 502 });
  }
  let offset = Number(receiptValue.received);
  if (receiptValue.complete === true) return;
  const handle = await fs.open(source, 'r');
  try {
    if (entry.size === 0) {
      query.set('offset', '0');
      await remoteJson(worker, `${chunkRoute}?${query.toString()}`, {
        method: 'PUT',
        body: new Uint8Array(),
        headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': '0' },
        timeoutMs: 30_000,
      });
      return;
    }
    while (offset < entry.size) {
      const length = Math.min(PROJECT_SYNC_CHUNK_BYTES, entry.size - offset);
      const buffer = Buffer.allocUnsafe(length);
      const { bytesRead } = await handle.read(buffer, 0, length, offset);
      if (bytesRead !== length) throw new ProjectSyncError('Source file changed during project sync', { status: 409 });
      query.set('offset', String(offset));
      await remoteJson(worker, `${chunkRoute}?${query.toString()}`, {
        method: 'PUT',
        body: buffer,
        headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(buffer.byteLength) },
        timeoutMs: 120_000,
      });
      offset += bytesRead;
      await progress(bytesRead);
    }
  } finally {
    await handle.close();
  }
}

async function pushChanges(options: {
  worker: WorkerNodeRecord;
  project: Project;
  projectRoot: string;
  profile: ProjectSyncProfileName;
  homeInstanceID: string;
  operationID: string;
  files: ProjectManifestEntry[];
  deletePaths: string[];
  preservePaths?: Array<{ path: string; preserve_as: string }>;
  progress: (bytes: number) => Promise<void>;
}) {
  for (const entry of options.files) {
    await uploadEntry(
      options.worker,
      options.project,
      options.projectRoot,
      options.profile,
      options.homeInstanceID,
      entry,
      options.progress,
    );
  }
  const result = await remoteJson<unknown>(options.worker, `${remoteProjectBase(options.project.id)}/commit`, {
    method: 'POST',
    body: JSON.stringify({
      operation_id: options.operationID,
      profile: options.profile,
      home_instance_id: options.homeInstanceID,
      files: options.files,
      delete_paths: options.deletePaths,
      preserve_paths: options.preservePaths || [],
    }),
    timeoutMs: 120_000,
  });
  if (!isObject(result) || result.committed !== true) {
    throw new ProjectSyncError('Worker did not confirm project sync commit', { status: 502 });
  }
}

type PullEntry = { source: ProjectManifestEntry; targetPath: string };

async function downloadEntry(options: {
  worker: WorkerNodeRecord;
  project: Project;
  projectRoot: string;
  profile: ProjectSyncProfileName;
  homeInstanceID: string;
  entry: PullEntry;
  progress: (bytes: number) => Promise<void>;
}) {
  const source = options.entry.source;
  let receipt = await getProjectSyncChunkReceipt(options.projectRoot, source.sha256, source.size);
  if (receipt.complete) return;
  if (source.size === 0) {
    await appendProjectSyncChunk(options.projectRoot, source.sha256, 0, 0, new Uint8Array());
    return;
  }
  while (!receipt.complete) {
    const end = Math.min(source.size - 1, receipt.received + PROJECT_SYNC_CHUNK_BYTES - 1);
    const query = new URLSearchParams({ profile: options.profile, home_instance_id: options.homeInstanceID });
    const response = await remoteFetch(
      options.worker,
      `${remoteBlobPath(options.project.id, source.path)}?${query.toString()}`,
      {
        headers: { Range: `bytes=${receipt.received}-${end}` },
        timeoutMs: 120_000,
      },
    );
    const bytes = new Uint8Array(await response.arrayBuffer());
    if (bytes.byteLength !== end - receipt.received + 1) {
      throw new ProjectSyncError('Worker returned an unexpected blob range length', { status: 502 });
    }
    receipt = await appendProjectSyncChunk(options.projectRoot, source.sha256, source.size, receipt.received, bytes);
    await options.progress(bytes.byteLength);
  }
}

async function pullChanges(options: {
  worker: WorkerNodeRecord;
  project: Project;
  projectRoot: string;
  profile: ProjectSyncProfileName;
  homeInstanceID: string;
  operationID: string;
  files: PullEntry[];
  deletePaths: string[];
  progress: (bytes: number) => Promise<void>;
}) {
  for (const entry of options.files) {
    await downloadEntry({ ...options, entry });
  }
  await commitProjectSyncPlan(options.projectRoot, {
    project_id: options.project.id,
    operation_id: `${options.operationID}-pull`,
    profile: options.profile,
    files: options.files.map(item => ({ ...item.source, path: item.targetPath })),
    delete_paths: options.deletePaths,
  });
}

function parseStoredResolutions(operation: ProjectSyncOperation) {
  try {
    const value: unknown = JSON.parse(operation.conflicts || '[]');
    if (!Array.isArray(value)) return new Map<string, ProjectSyncConflictResolution>();
    return new Map(
      value.flatMap(item => {
        if (!isObject(item) || typeof item.path !== 'string') return [];
        if (item.resolution !== 'keep-home' && item.resolution !== 'keep-worker' && item.resolution !== 'keep-both') return [];
        return [[item.path, item.resolution] as const];
      }),
    );
  } catch {
    return new Map<string, ProjectSyncConflictResolution>();
  }
}

function isRemoteFailure(error: unknown) {
  return error instanceof RemoteClientError || /fetch|worker|network|timeout|connect/i.test(error instanceof Error ? error.message : '');
}

async function deferOperation(operation: ProjectSyncOperation, status: 'waiting_for_job' | 'waiting_for_worker', error: string) {
  const retryCount = operation.retry_count + 1;
  const delay = status === 'waiting_for_job' ? 5_000 : Math.min(300_000, 5_000 * 2 ** Math.min(6, retryCount));
  return db.projectSyncOperations.update(operation.id, {
    status,
    phase: status,
    retry_count: retryCount,
    retry_at: new Date(Date.now() + delay),
    error: error.slice(0, 1000),
  });
}

async function synchronizeConflictResolutions(options: {
  operation: ProjectSyncOperation;
  worker: WorkerNodeRecord;
  project: Project;
  projectRoot: string;
  profile: ProjectSyncProfileName;
  homeInstanceID: string;
  workerInstanceID: string;
  conflicts: ProjectSyncConflict[];
  resolutions: Map<string, ProjectSyncConflictResolution>;
  progress: (bytes: number) => Promise<void>;
}) {
  const pull: PullEntry[] = [];
  const localDeletes: string[] = [];
  const push: ProjectManifestEntry[] = [];
  const remoteDeletes: string[] = [];
  const preserve: Array<{ path: string; preserve_as: string }> = [];
  for (const conflict of options.conflicts) {
    const resolution = options.resolutions.get(conflict.path);
    if (!resolution) continue;
    if (resolution === 'keep-home') {
      if (conflict.home) push.push(conflict.home);
      else remoteDeletes.push(conflict.path);
      continue;
    }
    if (resolution === 'keep-worker') {
      if (conflict.worker) pull.push({ source: conflict.worker, targetPath: conflict.path });
      else localDeletes.push(conflict.path);
      continue;
    }
    if (conflict.worker && conflict.home) {
      const keepPath = deterministicKeepBothPath(conflict.path, options.workerInstanceID);
      pull.push({ source: conflict.worker, targetPath: keepPath });
      push.push(conflict.home);
      preserve.push({ path: conflict.path, preserve_as: keepPath });
    } else if (conflict.worker) {
      const keepPath = deterministicKeepBothPath(conflict.path, options.workerInstanceID);
      pull.push({ source: conflict.worker, targetPath: keepPath });
      preserve.push({ path: conflict.path, preserve_as: keepPath });
      remoteDeletes.push(conflict.path);
    } else if (conflict.home) {
      push.push(conflict.home);
    }
  }
  if (pull.length || localDeletes.length) {
    await pullChanges({
      ...options,
      operationID: `${options.operation.id}-resolve`,
      files: pull,
      deletePaths: localDeletes,
    });
  }
  if (push.length || remoteDeletes.length || preserve.length) {
    await pushChanges({
      ...options,
      operationID: `${options.operation.id}-resolve`,
      files: push,
      deletePaths: remoteDeletes,
      preservePaths: preserve,
    });
  }
}

export async function runProjectSyncOperation(operationID: string): Promise<ProjectSyncOperation> {
  if (runningOperations.has(operationID)) {
    return (await db.projectSyncOperations.findById(operationID)) as ProjectSyncOperation;
  }
  runningOperations.add(operationID);
  let operation = await db.projectSyncOperations.findById(operationID);
  try {
    if (!operation) throw new ProjectSyncError('Project sync operation not found', { status: 404 });
    if (operation.status === 'completed' || operation.status === 'cancelled') return operation;
    const project = await resolveProject(operation.project_id, { intent: operation.profile === 'results' ? 'write' : 'execute' });
    const homeInstanceID = await assertLocalProjectHome(project);
    const activeJobs = await db.jobs.list({
      project_id: project.id,
      status: [...PROJECT_SYNC_TRANSFER_BLOCKING_JOB_STATUSES],
    });
    if (activeJobs.length > 0) {
      return deferOperation(operation, 'waiting_for_job', 'Project paths are active; sync will resume after jobs stop.');
    }
    const roots = await getProjectRoots(project);
    const worker = await getRemoteWorker(operation.worker_id);
    const capabilities = await fetchProjectSyncCapabilities(worker);
    if (!capabilities.profiles.includes(operation.profile)) {
      throw new ProjectSyncError(`Worker does not support the ${operation.profile} sync profile`, {
        status: 409,
        code: 'PROJECT_SYNC_WORKER_INCOMPATIBLE',
      });
    }
    let replica = await db.projectReplicas.findByProjectAndWorker(project.id, worker.id);
    if (!replica) replica = await linkProjectWorker(project.id, worker.id);
    await assertReplicaInstance(replica, capabilities);
    operation = await db.projectSyncOperations.update(operation.id, {
      status: 'running',
      phase: 'manifest',
      retry_at: null,
      error: null,
    });
    await db.projectReplicas.update(replica.id, { state: 'syncing', last_error: null });
    let homeManifest = await buildProjectSyncManifest(roots.root, project.id, operation.profile);
    let workerManifest = await fetchRemoteManifest(worker, project, operation.profile, homeInstanceID);
    const baseManifest = await readProjectSyncBaseManifest(roots.root, worker.id, operation.profile);
    const conflicts = detectProjectSyncConflicts(baseManifest, homeManifest, workerManifest);
    const resolutions = parseStoredResolutions(operation);
    const unresolved = conflicts.filter(conflict => !resolutions.has(conflict.path));
    if (unresolved.length > 0) {
      await db.projectReplicas.update(replica.id, {
        state: 'conflict',
        local_manifest_hash: homeManifest.hash,
        remote_manifest_hash: workerManifest.hash,
        last_error: `${unresolved.length} project sync conflict${unresolved.length === 1 ? '' : 's'} require resolution.`,
      });
      return db.projectSyncOperations.update(operation.id, {
        status: 'conflict',
        phase: 'conflict',
        base_manifest_hash: baseManifest?.hash ?? null,
        source_manifest_hash: operation.profile === 'results' ? workerManifest.hash : homeManifest.hash,
        target_manifest_hash: operation.profile === 'results' ? homeManifest.hash : workerManifest.hash,
        conflicts: JSON.stringify(conflicts),
        error: null,
      });
    }

    let bytesDone = 0;
    let filesDone = 0;
    const progress = async (bytes: number) => {
      bytesDone += bytes;
      operation = await db.projectSyncOperations.update(operation!.id, { bytes_done: bytesDone, phase: 'transfer' });
    };

    if (conflicts.length > 0) {
      await synchronizeConflictResolutions({
        operation,
        worker,
        project,
        projectRoot: roots.root,
        profile: operation.profile,
        homeInstanceID,
        workerInstanceID: capabilities.instance_id,
        conflicts,
        resolutions,
        progress,
      });
      homeManifest = await buildProjectSyncManifest(roots.root, project.id, operation.profile);
      workerManifest = await fetchRemoteManifest(worker, project, operation.profile, homeInstanceID);
    }

    const conflictPaths = new Set(conflicts.map(conflict => conflict.path));
    const source = operation.profile === 'results' ? workerManifest : homeManifest;
    const target = operation.profile === 'results' ? homeManifest : workerManifest;
    const diff = diffProjectSyncManifests(source, target);
    const files = diff.add_or_update.filter(entry => !conflictPaths.has(entry.path));
    const deletePaths = operation.profile === 'full' ? diff.delete.filter(filePath => !conflictPaths.has(filePath)) : [];
    const bytesTotal = files.reduce((sum, entry) => sum + entry.size, 0);
    operation = await db.projectSyncOperations.update(operation.id, {
      phase: 'transfer',
      files_total: files.length,
      files_done: 0,
      bytes_total: bytesTotal,
      bytes_done: bytesDone,
      base_manifest_hash: baseManifest?.hash ?? null,
      source_manifest_hash: source.hash,
      target_manifest_hash: target.hash,
      conflicts: JSON.stringify(conflicts.map(conflict => ({ ...conflict, resolution: resolutions.get(conflict.path) }))),
    });
    const fileProgress = async (bytes: number) => {
      await progress(bytes);
    };
    if (operation.profile === 'results') {
      await pullChanges({
        worker,
        project,
        projectRoot: roots.root,
        profile: operation.profile,
        homeInstanceID,
        operationID: operation.id,
        files: files.map(entry => ({ source: entry, targetPath: entry.path })),
        deletePaths,
        progress: fileProgress,
      });
    } else {
      await pushChanges({
        worker,
        project,
        projectRoot: roots.root,
        profile: operation.profile,
        homeInstanceID,
        operationID: operation.id,
        files,
        deletePaths,
        progress: fileProgress,
      });
    }
    if (operation.profile === 'full') {
      operation = await db.projectSyncOperations.update(operation.id, { phase: 'job-metadata' });
      await syncProjectJobSnapshot(worker, project, roots.root, homeInstanceID);
    }
    filesDone = files.length;
    homeManifest = await buildProjectSyncManifest(roots.root, project.id, operation.profile);
    workerManifest = await fetchRemoteManifest(worker, project, operation.profile, homeInstanceID);
    const baseline = operation.profile === 'results' ? workerManifest : homeManifest;
    await writeProjectSyncBaseManifest(roots.root, worker.id, operation.profile, baseline);
    await db.projectReplicas.update(replica.id, {
      state: 'in_sync',
      base_manifest_hash: baseline.hash,
      local_manifest_hash: homeManifest.hash,
      remote_manifest_hash: workerManifest.hash,
      last_synced_at: new Date(),
      last_error: null,
    });
    return db.projectSyncOperations.update(operation.id, {
      status: 'completed',
      phase: 'completed',
      files_done: filesDone,
      bytes_done: Math.max(bytesDone, bytesTotal),
      base_manifest_hash: baseline.hash,
      source_manifest_hash: operation.profile === 'results' ? workerManifest.hash : homeManifest.hash,
      target_manifest_hash: operation.profile === 'results' ? homeManifest.hash : workerManifest.hash,
      retry_at: null,
      error: null,
    });
  } catch (error) {
    if (!operation) throw error;
    const message = error instanceof Error ? error.message : 'Project sync failed';
    const replica = await db.projectReplicas.findByProjectAndWorker(operation.project_id, operation.worker_id).catch(() => null);
    if (isRemoteFailure(error)) {
      if (replica) await db.projectReplicas.update(replica.id, { state: 'waiting_for_worker', last_error: message }).catch(() => undefined);
      return deferOperation(operation, 'waiting_for_worker', message);
    }
    if (replica) await db.projectReplicas.update(replica.id, { state: 'error', last_error: message }).catch(() => undefined);
    return db.projectSyncOperations.update(operation.id, {
      status: 'failed',
      phase: 'failed',
      retry_at: null,
      error: message.slice(0, 1000),
    });
  } finally {
    runningOperations.delete(operationID);
  }
}

export async function resolveProjectSyncConflicts(
  projectIdentifier: string,
  operationID: string,
  resolutionsInput: Record<string, unknown>,
) {
  const project = await resolveProject(projectIdentifier, { intent: 'write' });
  await assertLocalProjectHome(project);
  const operation = await db.projectSyncOperations.findById(operationID);
  if (!operation || operation.project_id !== project.id) {
    throw new ProjectSyncError('Project sync operation not found', { status: 404 });
  }
  if (operation.status !== 'conflict') {
    throw new ProjectSyncError('Project sync operation is not waiting for conflict resolution', { status: 409 });
  }
  let conflicts: ProjectSyncConflict[];
  try {
    const value: unknown = JSON.parse(operation.conflicts || '[]');
    if (!Array.isArray(value)) throw new Error('invalid conflicts');
    conflicts = value as ProjectSyncConflict[];
  } catch {
    throw new ProjectSyncError('Stored conflict state is invalid', { status: 500 });
  }
  const updated = conflicts.map(conflict => {
    const resolution = resolutionsInput[conflict.path];
    if (resolution !== 'keep-home' && resolution !== 'keep-worker' && resolution !== 'keep-both') {
      throw new ProjectSyncError(`A valid resolution is required for ${conflict.path}`);
    }
    return { ...conflict, resolution };
  });
  return db.projectSyncOperations.update(operation.id, {
    status: 'queued',
    phase: 'queued',
    conflicts: JSON.stringify(updated),
    retry_at: new Date(),
    error: null,
  });
}

export async function processProjectSyncQueue() {
  const candidates = await db.projectSyncOperations.list({
    status: ['queued', 'waiting_for_job', 'waiting_for_worker'],
  });
  const now = Date.now();
  const next = candidates.find(operation => {
    if (runningOperations.has(operation.id)) return false;
    if (!operation.retry_at) return true;
    return new Date(operation.retry_at).getTime() <= now;
  });
  if (!next) return null;
  return runProjectSyncOperation(next.id);
}

export async function rehomeProject(projectIdentifier: string, workerID: string, expectedRevision: number) {
  const project = await resolveProject(projectIdentifier, { intent: 'lifecycle' });
  const oldHomeInstanceID = await assertLocalProjectHome(project);
  if (!Number.isSafeInteger(expectedRevision) || expectedRevision !== project.revision) {
    throw new ProjectSyncError('Project revision changed before rehome', {
      status: 409,
      code: 'PROJECT_SYNC_REVISION_CONFLICT',
    });
  }
  const activeJobs = await db.jobs.list({
    project_id: project.id,
    status: [...PROJECT_SYNC_LIFECYCLE_BLOCKING_JOB_STATUSES],
  });
  if (activeJobs.length > 0) {
    throw new ProjectSyncError('Project cannot be rehomed while jobs are active', {
      status: 409,
      code: 'PROJECT_SYNC_ACTIVE_JOBS',
    });
  }
  const operation = await createProjectSyncOperation(project.id, workerID, 'full');
  const completed = await runProjectSyncOperation(operation.id);
  if (completed.status !== 'completed') {
    throw new ProjectSyncError('A verified full sync is required before rehome', {
      status: 409,
      code: 'PROJECT_SYNC_REHOME_SYNC_REQUIRED',
      details: { operation: completed },
    });
  }
  const worker = await getRemoteWorker(workerID);
  const capabilities = await fetchProjectSyncCapabilities(worker);
  const replica = await db.projectReplicas.findByProjectAndWorker(project.id, workerID);
  if (!replica || replica.remote_instance_id !== capabilities.instance_id || replica.local_manifest_hash !== replica.remote_manifest_hash) {
    throw new ProjectSyncError('Replica manifest verification failed before rehome', {
      status: 409,
      code: 'PROJECT_SYNC_REHOME_SYNC_REQUIRED',
    });
  }
  const blockersAfterSync = await db.jobs.list({
    project_id: project.id,
    status: [...PROJECT_SYNC_LIFECYCLE_BLOCKING_JOB_STATUSES],
  });
  if (blockersAfterSync.length > 0) {
    throw new ProjectSyncError('Project jobs changed while rehome was synchronizing', {
      status: 409,
      code: 'PROJECT_SYNC_ACTIVE_JOBS',
      details: { job_ids: blockersAfterSync.map(job => job.id) },
    });
  }
  const projectRoots = await getProjectRoots(project);
  const verifiedJobSnapshot = await syncProjectJobSnapshot(worker, project, projectRoots.root, oldHomeInstanceID);
  const preparedValue = await remoteJson<unknown>(worker, `${remoteProjectBase(project.id)}/promote`, {
    method: 'POST',
    body: JSON.stringify({ action: 'prepare', home_instance_id: oldHomeInstanceID }),
    timeoutMs: 60_000,
  });
  if (
    !isObject(preparedValue) ||
    typeof preparedValue.token !== 'string' ||
    preparedValue.job_snapshot_hash !== verifiedJobSnapshot.hash
  ) {
    throw new ProjectSyncError('Worker returned a malformed rehome preparation', { status: 502 });
  }
  const rehomeToken = preparedValue.token;
  const blockersBeforeCommit = await db.jobs.list({
    project_id: project.id,
    status: [...PROJECT_SYNC_LIFECYCLE_BLOCKING_JOB_STATUSES],
  });
  const currentJobSnapshot = await buildPortableProjectJobSnapshot(project, projectRoots.root, oldHomeInstanceID);
  if (blockersBeforeCommit.length > 0 || currentJobSnapshot.hash !== verifiedJobSnapshot.hash) {
    await remoteJson(worker, `${remoteProjectBase(project.id)}/promote`, {
      method: 'POST',
      body: JSON.stringify({ action: 'cancel', home_instance_id: oldHomeInstanceID, token: rehomeToken }),
      timeoutMs: 30_000,
    }).catch(() => undefined);
    throw new ProjectSyncError('Project job metadata changed before rehome ownership commit', {
      status: 409,
      code: 'PROJECT_SYNC_JOB_SNAPSHOT_CHANGED',
    });
  }
  const updated = await db.projects.compareAndSet(
    project.id,
    { revision: project.revision, lifecycle_state: 'active' },
    {
      home_worker_id: workerID,
      home_instance_id: capabilities.instance_id,
      revision: project.revision + 1,
      operation_error: null,
    },
  );
  if (!updated) {
    await remoteJson(worker, `${remoteProjectBase(project.id)}/promote`, {
      method: 'POST',
      body: JSON.stringify({ action: 'cancel', home_instance_id: oldHomeInstanceID, token: rehomeToken }),
      timeoutMs: 30_000,
    }).catch(() => undefined);
    throw new ProjectSyncError('Project revision changed while rehome was being committed', {
      status: 409,
      code: 'PROJECT_SYNC_REVISION_CONFLICT',
    });
  }
  try {
    await remoteJson(worker, `${remoteProjectBase(project.id)}/promote`, {
      method: 'POST',
      body: JSON.stringify({ action: 'commit', home_instance_id: oldHomeInstanceID, token: rehomeToken }),
      timeoutMs: 60_000,
    });
  } catch (error) {
    const ownership = await remoteJson<unknown>(worker, `${remoteProjectBase(project.id)}/ownership`, {
      timeoutMs: 30_000,
    }).catch(() => null);
    if (!isObject(ownership) || ownership.home_instance_id !== capabilities.instance_id) {
      if (isObject(ownership) && ownership.home_instance_id === oldHomeInstanceID) {
        await db.projects.compareAndSet(
          project.id,
          { revision: updated.revision, lifecycle_state: 'active' },
          {
            home_worker_id: project.home_worker_id,
            home_instance_id: oldHomeInstanceID,
            revision: updated.revision + 1,
            operation_error: 'Rehome was rolled back because the target did not commit ownership.',
          },
        );
        await remoteJson(worker, `${remoteProjectBase(project.id)}/promote`, {
          method: 'POST',
          body: JSON.stringify({ action: 'cancel', home_instance_id: oldHomeInstanceID, token: rehomeToken }),
          timeoutMs: 30_000,
        }).catch(() => undefined);
      } else {
        await db.projects.update(project.id, {
          operation_error: 'Rehome ownership is uncertain; verify the target worker before changing project state.',
        });
      }
      throw error;
    }
  }
  await db.projectReplicas.update(replica.id, { role: 'home', state: 'in_sync', last_error: null });
  return { project: updated, operation: completed, home_instance_id: capabilities.instance_id };
}

export function projectSyncApiError(error: unknown, fallback: string) {
  if (error instanceof ProjectSyncError || error instanceof ProjectSyncProtocolError) {
    return {
      body: {
        error: error.message,
        code: error.code,
        ...(error.details === undefined ? {} : { details: error.details }),
      },
      status: error.status,
    };
  }
  console.error(fallback, error);
  return { body: { error: fallback, code: 'PROJECT_SYNC_INTERNAL_ERROR' }, status: 500 };
}
