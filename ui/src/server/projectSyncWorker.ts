import fs from 'fs/promises';
import path from 'path';
import { randomUUID } from 'crypto';
import { db } from './db';
import { getAITKInstanceID, getProjectRoots, PROJECT_FOLDERS, resolveProject } from './projects';
import { getProjectsRoot } from './settings';
import {
  collectCredentialConfigKeys,
  parsePortableProjectJobSnapshot,
  PROJECT_SYNC_CHUNK_BYTES,
  PROJECT_SYNC_LIFECYCLE_BLOCKING_JOB_STATUSES,
  PROJECT_SYNC_PROFILES,
  PROJECT_SYNC_PROTOCOL,
  ProjectSyncProtocolError,
  resolvePortableProjectConfig,
  type ProjectReplicaExecutionAuthorization,
  type PortableProjectJobSnapshot,
} from './projectSyncProtocol';
import type { Project } from '@/types';
import type { Job } from '@/types';

export type ProjectSyncCapabilities = {
  protocol: typeof PROJECT_SYNC_PROTOCOL;
  instance_id: string;
  chunk_bytes: number;
  profiles: typeof PROJECT_SYNC_PROFILES;
  features: {
    resumable_chunks: true;
    manifest_sha256: true;
    range_blobs: true;
    three_way_conflicts: true;
    atomic_file_commit: true;
  };
};

export async function getProjectSyncCapabilities(): Promise<ProjectSyncCapabilities> {
  return {
    protocol: PROJECT_SYNC_PROTOCOL,
    instance_id: await getAITKInstanceID(),
    chunk_bytes: PROJECT_SYNC_CHUNK_BYTES,
    profiles: PROJECT_SYNC_PROFILES,
    features: {
      resumable_chunks: true,
      manifest_sha256: true,
      range_blobs: true,
      three_way_conflicts: true,
      atomic_file_commit: true,
    },
  };
}

function safeIdentity(value: unknown, label: string) {
  const normalized = typeof value === 'string' ? value.trim() : '';
  if (!normalized || !/^[a-zA-Z0-9._-]{1,160}$/.test(normalized)) {
    throw new ProjectSyncProtocolError(`${label} is invalid`, { code: 'PROJECT_SYNC_IDENTITY_INVALID' });
  }
  return normalized;
}

function safeName(value: unknown, fallback: string) {
  const normalized = typeof value === 'string' ? value.trim().slice(0, 120) : '';
  return normalized || fallback;
}

function safeSlug(value: unknown, projectID: string) {
  const normalized = typeof value === 'string'
    ? value
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, 72)
    : '';
  return normalized || `replica-${projectID.slice(0, 8).toLowerCase()}`;
}

function isInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return !!relative && !relative.startsWith('..') && !path.isAbsolute(relative);
}

export async function linkProjectReplica(input: Record<string, unknown>) {
  const projectID = safeIdentity(input.project_id, 'project_id');
  const homeInstanceID = safeIdentity(input.home_instance_id, 'home_instance_id');
  const homeWorkerID = safeIdentity(input.home_worker_id || 'remote-home', 'home_worker_id');
  const localInstanceID = await getAITKInstanceID();
  if (homeInstanceID === localInstanceID) {
    throw new ProjectSyncProtocolError('An instance cannot register itself as an execution replica', {
      status: 409,
      code: 'PROJECT_SYNC_SELF_REPLICA',
    });
  }

  const existing = await db.projects.findById(projectID);
  if (existing) {
    if (existing.home_instance_id && existing.home_instance_id !== homeInstanceID) {
      throw new ProjectSyncProtocolError('Project identity is already linked to another authoritative home', {
        status: 409,
        code: 'PROJECT_SYNC_HOME_MISMATCH',
      });
    }
    const roots = await getProjectRoots(existing);
    await Promise.all(Object.values(roots).map(root => fs.mkdir(root, { recursive: true })));
    const updated = existing.home_instance_id
      ? existing
      : await db.projects.update(existing.id, { home_instance_id: homeInstanceID, home_worker_id: homeWorkerID });
    return { project: updated, created: false, instance_id: localInstanceID };
  }

  const requestedSlug = safeSlug(input.slug, projectID);
  const slugOwner = await db.projects.findBySlug(requestedSlug);
  const slug = slugOwner ? `${requestedSlug}-${projectID.slice(0, 8).toLowerCase()}` : requestedSlug;
  const storageRoot = path.resolve(await getProjectsRoot());
  const finalRoot = path.join(storageRoot, slug);
  if (!isInside(storageRoot, finalRoot)) {
    throw new ProjectSyncProtocolError('Replica root escapes the projects storage boundary', {
      status: 409,
      code: 'PROJECT_SYNC_ROOT_INVALID',
    });
  }
  const stagingRoot = path.join(storageRoot, '.aitk-staging', `replica-${projectID}`);
  const manifest = {
    format: 'aitk-project',
    version: 1,
    project_id: projectID,
    slug,
    created_at: new Date().toISOString(),
    replica: true,
    home_instance_id: homeInstanceID,
  };
  await fs.mkdir(path.dirname(stagingRoot), { recursive: true });
  await fs.rm(stagingRoot, { recursive: true, force: true });
  await fs.mkdir(stagingRoot, { recursive: false });
  let project: Project | null = null;
  try {
    await Promise.all(PROJECT_FOLDERS.map(zone => fs.mkdir(path.join(stagingRoot, zone), { recursive: false })));
    await fs.writeFile(path.join(stagingRoot, '.aitk-project.json'), `${JSON.stringify(manifest, null, 2)}\n`, {
      encoding: 'utf8',
      flag: 'wx',
    });
    await fs.mkdir(path.dirname(finalRoot), { recursive: true });
    await fs.rename(stagingRoot, finalRoot);
    project = await db.projects.create({
      id: projectID,
      slug,
      name: safeName(input.name, slug),
      description: typeof input.description === 'string' ? input.description.trim().slice(0, 500) : '',
      badge_asset: typeof input.badge_asset === 'string' ? input.badge_asset : null,
      root_path: finalRoot,
      storage_root_path: storageRoot,
      lifecycle_state: 'active',
      revision: Number.isInteger(input.revision) ? Math.max(1, Number(input.revision)) : 1,
      home_worker_id: homeWorkerID,
      home_instance_id: homeInstanceID,
    });
    return { project, created: true, instance_id: localInstanceID };
  } catch (error) {
    if (!project) {
      await fs.rm(stagingRoot, { recursive: true, force: true }).catch(() => undefined);
      const row = await db.projects.findById(projectID).catch(() => null);
      if (!row) await fs.rm(finalRoot, { recursive: true, force: true }).catch(() => undefined);
    }
    throw error;
  }
}

export async function assertExecutionReplica(projectIdentifier: string, expectedHomeInstanceID?: string | null) {
  const project = await resolveProject(projectIdentifier, { intent: 'read' });
  const localInstanceID = await getAITKInstanceID();
  if (!project.home_instance_id || project.home_instance_id === localInstanceID) {
    throw new ProjectSyncProtocolError('Project is authoritative on this instance, not an execution replica', {
      status: 409,
      code: 'PROJECT_SYNC_NOT_EXECUTION_REPLICA',
    });
  }
  if (expectedHomeInstanceID && project.home_instance_id !== expectedHomeInstanceID) {
    throw new ProjectSyncProtocolError('Replica authoritative-home identity changed', {
      status: 409,
      code: 'PROJECT_SYNC_HOME_MISMATCH',
    });
  }
  return project;
}

export async function removeExecutionReplica(projectIdentifier: string, expectedHomeInstanceID: string) {
  const project = await assertExecutionReplica(projectIdentifier, expectedHomeInstanceID);
  const activeJobs = await db.jobs.list({
    project_id: project.id,
    status: [...PROJECT_SYNC_LIFECYCLE_BLOCKING_JOB_STATUSES],
  });
  if (activeJobs.length > 0) {
    throw new ProjectSyncProtocolError('Replica has active jobs and cannot be removed', {
      status: 409,
      code: 'PROJECT_SYNC_ACTIVE_JOBS',
      details: { job_ids: activeJobs.map(job => job.id) },
    });
  }
  const allJobs = await db.jobs.list({ project_id: project.id });
  if (allJobs.length > 0) {
    throw new ProjectSyncProtocolError('Replica still owns job records and must be cleaned explicitly', {
      status: 409,
      code: 'PROJECT_SYNC_REPLICA_HAS_JOBS',
      details: { job_ids: allJobs.map(job => job.id) },
    });
  }
  const roots = await getProjectRoots(project);
  const quarantine = `${roots.root}.aitk-removing-${Date.now()}`;
  await fs.rename(roots.root, quarantine);
  try {
    await db.projectSyncOperations.deleteByProject(project.id);
    await db.projectReplicas.deleteByProject(project.id);
    await db.projects.delete(project.id);
    await fs.rm(quarantine, { recursive: true, force: true });
  } catch (error) {
    await fs.rename(quarantine, roots.root).catch(() => undefined);
    throw error;
  }
  return { project_id: project.id, removed: true, instance_id: await getAITKInstanceID() };
}

function safeRehomeToken(value: unknown) {
  const token = typeof value === 'string' ? value.trim() : '';
  if (!/^[a-f0-9-]{36}$/i.test(token)) {
    throw new ProjectSyncProtocolError('Invalid rehome token', { code: 'PROJECT_SYNC_REHOME_TOKEN_INVALID' });
  }
  return token;
}

async function rehomeMarkerPath(project: Project, token: string) {
  const roots = await getProjectRoots(project);
  return path.join(roots.root, '.aitk-sync', 'rehome', `${safeRehomeToken(token)}.json`);
}

export async function prepareExecutionReplicaPromotion(projectIdentifier: string, expectedHomeInstanceID: string) {
  const project = await assertExecutionReplica(projectIdentifier, expectedHomeInstanceID);
  const activeJobs = await db.jobs.list({
    project_id: project.id,
    status: [...PROJECT_SYNC_LIFECYCLE_BLOCKING_JOB_STATUSES],
  });
  if (activeJobs.length > 0) {
    throw new ProjectSyncProtocolError('Project cannot be rehomed while jobs are active', {
      status: 409,
      code: 'PROJECT_SYNC_ACTIVE_JOBS',
      details: { job_ids: activeJobs.map(job => job.id) },
    });
  }
  const jobSnapshot = await readImportedProjectJobSnapshot(project.id, expectedHomeInstanceID);
  const token = randomUUID();
  const markerPath = await rehomeMarkerPath(project, token);
  await fs.mkdir(path.dirname(markerPath), { recursive: true });
  await fs.writeFile(
    markerPath,
    JSON.stringify({
      token,
      project_id: project.id,
      expected_home_instance_id: expectedHomeInstanceID,
      expected_revision: project.revision,
      job_snapshot_hash: jobSnapshot.hash,
      created_at: new Date().toISOString(),
    }),
    { encoding: 'utf8', flag: 'wx' },
  );
  return {
    project_id: project.id,
    token,
    prepared: true,
    instance_id: await getAITKInstanceID(),
    job_snapshot_hash: jobSnapshot.hash,
  };
}

export async function commitExecutionReplicaPromotion(
  projectIdentifier: string,
  expectedHomeInstanceID: string,
  tokenValue: unknown,
) {
  const project = await assertExecutionReplica(projectIdentifier, expectedHomeInstanceID);
  const token = safeRehomeToken(tokenValue);
  const markerPath = await rehomeMarkerPath(project, token);
  let marker: Record<string, unknown>;
  try {
    const value: unknown = JSON.parse(await fs.readFile(markerPath, 'utf8'));
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('invalid marker');
    marker = value as Record<string, unknown>;
  } catch {
    throw new ProjectSyncProtocolError('Rehome preparation was not found', {
      status: 409,
      code: 'PROJECT_SYNC_REHOME_NOT_PREPARED',
    });
  }
  if (
    marker.project_id !== project.id ||
    marker.expected_home_instance_id !== expectedHomeInstanceID ||
    marker.expected_revision !== project.revision
  ) {
    throw new ProjectSyncProtocolError('Project changed after rehome preparation', {
      status: 409,
      code: 'PROJECT_SYNC_REVISION_CONFLICT',
    });
  }
  const jobSnapshot = await readImportedProjectJobSnapshot(project.id, expectedHomeInstanceID);
  if (marker.job_snapshot_hash !== jobSnapshot.hash) {
    throw new ProjectSyncProtocolError('Project job metadata changed after rehome preparation', {
      status: 409,
      code: 'PROJECT_SYNC_JOB_SNAPSHOT_CHANGED',
    });
  }
  const localInstanceID = await getAITKInstanceID();
  const updated = await db.projects.compareAndSet(
    project.id,
    { revision: project.revision, lifecycle_state: 'active' },
    {
      home_instance_id: localInstanceID,
      home_worker_id: 'local',
      revision: project.revision + 1,
      operation_error: null,
    },
  );
  if (!updated) {
    throw new ProjectSyncProtocolError('Project changed while rehome was being committed', {
      status: 409,
      code: 'PROJECT_SYNC_REVISION_CONFLICT',
    });
  }
  await fs.rm(markerPath, { force: true });
  return { project: updated, instance_id: localInstanceID, promoted: true };
}

export async function cancelExecutionReplicaPromotion(
  projectIdentifier: string,
  expectedHomeInstanceID: string,
  tokenValue: unknown,
) {
  const project = await assertExecutionReplica(projectIdentifier, expectedHomeInstanceID);
  const token = safeRehomeToken(tokenValue);
  await fs.rm(await rehomeMarkerPath(project, token), { force: true });
  return { project_id: project.id, token, cancelled: true };
}

export async function getProjectOwnership(projectIdentifier: string) {
  const project = await resolveProject(projectIdentifier, { intent: 'read' });
  return {
    project_id: project.id,
    home_instance_id: project.home_instance_id,
    home_worker_id: project.home_worker_id,
    local_instance_id: await getAITKInstanceID(),
    revision: project.revision,
  };
}

export async function linkProjectJobReplica(projectIdentifier: string, expectedHomeInstanceID: string, input: Record<string, unknown>) {
  const project = await assertExecutionReplica(projectIdentifier, expectedHomeInstanceID);
  const jobValue = input.job;
  if (!jobValue || typeof jobValue !== 'object' || Array.isArray(jobValue)) {
    throw new ProjectSyncProtocolError('job must be an object');
  }
  const job = jobValue as Record<string, unknown>;
  const jobID = safeIdentity(job.id, 'job.id');
  if (job.project_id !== project.id) {
    throw new ProjectSyncProtocolError('Job project identity does not match the linked replica', {
      status: 409,
      code: 'PROJECT_SYNC_CROSS_PROJECT_JOB',
    });
  }
  const roots = await getProjectRoots(project);
  const resolvedConfig = resolvePortableProjectConfig(input.job_config, roots.root, project.id);
  const existing = await db.jobs.findById(jobID);
  if (existing && existing.project_id !== project.id) {
    throw new ProjectSyncProtocolError('Job identity already belongs to another project', {
      status: 409,
      code: 'PROJECT_SYNC_JOB_ID_COLLISION',
    });
  }
  const patch = {
    name: safeName(job.name, jobID),
    project_id: project.id,
    worker_id: 'local',
    remote_job_id: null,
    gpu_ids: typeof job.gpu_ids === 'string' && job.gpu_ids.trim() ? job.gpu_ids : '0',
    job_config: JSON.stringify(resolvedConfig),
    status: 'stopped',
    stop: false,
    return_to_queue: false,
    step: typeof job.step === 'number' && Number.isFinite(job.step) ? Math.max(0, job.step) : 0,
    info: 'Linked from authoritative project home',
    speed_string: '',
    queue_position:
      typeof job.queue_position === 'number' && Number.isFinite(job.queue_position) ? job.queue_position : 0,
    pid: null,
    job_type: typeof job.job_type === 'string' ? job.job_type : 'train',
    job_ref: typeof job.job_ref === 'string' ? job.job_ref : null,
    save_now: false,
  };
  const linked = existing ? await db.jobs.update(existing.id, patch) : await db.jobs.create({ id: jobID, ...patch });
  return { job: linked, project_id: project.id, instance_id: await getAITKInstanceID() };
}

export type ReplicaExecutionAuthorization = ProjectReplicaExecutionAuthorization;

export async function authorizeReplicaJobExecution(job: Job, authorization: ReplicaExecutionAuthorization) {
  if (authorization.protocol !== PROJECT_SYNC_PROTOCOL || !job.project_id || job.project_id !== authorization.projectID) {
    throw new ProjectSyncProtocolError('Replica execution authorization does not match the job project', {
      status: 403,
      code: 'PROJECT_SYNC_EXECUTION_UNAUTHORIZED',
    });
  }
  if (job.worker_id !== 'local' || job.remote_job_id) {
    throw new ProjectSyncProtocolError('Only a local execution-replica job can use this authorization', {
      status: 409,
      code: 'PROJECT_SYNC_EXECUTION_INVALID_JOB',
    });
  }
  const project = await assertExecutionReplica(job.project_id, authorization.homeInstanceID);
  if (project.lifecycle_state !== 'active') {
    throw new ProjectSyncProtocolError('Replica project is not active', {
      status: 409,
      code: 'PROJECT_SYNC_EXECUTION_PROJECT_INACTIVE',
    });
  }
  return project;
}

function jobSnapshotPath(project: Project, homeInstanceID: string) {
  const safeHome = homeInstanceID.replace(/[^a-zA-Z0-9._-]/g, '_');
  return getProjectRoots(project).then(roots => path.join(roots.root, '.aitk-sync', 'job-snapshots', `${safeHome}.json`));
}

export async function importProjectJobSnapshot(
  projectIdentifier: string,
  expectedHomeInstanceID: string,
  input: unknown,
) {
  const project = await assertExecutionReplica(projectIdentifier, expectedHomeInstanceID);
  const snapshot = parsePortableProjectJobSnapshot(input, project.id, expectedHomeInstanceID);
  const activeJobs = await db.jobs.list({ project_id: project.id, status: ['starting', 'running', 'stopping'] });
  if (activeJobs.length > 0) {
    throw new ProjectSyncProtocolError('Project job metadata cannot be replaced while replica jobs are active', {
      status: 409,
      code: 'PROJECT_SYNC_ACTIVE_JOBS',
      details: { job_ids: activeJobs.map(job => job.id) },
    });
  }
  const roots = await getProjectRoots(project);
  const prepared = snapshot.jobs.map(portableJob => {
    const credentialKeys = collectCredentialConfigKeys(portableJob.job_config);
    if (credentialKeys.length > 0) {
      throw new ProjectSyncProtocolError('Project job snapshot contains credential material', {
        status: 409,
        code: 'PROJECT_SYNC_SECRET_BLOCKED',
        details: { job_id: portableJob.id, keys: credentialKeys },
      });
    }
    return {
      portableJob,
      resolvedConfig: resolvePortableProjectConfig(portableJob.job_config, roots.root, project.id),
    };
  });
  for (const item of prepared) {
    const existing = await db.jobs.findById(item.portableJob.id);
    if (existing && existing.project_id !== project.id) {
      throw new ProjectSyncProtocolError('A portable job identity belongs to another project on this worker', {
        status: 409,
        code: 'PROJECT_SYNC_JOB_ID_COLLISION',
        details: { job_id: item.portableJob.id },
      });
    }
  }
  for (const { portableJob, resolvedConfig } of prepared) {
    const patch = {
      name: portableJob.name,
      project_id: project.id,
      worker_id: 'local',
      remote_job_id: null,
      remote_sync_at: portableJob.remote_sync_at == null ? null : new Date(portableJob.remote_sync_at),
      remote_error: portableJob.remote_error,
      gpu_ids: portableJob.gpu_ids,
      job_config: JSON.stringify(resolvedConfig),
      // Queued work remains inert on an execution replica. Its authoritative
      // status is retained in the durable snapshot; rehome blocks queued jobs.
      status: portableJob.status === 'queued' ? 'stopped' : portableJob.status,
      stop: portableJob.stop,
      return_to_queue: portableJob.return_to_queue,
      step: portableJob.step,
      info: portableJob.info,
      speed_string: portableJob.speed_string,
      queue_position: portableJob.queue_position,
      pid: null,
      job_type: portableJob.job_type,
      job_ref: portableJob.job_ref,
      save_now: portableJob.save_now,
      created_at: new Date(portableJob.created_at),
      updated_at: new Date(portableJob.updated_at),
    };
    const existing = await db.jobs.findById(portableJob.id);
    if (existing) await db.jobs.update(existing.id, patch);
    else await db.jobs.create({ id: portableJob.id, ...patch });
    // Mongo create assigns fresh timestamps. The follow-up update makes the
    // portable dates authoritative for both database providers.
    await db.jobs.update(portableJob.id, patch);
  }
  const target = await jobSnapshotPath(project, expectedHomeInstanceID);
  await fs.mkdir(path.dirname(target), { recursive: true });
  const temporary = `${target}.${process.pid}.tmp`;
  await fs.writeFile(temporary, JSON.stringify(snapshot), 'utf8');
  await fs.rename(temporary, target).catch(async error => {
    if (process.platform !== 'win32') throw error;
    await fs.rm(target, { force: true });
    await fs.rename(temporary, target);
  });
  return { project_id: project.id, hash: snapshot.hash, job_count: snapshot.jobs.length, imported: true };
}

export async function readImportedProjectJobSnapshot(projectIdentifier: string, expectedHomeInstanceID: string) {
  const project = await assertExecutionReplica(projectIdentifier, expectedHomeInstanceID);
  const target = await jobSnapshotPath(project, expectedHomeInstanceID);
  let value: unknown;
  try {
    value = JSON.parse(await fs.readFile(target, 'utf8'));
  } catch {
    throw new ProjectSyncProtocolError('Project job snapshot was not imported', {
      status: 404,
      code: 'PROJECT_SYNC_JOB_SNAPSHOT_MISSING',
    });
  }
  const snapshot: PortableProjectJobSnapshot = parsePortableProjectJobSnapshot(
    value,
    project.id,
    expectedHomeInstanceID,
  );
  return { project_id: project.id, hash: snapshot.hash, job_count: snapshot.jobs.length };
}

export function projectSyncWorkerError(error: unknown, fallback: string) {
  if (error instanceof ProjectSyncProtocolError) {
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
