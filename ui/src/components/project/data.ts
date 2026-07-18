import { apiClient } from '@/utils/api';
import type { DatasetSummary } from '@/types';
import type {
  ProjectArtifact,
  ProjectArtifactAvailability,
  ProjectArtifactMediaKind,
  ProjectArtifactZone,
  ProjectCardSummary,
  ProjectIdentity,
  ProjectOverview,
  ProjectRecord,
  ProjectReplicaState,
  ProjectRoots,
  ProjectSummary,
  ProjectSummaryJob,
  ProjectSyncOperation,
  ProjectWorkflowState,
  ProjectZoneSummary,
} from './types';

type UnknownRecord = Record<string, unknown>;

const EMPTY_ROOTS: ProjectRoots = {
  root: '',
  datasets: '',
  configs: '',
  runs: '',
  outputs: '',
  models: '',
  assets: '',
  notes: '',
  cache: '',
};

const IMAGE_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'webp', 'jxl', 'gif', 'bmp', 'avif']);
const VIDEO_EXTENSIONS = new Set(['mp4', 'webm', 'mov', 'mkv']);
const AUDIO_EXTENSIONS = new Set(['mp3', 'wav', 'flac', 'ogg', 'm4a']);
const MODEL_EXTENSIONS = new Set(['safetensors', 'ckpt', 'pt', 'pth', 'bin', 'gguf']);
const TEXT_EXTENSIONS = new Set(['txt', 'caption', 'json', 'jsonc', 'yaml', 'yml', 'md', 'toml', 'log', 'csv']);
const ARCHIVE_EXTENSIONS = new Set(['zip', 'tar', 'gz', '7z']);
const PROJECT_ZONES = new Set<ProjectArtifactZone>([
  'datasets',
  'runs',
  'outputs',
  'models',
  'assets',
  'notes',
  'configs',
  'cache',
]);

export function isRecord(value: unknown): value is UnknownRecord {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function stringValue(value: unknown, fallback = '') {
  return typeof value === 'string' ? value : fallback;
}

function nullableString(value: unknown) {
  return typeof value === 'string' && value.trim() ? value : null;
}

function numberValue(value: unknown, fallback = 0) {
  const parsed = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function dateValue(value: unknown) {
  if (value instanceof Date) return value.toISOString();
  if (typeof value === 'string' || typeof value === 'number') {
    const date = new Date(value);
    if (!Number.isNaN(date.getTime())) return date.toISOString();
  }
  return new Date(0).toISOString();
}

function boolValue(value: unknown, fallback = false) {
  return typeof value === 'boolean' ? value : fallback;
}

function recordValue(value: unknown): UnknownRecord {
  return isRecord(value) ? value : {};
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function statusCode(error: unknown) {
  if (!isRecord(error)) return null;
  const response = recordValue(error.response);
  return numberValue(response.status, 0) || null;
}

function projectFromUnknown(value: unknown, fallbackID = ''): ProjectRecord {
  const raw = recordValue(value);
  const id = stringValue(raw.id, fallbackID);
  const slug = stringValue(raw.slug, id || 'project');
  return {
    id,
    slug,
    name: stringValue(raw.name, slug || 'Untitled project'),
    description: stringValue(raw.description),
    badge_asset: nullableString(raw.badge_asset ?? raw.badgeAsset),
    root_path: stringValue(raw.root_path ?? raw.rootPath ?? raw.storage_root),
    created_at: dateValue(raw.created_at ?? raw.createdAt),
    updated_at: dateValue(raw.updated_at ?? raw.updatedAt),
    lifecycle_state: stringValue(
      raw.lifecycle_state ?? raw.lifecycleState ?? raw.state,
      'active',
    ) as ProjectRecord['lifecycle_state'],
    state: stringValue(raw.state ?? raw.lifecycle_state ?? raw.lifecycleState, 'active') as ProjectRecord['state'],
    archived_at: nullableString(raw.archived_at ?? raw.archivedAt),
    revision: numberValue(raw.revision),
    storage_root_path: nullableString(raw.storage_root_path ?? raw.storageRootPath),
    operation_started_at: nullableString(raw.operation_started_at ?? raw.operationStartedAt),
    operation_status: nullableString(raw.operation_status ?? raw.operationStatus),
    operation_error: nullableString(raw.operation_error ?? raw.operationError),
    home_worker_id: nullableString(raw.home_worker_id ?? raw.homeWorkerId),
    home_instance_id: nullableString(raw.home_instance_id ?? raw.homeInstanceId),
    home_instance_name: nullableString(raw.home_instance_name ?? raw.homeInstanceName),
    storage_root: nullableString(raw.storage_root ?? raw.storageRoot ?? raw.root_path),
  };
}

function rootsFromUnknown(value: unknown): ProjectRoots {
  const raw = recordValue(value);
  return Object.fromEntries(
    Object.keys(EMPTY_ROOTS).map(key => [key, stringValue(raw[key], EMPTY_ROOTS[key as keyof ProjectRoots])]),
  ) as ProjectRoots;
}

function replicaFromUnknown(value: unknown, projectId: string, index: number): ProjectReplicaState {
  const raw = recordValue(value);
  return {
    id: stringValue(raw.id, `replica-${index}`),
    projectId: stringValue(raw.projectId ?? raw.project_id, projectId),
    instanceId: stringValue(raw.instanceId ?? raw.instance_id ?? raw.worker_id, 'local'),
    instanceName: stringValue(raw.instanceName ?? raw.instance_name ?? raw.worker_name, 'Local'),
    role: stringValue(raw.role, 'replica') === 'home' ? 'home' : 'replica',
    status: (() => {
      const state = stringValue(raw.status ?? raw.state, 'ready');
      if (state === 'in_sync' || state === 'dirty' || state === 'creating' || state === 'detached') return 'ready';
      if (state === 'waiting_for_job' || state === 'waiting_for_worker' || state === 'conflict') return 'waiting';
      if (state === 'syncing' || state === 'offline' || state === 'incompatible' || state === 'error') return state;
      return 'ready';
    })(),
    lastSyncedAt: nullableString(raw.lastSyncedAt ?? raw.last_synced_at),
    error: nullableString(raw.error),
  };
}

function syncOperationFromUnknown(value: unknown, projectId: string, index: number): ProjectSyncOperation {
  const raw = recordValue(value);
  return {
    id: stringValue(raw.id, `operation-${index}`),
    projectId: stringValue(raw.projectId ?? raw.project_id, projectId),
    profile: stringValue(raw.profile, 'full') as ProjectSyncOperation['profile'],
    status: (() => {
      const state = stringValue(raw.status, 'queued');
      return state === 'waiting_for_job' || state === 'waiting_for_worker' || state === 'conflict'
        ? 'waiting'
        : (state as ProjectSyncOperation['status']);
    })(),
    progress:
      numberValue(raw.files_total) > 0
        ? Math.round((numberValue(raw.files_done) / numberValue(raw.files_total)) * 100)
        : numberValue(raw.bytes_total) > 0
          ? Math.round((numberValue(raw.bytes_done) / numberValue(raw.bytes_total)) * 100)
          : numberValue(raw.progress),
    message: stringValue(raw.message ?? raw.error ?? raw.phase),
    createdAt: dateValue(raw.createdAt ?? raw.created_at),
    updatedAt: dateValue(raw.updatedAt ?? raw.updated_at),
  };
}

export function normalizeProjectIdentity(value: unknown, fallbackID = ''): ProjectIdentity {
  const raw = recordValue(value);
  const projectRaw = isRecord(raw.project) ? raw.project : raw;
  const project = projectFromUnknown(projectRaw, fallbackID);
  const replicasRaw = raw.replicas ?? recordValue(raw.replicaState).replicas;
  const operationsRaw = raw.syncOperations ?? raw.sync_operations;
  return {
    project,
    roots: rootsFromUnknown(raw.roots),
    replicas: arrayValue(replicasRaw).map((item, index) => replicaFromUnknown(item, project.id, index)),
    syncOperations: arrayValue(operationsRaw).map((item, index) => syncOperationFromUnknown(item, project.id, index)),
  };
}

function inferArtifactZone(relativePath: string, explicit: unknown): ProjectArtifactZone {
  const normalizedExplicit = stringValue(explicit).toLowerCase() as ProjectArtifactZone;
  if (PROJECT_ZONES.has(normalizedExplicit)) return normalizedExplicit;
  const segment = relativePath.replace(/\\/g, '/').split('/').filter(Boolean)[0]?.toLowerCase() as ProjectArtifactZone;
  return PROJECT_ZONES.has(segment) ? segment : 'assets';
}

function inferMediaKind(name: string, explicit: unknown, zone: ProjectArtifactZone): ProjectArtifactMediaKind {
  const explicitKind = stringValue(explicit).toLowerCase() as ProjectArtifactMediaKind;
  if (['image', 'video', 'audio', 'model', 'text', 'archive', 'other'].includes(explicitKind)) return explicitKind;
  const extension = name.toLowerCase().split('.').pop() || '';
  if (IMAGE_EXTENSIONS.has(extension)) return 'image';
  if (VIDEO_EXTENSIONS.has(extension)) return 'video';
  if (AUDIO_EXTENSIONS.has(extension)) return 'audio';
  if (MODEL_EXTENSIONS.has(extension) || zone === 'models') return 'model';
  if (TEXT_EXTENSIONS.has(extension)) return 'text';
  if (ARCHIVE_EXTENSIONS.has(extension)) return 'archive';
  return 'other';
}

export function normalizeProjectArtifact(value: unknown, projectId: string, index = 0): ProjectArtifact {
  const raw = recordValue(value);
  const rawPath = nullableString(raw.path ?? raw.absolutePath);
  const relativePath = stringValue(raw.relativePath ?? raw.relative_path ?? raw.portablePath, rawPath || '');
  const name = stringValue(raw.name, relativePath.replace(/\\/g, '/').split('/').pop() || `artifact-${index + 1}`);
  const zone = inferArtifactZone(relativePath, raw.zone ?? raw.artifactType ?? raw.artifact_type);
  const rawAvailability = stringValue(raw.availability, raw.remote === true ? 'remote' : 'local');
  const availability = (
    rawAvailability === 'offline' ? 'missing' : rawAvailability === 'syncing' ? 'remote' : rawAvailability
  ) as ProjectArtifactAvailability;
  const sourceKindRaw = stringValue(raw.sourceKind ?? raw.source_kind ?? raw.source).toLowerCase();
  const sourceKind = ['training', 'generation', 'import'].includes(sourceKindRaw)
    ? (sourceKindRaw as ProjectArtifact['sourceKind'])
    : sourceKindRaw === 'generated'
      ? 'generation'
      : sourceKindRaw === 'training-sample' || sourceKindRaw === 'model'
        ? 'training'
        : relativePath.toLowerCase().includes('sample') || relativePath.toLowerCase().includes('generation')
          ? 'generation'
          : zone === 'models' || zone === 'runs'
            ? 'training'
            : 'unknown';
  return {
    id: stringValue(raw.id, `${projectId}:${relativePath || index}`),
    projectId: stringValue(raw.projectId ?? raw.project_id, projectId),
    name,
    relativePath,
    portableRef: nullableString(raw.portableRef ?? raw.portable_ref),
    path: rawPath,
    kind: stringValue(raw.kind, 'file') === 'folder' ? 'folder' : 'file',
    zone,
    mediaKind: inferMediaKind(name, raw.mediaKind ?? raw.media_kind ?? raw.type ?? raw.kind, zone),
    size: numberValue(raw.size ?? raw.bytes),
    updatedAt: dateValue(raw.updatedAt ?? raw.updated_at ?? raw.modifiedAt ?? raw.modified_at),
    availability: ['local', 'remote', 'both', 'missing'].includes(availability) ? availability : 'local',
    workerId: nullableString(raw.workerId ?? raw.workerID ?? raw.worker_id),
    workerName: nullableString(raw.workerName ?? raw.worker_name),
    sourceRunId: nullableString(
      raw.sourceRunId ?? raw.sourceJobId ?? raw.source_run_id ?? raw.source_job_id ?? raw.job_id,
    ),
    sourceKind,
    previewUrl: nullableString(raw.previewUrl ?? raw.preview_url ?? raw.mediaUrl ?? raw.media_url),
    downloadUrl: nullableString(raw.downloadUrl ?? raw.download_url),
    metadata: recordValue(raw.metadata),
  };
}

function zoneSummaryFromUnknown(value: unknown): ProjectZoneSummary {
  const raw = recordValue(value);
  return {
    fileCount: numberValue(raw.fileCount ?? raw.file_count),
    folderCount: numberValue(raw.folderCount ?? raw.folder_count),
    mediaCount: numberValue(raw.mediaCount ?? raw.media_count),
    totalBytes: numberValue(raw.totalBytes ?? raw.total_bytes),
    recent: arrayValue(raw.recent).map(item => {
      const entry = recordValue(item);
      return {
        name: stringValue(entry.name),
        path: stringValue(entry.path),
        kind: stringValue(entry.kind, 'file') === 'folder' ? ('folder' as const) : ('file' as const),
        updatedAt: dateValue(entry.updatedAt ?? entry.updated_at),
        size: numberValue(entry.size),
      };
    }),
  };
}

function jobFromUnknown(value: unknown): ProjectSummaryJob | null {
  const raw = recordValue(value);
  const id = stringValue(raw.id);
  if (!id) return null;
  return {
    id,
    name: stringValue(raw.name, 'Untitled run'),
    project_id: nullableString(raw.project_id ?? raw.projectId),
    worker_id: stringValue(raw.worker_id ?? raw.workerId, 'local'),
    remote_job_id: nullableString(raw.remote_job_id ?? raw.remoteJobId),
    remote_sync_at: nullableString(raw.remote_sync_at ?? raw.remoteSyncAt),
    remote_error: nullableString(raw.remote_error ?? raw.remoteError),
    gpu_ids: stringValue(raw.gpu_ids ?? raw.gpuIds, '0'),
    created_at: dateValue(raw.created_at ?? raw.createdAt),
    updated_at: dateValue(raw.updated_at ?? raw.updatedAt),
    status: stringValue(raw.status, 'queued'),
    stop: boolValue(raw.stop),
    return_to_queue: boolValue(raw.return_to_queue ?? raw.returnToQueue),
    step: numberValue(raw.step),
    info: stringValue(raw.info),
    speed_string: stringValue(raw.speed_string ?? raw.speedString),
    queue_position: numberValue(raw.queue_position ?? raw.queuePosition),
    pid: raw.pid === null || raw.pid === undefined ? null : numberValue(raw.pid),
    job_type: stringValue(raw.job_type ?? raw.jobType, 'train'),
    job_ref: nullableString(raw.job_ref ?? raw.jobRef),
    save_now: boolValue(raw.save_now ?? raw.saveNow),
    sample_now: boolValue(raw.sample_now ?? raw.sampleNow),
    total_steps:
      raw.total_steps === null || raw.totalSteps === null
        ? null
        : numberValue(raw.total_steps ?? raw.totalSteps) || null,
  };
}

function datasetFromUnknown(value: unknown): DatasetSummary | null {
  const raw = recordValue(value);
  const name = stringValue(raw.name);
  if (!name) return null;
  return {
    name,
    encrypted: boolValue(raw.encrypted),
    itemCount: numberValue(raw.itemCount ?? raw.item_count),
    captionedItemCount: numberValue(raw.captionedItemCount ?? raw.captioned_item_count),
    missingCaptionCount: numberValue(raw.missingCaptionCount ?? raw.missing_caption_count),
    detectedCaptionExt: nullableString(raw.detectedCaptionExt ?? raw.detected_caption_ext),
    source: stringValue(raw.source, 'local') === 'remote' ? 'remote' : 'local',
    worker_id: stringValue(raw.worker_id ?? raw.workerId, 'local'),
    worker_name: nullableString(raw.worker_name ?? raw.workerName) || undefined,
    ref: nullableString(raw.ref) || undefined,
    path: nullableString(raw.path) || undefined,
    importSourcePath: nullableString(raw.importSourcePath ?? raw.import_source_path),
  };
}

export function normalizeProjectSummary(value: unknown, fallbackID = ''): ProjectSummary {
  const raw = recordValue(value);
  const identity = normalizeProjectIdentity(raw, fallbackID);
  const jobs = arrayValue(raw.jobs)
    .map(jobFromUnknown)
    .filter((job): job is ProjectSummaryJob => job !== null);
  const activeJob = jobFromUnknown(raw.activeJob ?? raw.active_job);
  const countsRaw = recordValue(raw.counts);
  const zonesRaw = recordValue(raw.zones);
  const datasets = arrayValue(raw.datasets)
    .map(datasetFromUnknown)
    .filter((item): item is DatasetSummary => item !== null);
  return {
    ...identity,
    datasets,
    jobs,
    activeJob,
    counts: {
      datasets: numberValue(countsRaw.datasets, datasets.length),
      jobs: numberValue(countsRaw.jobs, jobs.length),
      activeJobs: numberValue(countsRaw.activeJobs ?? countsRaw.active_jobs, activeJob ? 1 : 0),
      outputs: numberValue(countsRaw.outputs),
      models: numberValue(countsRaw.models),
    },
    zones: {
      inputs: zoneSummaryFromUnknown(zonesRaw.inputs ?? zonesRaw.datasets),
      runs: zoneSummaryFromUnknown(zonesRaw.runs),
      outputs: zoneSummaryFromUnknown(zonesRaw.outputs),
      models: zoneSummaryFromUnknown(zonesRaw.models),
    },
    recentActivity: arrayValue(raw.recentActivity ?? raw.recent_activity).map((item, index) => {
      const entry = recordValue(item);
      return {
        id: stringValue(entry.id, `activity-${index}`),
        label: stringValue(entry.label, 'Project updated'),
        detail: stringValue(entry.detail),
        kind: stringValue(entry.kind, 'project'),
        updatedAt: dateValue(entry.updatedAt ?? entry.updated_at),
      };
    }),
    fileTree: arrayValue(raw.fileTree ?? raw.file_tree).map(item => {
      const entry = recordValue(item);
      return {
        name: stringValue(entry.name),
        path: stringValue(entry.path),
        relativePath: stringValue(entry.relativePath ?? entry.relative_path),
        kind: stringValue(entry.kind, 'file') === 'folder' ? ('folder' as const) : ('file' as const),
        size: numberValue(entry.size),
        updatedAt: dateValue(entry.updatedAt ?? entry.updated_at),
      };
    }),
  };
}

function deriveWorkflowState(summary: Pick<ProjectSummary, 'counts' | 'datasets' | 'jobs'>): ProjectWorkflowState {
  if (summary.jobs.some(job => ['failed', 'error'].includes(job.status))) return 'attention';
  if (summary.counts.activeJobs > 0) return 'training';
  if (summary.counts.outputs > 0 || summary.counts.models > 0) return 'review';
  if (summary.counts.datasets > 0) {
    const missingCaptions = summary.datasets.some(dataset => (dataset.missingCaptionCount || 0) > 0);
    return missingCaptions ? 'preparing' : 'ready';
  }
  return 'empty';
}

function cardFromUnknown(value: unknown, fallbackID = ''): ProjectCardSummary {
  const raw = recordValue(value);
  const project = projectFromUnknown(raw.project ?? raw, fallbackID);
  const countsRaw = recordValue(raw.counts);
  const latestOutputRaw = raw.latestOutput ?? raw.latest_output;
  const workflowRaw = stringValue(raw.workflowState ?? raw.workflow_state ?? raw.workflow_stage).toLowerCase();
  const workflow = (
    workflowRaw === 'prepare' ? 'preparing' : workflowRaw === 'train' ? 'training' : workflowRaw
  ) as ProjectWorkflowState;
  const activeJob = recordValue(raw.active_job ?? raw.activeJob);
  return {
    project,
    roots: isRecord(raw.roots) ? rootsFromUnknown(raw.roots) : undefined,
    counts: {
      datasets: numberValue(countsRaw.datasets ?? raw.dataset_count),
      jobs: numberValue(countsRaw.jobs ?? countsRaw.runs ?? raw.run_count),
      activeJobs: numberValue(countsRaw.activeJobs ?? countsRaw.active_jobs, Object.keys(activeJob).length > 0 ? 1 : 0),
      outputs: numberValue(countsRaw.outputs ?? raw.output_count),
      models: numberValue(countsRaw.models ?? raw.model_count),
    },
    workflowState: ['empty', 'preparing', 'training', 'review', 'ready', 'attention'].includes(workflow)
      ? workflow
      : 'empty',
    latestOutput: isRecord(latestOutputRaw) ? normalizeProjectArtifact(latestOutputRaw, project.id) : null,
    totalBytes: numberValue(raw.totalBytes ?? raw.total_bytes ?? countsRaw.totalBytes),
    replicaWarnings: numberValue(raw.replicaWarnings ?? raw.replica_warnings),
  };
}

async function fetchLegacySummary(projectID: string) {
  const response = await apiClient.get(`/api/projects/${encodeURIComponent(projectID)}/summary`);
  return normalizeProjectSummary(response.data, projectID);
}

export async function loadProjectIdentity(projectID: string) {
  const response = await apiClient.get(`/api/projects/${encodeURIComponent(projectID)}`);
  return normalizeProjectIdentity(response.data, projectID);
}

export async function loadProjectArtifacts(
  projectID: string,
  zone?: ProjectArtifactZone,
  options: { mediaKind?: string; source?: string; availability?: string; query?: string } = {},
) {
  const response = await apiClient.get(`/api/projects/${encodeURIComponent(projectID)}/artifacts`, {
    params: {
      ...(zone ? { zone, kind: zone } : {}),
      ...(options.mediaKind && options.mediaKind !== 'all' ? { media: options.mediaKind } : {}),
      ...(options.source && options.source !== 'all' ? { source: options.source } : {}),
      ...(options.availability && options.availability !== 'all' ? { availability: options.availability } : {}),
      ...(options.query ? { q: options.query } : {}),
      limit: 200,
    },
  });
  const raw = recordValue(response.data);
  const source = raw.artifacts ?? raw.items ?? response.data;
  const artifacts = arrayValue(source).map((item, index) => normalizeProjectArtifact(item, projectID, index));
  if (zone === 'outputs') {
    return artifacts.filter(
      artifact =>
        artifact.zone === 'outputs' ||
        (artifact.zone === 'runs' && ['image', 'video', 'audio'].includes(artifact.mediaKind)),
    );
  }
  if (zone === 'models') {
    return artifacts.filter(artifact => artifact.zone === 'models' || artifact.mediaKind === 'model');
  }
  return zone ? artifacts.filter(artifact => artifact.zone === zone) : artifacts;
}

export async function loadProjectOverview(projectID: string): Promise<ProjectOverview> {
  let summary: ProjectSummary;
  let overviewRaw: UnknownRecord = {};
  try {
    const response = await apiClient.get(`/api/projects/${encodeURIComponent(projectID)}/overview`);
    overviewRaw = recordValue(response.data);
    summary = normalizeProjectSummary(overviewRaw.overview ?? response.data, projectID);
  } catch (error: unknown) {
    if (statusCode(error) !== 404) throw error;
    summary = await fetchLegacySummary(projectID);
  }

  const suppliedOutputs = arrayValue(overviewRaw.recentOutputs ?? overviewRaw.recent_outputs).map((item, index) =>
    normalizeProjectArtifact(item, summary.project.id, index),
  );
  const suppliedModels = arrayValue(overviewRaw.recentModels ?? overviewRaw.recent_models).map((item, index) =>
    normalizeProjectArtifact(item, summary.project.id, index),
  );
  let recentOutputs = suppliedOutputs;
  let recentModels = suppliedModels;
  if (recentOutputs.length === 0 || recentModels.length === 0) {
    try {
      const [outputArtifacts, modelArtifacts] = await Promise.all([
        recentOutputs.length === 0 ? loadProjectArtifacts(summary.project.id, 'outputs') : Promise.resolve([]),
        recentModels.length === 0 ? loadProjectArtifacts(summary.project.id, 'models') : Promise.resolve([]),
      ]);
      if (recentOutputs.length === 0) recentOutputs = outputArtifacts.slice(0, 8);
      if (recentModels.length === 0) recentModels = modelArtifacts.slice(0, 8);
    } catch {
      if (recentOutputs.length === 0) {
        recentOutputs = summary.zones.outputs.recent.map((item, index) =>
          normalizeProjectArtifact(
            { ...item, relativePath: `outputs/${item.name}`, zone: 'outputs' },
            summary.project.id,
            index,
          ),
        );
      }
      if (recentModels.length === 0) {
        recentModels = summary.zones.models.recent.map((item, index) =>
          normalizeProjectArtifact(
            { ...item, relativePath: `models/${item.name}`, zone: 'models' },
            summary.project.id,
            index,
          ),
        );
      }
    }
  }
  return {
    ...summary,
    replicas: arrayValue(overviewRaw.replicas).map((item, index) =>
      replicaFromUnknown(item, summary.project.id, index),
    ),
    recentOutputs,
    recentModels,
  };
}

export async function loadProjectCards(): Promise<ProjectCardSummary[]> {
  const response = await apiClient.get('/api/projects');
  const raw = recordValue(response.data);
  const entries = arrayValue(raw.cards ?? raw.projects ?? response.data);
  return Promise.all(
    entries.map(async (entry, index) => {
      const card = cardFromUnknown(entry, `project-${index}`);
      const entryRecord = recordValue(entry);
      if (isRecord(entryRecord.counts) || 'workflowState' in entryRecord || 'workflow_state' in entryRecord)
        return card;
      try {
        const summary = await fetchLegacySummary(card.project.id || card.project.slug);
        const latest = summary.zones.outputs.recent[0];
        let latestOutput = latest
          ? normalizeProjectArtifact(
              { ...latest, relativePath: `outputs/${latest.name}`, zone: 'outputs' },
              summary.project.id,
            )
          : null;
        if (!latestOutput || !['image', 'video', 'audio'].includes(latestOutput.mediaKind)) {
          const outputArtifacts = await loadProjectArtifacts(summary.project.id, 'outputs').catch(() => []);
          latestOutput =
            outputArtifacts.find(item => ['image', 'video', 'audio'].includes(item.mediaKind)) || latestOutput;
        }
        return {
          ...card,
          project: summary.project,
          roots: summary.roots,
          counts: summary.counts,
          workflowState: deriveWorkflowState(summary),
          latestOutput,
          totalBytes: Object.values(summary.zones).reduce((total, zoneSummary) => total + zoneSummary.totalBytes, 0),
        };
      } catch {
        return card;
      }
    }),
  );
}

export function projectIsArchived(project: ProjectRecord | null | undefined) {
  const state = project?.lifecycle_state || project?.state;
  return state === 'archived' || !!project?.archived_at;
}

export function artifactPreviewUrl(artifact: ProjectArtifact) {
  return artifact.previewUrl;
}

export function artifactDownloadUrl(artifact: ProjectArtifact) {
  return artifact.downloadUrl;
}
