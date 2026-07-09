import type { DatasetSummary, Job, Project } from '@/types';

export const PROJECT_LIFECYCLE_STATES = ['creating', 'active', 'archived', 'relocating', 'purging', 'error'] as const;
export type ProjectLifecycleState = (typeof PROJECT_LIFECYCLE_STATES)[number];

export type ProjectRoots = {
  root: string;
  datasets: string;
  configs: string;
  runs: string;
  outputs: string;
  models: string;
  assets: string;
  notes: string;
  cache: string;
};

export type ProjectRecord = Pick<
  Project,
  'id' | 'slug' | 'name' | 'description' | 'badge_asset' | 'root_path' | 'created_at' | 'updated_at'
> & {
  lifecycle_state?: ProjectLifecycleState;
  state?: ProjectLifecycleState;
  archived_at?: string | Date | null;
  revision?: number;
  storage_root_path?: string | null;
  operation_started_at?: string | Date | null;
  operation_status?: string | null;
  operation_error?: string | null;
  home_worker_id?: string | null;
  home_instance_id?: string | null;
  home_instance_name?: string | null;
  storage_root?: string | null;
};

export type ProjectFileTreeItem = {
  name: string;
  path: string;
  relativePath: string;
  kind: 'file' | 'folder';
  size: number;
  updatedAt: string;
};

export type ProjectSummaryJob = Pick<
  Job,
  | 'id'
  | 'name'
  | 'project_id'
  | 'worker_id'
  | 'remote_job_id'
  | 'remote_sync_at'
  | 'remote_error'
  | 'gpu_ids'
  | 'created_at'
  | 'updated_at'
  | 'status'
  | 'stop'
  | 'return_to_queue'
  | 'step'
  | 'info'
  | 'speed_string'
  | 'queue_position'
  | 'pid'
  | 'job_type'
  | 'job_ref'
  | 'save_now'
> & {
  total_steps?: number | null;
};

export type ProjectZoneSummary = {
  fileCount: number;
  folderCount: number;
  mediaCount: number;
  totalBytes: number;
  recent: Array<{ name: string; path: string; kind: 'file' | 'folder'; updatedAt: string; size: number }>;
};

export type ProjectReplicaState = {
  id: string;
  projectId: string;
  instanceId: string;
  instanceName: string;
  role: 'home' | 'replica';
  status: 'ready' | 'syncing' | 'waiting' | 'offline' | 'incompatible' | 'error';
  lastSyncedAt: string | null;
  error: string | null;
};

export type ProjectSyncOperation = {
  id: string;
  projectId: string;
  profile: 'full' | 'launch' | 'results';
  status: 'queued' | 'running' | 'waiting' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  message: string;
  createdAt: string;
  updatedAt: string;
};

export type ProjectIdentity = {
  project: ProjectRecord;
  roots: ProjectRoots;
  replicas: ProjectReplicaState[];
  syncOperations: ProjectSyncOperation[];
};

export type ProjectSummary = {
  project: ProjectRecord;
  roots: ProjectRoots;
  datasets: DatasetSummary[];
  jobs: ProjectSummaryJob[];
  activeJob: ProjectSummaryJob | null;
  counts: {
    datasets: number;
    jobs: number;
    activeJobs: number;
    outputs: number;
    models: number;
  };
  zones: Record<'inputs' | 'runs' | 'outputs' | 'models', ProjectZoneSummary>;
  recentActivity: Array<{ id: string; label: string; detail: string; kind: string; updatedAt: string }>;
  fileTree: ProjectFileTreeItem[];
};

export type ProjectArtifactZone = 'datasets' | 'runs' | 'outputs' | 'models' | 'assets' | 'notes' | 'configs' | 'cache';
export type ProjectArtifactMediaKind = 'image' | 'video' | 'audio' | 'model' | 'text' | 'archive' | 'other';
export type ProjectArtifactAvailability = 'local' | 'remote' | 'both' | 'missing';

export type ProjectArtifact = {
  id: string;
  projectId: string;
  name: string;
  relativePath: string;
  portableRef: string | null;
  path: string | null;
  kind: 'file' | 'folder';
  zone: ProjectArtifactZone;
  mediaKind: ProjectArtifactMediaKind;
  size: number;
  updatedAt: string;
  availability: ProjectArtifactAvailability;
  workerId: string | null;
  workerName: string | null;
  sourceRunId: string | null;
  sourceKind: 'training' | 'generation' | 'import' | 'unknown';
  previewUrl: string | null;
  downloadUrl: string | null;
  metadata: Record<string, unknown>;
};

export type ProjectOverview = ProjectSummary & {
  replicas: ProjectReplicaState[];
  recentOutputs: ProjectArtifact[];
  recentModels: ProjectArtifact[];
};

export type ProjectWorkflowState = 'empty' | 'preparing' | 'training' | 'review' | 'ready' | 'attention';

export type ProjectCardSummary = {
  project: ProjectRecord;
  roots?: ProjectRoots;
  counts: {
    datasets: number;
    jobs: number;
    activeJobs: number;
    outputs: number;
    models: number;
  };
  workflowState: ProjectWorkflowState;
  latestOutput: ProjectArtifact | null;
  totalBytes: number;
  replicaWarnings: number;
};

export type ProjectErrorCode =
  | 'PROJECT_NOT_FOUND'
  | 'PROJECT_ARCHIVED'
  | 'PROJECTS_DISABLED'
  | 'PROJECT_CONFLICT'
  | 'PROJECT_SYNC_REQUIRED'
  | 'PROJECT_INVALID_SCOPE'
  | 'PROJECT_OPERATION_FAILED';
