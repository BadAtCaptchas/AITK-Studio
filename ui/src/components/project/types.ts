import type { DatasetSummary, Job, Project } from '@/types';

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

export type ProjectSummary = {
  project: Project;
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
