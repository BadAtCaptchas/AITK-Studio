export const QUEUE_TABS = ['active', 'history', 'failed', 'all'] as const;
export type QueueTab = (typeof QUEUE_TABS)[number];
export const QUEUE_SORTS = ['newest', 'oldest', 'name'] as const;
export type QueueSort = (typeof QUEUE_SORTS)[number];

export interface QueueWorkspaceState {
  tab: QueueTab;
  sort: QueueSort;
  filter: string;
  selectedJobID: string | null;
  scrollTop: number;
  loadedCount: number;
}

const initialState: QueueWorkspaceState = {
  tab: 'active',
  sort: 'newest',
  filter: '',
  selectedJobID: null,
  scrollTop: 0,
  loadedCount: 50,
};

/** Route-return state only. Never persist job payloads, configurations or search text to storage. */
export function createQueueWorkspaceMemory() {
  let state = { ...initialState };
  return {
    read: (): QueueWorkspaceState => ({ ...state }),
    write: (patch: Partial<QueueWorkspaceState>): void => {
      // Explicit fields prevent accidental retention of a job/config payload from a caller.
      state = {
        tab: patch.tab ?? state.tab,
        sort: patch.sort ?? state.sort,
        filter: patch.filter ?? state.filter,
        selectedJobID: patch.selectedJobID === undefined ? state.selectedJobID : patch.selectedJobID,
        scrollTop: Math.max(0, patch.scrollTop ?? state.scrollTop),
        loadedCount: Math.max(50, patch.loadedCount ?? state.loadedCount),
      };
    },
  };
}

const memory = createQueueWorkspaceMemory();
export const readQueueWorkspace = (): QueueWorkspaceState =>
  typeof window === 'undefined' ? { ...initialState } : memory.read();
export const rememberQueueWorkspace = (patch: Partial<QueueWorkspaceState>): void => {
  if (typeof window !== 'undefined') memory.write(patch);
};

type QueueOrderJob = { id: string; status: string; worker_id: string; gpu_ids: string; queue_position?: number | null };

/** Paging can finish before a concurrent queue insertion is reflected in the loaded rows. */
export function hasCompleteQueuedJobs(jobs: readonly QueueOrderJob[], queuedCount: number | null | undefined): boolean {
  return (
    Number.isSafeInteger(queuedCount) &&
    queuedCount !== null &&
    queuedCount !== undefined &&
    queuedCount >= 0 &&
    jobs.filter(job => job.status === 'queued').length === queuedCount
  );
}

/** A draft is valid only while the lane still has the order the user started editing. */
export function matchesQueuedLaneOrder(
  jobs: readonly QueueOrderJob[],
  expectedJobIDs: readonly string[],
  workerID: string,
  gpuIDs: string,
): boolean {
  const queued = jobs
    .filter(job => job.status === 'queued' && job.worker_id === workerID && job.gpu_ids === gpuIDs)
    .sort((a, b) => (a.queue_position ?? Number.MAX_SAFE_INTEGER) - (b.queue_position ?? Number.MAX_SAFE_INTEGER));
  return queued.length === expectedJobIDs.length && queued.every((job, index) => job.id === expectedJobIDs[index]);
}

/** Reorder only queued jobs within the supplied lane; running jobs are never part of the payload. */
export function moveQueuedJob(
  jobs: readonly QueueOrderJob[],
  jobID: string,
  targetJobID: string,
  workerID: string,
  gpuIDs: string,
): string[] | null {
  const queued = jobs.filter(job => job.status === 'queued' && job.worker_id === workerID && job.gpu_ids === gpuIDs);
  const ids = queued.map(job => job.id);
  const from = ids.indexOf(jobID);
  const to = ids.indexOf(targetJobID);
  if (from < 0 || to < 0 || from === to) return null;
  ids.splice(to, 0, ...ids.splice(from, 1));
  return ids;
}

export function clampInspectorWidth(value: number): number {
  return Number.isFinite(value) ? Math.min(520, Math.max(280, value)) : 360;
}
