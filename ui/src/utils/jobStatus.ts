export const ACTIVE_JOB_STATUSES: string[] = ['queued', 'starting', 'remote-starting', 'running', 'stopping'];
export function isActiveJobStatus(status: string): boolean {
  return ACTIVE_JOB_STATUSES.includes(status);
}
