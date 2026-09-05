/** Compatibility boundary for clients predating the global-only application. */
const obsoleteKeys = new Set([
  'includeProjectActive',
  'project_id',
  'projectID',
  'projectId',
  'project_id_override',
  'project_sync',
  'remote_project_id',
  'source_project_id',
  'destination_project_id',
  'include_project_active',
  'PROJECTS_ENABLED',
  'PROJECTS_FOLDER',
]);

export class ObsoleteWorkspaceError extends Error {
  readonly status = 400;
  readonly code = 'PROJECT_WORKSPACES_REMOVED';
  constructor() {
    super('Project workspaces have been removed. Select a global resource and retry.');
    this.name = 'ObsoleteWorkspaceError';
  }
}

export function hasObsoleteWorkspaceScope(value: unknown): boolean {
  if (typeof value === 'string') return value.startsWith('aitk-project://');
  if (!value || typeof value !== 'object') return false;
  if (value instanceof URLSearchParams || (typeof FormData !== 'undefined' && value instanceof FormData)) {
    return (
      Array.from(value.entries()).some(([key, entry]) => obsoleteKeys.has(key) || hasObsoleteWorkspaceScope(entry)) ||
      (value.has('scope') && value.get('scope') !== 'global')
    );
  }
  if (Array.isArray(value)) return value.some(hasObsoleteWorkspaceScope);
  return Object.entries(value).some(
    ([key, nested]) =>
      obsoleteKeys.has(key) ||
      (key === 'scope' && (nested === 'project' || nested === 'all')) ||
      hasObsoleteWorkspaceScope(nested),
  );
}

export function assertGlobalPayload<T>(value: T): T {
  if (hasObsoleteWorkspaceScope(value)) throw new ObsoleteWorkspaceError();
  return value;
}

/** Older workers may return scoped records; never mirror them into global data. */
export function isLegacyScopedRecord(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false;
  const record = value as Record<string, unknown>;
  return ['project_id', 'projectID', 'projectId', 'remote_project_id'].some(
    key => record[key] !== undefined && record[key] !== null && record[key] !== '',
  );
}

export function hasObsoleteWorkspaceHeaders(headers: Headers): boolean {
  return Array.from(headers.keys()).some(key => key.toLowerCase().startsWith('x-aitk-project-'));
}
