import type { JobConfig } from '../types';

export type TrainingWorkspace = 'training' | 'continuation';
export const trainingWorkspacePath = (workspace: TrainingWorkspace): string =>
  workspace === 'continuation' ? '/loras/continue' : '/jobs/new';

export function trainingWorkspaceForConfig(config: JobConfig): TrainingWorkspace {
  return config.config.process.some(process => process.network?.continuation !== undefined) ? 'continuation' : 'training';
}

export function trainingWorkspaceDraftKey(workspace: TrainingWorkspace, query: Pick<URLSearchParams, 'get'>): string {
  const key = query.get('migrateId') ? `migrate:${query.get('migrateId')}` : query.get('id') ? `edit:${query.get('id')}` :
    query.get('cloneId') ? `clone:${query.get('cloneId')}` : 'new';
  return workspace === 'continuation' ? `continuation:${key}` : key;
}

export function trainingJobEditorPath(jobConfig: string): string {
  try {
    const value: unknown = JSON.parse(jobConfig);
    if (value && typeof value === 'object' && 'config' in value && value.config && typeof value.config === 'object' &&
        'process' in value.config && Array.isArray(value.config.process) && value.config.process.some((item: unknown) =>
          item && typeof item === 'object' && 'network' in item && item.network && typeof item.network === 'object' &&
          'continuation' in item.network && item.network.continuation !== undefined)) return trainingWorkspacePath('continuation');
  } catch { /* The source loader reports invalid stored configs. */ }
  return trainingWorkspacePath('training');
}
