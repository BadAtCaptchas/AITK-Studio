import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from './settings';
import { getProjectRoots, resolveOptionalProject } from './projects';
import type { Project } from '@/types';

export class DatasetScopeError extends Error {
  status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = 'DatasetScopeError';
    this.status = status;
  }
}

export type DatasetScope = {
  project: Project | null;
  projectID: string | null;
  datasetsRoot: string;
  trainingRoot: string;
};

export function projectIDFromSearchParams(request: Request) {
  try {
    return new URL(request.url).searchParams.get('project_id');
  } catch {
    return null;
  }
}

export async function resolveDatasetScope(projectIdentifier: unknown): Promise<DatasetScope> {
  const project = await resolveOptionalProject(projectIdentifier);
  if (project) {
    const roots = await getProjectRoots(project);
    return {
      project,
      projectID: project.id,
      datasetsRoot: roots.datasets,
      trainingRoot: roots.runs,
    };
  }

  return {
    project: null,
    projectID: null,
    datasetsRoot: await getDatasetsRoot(),
    trainingRoot: await getTrainingFolder(),
  };
}

export function rejectRemoteProjectScope(workerID: string, projectIdentifier: unknown) {
  if (workerID !== 'local' && typeof projectIdentifier === 'string' && projectIdentifier.trim()) {
    throw new DatasetScopeError('Project-scoped dataset editing is only available on the local worker.', 400);
  }
}

export function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}
