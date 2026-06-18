import path from 'path';
import {
  areProjectsEnabled,
  getDatasetsRoot,
  getTrainingFolder,
  isProjectSpacesDisabledError,
  PROJECT_SPACES_DISABLED_MESSAGE,
} from './settings';
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

export function hasProjectScopeIdentifier(projectIdentifier: unknown) {
  return typeof projectIdentifier === 'string' && projectIdentifier.trim().length > 0;
}

export async function assertProjectScopeEnabled(projectIdentifier: unknown) {
  if (hasProjectScopeIdentifier(projectIdentifier) && !(await areProjectsEnabled())) {
    throw new DatasetScopeError(PROJECT_SPACES_DISABLED_MESSAGE, 403);
  }
}

export async function resolveDatasetScope(projectIdentifier: unknown): Promise<DatasetScope> {
  let project: Project | null = null;
  try {
    project = await resolveOptionalProject(projectIdentifier);
  } catch (error) {
    if (isProjectSpacesDisabledError(error)) {
      throw new DatasetScopeError(PROJECT_SPACES_DISABLED_MESSAGE, 403);
    }
    throw error;
  }
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
  if (workerID !== 'local' && hasProjectScopeIdentifier(projectIdentifier)) {
    throw new DatasetScopeError('Project-scoped dataset editing is only available on the local worker.', 400);
  }
}

export function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}
