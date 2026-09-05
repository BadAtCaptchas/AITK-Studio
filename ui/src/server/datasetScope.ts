import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from './settings';

export class DatasetScopeError extends Error {
  constructor(
    message: string,
    public status = 400,
    public code?: string,
    public details?: unknown,
  ) {
    super(message);
    this.name = 'DatasetScopeError';
  }
}
export type DatasetScope = { datasetsRoot: string; trainingRoot: string };
export async function resolveDatasetScope(): Promise<DatasetScope> {
  const [datasetsRoot, trainingRoot] = await Promise.all([getDatasetsRoot(), getTrainingFolder()]);
  return { datasetsRoot, trainingRoot };
}
export function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
}
