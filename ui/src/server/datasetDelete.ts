import fsp from 'fs/promises';
import path from 'path';

export class DatasetDeleteError extends Error {
  status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = 'DatasetDeleteError';
    this.status = status;
  }
}

export type DatasetDeleteResult = {
  success: true;
  deleted: boolean;
  path: string;
};

export function resolveDatasetDeletePath(datasetsRoot: string, target: unknown) {
  if (typeof target !== 'string' || target.trim().length === 0) {
    throw new DatasetDeleteError('Invalid dataset path');
  }

  const resolvedRoot = path.resolve(datasetsRoot);
  const resolvedPath = path.resolve(resolvedRoot, target);
  const relativePath = path.relative(resolvedRoot, resolvedPath);

  if (relativePath === '' || relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    throw new DatasetDeleteError('Invalid dataset path');
  }

  return resolvedPath;
}

export async function deleteDatasetFolder(
  datasetsRoot: string,
  target: unknown,
): Promise<DatasetDeleteResult> {
  const datasetPath = resolveDatasetDeletePath(datasetsRoot, target);
  const stat = await fsp.stat(datasetPath).catch(() => null);
  if (!stat) {
    return { success: true, deleted: false, path: datasetPath };
  }

  await fsp.rm(datasetPath, { recursive: true, force: true });
  return { success: true, deleted: true, path: datasetPath };
}
