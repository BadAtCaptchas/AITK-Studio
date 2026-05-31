import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { cleanDatasetName, listDatasetSummaries, resolveDatasetFolder } from './encryptedDatasets';
import type { DatasetSummary } from '../types';

export class DatasetRenameError extends Error {
  status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = 'DatasetRenameError';
    this.status = status;
  }
}

export type DatasetRenameResult = {
  success: true;
  oldName: string;
  name: string;
  path: string;
  dataset: DatasetSummary;
};

async function sameExistingDirectory(left: string, right: string) {
  const [leftReal, rightReal] = await Promise.all([
    fsp.realpath(left).catch(() => null),
    fsp.realpath(right).catch(() => null),
  ]);
  if (!leftReal || !rightReal) return false;
  const normalize = process.platform === 'win32' ? (value: string) => value.toLowerCase() : (value: string) => value;
  return normalize(path.resolve(leftReal)) === normalize(path.resolve(rightReal));
}

async function datasetSummary(datasetsRoot: string, datasetName: string, datasetPath: string): Promise<DatasetSummary> {
  const dataset = (await listDatasetSummaries(datasetsRoot)).find(item => item.name === datasetName);
  return (
    dataset || {
      name: datasetName,
      encrypted: false,
      source: 'local',
      worker_id: 'local',
      worker_name: 'Local',
      ref: `aitk-dataset://local/${encodeURIComponent(datasetName)}`,
      path: datasetPath,
    }
  );
}

export async function renameDatasetFolder(
  datasetsRoot: string,
  rawOldName: unknown,
  rawNewName: unknown,
): Promise<DatasetRenameResult> {
  const oldNameInput = typeof rawOldName === 'string' ? rawOldName.trim() : '';
  const newName = cleanDatasetName(typeof rawNewName === 'string' ? rawNewName : '');

  if (!oldNameInput) {
    throw new DatasetRenameError('Dataset name is required');
  }
  if (!newName) {
    throw new DatasetRenameError('New dataset name is required');
  }

  let oldPath: string;
  let newPath: string;
  try {
    oldPath = resolveDatasetFolder(datasetsRoot, oldNameInput);
    newPath = resolveDatasetFolder(datasetsRoot, newName);
  } catch {
    throw new DatasetRenameError('Invalid dataset name');
  }

  const oldStat = await fsp.stat(oldPath).catch(() => null);
  if (!oldStat?.isDirectory()) {
    throw new DatasetRenameError('Dataset not found', 404);
  }

  const actualOldName = path.basename(oldPath);
  const targetStat = await fsp.stat(newPath).catch(() => null);
  if (targetStat && !(await sameExistingDirectory(oldPath, newPath))) {
    throw new DatasetRenameError('A dataset with that name already exists', 409);
  }

  if (actualOldName !== newName) {
    await fsp.rename(oldPath, newPath);
  }

  const finalPath = fs.existsSync(newPath) ? newPath : oldPath;
  return {
    success: true,
    oldName: actualOldName,
    name: fs.existsSync(newPath) ? newName : actualOldName,
    path: finalPath,
    dataset: await datasetSummary(datasetsRoot, fs.existsSync(newPath) ? newName : actualOldName, finalPath),
  };
}
