import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { resolveDatasetDirectoryInsideRoot, isPathInside } from './remoteCaptionSecurity';

function safeDatasetCopyName(baseName: string, suffix: string) {
  const safeBase = baseName
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 80);
  const safeSuffix = suffix
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 40);
  const stamp = new Date().toISOString().replace(/-|:|T|Z|\./g, '').slice(0, 14);
  return `${safeBase || 'dataset'}_${safeSuffix || 'copy'}_${stamp}`;
}

function normalizeRequestedDatasetName(value: string) {
  const name = value.trim();
  if (!name) return '';
  if (name === '.' || name.includes('..') || /[\\/]/.test(name)) {
    throw new Error('Dataset copy destination cannot contain path separators or "..".');
  }
  if (/[<>:"|?*\x00-\x1f]/.test(name)) {
    throw new Error('Dataset copy destination contains invalid filename characters.');
  }
  return name;
}

async function uniqueDatasetPath(datasetsRoot: string, requestedName: string) {
  const root = path.resolve(datasetsRoot);
  await fsp.mkdir(root, { recursive: true });

  let candidateName = normalizeRequestedDatasetName(requestedName);
  if (!candidateName) throw new Error('Dataset copy destination is required');

  let candidatePath = path.resolve(root, candidateName);
  let counter = 2;
  while (fs.existsSync(candidatePath)) {
    candidateName = `${requestedName}_${counter}`;
    candidatePath = path.resolve(root, candidateName);
    counter += 1;
  }
  if (!isPathInside(root, candidatePath) || candidatePath === root) {
    throw new Error('Invalid dataset copy destination');
  }
  return { name: candidateName, path: candidatePath };
}

export async function copyDatasetBetweenRoots({
  datasetPath,
  sourceDatasetsRoot,
  destinationDatasetsRoot,
  requestedName,
  suffix = 'copy',
}: {
  datasetPath: string;
  sourceDatasetsRoot: string;
  destinationDatasetsRoot: string;
  requestedName?: string;
  suffix?: string;
}) {
  if (!datasetPath.trim()) {
    throw new Error('datasetPath is required');
  }

  const sourcePath = await resolveDatasetDirectoryInsideRoot(path.resolve(datasetPath), sourceDatasetsRoot);
  const sourceName = path.basename(sourcePath);
  const destinationName = requestedName?.trim()
    ? normalizeRequestedDatasetName(requestedName)
    : safeDatasetCopyName(sourceName, suffix);
  const destination = await uniqueDatasetPath(destinationDatasetsRoot, destinationName);

  await fsp.cp(sourcePath, destination.path, {
    recursive: true,
    force: false,
    errorOnExist: true,
    verbatimSymlinks: false,
  });

  return destination;
}
