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

type DatasetCopyEntry = { relativePath: string; kind: 'directory' | 'file' };

async function collectSafeDatasetCopyTree(sourceRoot: string) {
  const canonicalRoot = await fsp.realpath(sourceRoot);
  const collected: DatasetCopyEntry[] = [];

  const walk = async (relativePath = ''): Promise<void> => {
    const directory = relativePath ? path.join(sourceRoot, relativePath) : sourceRoot;
    const entries = await fsp.readdir(directory, { withFileTypes: true });
    for (const entry of entries) {
      const childRelativePath = relativePath ? path.join(relativePath, entry.name) : entry.name;
      const childPath = path.join(sourceRoot, childRelativePath);
      const stat = await fsp.lstat(childPath);
      if (stat.isSymbolicLink()) {
        throw new Error(`Dataset copy cannot include symbolic links or junctions: ${childRelativePath}`);
      }
      const canonicalChild = await fsp.realpath(childPath);
      if (!isPathInside(canonicalRoot, canonicalChild)) {
        throw new Error(`Dataset copy source escaped its declared root: ${childRelativePath}`);
      }
      if (stat.isDirectory()) {
        collected.push({ relativePath: childRelativePath, kind: 'directory' });
        await walk(childRelativePath);
      } else if (stat.isFile()) {
        collected.push({ relativePath: childRelativePath, kind: 'file' });
      } else {
        throw new Error(`Dataset copy cannot include special filesystem entries: ${childRelativePath}`);
      }
    }
  };

  await walk();
  return { canonicalRoot, entries: collected };
}

async function copySafeDatasetTree(
  sourceRoot: string,
  destinationRoot: string,
  entries: DatasetCopyEntry[],
  canonicalSourceRoot: string,
) {
  await fsp.mkdir(destinationRoot);
  for (const entry of entries.filter(item => item.kind === 'directory')) {
    await fsp.mkdir(path.join(destinationRoot, entry.relativePath), { recursive: true });
  }
  for (const entry of entries.filter(item => item.kind === 'file')) {
    const source = path.join(sourceRoot, entry.relativePath);
    const sourceStat = await fsp.lstat(source);
    if (!sourceStat.isFile() || sourceStat.isSymbolicLink()) {
      throw new Error(`Dataset copy source changed during transfer: ${entry.relativePath}`);
    }
    const canonicalSource = await fsp.realpath(source);
    if (!isPathInside(canonicalSourceRoot, canonicalSource)) {
      throw new Error(`Dataset copy source escaped its declared root: ${entry.relativePath}`);
    }
    const destination = path.join(destinationRoot, entry.relativePath);
    if (!isPathInside(destinationRoot, destination)) {
      throw new Error(`Dataset copy destination escaped its declared root: ${entry.relativePath}`);
    }
    await fsp.copyFile(source, destination, fs.constants.COPYFILE_EXCL);
  }
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
  if (isPathInside(sourcePath, destination.path) || isPathInside(destination.path, sourcePath)) {
    throw new Error('Dataset copy source and destination cannot contain one another');
  }
  const tree = await collectSafeDatasetCopyTree(sourcePath);
  try {
    await copySafeDatasetTree(sourcePath, destination.path, tree.entries, tree.canonicalRoot);
  } catch (error) {
    const destinationRoot = path.resolve(destinationDatasetsRoot);
    const safeDestination = path.resolve(destination.path);
    if (isPathInside(destinationRoot, safeDestination) && safeDestination !== destinationRoot) {
      await fsp.rm(safeDestination, { recursive: true, force: true }).catch(() => undefined);
    }
    throw error;
  }

  return destination;
}
