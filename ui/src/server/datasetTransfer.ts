import archiver from 'archiver';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import yauzl from 'yauzl';
import { copyDatasetBetweenRoots } from './datasetCopy';
import { deleteDatasetFolder } from './datasetDelete';
import { resolveDatasetScope, type DatasetScope } from './datasetScope';
import { isEncryptedDatasetFolder, listDatasetSummaries } from './encryptedDatasets';
import { db } from './db';
import {
  isPathInside as isArchivePathInside,
  listFilesRecursive,
  safeNameSegment,
  shouldIncludeDatasetExportPath,
  validateArchiveEntryName,
} from './trainingJobTransfer';
import type { DatasetSummary, Job } from '../types';

export const DATASET_EXPORT_FORMAT = 'ai-toolkit-dataset-export';
export const DATASET_EXPORT_VERSION = 1;

export type DatasetExportManifest = {
  format: typeof DATASET_EXPORT_FORMAT;
  version: typeof DATASET_EXPORT_VERSION;
  exportedAt: string;
  source: {
    app: 'ai-toolkit';
    datasetName: string;
  };
  dataset: {
    name: string;
    archivePath: 'dataset';
    encrypted: boolean;
  };
};

export function datasetExportFileName(datasetName: string) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  return `${safeNameSegment(datasetName, 'dataset')}_${timestamp}.aitk-dataset.zip`;
}

export async function createDatasetExportArchive(datasetName: string, datasetFolder: string, outputPath: string) {
  const realDatasetFolder = await fsp.realpath(datasetFolder).catch(() => path.resolve(datasetFolder));
  const files = await listFilesRecursive(realDatasetFolder, shouldIncludeDatasetExportPath);
  const manifest: DatasetExportManifest = {
    format: DATASET_EXPORT_FORMAT,
    version: DATASET_EXPORT_VERSION,
    exportedAt: new Date().toISOString(),
    source: {
      app: 'ai-toolkit',
      datasetName,
    },
    dataset: {
      name: datasetName,
      archivePath: 'dataset',
      encrypted: isEncryptedDatasetFolder(realDatasetFolder),
    },
  };

  await fsp.mkdir(path.dirname(outputPath), { recursive: true });
  await new Promise<void>((resolve, reject) => {
    const output = fs.createWriteStream(outputPath);
    const archive = archiver('zip', { zlib: { level: 9 } });

    output.on('close', resolve);
    output.on('error', reject);
    archive.on('error', reject);
    archive.pipe(output);
    archive.append(JSON.stringify(manifest, null, 2), { name: 'manifest.json' });
    if (files.length === 0) archive.append('', { name: 'dataset/.empty' });
    for (const relativePath of files) {
      archive.file(path.join(realDatasetFolder, relativePath), {
        name: path.posix.join('dataset', relativePath.replace(/\\/g, '/')),
      });
    }
    archive.finalize().catch(reject);
  });

  return manifest;
}

export async function extractZipSafely(zipPath: string, destination: string) {
  const destinationRoot = path.resolve(destination);
  await fsp.mkdir(destinationRoot, { recursive: true });

  await new Promise<void>((resolve, reject) => {
    let settled = false;
    const fail = (error: Error) => {
      if (!settled) {
        settled = true;
        reject(error);
      }
    };

    yauzl.open(zipPath, { lazyEntries: true, validateEntrySizes: true }, (openError, zipFile) => {
      if (openError || !zipFile) {
        fail(openError || new Error('Could not open archive'));
        return;
      }

      zipFile.on('error', fail);
      zipFile.on('end', () => {
        if (!settled) {
          settled = true;
          resolve();
        }
      });

      zipFile.readEntry();
      zipFile.on('entry', entry => {
        let normalizedName: string;
        try {
          normalizedName = validateArchiveEntryName(entry.fileName);
        } catch (error) {
          zipFile.close();
          fail(error as Error);
          return;
        }

        const targetPath = path.resolve(destinationRoot, ...normalizedName.split('/'));
        if (!isArchivePathInside(destinationRoot, targetPath)) {
          zipFile.close();
          fail(new Error(`Archive entry escapes import folder: ${entry.fileName}`));
          return;
        }

        if (/\/$/.test(normalizedName)) {
          fsp
            .mkdir(targetPath, { recursive: true })
            .then(() => zipFile.readEntry())
            .catch(error => {
              zipFile.close();
              fail(error);
            });
          return;
        }

        zipFile.openReadStream(entry, (streamError, readStream) => {
          if (streamError || !readStream) {
            zipFile.close();
            fail(streamError || new Error(`Could not read archive entry: ${entry.fileName}`));
            return;
          }

          fsp
            .mkdir(path.dirname(targetPath), { recursive: true })
            .then(() => {
              const writeStream = fs.createWriteStream(targetPath, { flags: 'wx' });
              writeStream.on('error', error => {
                zipFile.close();
                fail(error);
              });
              writeStream.on('close', () => zipFile.readEntry());
              readStream.on('error', error => {
                zipFile.close();
                fail(error);
              });
              readStream.pipe(writeStream);
            })
            .catch(error => {
              zipFile.close();
              fail(error);
            });
        });
      });
    });
  });
}

export async function readDatasetExportManifest(extractRoot: string) {
  const text = await fsp.readFile(path.join(extractRoot, 'manifest.json'), 'utf-8');
  const manifest = JSON.parse(text) as DatasetExportManifest;
  if (manifest.format !== DATASET_EXPORT_FORMAT || manifest.version !== DATASET_EXPORT_VERSION) {
    throw new Error('Unsupported dataset export archive');
  }
  return manifest;
}

export function getExtractedDatasetPath(extractRoot: string, archivePath: string) {
  const normalized = validateArchiveEntryName(archivePath);
  const resolved = path.resolve(extractRoot, ...normalized.split('/'));
  if (!isArchivePathInside(extractRoot, resolved)) {
    throw new Error(`Archive path escapes import folder: ${archivePath}`);
  }
  return resolved;
}

export type DatasetTransferOperation = 'copy' | 'move';

export type DatasetTransferRequest = {
  sourceProjectID: unknown;
  operation: unknown;
  all?: unknown;
  datasetNames?: unknown;
};

export type DatasetTransferItemResult = {
  sourceName: string;
  destinationName: string | null;
  sourcePath: string;
  destinationPath: string | null;
  copied: boolean;
  deleted: boolean;
  rewrittenJobCount: number;
  error?: string;
};

export type DatasetTransferResponse = {
  operation: DatasetTransferOperation;
  sourceProjectID: string;
  all: boolean;
  results: DatasetTransferItemResult[];
  copiedCount: number;
  movedCount: number;
  deletedCount: number;
  failedCount: number;
  rewrittenJobCount: number;
};

export type DatasetTransferTarget = {
  name: string;
  sourcePath: string;
};

export type DatasetPathMapping = {
  sourceName: string;
  destinationName: string;
  sourcePath: string;
  destinationPath: string;
};

export type RewriteProjectJobRefsResult = {
  rewrittenJobCount: number;
  rewrittenJobIDs: string[];
};

export type DatasetTransferDeps = {
  resolveDatasetScope: (projectIdentifier: unknown) => Promise<DatasetScope>;
  listDatasetSummaries: (datasetsRoot: string) => Promise<DatasetSummary[]>;
  copyDatasetBetweenRoots: typeof copyDatasetBetweenRoots;
  deleteDatasetFolder: typeof deleteDatasetFolder;
  listProjectJobs: (projectID: string) => Promise<Job[]>;
  updateJobConfig: (jobID: string, jobConfig: string) => Promise<unknown>;
};

export class DatasetTransferError extends Error {
  status: number;

  constructor(message: string, status = 400) {
    super(message);
    this.name = 'DatasetTransferError';
    this.status = status;
  }
}

const DATASET_PATH_FIELDS = [
  'folder_path',
  'dataset_path',
  'control_path',
  'control_path_1',
  'control_path_2',
  'control_path_3',
  'mask_path',
  'unconditional_path',
  'inpaint_path',
  'clip_image_path',
] as const;

const MOVE_BLOCKING_STATUSES = new Set(['running', 'stopping']);

const defaultDeps: DatasetTransferDeps = {
  resolveDatasetScope,
  listDatasetSummaries,
  copyDatasetBetweenRoots,
  deleteDatasetFolder,
  listProjectJobs: projectID => db.jobs.list({ project_id: projectID }),
  updateJobConfig: (jobID, jobConfig) => db.jobs.update(jobID, { job_config: jobConfig }),
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isProtocolPath(value: string) {
  return /^[a-z][a-z0-9+.-]*:\/\//i.test(value) || value.startsWith('aitk-');
}

function isPathInside(parent: string, child: string) {
  const relative = path.relative(path.resolve(parent), path.resolve(child));
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

function normalizeSourceProjectID(value: unknown) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new DatasetTransferError('source_project_id is required');
  }
  return value.trim();
}

function normalizeOperation(value: unknown): DatasetTransferOperation {
  if (value === 'copy' || value === 'move') return value;
  throw new DatasetTransferError('operation must be "copy" or "move"');
}

function normalizeDatasetName(value: unknown) {
  if (typeof value !== 'string') {
    throw new DatasetTransferError('dataset_names must contain dataset names');
  }
  const name = value.trim();
  if (!name) {
    throw new DatasetTransferError('Dataset name is required');
  }
  if (name === '.' || name.includes('..') || /[\\/]/.test(name)) {
    throw new DatasetTransferError('Dataset name cannot contain path separators or "..".');
  }
  if (/[<>:"|?*\x00-\x1f]/.test(name)) {
    throw new DatasetTransferError('Dataset name contains invalid filename characters.');
  }
  return name;
}

function normalizeDatasetNames(value: unknown) {
  if (!Array.isArray(value) || value.length === 0) {
    throw new DatasetTransferError('dataset_names must include at least one dataset');
  }

  const seen = new Set<string>();
  const names: string[] = [];
  value.forEach(rawName => {
    const name = normalizeDatasetName(rawName);
    const key = name.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      names.push(name);
    }
  });
  return names;
}

async function realpathOrResolve(targetPath: string) {
  return fsp.realpath(targetPath).catch(() => path.resolve(targetPath));
}

async function selectTransferTargets(
  sourceDatasetsRoot: string,
  all: boolean,
  datasetNames: unknown,
  deps: DatasetTransferDeps,
): Promise<DatasetTransferTarget[]> {
  if (all) {
    const summaries = await deps.listDatasetSummaries(sourceDatasetsRoot);
    return summaries.map(summary => ({
      name: summary.name,
      sourcePath: summary.path || path.join(sourceDatasetsRoot, summary.name),
    }));
  }

  return normalizeDatasetNames(datasetNames).map(name => ({
    name,
    sourcePath: path.join(sourceDatasetsRoot, name),
  }));
}

function resolveMappedPath(value: string, mappings: DatasetPathMapping[]) {
  if (!value.trim() || isProtocolPath(value)) return null;

  const resolvedValue = path.resolve(value);
  for (const mapping of mappings) {
    const sourcePath = path.resolve(mapping.sourcePath);
    if (!isPathInside(sourcePath, resolvedValue)) continue;

    const relativePath = path.relative(sourcePath, resolvedValue);
    return relativePath ? path.join(mapping.destinationPath, relativePath) : mapping.destinationPath;
  }

  return null;
}

function rewriteDatasetPathValue(value: unknown, mappings: DatasetPathMapping[]): { value: unknown; changed: boolean } {
  if (typeof value === 'string') {
    const rewritten = resolveMappedPath(value, mappings);
    return rewritten ? { value: rewritten, changed: rewritten !== value } : { value, changed: false };
  }

  if (Array.isArray(value)) {
    let changed = false;
    const rewritten = value.map(item => {
      const result = rewriteDatasetPathValue(item, mappings);
      changed = changed || result.changed;
      return result.value;
    });
    return { value: rewritten, changed };
  }

  return { value, changed: false };
}

function pathValueReferencesSources(value: unknown, sourcePaths: string[]): boolean {
  if (typeof value === 'string') {
    if (!value.trim() || isProtocolPath(value)) return false;
    const resolvedValue = path.resolve(value);
    return sourcePaths.some(sourcePath => isPathInside(sourcePath, resolvedValue));
  }

  if (Array.isArray(value)) {
    return value.some(item => pathValueReferencesSources(item, sourcePaths));
  }

  return false;
}

function rewriteProcessDatasetRefs(processConfig: Record<string, unknown>, mappings: DatasetPathMapping[]) {
  let changed = false;
  const datasets = Array.isArray(processConfig.datasets) ? processConfig.datasets : [];

  datasets.forEach(dataset => {
    if (!isRecord(dataset)) return;
    DATASET_PATH_FIELDS.forEach(field => {
      if (!(field in dataset)) return;
      const result = rewriteDatasetPathValue(dataset[field], mappings);
      if (result.changed) {
        dataset[field] = result.value;
        changed = true;
      }
    });
  });

  if (isRecord(processConfig.caption) && 'path_to_caption' in processConfig.caption) {
    const result = rewriteDatasetPathValue(processConfig.caption.path_to_caption, mappings);
    if (result.changed) {
      processConfig.caption.path_to_caption = result.value;
      changed = true;
    }
  }

  return changed;
}

function processReferencesSources(processConfig: Record<string, unknown>, sourcePaths: string[]) {
  const datasets = Array.isArray(processConfig.datasets) ? processConfig.datasets : [];
  const datasetReference = datasets.some(dataset => {
    if (!isRecord(dataset)) return false;
    return DATASET_PATH_FIELDS.some(field => field in dataset && pathValueReferencesSources(dataset[field], sourcePaths));
  });
  if (datasetReference) return true;

  return (
    isRecord(processConfig.caption) &&
    'path_to_caption' in processConfig.caption &&
    pathValueReferencesSources(processConfig.caption.path_to_caption, sourcePaths)
  );
}

function getProcessConfigs(jobConfig: unknown) {
  if (!isRecord(jobConfig) || !isRecord(jobConfig.config) || !Array.isArray(jobConfig.config.process)) {
    return [];
  }
  return jobConfig.config.process.filter(isRecord);
}

export function rewriteJobConfigDatasetRefs(jobConfig: unknown, mappings: DatasetPathMapping[]) {
  let changed = false;
  getProcessConfigs(jobConfig).forEach(processConfig => {
    changed = rewriteProcessDatasetRefs(processConfig, mappings) || changed;
  });
  return { jobConfig, changed };
}

export function jobConfigReferencesDatasetSources(jobConfig: unknown, sourcePaths: string[]) {
  return getProcessConfigs(jobConfig).some(processConfig => processReferencesSources(processConfig, sourcePaths));
}

function rawJobConfigMentionsSources(rawJobConfig: string, sourcePaths: string[]) {
  return sourcePaths.some(sourcePath => {
    const resolved = path.resolve(sourcePath);
    return rawJobConfig.includes(resolved) || rawJobConfig.includes(resolved.replace(/\\/g, '\\\\'));
  });
}

async function assertNoActiveJobReferences(projectID: string, sourcePaths: string[], deps: DatasetTransferDeps) {
  const jobs = await deps.listProjectJobs(projectID);
  const blockingJobs: string[] = [];

  jobs.forEach(job => {
    if (!MOVE_BLOCKING_STATUSES.has(job.status)) return;
    try {
      if (jobConfigReferencesDatasetSources(JSON.parse(job.job_config), sourcePaths)) {
        blockingJobs.push(job.name || job.id);
      }
    } catch {
      if (rawJobConfigMentionsSources(job.job_config, sourcePaths)) {
        blockingJobs.push(job.name || job.id);
      }
    }
  });

  if (blockingJobs.length > 0) {
    throw new DatasetTransferError(
      `Cannot move project datasets while running or stopping jobs reference them: ${blockingJobs.join(', ')}`,
      409,
    );
  }
}

export async function rewriteProjectJobDatasetRefs(
  projectID: string,
  mappings: DatasetPathMapping[],
  deps: DatasetTransferDeps = defaultDeps,
): Promise<RewriteProjectJobRefsResult> {
  const jobs = await deps.listProjectJobs(projectID);
  const rewrittenJobIDs: string[] = [];

  for (const job of jobs) {
    let parsedConfig: unknown;
    try {
      parsedConfig = JSON.parse(job.job_config);
    } catch (error) {
      if (rawJobConfigMentionsSources(job.job_config, mappings.map(mapping => mapping.sourcePath))) {
        throw new DatasetTransferError(`Job "${job.name || job.id}" has invalid JSON config and references a moved dataset.`);
      }
      continue;
    }

    const result = rewriteJobConfigDatasetRefs(parsedConfig, mappings);
    if (!result.changed) continue;

    await deps.updateJobConfig(job.id, JSON.stringify(result.jobConfig));
    rewrittenJobIDs.push(job.id);
  }

  return {
    rewrittenJobCount: rewrittenJobIDs.length,
    rewrittenJobIDs,
  };
}

export async function transferProjectDatasetsToGlobal(
  request: DatasetTransferRequest,
  deps: DatasetTransferDeps = defaultDeps,
): Promise<DatasetTransferResponse> {
  const sourceProjectID = normalizeSourceProjectID(request.sourceProjectID);
  const operation = normalizeOperation(request.operation);
  const all = request.all === true;
  const sourceScope = await deps.resolveDatasetScope(sourceProjectID);
  if (!sourceScope.projectID) {
    throw new DatasetTransferError('source_project_id must resolve to a project');
  }
  const destinationScope = await deps.resolveDatasetScope(null);

  const targets = await selectTransferTargets(sourceScope.datasetsRoot, all, request.datasetNames, deps);
  const resolvedTargets = await Promise.all(
    targets.map(async target => ({
      ...target,
      sourcePath: await realpathOrResolve(target.sourcePath),
    })),
  );

  if (operation === 'move') {
    await assertNoActiveJobReferences(
      sourceScope.projectID,
      resolvedTargets.map(target => target.sourcePath),
      deps,
    );
  }

  const results: DatasetTransferItemResult[] = [];

  for (const target of resolvedTargets) {
    const result: DatasetTransferItemResult = {
      sourceName: target.name,
      destinationName: null,
      sourcePath: target.sourcePath,
      destinationPath: null,
      copied: false,
      deleted: false,
      rewrittenJobCount: 0,
    };

    try {
      const destination = await deps.copyDatasetBetweenRoots({
        datasetPath: target.sourcePath,
        sourceDatasetsRoot: sourceScope.datasetsRoot,
        destinationDatasetsRoot: destinationScope.datasetsRoot,
        requestedName: target.name,
      });
      result.destinationName = destination.name;
      result.destinationPath = destination.path;
      result.copied = true;

      if (operation === 'move') {
        try {
          const rewriteResult = await rewriteProjectJobDatasetRefs(
            sourceScope.projectID,
            [
              {
                sourceName: target.name,
                destinationName: destination.name,
                sourcePath: target.sourcePath,
                destinationPath: destination.path,
              },
            ],
            deps,
          );
          result.rewrittenJobCount = rewriteResult.rewrittenJobCount;
          const deleteResult = await deps.deleteDatasetFolder(sourceScope.datasetsRoot, target.name);
          result.deleted = deleteResult.deleted;
        } catch (error) {
          result.error = `Copied to global, but kept the project dataset: ${
            error instanceof Error ? error.message : 'move finalization failed'
          }`;
        }
      }
    } catch (error) {
      result.error = error instanceof Error ? error.message : 'Dataset transfer failed';
    }

    results.push(result);
  }

  return {
    operation,
    sourceProjectID: sourceScope.projectID,
    all,
    results,
    copiedCount: results.filter(result => result.copied).length,
    movedCount: operation === 'move' ? results.filter(result => result.deleted).length : 0,
    deletedCount: results.filter(result => result.deleted).length,
    failedCount: results.filter(result => result.error).length,
    rewrittenJobCount: results.reduce((total, result) => total + result.rewrittenJobCount, 0),
  };
}
