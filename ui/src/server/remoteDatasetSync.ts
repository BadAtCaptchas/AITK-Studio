import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import type { WorkerNodeRecord } from './db';
import { getDatasetsRoot } from './settings';
import {
  createDatasetExportArchive,
  datasetExportFileName,
} from './datasetTransfer';
import { remoteJson, uploadDatasetArchiveToWorker } from './remoteClient';
import {
  cloneJson,
  collectDatasetReferences,
  collectSameWorkerRemoteDatasetReferences,
  isPathInside,
  isRemoteReference,
  resolveConfigPath,
  safeNameSegment,
  setConfigPathValue,
  rewriteSameWorkerRemoteDatasetRefs,
} from './trainingJobTransfer';
import type { DatasetSummary } from '../types';

export type RemoteDatasetSyncStatus =
  | 'checking-datasets'
  | 'zipping-dataset'
  | 'uploading-dataset';

export type RemoteDatasetSyncProgress = {
  status: RemoteDatasetSyncStatus;
  message: string;
  percent: number;
  datasetName: string | null;
  bytesProcessed?: number;
  bytesTotal?: number;
};

export type RemoteDatasetSyncMapping = {
  datasetName: string;
  localDatasetPath: string;
  remoteDatasetName: string;
  remoteDatasetPath: string;
  uploaded: boolean;
};

export type RemoteDatasetSyncResult = {
  jobConfig: any;
  mappings: RemoteDatasetSyncMapping[];
  warnings: string[];
};

type LocalDatasetReference = {
  configPath: string;
  relativePath: string;
};

type LocalDatasetGroup = {
  datasetName: string;
  localDatasetPath: string;
  refs: LocalDatasetReference[];
};

type FileUploadProgress = {
  loaded: number;
  total: number;
};

type RemoteDatasetSyncDeps = {
  getDatasetsRoot: () => Promise<string>;
  listRemoteDatasets: (worker: WorkerNodeRecord) => Promise<DatasetSummary[]>;
  createDatasetArchive: (datasetName: string, datasetFolder: string, outputPath: string) => Promise<unknown>;
  uploadDatasetArchive: (
    worker: WorkerNodeRecord,
    zipPath: string,
    preferredName: string,
    onProgress?: (progress: FileUploadProgress) => void,
  ) => Promise<{ dataset?: { name?: string; encrypted?: boolean; path?: string }; path?: string; renamed?: boolean }>;
  stat: (targetPath: string) => Promise<fs.Stats>;
  realpath: (targetPath: string) => Promise<string>;
  rmPath: (targetPath: string, options: { force?: boolean; recursive?: boolean }) => Promise<void>;
};

const defaultDeps: RemoteDatasetSyncDeps = {
  getDatasetsRoot,
  listRemoteDatasets: worker => remoteJson<DatasetSummary[]>(worker, '/api/datasets/list'),
  createDatasetArchive: createDatasetExportArchive,
  uploadDatasetArchive: uploadDatasetArchiveToWorker,
  stat: targetPath => fsp.stat(targetPath),
  realpath: targetPath => fsp.realpath(targetPath),
  rmPath: (targetPath, options) => fsp.rm(targetPath, options),
};

function splitRelativePath(relativePath: string) {
  return relativePath.split(/[\\/]+/).filter(Boolean);
}

function joinRemotePath(basePath: string, relativePath: string) {
  const cleanedRelativePath = relativePath.replace(/\\/g, '/').replace(/^\/+/, '');
  if (!cleanedRelativePath) return basePath;

  const separator = basePath.includes('\\') && !basePath.includes('/') ? '\\' : '/';
  const base = basePath.replace(/[\\/]+$/, '');
  return `${base}${separator}${cleanedRelativePath.split('/').join(separator)}`;
}

async function realpathOrResolve(targetPath: string, deps: RemoteDatasetSyncDeps) {
  try {
    return await deps.realpath(targetPath);
  } catch {
    return path.resolve(targetPath);
  }
}

async function collectLocalDatasetGroups(
  jobConfig: any,
  datasetsRoot: string,
  deps: RemoteDatasetSyncDeps,
) {
  const warnings: string[] = [];
  const groups = new Map<string, LocalDatasetGroup>();
  const datasetsRootReal = await realpathOrResolve(datasetsRoot, deps);

  for (const ref of collectDatasetReferences(jobConfig)) {
    if (isRemoteReference(ref.value)) continue;

    const absolutePath = resolveConfigPath(ref.value);
    try {
      await deps.stat(absolutePath);
    } catch {
      warnings.push(`Dataset path not found and was not uploaded: ${ref.value}`);
      continue;
    }

    const realTargetPath = await realpathOrResolve(absolutePath, deps);
    if (!isPathInside(datasetsRootReal, realTargetPath)) {
      warnings.push(`Dataset path is outside the datasets folder and was not uploaded: ${ref.value}`);
      continue;
    }

    const relativeToRoot = path.relative(datasetsRootReal, realTargetPath);
    const [datasetName] = splitRelativePath(relativeToRoot);
    if (!datasetName) {
      warnings.push(`Dataset path does not resolve to a dataset folder and was not uploaded: ${ref.value}`);
      continue;
    }

    const localDatasetPath = path.join(datasetsRootReal, datasetName);
    const datasetStat = await deps.stat(localDatasetPath).catch(() => null);
    if (!datasetStat?.isDirectory()) {
      warnings.push(`Dataset path is not inside a dataset folder and was not uploaded: ${ref.value}`);
      continue;
    }

    const groupKey = localDatasetPath.toLowerCase();
    const group =
      groups.get(groupKey) ||
      ({
        datasetName,
        localDatasetPath,
        refs: [],
      } satisfies LocalDatasetGroup);
    group.refs.push({
      configPath: ref.configPath,
      relativePath: path.relative(localDatasetPath, realTargetPath),
    });
    groups.set(groupKey, group);
  }

  return { groups: Array.from(groups.values()), warnings };
}

function applyRemoteDatasetPath(jobConfig: any, group: LocalDatasetGroup, remoteDatasetPath: string) {
  for (const ref of group.refs) {
    setConfigPathValue(jobConfig, ref.configPath, joinRemotePath(remoteDatasetPath, ref.relativePath));
  }
}

function progressForUpload(uploadIndex: number, uploadTotal: number, loaded: number, total: number) {
  if (uploadTotal <= 0 || total <= 0) return 15;
  const uploadSpan = 45 / uploadTotal;
  const uploadStart = 15 + uploadSpan * uploadIndex;
  return uploadStart + uploadSpan * Math.min(1, loaded / total);
}

export async function syncRemoteDatasetsForJobConfig(
  rawJobConfig: any,
  worker: WorkerNodeRecord,
  options: {
    onProgress?: (progress: RemoteDatasetSyncProgress) => void;
    deps?: Partial<RemoteDatasetSyncDeps>;
  } = {},
): Promise<RemoteDatasetSyncResult> {
  const deps = { ...defaultDeps, ...(options.deps || {}) };
  const datasetsRoot = await deps.getDatasetsRoot();
  const { groups, warnings } = await collectLocalDatasetGroups(rawJobConfig, datasetsRoot, deps);
  const sameWorkerRemoteRefs = collectSameWorkerRemoteDatasetReferences(rawJobConfig, worker.id);

  if (groups.length === 0 && sameWorkerRemoteRefs.length === 0) {
    return { jobConfig: rawJobConfig, mappings: [], warnings };
  }

  options.onProgress?.({
    status: 'checking-datasets',
    message: 'Checking remote datasets',
    percent: 5,
    datasetName: null,
  });

  const remoteDatasets = await deps.listRemoteDatasets(worker);
  const remoteDatasetByName = new Map((Array.isArray(remoteDatasets) ? remoteDatasets : []).map(dataset => [dataset.name, dataset]));
  let jobConfig = cloneJson(rawJobConfig);

  if (sameWorkerRemoteRefs.length > 0) {
    jobConfig = rewriteSameWorkerRemoteDatasetRefs(jobConfig, {
      workerID: worker.id,
      workerName: worker.name,
      datasets: Array.isArray(remoteDatasets) ? remoteDatasets : [],
    });
  }

  const missingGroups = groups.filter(group => !remoteDatasetByName.has(group.datasetName));
  const exportRoot = path.join(datasetsRoot, '.aitk-remote-dataset-sync');
  const mappings: RemoteDatasetSyncMapping[] = [];

  for (const group of groups) {
    let remoteDataset = remoteDatasetByName.get(group.datasetName);
    let uploaded = false;

    if (!remoteDataset) {
      const uploadIndex = missingGroups.findIndex(missing => missing.localDatasetPath === group.localDatasetPath);
      const safeName = safeNameSegment(group.datasetName, 'dataset');
      const zipPath = path.join(exportRoot, datasetExportFileName(safeName));

      options.onProgress?.({
        status: 'zipping-dataset',
        message: `Preparing dataset ${group.datasetName}`,
        percent: 10,
        datasetName: group.datasetName,
        bytesProcessed: 0,
        bytesTotal: 0,
      });

      try {
        await deps.createDatasetArchive(group.datasetName, group.localDatasetPath, zipPath);
        const zipStat = await deps.stat(zipPath);
        options.onProgress?.({
          status: 'uploading-dataset',
          message: `Uploading dataset ${group.datasetName}`,
          percent: progressForUpload(Math.max(uploadIndex, 0), missingGroups.length, 0, zipStat.size),
          datasetName: group.datasetName,
          bytesProcessed: 0,
          bytesTotal: zipStat.size,
        });

        const imported = await deps.uploadDatasetArchive(worker, zipPath, group.datasetName, progress => {
          options.onProgress?.({
            status: 'uploading-dataset',
            message: `Uploading dataset ${group.datasetName}`,
            percent: progressForUpload(
              Math.max(uploadIndex, 0),
              missingGroups.length,
              progress.loaded,
              progress.total,
            ),
            datasetName: group.datasetName,
            bytesProcessed: progress.loaded,
            bytesTotal: progress.total,
          });
        });

        uploaded = true;
        remoteDataset = {
          name: imported.dataset?.name || group.datasetName,
          encrypted: imported.dataset?.encrypted === true,
          path: imported.path || imported.dataset?.path,
        };
        remoteDatasetByName.set(group.datasetName, remoteDataset);
        if (remoteDataset.name !== group.datasetName || imported.renamed) {
          warnings.push(`Remote worker renamed uploaded dataset "${group.datasetName}" to "${remoteDataset.name}".`);
        }
      } finally {
        await deps.rmPath(zipPath, { force: true }).catch(() => undefined);
      }
    }

    const remoteDatasetPath = remoteDataset?.path;
    if (!remoteDatasetPath) {
      throw new Error(`Remote dataset "${group.datasetName}" on worker "${worker.name}" did not report a local path.`);
    }

    applyRemoteDatasetPath(jobConfig, group, remoteDatasetPath);
    mappings.push({
      datasetName: group.datasetName,
      localDatasetPath: group.localDatasetPath,
      remoteDatasetName: remoteDataset.name || group.datasetName,
      remoteDatasetPath,
      uploaded,
    });
  }

  return { jobConfig, mappings, warnings };
}
