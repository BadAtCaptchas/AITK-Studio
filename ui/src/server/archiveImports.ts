import { configContractErrors } from '../domain/configContract';
import fs from 'fs/promises';
import path from 'path';
import { createHash } from 'crypto';
import { db, getDatabaseConfig } from './db';
import { getTrainingFolder } from './settings';
import { resolveDatasetScope } from './datasetScope';
import { isRecord } from './commandInput';
import { assertGlobalPayload } from '../utils/obsoleteWorkspaceGuard';
import { deviceIds, validJobName } from '../utils/jobIdentity';
import { extractZipSafely } from './safeArchive';
import {
  getUploadManifest,
  setUploadState,
  ensureUploadDirectory,
  assembleManifestChunks,
} from './archiveUploadManifest';
import {
  beginOperationCommit,
  checkpointOperation,
  getOperation,
  type Operation,
  type OperationInput,
} from './operations';
import { withProcessLease } from './processLease';
import { isPathWithinRoot } from './pathContainment';
import { getExtractedDatasetPath, readDatasetExportManifest } from './datasetTransfer';
import { isEncryptedDatasetFolder } from './encryptedDatasets';
import { listLayeredImages, validateLayeredImageAssets } from './layeredImages';
import {
  safeNameSegment,
  validateArchiveEntryName,
  renameImportedTrainingFiles,
  refreshImportedLatestCheckpoint,
  rewriteJobConfigForTarget,
  TRAINING_JOB_EXPORT_FORMAT,
  TRAINING_JOB_EXPORT_VERSION,
  type TrainingJobExportManifest,
} from './trainingJobTransfer';

export const MAX_IMPORT_BYTES = 64 * 1024 ** 3;
export type ArchiveInput = Extract<OperationInput, { kind: 'dataset-import' | 'job-import' }>;
export async function importRoot(kind: ArchiveInput['kind']): Promise<string> {
  const parent = kind === 'job-import' ? await getTrainingFolder() : (await resolveDatasetScope()).datasetsRoot;
  await fs.mkdir(parent, { recursive: true });
  return path.join(
    await fs.realpath(parent),
    kind === 'job-import' ? '.aitk-job-import-chunks' : '.aitk-dataset-import-archive-chunks',
  );
}
async function readJson(filename: string): Promise<unknown> {
  if ((await fs.stat(filename)).size > 2 * 1024 ** 2) throw new Error('Archive metadata exceeds the JSON limit');
  return assertGlobalPayload(JSON.parse(await fs.readFile(filename, 'utf8')) as unknown);
}
function extracted(root: string, relative: string): string {
  const target = path.resolve(root, ...validateArchiveEntryName(relative).split('/'));
  if (!isPathWithinRoot(root, target) || target === root) throw new Error('Archive payload escapes staging');
  return target;
}
async function reserveName(operation: Operation, root: string, preferred: string, slot: string): Promise<string> {
  const fresh = await getOperation(operation.id);
  const stored = fresh?.checkpoint[slot];
  if (typeof stored === 'string') {
    if (!validJobName(stored)) throw new Error('Invalid reserved publication name');
    return stored;
  }
  return withProcessLease(`publication-root:${createHash('sha256').update(root).digest('hex')}`, async () => {
    const base = safeNameSegment(preferred, 'imported').replace(/[. ]+$/, '');
    for (let index = 0; index < 10_000; index++) {
      const name = index === 0 ? base : `${base}_${index}`;
      if (!validJobName(name)) continue;
      const key = `publication:${createHash('sha256').update(`${root}/${name}`.toLowerCase()).digest('hex')}`;
      if (await fs.stat(path.join(root, name)).catch(() => null)) continue;
      if (slot === 'jobName' && (await db.jobs.findByName(name))) continue;
      const owner = await db.runtime.get(key);
      if (
        owner?.value !== `${operation.id}:${slot}` &&
        !(await db.runtime.compareAndSwap(key, null, `${operation.id}:${slot}`))
      )
        continue;
      await checkpointOperation(operation.id, 'publication-reserved', { [slot]: name });
      return name;
    }
    throw new Error('No publication name is available');
  });
}
async function publishDirectory(source: string, destination: string, operationID: string): Promise<void> {
  const marker = '.aitk-import-operation.json';
  const existing = await fs.stat(destination).catch(() => null);
  if (existing) {
    const identity = await readJson(path.join(destination, marker)).catch(() => null);
    if (!isRecord(identity) || identity.operationID !== operationID)
      throw new Error('Import destination is already occupied');
    return;
  }
  await beginOperationCommit(operationID.split(':')[0], 'publishing');
  await fs.mkdir(source, { recursive: true });
  await fs.writeFile(path.join(source, marker), JSON.stringify({ operationID }));
  // Same filesystem as staging; the durable marker proves publication on replay.
  await fs.rename(source, destination);
}

function trainingManifest(value: unknown): TrainingJobExportManifest {
  if (
    !isRecord(value) ||
    value.format !== TRAINING_JOB_EXPORT_FORMAT ||
    value.version !== TRAINING_JOB_EXPORT_VERSION ||
    !isRecord(value.source) ||
    typeof value.source.jobName !== 'string' ||
    !isRecord(value.training) ||
    value.training.archivePath !== 'training' ||
    !isRecord(value.datasets) ||
    typeof value.datasets.included !== 'boolean' ||
    !Array.isArray(value.datasets.mappings) ||
    !isRecord(value.models) ||
    !Array.isArray(value.models.references) ||
    !Array.isArray(value.warnings) ||
    !value.warnings.every((x: unknown) => typeof x === 'string')
  )
    throw new Error('Unsupported training job export manifest');
  for (const mapping of value.datasets.mappings) {
    if (
      !isRecord(mapping) ||
      typeof mapping.archivePath !== 'string' ||
      typeof mapping.originalPath !== 'string' ||
      !Array.isArray(mapping.targetConfigPaths) ||
      !mapping.targetConfigPaths.every(
        (x: unknown) =>
          typeof x === 'string' && /^config\.process\[\d+\]\.datasets\[\d+\]\.[a-z_]+(?:\[\d+\])?$/.test(x),
      )
    )
      throw new Error('Invalid dataset archive mapping');
  }
  if (value.training.latestCheckpointPath !== null && typeof value.training.latestCheckpointPath !== 'string')
    throw new Error('Invalid checkpoint path');
  return value as TrainingJobExportManifest;
}

export async function executeArchiveImport(operation: Operation): Promise<unknown> {
  const input = operation.input;
  if (input.kind === 'remote-start') throw new Error('Not an archive operation');
  const root = await importRoot(input.kind);
  if (path.resolve(input.root) !== root)
    throw new Error('Configured storage root changed; restore it before retrying this import');
  const work = await ensureUploadDirectory(root, input.uploadID),
    zipPath = path.join(work, 'upload.zip'),
    extractRoot = path.join(work, 'extract');
  if (!isPathWithinRoot(root, work)) throw new Error('Invalid upload workspace');
  const stableJobID = `import-${operation.id}`;
  if (input.kind === 'job-import') {
    const existing = await db.jobs.findById(stableJobID);
    if (existing) return { job: existing, warnings: [] };
  }
  if (input.chunked) {
    const manifest = await getUploadManifest(root, input.uploadID);
    if (!manifest) throw new Error('Upload manifest is unavailable');
    if (manifest.state !== 'importing')
      await assembleManifestChunks(root, input.uploadID, input.chunksTotal, zipPath, {
        maxBytes: MAX_IMPORT_BYTES,
        expectedBytes: input.expectedBytes ?? undefined,
      });
    await setUploadState(root, input.uploadID, 'importing');
  }
  await checkpointOperation(operation.id, 'extracting');
  // Only disposable extraction state is rebuilt; already published user data is never removed.
  await fs.rm(extractRoot, { recursive: true, force: true });
  await extractZipSafely(zipPath, extractRoot);
  const { datasetsRoot } = await resolveDatasetScope();
  let result: unknown;
  if (input.kind === 'dataset-import') {
    const manifest = await readDatasetExportManifest(extractRoot);
    await listLayeredImages(getExtractedDatasetPath(extractRoot, manifest.dataset.archivePath));
    const preferred = input.preferredName || manifest.dataset.name || 'dataset';
    const name = await reserveName(operation, datasetsRoot, preferred, 'datasetName');
    const destination = path.join(datasetsRoot, name);
    await checkpointOperation(operation.id, 'publishing-dataset');
    await publishDirectory(
      getExtractedDatasetPath(extractRoot, manifest.dataset.archivePath),
      destination,
      operation.id,
    );
    result = {
      dataset: {
        name,
        encrypted: isEncryptedDatasetFolder(destination),
        source: 'local',
        worker_id: 'local',
        worker_name: 'Local',
        ref: `aitk-dataset://local/${encodeURIComponent(name)}`,
        path: destination,
      },
      path: destination,
      manifest,
      renamed: name !== preferred,
    };
  } else {
    const manifest = trainingManifest(await readJson(path.join(extractRoot, 'manifest.json')));
    const sourceJob = await readJson(path.join(extractRoot, 'job.json'));
    const sourceConfig = await readJson(path.join(extractRoot, 'job_config.json'));
    if (
      !isRecord(sourceJob) ||
      !isRecord(sourceConfig) ||
      !isRecord(sourceConfig.config) ||
      !Array.isArray(sourceConfig.config.process)
    )
      throw new Error('Invalid imported job configuration');
    const errors = configContractErrors(sourceConfig);
    if (errors.length) throw new Error(errors[0]);
    const sourceName = typeof sourceJob.name === 'string' ? sourceJob.name : manifest.source.jobName;
    const trainingRoot = await getTrainingFolder();
    const name = await reserveName(operation, trainingRoot, sourceName, 'jobName');
    const storageKey = `job-${stableJobID}`,
      destination = path.join(trainingRoot, storageKey);
    const mappings = new Map<string, string>();
    if (manifest.datasets.included)
      for (let index = 0; index < manifest.datasets.mappings.length; index++) {
        const mapping = manifest.datasets.mappings[index];
        const datasetName = await reserveName(
          operation,
          datasetsRoot,
          path.basename(mapping.originalPath || mapping.archivePath),
          `dataset-${index}`,
        );
        const target = path.join(datasetsRoot, datasetName);
        await validateLayeredImageAssets(extracted(extractRoot, mapping.archivePath));
        await publishDirectory(extracted(extractRoot, mapping.archivePath), target, `${operation.id}:dataset-${index}`);
        for (const configPath of mapping.targetConfigPaths) mappings.set(configPath, target);
      }
    const rewritten = rewriteJobConfigForTarget(sourceConfig, {
      jobName: storageKey,
      trainingFolder: trainingRoot,
      sqliteDbPath: getDatabaseConfig().sqlitePath,
      datasetPathByConfigPath: mappings,
    });
    const trainingSource = extracted(extractRoot, manifest.training.archivePath);
    const sourceStorage = typeof sourceJob.storage_key === 'string' ? sourceJob.storage_key : sourceName;
    await fs.mkdir(trainingSource, { recursive: true });
    await renameImportedTrainingFiles(trainingSource, sourceStorage, storageKey);
    await refreshImportedLatestCheckpoint(
      trainingSource,
      manifest.training.latestCheckpointPath,
      sourceStorage,
      storageKey,
    );
    await fs.writeFile(path.join(trainingSource, '.job_config.json'), JSON.stringify(rewritten, null, 2));
    await fs.writeFile(
      path.join(trainingSource, 'import_manifest.json'),
      JSON.stringify({ importedAt: new Date().toISOString(), importedName: name, sourceName, manifest }),
    );
    await checkpointOperation(operation.id, 'publishing-job', { remoteJobID: stableJobID });
    await publishDirectory(trainingSource, destination, operation.id);
    const gpuIds =
      process.platform === 'darwin' ? 'mps' : deviceIds(input.gpuIds || sourceJob.gpu_ids || '0').join(',');
    const importedStep = Number(
      manifest.training.latestCheckpointStep ?? sourceJob.step ?? manifest.training.dbStep ?? 0,
    );
    if (!Number.isSafeInteger(importedStep) || importedStep < 0) throw new Error('Invalid imported training step');
    const job = await db.jobs.create({
      id: stableJobID,
      name,
      storage_key: storageKey,
      gpu_ids: gpuIds,
      job_config: JSON.stringify(rewritten),
      status: 'stopped',
      step: importedStep,
      info: `Imported from ${sourceName}`,
      queue_position: (await db.jobs.maxQueuePosition()) + 1000,
      job_type: 'train',
    });
    result = { job, warnings: manifest.warnings };
  }
  await checkpointOperation(operation.id, 'published', { published: true });
  if (input.chunked) await setUploadState(root, input.uploadID, 'completed');
  // Keep the assembled archive until the operation result is committed by the worker.
  return result;
}
