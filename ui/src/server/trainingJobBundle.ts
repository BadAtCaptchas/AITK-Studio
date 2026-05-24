import archiver from 'archiver';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';
import { db } from '@/server/db';
import {
  TRAINING_JOB_EXPORT_FORMAT,
  TRAINING_JOB_EXPORT_VERSION,
  collectDatasetArchiveMappings,
  collectModelReferences,
  findLatestCheckpoint,
  isCheckpointExportPath,
  listFilesRecursive,
  makeExportFileName,
  resolveConfigPath,
  shouldIncludeDatasetExportPath,
  shouldIncludeTrainingExportPath,
  type TrainingJobExportManifest,
} from '@/server/trainingJobTransfer';

type ArchiveFileEntry = {
  sourcePath: string;
  archivePath: string;
};

function toArchivePath(...segments: string[]) {
  return path.posix.join(...segments.map(segment => segment.replace(/\\/g, '/')));
}

async function collectFilesForArchive(
  sourcePath: string,
  archivePath: string,
  isDirectory: boolean,
  filter?: (absolutePath: string, relativePath: string) => boolean,
) {
  if (!fs.existsSync(sourcePath)) return [];
  if (!isDirectory) return [{ sourcePath, archivePath: archivePath.replace(/\\/g, '/') }];

  const files = await listFilesRecursive(sourcePath, filter || (() => true));
  return files.map(relativePath => ({
    sourcePath: path.join(sourcePath, relativePath),
    archivePath: toArchivePath(archivePath, relativePath),
  }));
}

async function writeZip(outputPath: string, entries: ArchiveFileEntry[], jsonEntries: Array<{ archivePath: string; value: unknown }>) {
  await fsp.mkdir(path.dirname(outputPath), { recursive: true });
  await new Promise<void>((resolve, reject) => {
    const output = fs.createWriteStream(outputPath);
    const archive = archiver('zip', { zlib: { level: 9 } });

    output.on('close', resolve);
    output.on('error', reject);
    archive.on('error', reject);
    archive.pipe(output);

    for (const entry of jsonEntries) {
      archive.append(JSON.stringify(entry.value, null, 2), { name: entry.archivePath });
    }

    if (!entries.some(entry => entry.archivePath === 'training/.empty')) {
      archive.append('', { name: 'training/.empty' });
    }

    for (const entry of entries) {
      archive.file(entry.sourcePath, { name: entry.archivePath });
    }

    archive.finalize().catch(reject);
  });
}

export async function createRemoteTrainingJobBundle(
  jobID: string,
  options: { includeDatasets?: boolean; checkpointMode?: 'latest' | 'all' } = {},
) {
  const includeDatasets = options.includeDatasets !== false;
  const checkpointMode = options.checkpointMode || 'all';
  const job = await db.jobs.findById(jobID);
  if (!job) {
    const error = new Error('Job not found');
    (error as any).status = 404;
    throw error;
  }
  if (job.job_type !== 'train') {
    const error = new Error('Only training jobs can be sent to remote workers');
    (error as any).status = 400;
    throw error;
  }

  const jobConfig = JSON.parse(job.job_config);
  const trainingRoot = await getTrainingFolder();
  const datasetsRoot = await getDatasetsRoot();
  const jobFolder = path.join(trainingRoot, job.name);
  const warnings: string[] = [];

  const latestCheckpoint = await findLatestCheckpoint(jobFolder, job.step);
  const latestCheckpointRelativePath = latestCheckpoint.relativePath?.replace(/\\/g, '/') ?? null;
  if (!fs.existsSync(jobFolder)) {
    warnings.push('Training folder does not exist yet; remote bundle will start from an empty training state.');
  }
  if (fs.existsSync(jobFolder) && !latestCheckpoint.relativePath) {
    warnings.push('No checkpoint file was found in the training folder.');
  }

  const { mappings: datasetMappings, warnings: datasetWarnings } = await collectDatasetArchiveMappings(
    jobConfig,
    includeDatasets,
    datasetsRoot,
  );
  warnings.push(...datasetWarnings);

  const manifest: TrainingJobExportManifest = {
    format: TRAINING_JOB_EXPORT_FORMAT,
    version: TRAINING_JOB_EXPORT_VERSION,
    exportedAt: new Date().toISOString(),
    source: {
      app: 'ai-toolkit',
      jobId: job.id,
      jobName: job.name,
    },
    options: {
      includeDatasets,
      includeBaseModels: false,
      checkpointMode,
    },
    training: {
      archivePath: 'training',
      dbStep: job.step,
      latestCheckpointPath: latestCheckpointRelativePath ? path.posix.join('training', latestCheckpointRelativePath) : null,
      latestCheckpointStep: latestCheckpoint.step,
      optimizerIncluded: fs.existsSync(path.join(jobFolder, 'optimizer.pt')),
      status: job.status,
    },
    datasets: {
      included: includeDatasets,
      mappings: datasetMappings,
    },
    models: {
      references: collectModelReferences(jobConfig),
    },
    warnings,
  };

  const trainingFiles = await collectFilesForArchive(
    jobFolder,
    'training',
    true,
    (_absolutePath, relativePath) => {
      if (!shouldIncludeTrainingExportPath(_absolutePath, relativePath)) return false;
      if (checkpointMode === 'all' || !isCheckpointExportPath(relativePath)) return true;
      return relativePath.replace(/\\/g, '/') === latestCheckpointRelativePath;
    },
  );

  const datasetFiles = (
    await Promise.all(
      datasetMappings.map(mapping =>
        collectFilesForArchive(
          resolveConfigPath(mapping.originalPath),
          mapping.archivePath,
          mapping.isDirectory,
          shouldIncludeDatasetExportPath,
        ),
      ),
    )
  ).flat();

  const jobJson = {
    id: job.id,
    name: job.name,
    gpu_ids: job.gpu_ids,
    created_at: job.created_at,
    updated_at: job.updated_at,
    status: job.status,
    stop: job.stop,
    return_to_queue: job.return_to_queue,
    step: job.step,
    info: job.info,
    speed_string: job.speed_string,
    queue_position: job.queue_position,
    job_type: job.job_type,
    job_ref: job.job_ref,
  };

  const bundleRoot = path.join(trainingRoot, '.aitk-remote-bundles');
  const zipPath = path.join(bundleRoot, makeExportFileName(job.name));
  await writeZip(
    zipPath,
    [...trainingFiles, ...datasetFiles],
    [
      { archivePath: 'manifest.json', value: manifest },
      { archivePath: 'job.json', value: jobJson },
      { archivePath: 'job_config.json', value: jobConfig },
    ],
  );

  return { zipPath, fileName: path.basename(zipPath), warnings };
}
