import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import archiver from 'archiver';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { getTrainingFolder } from '@/server/settings';
import {
  TRAINING_JOB_EXPORT_FORMAT,
  TRAINING_JOB_EXPORT_VERSION,
  collectDatasetArchiveMappings,
  collectModelReferences,
  findLatestCheckpoint,
  listFilesRecursive,
  makeExportFileName,
  resolveConfigPath,
  shouldIncludeTrainingExportPath,
  type TrainingJobExportManifest,
} from '@/server/trainingJobTransfer';
import {
  createTrainingJobExportProgress,
  updateTrainingJobExportProgress,
  type TrainingJobExportProgressSnapshot,
} from '@/server/trainingJobExportProgress';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const prisma = new PrismaClient();

type ExportBody = {
  includeDatasets?: boolean;
  background?: boolean;
};

type ExportProgressUpdate = Partial<
  Pick<
    TrainingJobExportProgressSnapshot,
    | 'status'
    | 'message'
    | 'percent'
    | 'entriesProcessed'
    | 'entriesTotal'
    | 'bytesProcessed'
    | 'bytesTotal'
    | 'zipPath'
    | 'fileName'
    | 'warnings'
    | 'error'
  >
>;

type ArchiveFileEntry = {
  sourcePath: string;
  archivePath: string;
  size: number;
};

type ArchiveJsonEntry = {
  archivePath: string;
  content: string;
  size: number;
};

function toArchivePath(...segments: string[]) {
  return path.posix.join(...segments.map(segment => segment.replace(/\\/g, '/')));
}

async function collectArchiveFilesForPath(
  sourcePath: string,
  archivePath: string,
  isDirectory: boolean,
  filter?: (absolutePath: string, relativePath: string) => boolean,
) {
  if (!isDirectory) {
    const stat = await fsp.stat(sourcePath);
    return [{ sourcePath, archivePath: archivePath.replace(/\\/g, '/'), size: stat.size }];
  }

  const files = await listFilesRecursive(sourcePath, filter || (() => true));
  return Promise.all(
    files.map(async relativePath => {
      const absolutePath = path.join(sourcePath, relativePath);
      const stat = await fsp.stat(absolutePath);
      return {
        sourcePath: absolutePath,
        archivePath: toArchivePath(archivePath, relativePath),
        size: stat.size,
      };
    }),
  );
}

function jsonArchiveEntry(archivePath: string, value: unknown): ArchiveJsonEntry {
  const content = JSON.stringify(value, null, 2);
  return {
    archivePath,
    content,
    size: Buffer.byteLength(content),
  };
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${units[unitIndex]}`;
}

async function performTrainingJobExport(
  jobID: string,
  includeDatasets: boolean,
  onProgress?: (progress: ExportProgressUpdate) => void,
) {
  onProgress?.({ status: 'preparing', message: 'Preparing export', percent: 2 });

  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) {
    const error = new Error('Job not found');
    (error as any).status = 404;
    throw error;
  }
  if (job.job_type !== 'train') {
    const error = new Error('Only training jobs can be exported');
    (error as any).status = 400;
    throw error;
  }

  const warnings: string[] = [];
  const jobConfig = JSON.parse(job.job_config);
  const trainingRoot = await getTrainingFolder();
  const jobFolder = path.join(trainingRoot, job.name);
  if (!fs.existsSync(jobFolder)) {
    const error = new Error('Training folder not found');
    (error as any).status = 404;
    throw error;
  }

  const latestCheckpoint = await findLatestCheckpoint(jobFolder, job.step);
  const optimizerIncluded = fs.existsSync(path.join(jobFolder, 'optimizer.pt'));
  if (job.status === 'running') {
    warnings.push('Job is running; export includes only the latest checkpoint and optimizer already saved to disk.');
    if (latestCheckpoint.step !== null && job.step > latestCheckpoint.step) {
      warnings.push(`Current DB step is ${job.step}, but the latest detected checkpoint step is ${latestCheckpoint.step}.`);
    }
  }
  if (!latestCheckpoint.relativePath) {
    warnings.push('No checkpoint file was found in the training folder.');
  }
  if (!optimizerIncluded) {
    warnings.push('No optimizer.pt file was found; the imported job may resume without optimizer state.');
  }

  const { mappings: datasetMappings, warnings: datasetWarnings } = await collectDatasetArchiveMappings(
    jobConfig,
    includeDatasets,
  );
  warnings.push(...datasetWarnings);
  onProgress?.({ status: 'preparing', message: 'Scanning files', percent: 5, warnings });

  const modelReferences = collectModelReferences(jobConfig);
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
    },
    training: {
      archivePath: 'training',
      dbStep: job.step,
      latestCheckpointPath: latestCheckpoint.relativePath
        ? path.posix.join('training', latestCheckpoint.relativePath.replace(/\\/g, '/'))
        : null,
      latestCheckpointStep: latestCheckpoint.step,
      optimizerIncluded,
      status: job.status,
    },
    datasets: {
      included: includeDatasets,
      mappings: datasetMappings,
    },
    models: {
      references: modelReferences,
    },
    warnings,
  };

  const trainingFiles = await collectArchiveFilesForPath(
    jobFolder,
    'training',
    true,
    shouldIncludeTrainingExportPath,
  );
  const datasetFiles = (
    await Promise.all(
      datasetMappings.map(mapping =>
        collectArchiveFilesForPath(resolveConfigPath(mapping.originalPath), mapping.archivePath, mapping.isDirectory),
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

  const jsonEntries = [
    jsonArchiveEntry('manifest.json', manifest),
    jsonArchiveEntry('job.json', jobJson),
    jsonArchiveEntry('job_config.json', jobConfig),
  ];
  const fileEntries: ArchiveFileEntry[] = [...trainingFiles, ...datasetFiles];
  const totalEntries = jsonEntries.length + fileEntries.length;
  const totalBytes =
    jsonEntries.reduce((total, entry) => total + entry.size, 0) +
    fileEntries.reduce((total, entry) => total + entry.size, 0);

  const fileName = makeExportFileName(job.name);
  const outputPath = path.join(jobFolder, fileName);
  const tempPath = path.join(jobFolder, `.aitk-export-${Date.now()}.tmp`);
  if (fs.existsSync(outputPath)) {
    await fsp.unlink(outputPath);
  }

  onProgress?.({
    status: 'zipping',
    message: `Zipping 0 / ${totalEntries} files`,
    percent: totalEntries === 0 ? 95 : 8,
    entriesProcessed: 0,
    entriesTotal: totalEntries,
    bytesProcessed: 0,
    bytesTotal: totalBytes,
    warnings,
  });

  await new Promise<void>((resolve, reject) => {
    const output = fs.createWriteStream(tempPath);
    const archive = archiver('zip', { zlib: { level: 9 } });
    let lastProgressUpdate = 0;

    output.on('close', resolve);
    output.on('error', reject);
    archive.on('error', reject);
    archive.on('progress', progress => {
      const now = Date.now();
      if (now - lastProgressUpdate < 250 && progress.entries.processed < totalEntries) return;
      lastProgressUpdate = now;

      const entriesProcessed = Math.min(progress.entries.processed, totalEntries);
      const bytesProcessed = Math.min(progress.fs.processedBytes, totalBytes);
      const ratio =
        totalBytes > 0
          ? bytesProcessed / totalBytes
          : totalEntries > 0
            ? entriesProcessed / totalEntries
            : 1;
      const percent = Math.min(95, Math.max(8, Math.round(8 + ratio * 87)));

      onProgress?.({
        status: 'zipping',
        message: `Zipping ${entriesProcessed} / ${totalEntries} files (${formatBytes(bytesProcessed)} / ${formatBytes(totalBytes)})`,
        percent,
        entriesProcessed,
        entriesTotal: totalEntries,
        bytesProcessed,
        bytesTotal: totalBytes,
      });
    });
    archive.pipe(output);

    for (const entry of jsonEntries) {
      archive.append(entry.content, { name: entry.archivePath });
    }
    for (const entry of fileEntries) {
      archive.file(entry.sourcePath, { name: entry.archivePath });
    }

    archive.finalize().catch(reject);
  });

  onProgress?.({
    status: 'finalizing',
    message: 'Finalizing archive',
    percent: 98,
    entriesProcessed: totalEntries,
    entriesTotal: totalEntries,
    bytesProcessed: totalBytes,
    bytesTotal: totalBytes,
  });

  await fsp.rename(tempPath, outputPath);

  return {
    zipPath: outputPath,
    fileName,
    warnings,
  };
}

async function runBackgroundExport(exportID: string, jobID: string, includeDatasets: boolean) {
  try {
    const result = await performTrainingJobExport(jobID, includeDatasets, progress =>
      updateTrainingJobExportProgress(exportID, progress),
    );
    updateTrainingJobExportProgress(exportID, {
      status: 'completed',
      message: 'Export ready',
      percent: 100,
      zipPath: result.zipPath,
      fileName: result.fileName,
      warnings: result.warnings,
    });
  } catch (error) {
    console.error('Background training job export failed:', error);
    updateTrainingJobExportProgress(exportID, {
      status: 'failed',
      message: 'Export failed',
      error: error instanceof Error ? error.message : 'Failed to export training job',
    });
  }
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  try {
    const body = ((await request.json().catch(() => ({}))) || {}) as ExportBody;
    const includeDatasets = body.includeDatasets === true;
    const background = body.background === true;

    if (background) {
      const progress = createTrainingJobExportProgress(jobID, includeDatasets);
      runBackgroundExport(progress.exportID, jobID, includeDatasets);
      return NextResponse.json(
        {
          exportID: progress.exportID,
          statusUrl: `/api/jobs/${jobID}/export/${progress.exportID}`,
          progress,
        },
        { status: 202 },
      );
    }

    const result = await performTrainingJobExport(jobID, includeDatasets);
    return NextResponse.json(result);
  } catch (error) {
    console.error('Training job export failed:', error);
    const status = typeof (error as any)?.status === 'number' ? (error as any).status : 500;
    const message = error instanceof Error ? error.message : 'Failed to export training job';
    return NextResponse.json({ error: message }, { status });
  }
}
