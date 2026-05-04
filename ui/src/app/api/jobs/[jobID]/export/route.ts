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
  makeExportFileName,
  resolveConfigPath,
  shouldIncludeTrainingExportPath,
  type TrainingJobExportManifest,
} from '@/server/trainingJobTransfer';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const prisma = new PrismaClient();

type ExportBody = {
  includeDatasets?: boolean;
};

function archiveDirectory(
  archive: archiver.Archiver,
  sourcePath: string,
  archivePath: string,
  filter?: (absolutePath: string, relativePath: string) => boolean,
) {
  archive.directory(sourcePath, archivePath, entry => {
    if (!entry.name) return entry;
    if (!filter) return entry;
    const relativePath = entry.name.replace(new RegExp(`^${archivePath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}/?`), '');
    const absolutePath = path.join(sourcePath, relativePath);
    return filter(absolutePath, relativePath) ? entry : false;
  });
}

function archiveDatasetPath(archive: archiver.Archiver, sourcePath: string, archivePath: string, isDirectory: boolean) {
  if (isDirectory) {
    archive.directory(sourcePath, archivePath);
  } else {
    archive.file(sourcePath, { name: archivePath });
  }
}

export async function POST(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  try {
    const body = ((await request.json().catch(() => ({}))) || {}) as ExportBody;
    const includeDatasets = body.includeDatasets === true;

    const job = await prisma.job.findUnique({ where: { id: jobID } });
    if (!job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }
    if (job.job_type !== 'train') {
      return NextResponse.json({ error: 'Only training jobs can be exported' }, { status: 400 });
    }

    const warnings: string[] = [];
    const jobConfig = JSON.parse(job.job_config);
    const trainingRoot = await getTrainingFolder();
    const jobFolder = path.join(trainingRoot, job.name);
    if (!fs.existsSync(jobFolder)) {
      return NextResponse.json({ error: 'Training folder not found' }, { status: 404 });
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

    const fileName = makeExportFileName(job.name);
    const outputPath = path.join(jobFolder, fileName);
    const tempPath = path.join(jobFolder, `.aitk-export-${Date.now()}.tmp`);
    if (fs.existsSync(outputPath)) {
      await fsp.unlink(outputPath);
    }

    await new Promise<void>((resolve, reject) => {
      const output = fs.createWriteStream(tempPath);
      const archive = archiver('zip', { zlib: { level: 9 } });

      output.on('close', resolve);
      output.on('error', reject);
      archive.on('error', reject);
      archive.pipe(output);

      archive.append(JSON.stringify(manifest, null, 2), { name: 'manifest.json' });
      archive.append(
        JSON.stringify(
          {
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
          },
          null,
          2,
        ),
        { name: 'job.json' },
      );
      archive.append(JSON.stringify(jobConfig, null, 2), { name: 'job_config.json' });

      archiveDirectory(archive, jobFolder, 'training', shouldIncludeTrainingExportPath);
      for (const mapping of datasetMappings) {
        archiveDatasetPath(archive, resolveConfigPath(mapping.originalPath), mapping.archivePath, mapping.isDirectory);
      }

      archive.finalize().catch(reject);
    });

    await fsp.rename(tempPath, outputPath);

    return NextResponse.json({
      zipPath: outputPath,
      fileName,
      warnings,
    });
  } catch (error) {
    console.error('Training job export failed:', error);
    return NextResponse.json({ error: 'Failed to export training job' }, { status: 500 });
  }
}
