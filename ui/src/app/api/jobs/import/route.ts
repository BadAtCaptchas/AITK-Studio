import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import yauzl from 'yauzl';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { isMac } from '@/helpers/basic';
import {
  getDatasetsRoot,
  getTrainingFolder,
  isProjectSpacesDisabledError,
  PROJECT_SPACES_DISABLED_MESSAGE,
} from '@/server/settings';
import { TOOLKIT_ROOT } from '@/paths';
import {
  TRAINING_JOB_EXPORT_FORMAT,
  TRAINING_JOB_EXPORT_VERSION,
  cloneJson,
  isPathInside,
  nextAvailablePath,
  refreshImportedLatestCheckpoint,
  renameImportedTrainingFiles,
  resolveConfigPath,
  rewriteJobConfigForTarget,
  safeNameSegment,
  validateArchiveEntryName,
  type TrainingJobExportManifest,
} from '@/server/trainingJobTransfer';
import {
  archiveUploadMode,
  assembleArchiveUploadChunks,
  cleanupOldArchiveUploadChunks,
  readArchiveUploadChunksTotal,
  readArchiveUploadID,
  saveArchiveUploadChunk,
} from '@/server/archiveUploadChunks';
import { db } from '@/server/db';
import { ensureProjectFolders, resolveOptionalProject } from '@/server/projects';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

async function readJsonFile<T>(filePath: string): Promise<T> {
  return JSON.parse(await fsp.readFile(filePath, 'utf8')) as T;
}

async function extractZipSafely(zipPath: string, destination: string) {
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
        if (!isPathInside(destinationRoot, targetPath)) {
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

function getExtractedArchivePath(extractRoot: string, archivePath: string) {
  const normalized = validateArchiveEntryName(archivePath);
  const resolved = path.resolve(extractRoot, ...normalized.split('/'));
  if (!isPathInside(extractRoot, resolved)) {
    throw new Error(`Archive path escapes import folder: ${archivePath}`);
  }
  return resolved;
}

async function getAvailableJobName(sourceName: string, trainingRoot: string, projectID: string | null) {
  const baseName = safeNameSegment(sourceName, 'imported_job');
  const candidates = [baseName, `${baseName}_imported`];
  let suffix = 2;

  while (true) {
    const candidate = candidates.shift() || `${baseName}_imported_${suffix++}`;
    const existingJob = await db.jobs.findByNameInScope(candidate, projectID);
    const candidateFolder = path.join(trainingRoot, candidate);
    if (!existingJob && !fs.existsSync(candidateFolder)) {
      return candidate;
    }
  }
}

async function copyArchivePath(sourcePath: string, targetPath: string) {
  await fsp.mkdir(path.dirname(targetPath), { recursive: true });
  await fsp.cp(sourcePath, targetPath, { recursive: true, force: false, errorOnExist: true });
}

function isMultipartRequest(request: NextRequest) {
  return (request.headers.get('content-type') || '').toLowerCase().includes('multipart/form-data');
}

async function saveJobArchiveUpload(request: NextRequest, uploadPath: string) {
  const url = new URL(request.url);
  let gpuIdsRaw: FormDataEntryValue | string | null =
    url.searchParams.get('gpu_ids') || request.headers.get('x-aitk-gpu-ids');

  await fsp.mkdir(path.dirname(uploadPath), { recursive: true });

  if (isMultipartRequest(request)) {
    const formData = await request.formData();
    const file = formData.get('file');
    if (!(file instanceof File)) {
      throw new Error('file is required');
    }
    gpuIdsRaw = formData.get('gpu_ids') || gpuIdsRaw;
    await fsp.writeFile(uploadPath, Buffer.from(await file.arrayBuffer()));
    return gpuIdsRaw;
  }

  if (!request.body) {
    throw new Error('file is required');
  }

  await pipeline(
    Readable.fromWeb(request.body as any),
    fs.createWriteStream(uploadPath),
  );
  return gpuIdsRaw;
}

export async function POST(request: NextRequest) {
  let workRoot: string | null = null;

  try {
    const project = await resolveOptionalProject(
      request.nextUrl.searchParams.get('project_id') || request.headers.get('x-aitk-project-id'),
    );
    const projectRoots = project ? await ensureProjectFolders(project) : null;
    const trainingRoot = projectRoots?.runs || (await getTrainingFolder());
    const datasetsRoot = projectRoots?.datasets || (await getDatasetsRoot());
    await fsp.mkdir(trainingRoot, { recursive: true });
    await fsp.mkdir(datasetsRoot, { recursive: true });

    const chunkUploadRoot = path.join(trainingRoot, '.aitk-job-import-chunks');
    const uploadMode = archiveUploadMode(request);
    if (uploadMode === 'chunk') {
      await cleanupOldArchiveUploadChunks(chunkUploadRoot);
      return NextResponse.json(await saveArchiveUploadChunk(request, chunkUploadRoot));
    }

    const importId = uploadMode === 'complete' ? readArchiveUploadID(request) : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    workRoot =
      uploadMode === 'complete'
        ? path.join(chunkUploadRoot, importId)
        : path.join(trainingRoot, `.aitk-import-${importId}`);
    const uploadPath = path.join(workRoot, 'upload.zip');
    const extractRoot = path.join(workRoot, 'extract');
    const requestedGpuIds =
      uploadMode === 'complete'
        ? request.nextUrl.searchParams.get('gpu_ids') || request.headers.get('x-aitk-gpu-ids')
        : await saveJobArchiveUpload(request, uploadPath);
    if (uploadMode === 'complete') {
      await assembleArchiveUploadChunks(chunkUploadRoot, importId, readArchiveUploadChunksTotal(request), uploadPath);
    }
    await extractZipSafely(uploadPath, extractRoot);

    const manifest = await readJsonFile<TrainingJobExportManifest>(path.join(extractRoot, 'manifest.json'));
    if (manifest.format !== TRAINING_JOB_EXPORT_FORMAT || manifest.version !== TRAINING_JOB_EXPORT_VERSION) {
      return NextResponse.json({ error: 'Unsupported training job export archive' }, { status: 400 });
    }

    const sourceJob = await readJsonFile<any>(path.join(extractRoot, 'job.json'));
    const sourceJobConfig = await readJsonFile<any>(path.join(extractRoot, 'job_config.json'));
    const sourceName = sourceJob?.name || sourceJobConfig?.config?.name || manifest.source.jobName;
    const importedName = await getAvailableJobName(sourceName, trainingRoot, project?.id || null);
    let warnings = [...(manifest.warnings || [])];
    const modelReferences = Array.isArray(manifest.models?.references) ? manifest.models.references : [];

    if (importedName !== safeNameSegment(sourceName, 'imported_job')) {
      warnings.push(`Imported job was renamed to "${importedName}" because the original name was unavailable.`);
    }

    const trainingSource = getExtractedArchivePath(extractRoot, manifest.training.archivePath);
    const finalTrainingFolder = path.join(trainingRoot, importedName);
    await copyArchivePath(trainingSource, finalTrainingFolder);
    await renameImportedTrainingFiles(finalTrainingFolder, sourceName, importedName);
    await refreshImportedLatestCheckpoint(
      finalTrainingFolder,
      manifest.training.latestCheckpointPath,
      sourceName,
      importedName,
    );

    const datasetPathByConfigPath = new Map<string, string>();
    if (manifest.datasets.included) {
      for (const mapping of manifest.datasets.mappings) {
        const datasetSource = getExtractedArchivePath(extractRoot, mapping.archivePath);
        if (!fs.existsSync(datasetSource)) {
          warnings.push(`Dataset payload missing from archive: ${mapping.archivePath}`);
          continue;
        }

        const preferredName = path.basename(mapping.originalPath || mapping.archivePath);
        const datasetTarget = await nextAvailablePath(datasetsRoot, preferredName);
        await copyArchivePath(datasetSource, datasetTarget);
        for (const configPath of mapping.targetConfigPaths) {
          datasetPathByConfigPath.set(configPath, datasetTarget);
        }
      }
    }

    for (const reference of modelReferences) {
      if (!reference.isLocal) continue;
      const targetPath = resolveConfigPath(reference.value);
      if (!fs.existsSync(targetPath)) {
        warnings.push(`Local model reference is not present on this system: ${reference.value}`);
      }
    }

    let gpuIds = typeof requestedGpuIds === 'string' && requestedGpuIds.trim() ? requestedGpuIds : sourceJob?.gpu_ids || '0';
    if (isMac()) {
      gpuIds = 'mps';
    }

    const rewrittenConfig = rewriteJobConfigForTarget(sourceJobConfig, {
      jobName: importedName,
      trainingFolder: trainingRoot,
      sqliteDbPath: path.join(TOOLKIT_ROOT, 'aitk_db.db'),
      datasetPathByConfigPath,
    });

    await fsp.writeFile(path.join(finalTrainingFolder, '.job_config.json'), JSON.stringify(rewrittenConfig, null, 2));
    await fsp.writeFile(
      path.join(finalTrainingFolder, 'import_manifest.json'),
      JSON.stringify(
        {
          importedAt: new Date().toISOString(),
          importedName,
          sourceName,
          manifest: cloneJson(manifest),
        },
        null,
        2,
      ),
    );

    const queuePosition = (await db.jobs.maxQueuePosition()) + 1000;
    const importedStep = manifest.training.latestCheckpointStep ?? sourceJob?.step ?? manifest.training.dbStep ?? 0;

    const job = await db.jobs.create({
      name: importedName,
      gpu_ids: gpuIds,
      job_config: JSON.stringify(rewrittenConfig),
      status: 'stopped',
      stop: false,
      return_to_queue: false,
      step: Number(importedStep) || 0,
      info: `Imported from ${sourceName}`,
      speed_string: '',
      queue_position: queuePosition,
      job_type: 'train',
      project_id: project?.id || null,
    });

    return NextResponse.json({ job, warnings });
  } catch (error) {
    if (isProjectSpacesDisabledError(error)) {
      return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
    }
    console.error('Training job import failed:', error);
    const message = error instanceof Error ? error.message : 'Failed to import training job';
    const status = message === 'file is required' || message.startsWith('Invalid archive upload') ? 400 : 500;
    return NextResponse.json({ error: message }, { status });
  } finally {
    if (workRoot && fs.existsSync(workRoot)) {
      await fsp.rm(workRoot, { recursive: true, force: true });
    }
  }
}
