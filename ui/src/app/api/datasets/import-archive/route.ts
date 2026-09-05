import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { NextRequest, NextResponse } from 'next/server';
import { DatasetScopeError, resolveDatasetScope } from '@/server/datasetScope';
import { extractZipSafely, getExtractedDatasetPath, readDatasetExportManifest } from '@/server/datasetTransfer';
import {
  archiveUploadMode,
  assembleArchiveUploadChunks,
  cleanupOldArchiveUploadChunks,
  createArchiveUploadImportStatus,
  getArchiveUploadImportStatus,
  readArchiveUploadFileBytes,
  readArchiveUploadChunksTotal,
  readArchiveUploadID,
  saveArchiveUploadChunk,
  updateArchiveUploadImportStatus,
} from '@/server/archiveUploadChunks';
import { isEncryptedDatasetFolder, listDatasetSummaries } from '@/server/encryptedDatasets';
import {
  cleanupStagedUpload,
  InvalidUploadError,
  moveStagedUpload,
  streamRequestToStagingFile,
} from '@/server/streamedUpload';
import { nextAvailablePath, safeNameSegment } from '@/server/trainingJobTransfer';
import type { DatasetSummary } from '@/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MAX_DATASET_ARCHIVE_BYTES = 64 * 1024 * 1024 * 1024;

async function copyArchivePath(sourcePath: string, targetPath: string) {
  await fsp.mkdir(path.dirname(targetPath), { recursive: true });
  await fsp.cp(sourcePath, targetPath, { recursive: true, force: false, errorOnExist: true });
}

async function saveDatasetArchiveUpload(request: NextRequest, uploadPath: string) {
  const url = new URL(request.url);
  const preferredNameRaw = url.searchParams.get('preferredName') || request.headers.get('x-aitk-preferred-name');
  const contentType = (request.headers.get('content-type') || '').split(';', 1)[0].trim().toLowerCase();
  if (!['application/octet-stream', 'application/zip', 'application/x-zip-compressed'].includes(contentType)) {
    const error = new InvalidUploadError('Dataset archives must use a streamed binary request');
    error.status = 415;
    throw error;
  }

  let stagingPath: string | null = null;
  try {
    const staged = await streamRequestToStagingFile(request, path.dirname(uploadPath), {
      maxBytes: MAX_DATASET_ARCHIVE_BYTES,
      prefix: 'dataset-archive',
    });
    stagingPath = staged.stagingPath;
    await moveStagedUpload(stagingPath, uploadPath);
    stagingPath = null;
    return preferredNameRaw;
  } finally {
    await cleanupStagedUpload(stagingPath);
  }
}

async function importDatasetArchiveFromZip(
  uploadPath: string,
  extractRoot: string,
  datasetsRoot: string,
  preferredNameRaw: string | null,
) {
  await extractZipSafely(uploadPath, extractRoot);

  const manifest = await readDatasetExportManifest(extractRoot);
  const datasetSource = getExtractedDatasetPath(extractRoot, manifest.dataset.archivePath);
  if (!fs.existsSync(datasetSource) || !fs.statSync(datasetSource).isDirectory()) {
    const error = new Error('Dataset payload missing from archive');
    error.name = 'DatasetArchiveImportError';
    throw error;
  }

  const preferredName =
    typeof preferredNameRaw === 'string' && preferredNameRaw.trim()
      ? safeNameSegment(preferredNameRaw, 'dataset')
      : manifest.dataset.name || 'dataset';
  const targetPath = await nextAvailablePath(datasetsRoot, preferredName);
  await copyArchivePath(datasetSource, targetPath);

  const importedName = path.basename(targetPath);
  const allDatasets = await listDatasetSummaries(datasetsRoot);
  const imported = allDatasets.find(dataset => dataset.name === importedName);
  const dataset: DatasetSummary = imported || {
    name: importedName,
    encrypted: isEncryptedDatasetFolder(targetPath),
    source: 'local',
    worker_id: 'local',
    worker_name: 'Local',
    ref: `aitk-dataset://local/${encodeURIComponent(importedName)}`,
    path: targetPath,
  };

  return {
    dataset,
    path: targetPath,
    manifest,
    renamed: importedName !== preferredName,
  };
}

function isBackgroundImportRequest(request: NextRequest) {
  return request.nextUrl.searchParams.get('background') === '1';
}

export async function GET(request: NextRequest) {
  if (archiveUploadMode(request) !== 'status') {
    return NextResponse.json({ error: 'Not found' }, { status: 404 });
  }

  try {
    const status = getArchiveUploadImportStatus(readArchiveUploadID(request));
    if (!status) {
      return NextResponse.json({ error: 'Archive import status not found' }, { status: 404 });
    }
    return NextResponse.json(status);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to read archive import status' },
      { status: 400 },
    );
  }
}

export async function POST(request: NextRequest) {
  let workRoot: string | null = null;

  try {
    const { datasetsRoot } = await resolveDatasetScope();
    await fsp.mkdir(datasetsRoot, { recursive: true });

    const chunkUploadRoot = path.join(datasetsRoot, '.aitk-dataset-import-archive-chunks');
    const uploadMode = archiveUploadMode(request);
    if (uploadMode === 'chunk') {
      await cleanupOldArchiveUploadChunks(chunkUploadRoot);
      return NextResponse.json(
        await saveArchiveUploadChunk(request, chunkUploadRoot, {
          maxArchiveBytes: MAX_DATASET_ARCHIVE_BYTES,
        }),
      );
    }

    const importID =
      uploadMode === 'complete' ? readArchiveUploadID(request) : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    workRoot =
      uploadMode === 'complete'
        ? path.join(chunkUploadRoot, importID)
        : path.join(datasetsRoot, `.aitk-dataset-import-archive-${importID}`);
    const uploadPath = path.join(workRoot, 'dataset.zip');
    const extractRoot = path.join(workRoot, 'extract');
    const preferredNameRaw =
      uploadMode === 'complete'
        ? request.nextUrl.searchParams.get('preferredName') || request.headers.get('x-aitk-preferred-name')
        : await saveDatasetArchiveUpload(request, uploadPath);
    if (uploadMode === 'complete') {
      const expectedBytes = readArchiveUploadFileBytes(request, MAX_DATASET_ARCHIVE_BYTES);
      if (isBackgroundImportRequest(request)) {
        const backgroundWorkRoot = workRoot;
        const chunksTotal = readArchiveUploadChunksTotal(request);
        workRoot = null;
        createArchiveUploadImportStatus(importID);
        void (async () => {
          try {
            await assembleArchiveUploadChunks(chunkUploadRoot, importID, chunksTotal, uploadPath, {
              maxBytes: MAX_DATASET_ARCHIVE_BYTES,
              expectedBytes: expectedBytes ?? undefined,
            });
            const result = await importDatasetArchiveFromZip(uploadPath, extractRoot, datasetsRoot, preferredNameRaw);
            updateArchiveUploadImportStatus(importID, { status: 'completed', result });
          } catch (error) {
            updateArchiveUploadImportStatus(importID, {
              status: 'failed',
              error: error instanceof Error ? error.message : 'Failed to import dataset archive',
            });
          } finally {
            await fsp.rm(backgroundWorkRoot, { recursive: true, force: true }).catch(() => undefined);
          }
        })();
        return NextResponse.json(getArchiveUploadImportStatus(importID));
      }
      await assembleArchiveUploadChunks(chunkUploadRoot, importID, readArchiveUploadChunksTotal(request), uploadPath, {
        maxBytes: MAX_DATASET_ARCHIVE_BYTES,
        expectedBytes: expectedBytes ?? undefined,
      });
    }

    return NextResponse.json(
      await importDatasetArchiveFromZip(uploadPath, extractRoot, datasetsRoot, preferredNameRaw),
    );
  } catch (error) {
    console.error('Dataset archive import failed:', error);
    const message = error instanceof Error ? error.message : 'Failed to import dataset archive';
    let status =
      error && typeof error === 'object' && 'status' in error && typeof error.status === 'number' ? error.status : 500;
    if (error instanceof DatasetScopeError) {
      status = error.status;
    } else if (
      message === 'file is required' ||
      message.startsWith('Invalid archive upload') ||
      message === 'Dataset payload missing from archive'
    ) {
      status = 400;
    }
    return NextResponse.json(
      {
        error: message,
        ...(error instanceof DatasetScopeError && error.code ? { code: error.code } : {}),
        ...(error instanceof DatasetScopeError && error.details !== undefined ? { details: error.details } : {}),
      },
      { status },
    );
  } finally {
    if (workRoot) {
      await fsp.rm(workRoot, { recursive: true, force: true }).catch(() => undefined);
    }
  }
}
