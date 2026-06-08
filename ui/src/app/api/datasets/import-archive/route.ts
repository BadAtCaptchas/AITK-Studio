import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { NextRequest, NextResponse } from 'next/server';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { getDatasetsRoot } from '@/server/settings';
import {
  extractZipSafely,
  getExtractedDatasetPath,
  readDatasetExportManifest,
} from '@/server/datasetTransfer';
import { isEncryptedDatasetFolder, listDatasetSummaries } from '@/server/encryptedDatasets';
import { nextAvailablePath, safeNameSegment } from '@/server/trainingJobTransfer';
import type { DatasetSummary } from '@/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

async function copyArchivePath(sourcePath: string, targetPath: string) {
  await fsp.mkdir(path.dirname(targetPath), { recursive: true });
  await fsp.cp(sourcePath, targetPath, { recursive: true, force: false, errorOnExist: true });
}

function isMultipartRequest(request: NextRequest) {
  return (request.headers.get('content-type') || '').toLowerCase().includes('multipart/form-data');
}

async function saveDatasetArchiveUpload(request: NextRequest, uploadPath: string) {
  const url = new URL(request.url);
  let preferredNameRaw: FormDataEntryValue | string | null =
    url.searchParams.get('preferredName') || request.headers.get('x-aitk-preferred-name');

  await fsp.mkdir(path.dirname(uploadPath), { recursive: true });

  if (isMultipartRequest(request)) {
    const formData = await request.formData();
    const file = formData.get('file');
    if (!(file instanceof File)) {
      throw new Error('file is required');
    }
    preferredNameRaw = formData.get('preferredName') || preferredNameRaw;
    await fsp.writeFile(uploadPath, Buffer.from(await file.arrayBuffer()));
    return preferredNameRaw;
  }

  if (!request.body) {
    throw new Error('file is required');
  }

  await pipeline(
    Readable.fromWeb(request.body as any),
    fs.createWriteStream(uploadPath),
  );
  return preferredNameRaw;
}

export async function POST(request: NextRequest) {
  const datasetsRoot = await getDatasetsRoot();
  await fsp.mkdir(datasetsRoot, { recursive: true });

  const importID = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const workRoot = path.join(datasetsRoot, `.aitk-dataset-import-archive-${importID}`);
  const uploadPath = path.join(workRoot, 'dataset.zip');
  const extractRoot = path.join(workRoot, 'extract');

  try {
    const preferredNameRaw = await saveDatasetArchiveUpload(request, uploadPath);
    await extractZipSafely(uploadPath, extractRoot);

    const manifest = await readDatasetExportManifest(extractRoot);
    const datasetSource = getExtractedDatasetPath(extractRoot, manifest.dataset.archivePath);
    if (!fs.existsSync(datasetSource) || !fs.statSync(datasetSource).isDirectory()) {
      return NextResponse.json({ error: 'Dataset payload missing from archive' }, { status: 400 });
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
    const dataset: DatasetSummary =
      imported || {
        name: importedName,
        encrypted: isEncryptedDatasetFolder(targetPath),
        source: 'local',
        worker_id: 'local',
        worker_name: 'Local',
        ref: `aitk-dataset://local/${encodeURIComponent(importedName)}`,
        path: targetPath,
      };

    return NextResponse.json({
      dataset,
      path: targetPath,
      manifest,
      renamed: importedName !== preferredName,
    });
  } catch (error) {
    console.error('Dataset archive import failed:', error);
    const status = error instanceof Error && error.message === 'file is required' ? 400 : 500;
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to import dataset archive' },
      { status },
    );
  } finally {
    await fsp.rm(workRoot, { recursive: true, force: true }).catch(() => undefined);
  }
}
