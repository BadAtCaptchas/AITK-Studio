import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { NextRequest, NextResponse } from 'next/server';
import { getDatasetsRoot } from '@/server/settings';
import { getRemoteWorker, remoteFetch } from '@/server/remoteClient';
import {
  extractZipSafely,
  getExtractedDatasetPath,
  readDatasetExportManifest,
} from '@/server/datasetTransfer';
import { isEncryptedDatasetFolder, listDatasetSummaries } from '@/server/encryptedDatasets';
import { nextAvailablePath } from '@/server/trainingJobTransfer';
import type { DatasetSummary } from '@/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

async function writeResponseBodyToFile(response: Response, targetPath: string) {
  if (!response.body) throw new Error('Remote worker returned an empty dataset archive');
  await pipeline(Readable.fromWeb(response.body as any), fs.createWriteStream(targetPath));
}

export async function POST(request: NextRequest) {
  const datasetsRoot = await getDatasetsRoot();
  await fsp.mkdir(datasetsRoot, { recursive: true });

  const importID = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const workRoot = path.join(datasetsRoot, `.aitk-dataset-import-${importID}`);
  const uploadPath = path.join(workRoot, 'dataset.zip');
  const extractRoot = path.join(workRoot, 'extract');

  try {
    const body = await request.json();
    const workerID = typeof body?.worker_id === 'string' ? body.worker_id : '';
    const datasetName = typeof body?.datasetName === 'string' ? body.datasetName : '';
    if (!workerID || !datasetName) {
      return NextResponse.json({ error: 'worker_id and datasetName are required' }, { status: 400 });
    }

    const worker = await getRemoteWorker(workerID);
    await fsp.mkdir(workRoot, { recursive: true });
    const remoteResponse = await remoteFetch(worker, '/api/datasets/export', {
      method: 'POST',
      body: JSON.stringify({ datasetName }),
      headers: { 'Content-Type': 'application/json' },
    });
    await writeResponseBodyToFile(remoteResponse, uploadPath);
    await extractZipSafely(uploadPath, extractRoot);

    const manifest = await readDatasetExportManifest(extractRoot);
    const datasetSource = getExtractedDatasetPath(extractRoot, manifest.dataset.archivePath);
    if (!fs.existsSync(datasetSource) || !fs.statSync(datasetSource).isDirectory()) {
      return NextResponse.json({ error: 'Dataset payload missing from archive' }, { status: 400 });
    }

    const targetPath = await nextAvailablePath(datasetsRoot, manifest.dataset.name || datasetName);
    await fsp.cp(datasetSource, targetPath, { recursive: true, force: false, errorOnExist: true });

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
      renamed: importedName !== manifest.dataset.name,
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to import remote dataset' },
      { status: 500 },
    );
  } finally {
    await fsp.rm(workRoot, { recursive: true, force: true }).catch(() => undefined);
  }
}
