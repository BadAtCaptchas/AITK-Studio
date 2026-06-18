import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { isEncryptedDatasetFolder, readEncryptedManifest, resolveDatasetFolder } from '@/server/encryptedDatasets';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { makeSignedRemoteDatasetAssetRef } from '@/server/remoteDatasetAssetAccess';
import { DATASET_TEXT_CAPTION_EXTENSIONS } from '@/server/captionFiles';
import { assertProjectScopeEnabled, DatasetScopeError, rejectRemoteProjectScope, resolveDatasetScope } from '@/server/datasetScope';

export async function POST(request: Request) {
  const body = await request.json();
  const { datasetName } = body;
  const workerID = typeof body?.worker_id === 'string' ? body.worker_id : 'local';
  const projectID = body?.project_id;

  try {
    await assertProjectScopeEnabled(projectID);
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: error.status || 400 });
  }

  if (!isLocalWorker(workerID)) {
    try {
      rejectRemoteProjectScope(workerID, projectID);
    } catch (error: any) {
      return NextResponse.json({ error: error.message }, { status: error.status || 400 });
    }
    const worker = await getRemoteWorker(workerID);
    const data: any = await remoteJson(worker, '/api/datasets/listImages', {
      method: 'POST',
      body: JSON.stringify({ datasetName }),
    });
    if (Array.isArray(data?.images)) {
      data.images = data.images.map((image: any) => ({
        ...image,
        img_path:
          typeof image?.img_path === 'string'
            ? makeSignedRemoteDatasetAssetRef(workerID, 'img', image.img_path)
            : image?.img_path,
      }));
    }
    return NextResponse.json(data);
  }

  let datasetFolder: string;
  let datasetsPath: string;
  try {
    const scope = await resolveDatasetScope(projectID);
    datasetsPath = scope.datasetsRoot;
    datasetFolder = resolveDatasetFolder(datasetsPath, datasetName);
  } catch (error: any) {
    if (error instanceof DatasetScopeError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    return NextResponse.json({ error: error?.message || 'Invalid dataset name' }, { status: 400 });
  }

  try {
    // Check if folder exists
    if (!fs.existsSync(datasetFolder)) {
      return NextResponse.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
    }

    if (isEncryptedDatasetFolder(datasetFolder)) {
      return NextResponse.json({
        encrypted: true,
        manifest: await readEncryptedManifest(datasetFolder),
        images: [],
      });
    }

    // Find all editable dataset items recursively
    const imageFiles = findImagesRecursively(datasetFolder);

    // Sort server-side so the client doesn't have to sort large lists
    imageFiles.sort((a, b) => a.localeCompare(b));

    // Format response
    const result = imageFiles.map(imgPath => ({
      img_path: imgPath,
    }));

    return NextResponse.json({ images: result });
  } catch (error) {
    console.error('Error finding images:', error);
    return NextResponse.json({ error: 'Failed to process request' }, { status: 500 });
  }
}

/**
 * Recursively finds all image files in a directory and its subdirectories
 * @param dir Directory to search
 * @returns Array of absolute paths to image files
 */
function findImagesRecursively(dir: string): string[] {
  const imageExtensions = [
    '.png',
    '.jpg',
    '.jpeg',
    '.webp',
    '.jxl',
    '.gif',
    '.bmp',
    '.mp4',
    '.avi',
    '.mov',
    '.mkv',
    '.wmv',
    '.m4v',
    '.flv',
    '.webm',
    '.mp3',
    '.wav',
    '.flac',
    '.ogg',
    '.m4a',
    '.aac',
  ];
  const mediaStems = new Set<string>();
  const candidateTextFiles: string[] = [];
  let results: string[] = [];

  // withFileTypes avoids a separate statSync per entry — a big win on large datasets
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const name = entry.name;
    if (name.startsWith('.')) continue;
    const itemPath = path.join(dir, name);

    if (entry.isDirectory()) {
      if (name === '_controls') continue;
      results = results.concat(findImagesRecursively(itemPath));
    } else if (entry.isFile()) {
      const ext = path.extname(name).toLowerCase();
      if (imageExtensions.includes(ext)) {
        mediaStems.add(itemPath.slice(0, -ext.length).toLowerCase());
        results.push(itemPath);
      } else if (DATASET_TEXT_CAPTION_EXTENSIONS.includes(ext)) {
        candidateTextFiles.push(itemPath);
      }
    }
  }

  for (const textPath of candidateTextFiles) {
    const ext = path.extname(textPath).toLowerCase();
    const stem = textPath.slice(0, -ext.length).toLowerCase();
    if (!mediaStems.has(stem)) {
      results.push(textPath);
    }
  }

  return results;
}
