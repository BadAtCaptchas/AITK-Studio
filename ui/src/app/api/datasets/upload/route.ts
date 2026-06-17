// src/app/api/datasets/upload/route.ts
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import { writeFile, mkdir } from 'fs/promises';
import { basename, dirname, extname, isAbsolute, join, resolve, relative, sep } from 'path';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import {
  isEncryptedDatasetFolder,
  resolveEncryptedObjectPath,
  validateEncryptedManifest,
  writeEncryptedManifest,
} from '@/server/encryptedDatasets';
import { rejectRemoteProjectScope, resolveDatasetScope } from '@/server/datasetScope';

function cleanPathSegment(segment: string, fallback: string) {
  const cleaned = segment
    .trim()
    .replace(/[<>:"/\\|?*\x00-\x1f]/g, '_')
    .replace(/\s+/g, '_')
    .replace(/^\.+/, '')
    .slice(0, 120);
  return cleaned || fallback;
}

function cleanUploadFileName(fileName: string) {
  const base = basename(fileName || 'file');
  const ext = extname(base);
  const stem = ext ? base.slice(0, -ext.length) : base;
  return `${cleanPathSegment(stem, 'file')}${ext.replace(/[^a-zA-Z0-9._-]/g, '_')}`;
}

function cleanRelativeUploadPath(relativePath: string, fallbackName: string) {
  const normalized = relativePath.replace(/\\/g, '/');
  const rawParts = normalized.split('/').filter(Boolean);
  const fallbackFileName = cleanUploadFileName(fallbackName);

  if (rawParts.length === 0 || normalized.startsWith('/') || rawParts.some(part => part === '..')) {
    return fallbackFileName;
  }

  const parts = rawParts.map((part, index) =>
    index === rawParts.length - 1 ? cleanUploadFileName(part) : cleanPathSegment(part, `folder_${index + 1}`),
  );
  return join(...parts);
}

function nextAvailableFilePath(uploadDir: string, relativeFilePath: string) {
  const targetDir = resolve(uploadDir, dirname(relativeFilePath));
  const fileName = basename(relativeFilePath);
  const ext = extname(fileName);
  const stem = ext ? fileName.slice(0, -ext.length) : fileName;
  let candidate = resolve(targetDir, fileName);
  let suffix = 2;
  while (fs.existsSync(candidate)) {
    candidate = resolve(targetDir, `${stem}_${suffix}${ext}`);
    suffix += 1;
  }
  return candidate;
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const workerID = (formData.get('worker_id') as string) || 'local';
    rejectRemoteProjectScope(workerID, formData.get('project_id'));
    const { datasetsRoot } = await resolveDatasetScope(formData.get('project_id'));
    if (!datasetsRoot) {
      return NextResponse.json({ error: 'Datasets path not found' }, { status: 500 });
    }
    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      const remoteFormData = new FormData();
      formData.forEach((value, key) => {
        if (key !== 'worker_id') remoteFormData.append(key, value);
      });
      return NextResponse.json(
        await remoteJson(worker, '/api/datasets/upload', {
          method: 'POST',
          body: remoteFormData,
        }),
      );
    }

    const files = formData.getAll('files');
    const datasetName = (formData.get('datasetName') as string)?.trim();
    const encrypted = formData.get('encrypted') === '1';
    const preserveRelativePaths = formData.get('preserveRelativePaths') === '1';
    const failIfDatasetExists = formData.get('failIfDatasetExists') === '1';
    const relativePathsText = formData.get('relativePaths');
    let relativePaths: string[] = [];
    if (typeof relativePathsText === 'string' && relativePathsText.trim()) {
      const parsed = JSON.parse(relativePathsText);
      if (!Array.isArray(parsed)) {
        return NextResponse.json({ error: 'relativePaths must be an array' }, { status: 400 });
      }
      relativePaths = parsed.map(value => (typeof value === 'string' ? value : ''));
    }

    if (!files || files.length === 0) {
      return NextResponse.json({ error: 'No files provided' }, { status: 400 });
    }

    if (!datasetName) {
      return NextResponse.json({ error: 'Dataset name is required' }, { status: 400 });
    }

    // Create upload directory if it doesn't exist
    const resolvedDatasetsRoot = resolve(datasetsRoot);
    const uploadDir = resolve(resolvedDatasetsRoot, datasetName);
    const uploadDirRelative = relative(resolvedDatasetsRoot, uploadDir);

    if (
      uploadDirRelative === '' ||
      uploadDirRelative === '.' ||
      uploadDirRelative.startsWith('..') ||
      uploadDirRelative.includes(`..${sep}`) ||
      isAbsolute(uploadDirRelative)
    ) {
      return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
    }

    if (
      failIfDatasetExists &&
      fs.existsSync(uploadDir) &&
      fs.statSync(uploadDir).isDirectory() &&
      fs.readdirSync(uploadDir).length > 0
    ) {
      return NextResponse.json({ error: 'Dataset already exists' }, { status: 409 });
    }

    await mkdir(uploadDir, { recursive: true });

    if (encrypted) {
      if (!isEncryptedDatasetFolder(uploadDir)) {
        return NextResponse.json({ error: 'Encrypted dataset not found' }, { status: 404 });
      }
      const manifestText = formData.get('manifest');
      const objectPathsText = formData.get('objectPaths');
      if (typeof manifestText !== 'string' || typeof objectPathsText !== 'string') {
        return NextResponse.json({ error: 'Encrypted upload requires a manifest and object paths' }, { status: 400 });
      }
      const manifest = validateEncryptedManifest(JSON.parse(manifestText));
      const objectPaths = JSON.parse(objectPathsText);
      if (!Array.isArray(objectPaths) || objectPaths.length !== files.length) {
        return NextResponse.json({ error: 'Object path count does not match uploaded files' }, { status: 400 });
      }

      await mkdir(join(uploadDir, 'objects'), { recursive: true });
      const savedObjects: string[] = [];
      for (let i = 0; i < files.length; i++) {
        const file = files[i] as any;
        const objectPath = objectPaths[i];
        if (typeof objectPath !== 'string') {
          return NextResponse.json({ error: 'Invalid encrypted object path' }, { status: 400 });
        }
        const resolvedObjectPath = resolveEncryptedObjectPath(uploadDir, objectPath);
        const bytes = await file.arrayBuffer();
        await writeFile(resolvedObjectPath, Buffer.from(bytes));
        savedObjects.push(objectPath);
      }
      await writeEncryptedManifest(uploadDir, manifest);
      return NextResponse.json({
        message: 'Encrypted files uploaded successfully',
        objects: savedObjects,
      });
    }

    if (isEncryptedDatasetFolder(uploadDir)) {
      return NextResponse.json({ error: 'Plain uploads are not allowed for encrypted datasets' }, { status: 400 });
    }

    const savedFiles: string[] = [];
    
    // Process files sequentially to avoid overwhelming the system
    for (let i = 0; i < files.length; i++) {
      const file = files[i] as any;
      const bytes = await file.arrayBuffer();
      const buffer = Buffer.from(bytes);

      const requestedPath =
        preserveRelativePaths && relativePaths[i]
          ? cleanRelativeUploadPath(relativePaths[i], file.name)
          : cleanUploadFileName(file.name);
      const filePath = nextAvailableFilePath(uploadDir, requestedPath);
      const filePathRelative = relative(uploadDir, filePath);
      if (
        filePathRelative === '' ||
        filePathRelative.startsWith('..') ||
        filePathRelative.includes(`..${sep}`) ||
        isAbsolute(filePathRelative)
      ) {
        return NextResponse.json({ error: 'Invalid upload path' }, { status: 400 });
      }

      await mkdir(dirname(filePath), { recursive: true });
      await writeFile(filePath, buffer);
      savedFiles.push(filePathRelative);
    }

    return NextResponse.json({
      message: 'Files uploaded successfully',
      files: savedFiles,
    });
  } catch (error: any) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: error?.message || 'Error uploading files' },
      { status: typeof error?.status === 'number' ? error.status : 500 },
    );
  }
}

// Increase payload size limit (default is 4mb)
export const config = {
  api: {
    bodyParser: false,
    responseLimit: '50mb',
  },
};
