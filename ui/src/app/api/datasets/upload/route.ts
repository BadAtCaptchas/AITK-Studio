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
  writeDatasetImportMetadata,
  writeEncryptedManifest,
} from '@/server/encryptedDatasets';
import { assertProjectScopeEnabled, rejectRemoteProjectScope, resolveDatasetScope } from '@/server/datasetScope';
import {
  cleanupStagedUpload,
  decodedUploadHeader,
  InvalidUploadError,
  moveStagedUpload,
  moveStagedUploadNoReplace,
  streamRequestToStagingFile,
} from '@/server/streamedUpload';

export const runtime = 'nodejs';

const MAX_STREAMED_FILE_BYTES = 2 * 1024 * 1024 * 1024;
const MAX_MULTIPART_REQUEST_BYTES = 256 * 1024 * 1024;
const MAX_MULTIPART_FILE_BYTES = 128 * 1024 * 1024;
const MAX_MULTIPART_FILES = 256;

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

async function writeDatasetImportMetadataBestEffort(uploadDir: string, sourceFolderPath: string) {
  try {
    await writeDatasetImportMetadata(uploadDir, sourceFolderPath);
  } catch (error) {
    console.warn('Could not write dataset import metadata:', error);
  }
}

function resolveDatasetUploadDir(datasetsRoot: string, datasetName: string) {
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
    throw new InvalidUploadError('Invalid dataset name');
  }
  return uploadDir;
}

function assertMultipartRequestLength(request: NextRequest) {
  const rawContentLength = request.headers.get('content-length');
  if (!rawContentLength) {
    const error = new Error('Content-Length is required for multipart dataset uploads');
    Object.assign(error, { status: 411 });
    throw error;
  }
  const contentLength = Number(rawContentLength);
  if (!Number.isSafeInteger(contentLength) || contentLength < 0) {
    const error = new Error('Invalid Content-Length header');
    Object.assign(error, { status: 400 });
    throw error;
  }
  if (contentLength > MAX_MULTIPART_REQUEST_BYTES) {
    const error = new Error('Multipart dataset upload is too large; upload files individually');
    Object.assign(error, { status: 413 });
    throw error;
  }
}

function assertMultipartUploadBounds(files: FormDataEntryValue[]) {
  if (files.length > MAX_MULTIPART_FILES) {
    const error = new Error(`Dataset upload may contain at most ${MAX_MULTIPART_FILES} files`);
    Object.assign(error, { status: 413 });
    throw error;
  }
  let totalBytes = 0;
  for (const entry of files) {
    if (!(entry instanceof File)) {
      const error = new Error('Dataset upload contains an invalid file');
      Object.assign(error, { status: 400 });
      throw error;
    }
    if (entry.size > MAX_MULTIPART_FILE_BYTES) {
      const error = new Error('Multipart dataset file is too large; upload it individually');
      Object.assign(error, { status: 413 });
      throw error;
    }
    totalBytes += entry.size;
    if (totalBytes > MAX_MULTIPART_REQUEST_BYTES) {
      const error = new Error('Multipart dataset upload is too large; upload files individually');
      Object.assign(error, { status: 413 });
      throw error;
    }
  }
}

function isDestinationCollision(error: unknown) {
  return error && typeof error === 'object' && 'code' in error && error.code === 'EEXIST';
}

function assertedRelativeUploadPath(uploadDir: string, targetPath: string) {
  const targetRelative = relative(uploadDir, targetPath);
  if (
    targetRelative === '' ||
    targetRelative.startsWith('..') ||
    targetRelative.includes(`..${sep}`) ||
    isAbsolute(targetRelative)
  ) {
    throw new InvalidUploadError('Invalid upload path');
  }
  return targetRelative;
}

function forwardedStreamHeaders(request: NextRequest) {
  const headers = new Headers();
  for (const name of [
    'content-type',
    'content-length',
    'x-aitk-dataset-name',
    'x-aitk-file-name',
    'x-aitk-project-id',
    'x-aitk-relative-path',
    'x-aitk-source-folder',
    'x-aitk-fail-if-dataset-exists',
    'x-aitk-encrypted-object-path',
  ]) {
    const value = request.headers.get(name);
    if (value !== null) headers.set(name, value);
  }
  return headers;
}

async function streamPlainDatasetFile(request: NextRequest) {
  const datasetName = decodedUploadHeader(request, 'x-aitk-dataset-name', 256).trim();
  const workerID = decodedUploadHeader(request, 'x-aitk-worker-id', 256).trim() || 'local';
  const projectID = decodedUploadHeader(request, 'x-aitk-project-id', 256).trim() || null;
  const originalFilename = decodedUploadHeader(request, 'x-aitk-file-name', 512);
  const requestedRelativePath = decodedUploadHeader(request, 'x-aitk-relative-path', 2_048);
  const sourceFolderPath = decodedUploadHeader(request, 'x-aitk-source-folder', 4_096);
  const encryptedObjectPath = decodedUploadHeader(request, 'x-aitk-encrypted-object-path', 2_048);
  const failIfDatasetExists = request.headers.get('x-aitk-fail-if-dataset-exists') === '1';

  if (!datasetName) return NextResponse.json({ error: 'Dataset name is required' }, { status: 400 });
  if (!originalFilename) return NextResponse.json({ error: 'Upload filename is required' }, { status: 400 });

  await assertProjectScopeEnabled(projectID);
  rejectRemoteProjectScope(workerID, projectID);
  if (!isLocalWorker(workerID)) {
    const worker = await getRemoteWorker(workerID);
    const result = await remoteJson(worker, '/api/datasets/upload', {
      method: 'POST',
      headers: forwardedStreamHeaders(request),
      body: request.body,
      duplex: 'half',
    } as RequestInit & { duplex: 'half' });
    return NextResponse.json(result);
  }

  const { datasetsRoot } = await resolveDatasetScope(projectID);
  const uploadDir = resolveDatasetUploadDir(datasetsRoot, datasetName);
  if (
    failIfDatasetExists &&
    fs.existsSync(uploadDir) &&
    fs.statSync(uploadDir).isDirectory() &&
    fs.readdirSync(uploadDir).length > 0
  ) {
    return NextResponse.json({ error: 'Dataset already exists' }, { status: 409 });
  }
  if (encryptedObjectPath) {
    if (!isEncryptedDatasetFolder(uploadDir)) {
      return NextResponse.json({ error: 'Encrypted dataset not found' }, { status: 404 });
    }
    const targetPath = resolveEncryptedObjectPath(uploadDir, encryptedObjectPath);
    let stagingPath: string | null = null;
    try {
      const staged = await streamRequestToStagingFile(request, dirname(targetPath), {
        maxBytes: MAX_STREAMED_FILE_BYTES,
        prefix: 'encrypted-object-upload',
      });
      stagingPath = staged.stagingPath;
      await moveStagedUpload(stagingPath, targetPath);
      stagingPath = null;
      await writeDatasetImportMetadataBestEffort(uploadDir, sourceFolderPath);
      return NextResponse.json({ message: 'Encrypted object uploaded successfully', objects: [encryptedObjectPath] });
    } finally {
      await cleanupStagedUpload(stagingPath);
    }
  }
  if (isEncryptedDatasetFolder(uploadDir)) {
    return NextResponse.json({ error: 'Plain uploads are not allowed for encrypted datasets' }, { status: 400 });
  }

  const relativeFilePath = requestedRelativePath
    ? cleanRelativeUploadPath(requestedRelativePath, originalFilename)
    : cleanUploadFileName(originalFilename);
  let targetPath = nextAvailableFilePath(uploadDir, relativeFilePath);
  assertedRelativeUploadPath(uploadDir, targetPath);

  let stagingPath: string | null = null;
  try {
    const staged = await streamRequestToStagingFile(request, dirname(targetPath), {
      maxBytes: MAX_STREAMED_FILE_BYTES,
      prefix: 'dataset-upload',
    });
    const pendingUploadPath = staged.stagingPath;
    stagingPath = pendingUploadPath;
    await mkdir(uploadDir, { recursive: true });
    let committed = false;
    for (let attempt = 0; attempt < 1_000; attempt += 1) {
      try {
        await moveStagedUploadNoReplace(pendingUploadPath, targetPath);
        stagingPath = null;
        committed = true;
        break;
      } catch (error) {
        if (!isDestinationCollision(error)) throw error;
        targetPath = nextAvailableFilePath(uploadDir, relativeFilePath);
        assertedRelativeUploadPath(uploadDir, targetPath);
      }
    }
    if (!committed) {
      throw new InvalidUploadError('Could not reserve a unique upload filename');
    }
    const targetRelative = assertedRelativeUploadPath(uploadDir, targetPath);
    await writeDatasetImportMetadataBestEffort(uploadDir, sourceFolderPath);
    return NextResponse.json({ message: 'File uploaded successfully', files: [targetRelative] });
  } finally {
    await cleanupStagedUpload(stagingPath);
  }
}

export async function POST(request: NextRequest) {
  try {
    const contentType = (request.headers.get('content-type') || '').split(';', 1)[0].trim().toLowerCase();
    if (contentType === 'application/octet-stream') {
      return await streamPlainDatasetFile(request);
    }
    if (contentType !== 'multipart/form-data') {
      return NextResponse.json({ error: 'Unsupported dataset upload content type' }, { status: 415 });
    }
    // Next's FormData parser buffers the request, so reject an oversized body
    // from its declared length before parsing it into memory.
    assertMultipartRequestLength(request);
    const formData = await request.formData();
    const workerValue = formData.get('worker_id');
    const workerID = typeof workerValue === 'string' && workerValue ? workerValue : 'local';
    const projectValue = formData.get('project_id');
    const projectID = typeof projectValue === 'string' && projectValue ? projectValue : null;
    await assertProjectScopeEnabled(projectID);
    rejectRemoteProjectScope(workerID, projectID);
    const files = formData.getAll('files');
    assertMultipartUploadBounds(files);
    const datasetNameValue = formData.get('datasetName');
    const datasetName = typeof datasetNameValue === 'string' ? datasetNameValue.trim() : '';
    if (files.length === 0) {
      return NextResponse.json({ error: 'No files provided' }, { status: 400 });
    }
    if (!datasetName) {
      return NextResponse.json({ error: 'Dataset name is required' }, { status: 400 });
    }
    const { datasetsRoot } = await resolveDatasetScope(projectID);
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

    const encrypted = formData.get('encrypted') === '1';
    const preserveRelativePaths = formData.get('preserveRelativePaths') === '1';
    const failIfDatasetExists = formData.get('failIfDatasetExists') === '1';
    const rawSourceFolderPath = formData.get('sourceFolderPath');
    const sourceFolderPath = typeof rawSourceFolderPath === 'string' ? rawSourceFolderPath : '';
    const relativePathsText = formData.get('relativePaths');
    let relativePaths: string[] = [];
    if (typeof relativePathsText === 'string' && relativePathsText.trim()) {
      const parsed = JSON.parse(relativePathsText);
      if (!Array.isArray(parsed)) {
        return NextResponse.json({ error: 'relativePaths must be an array' }, { status: 400 });
      }
      relativePaths = parsed.map(value => (typeof value === 'string' ? value : ''));
    }

    // Create upload directory if it doesn't exist
    const uploadDir = resolveDatasetUploadDir(datasetsRoot, datasetName);

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
        const file = files[i] as File;
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
      await writeDatasetImportMetadataBestEffort(uploadDir, sourceFolderPath);
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
      const file = files[i] as File;
      const bytes = await file.arrayBuffer();
      const buffer = Buffer.from(bytes);

      const requestedPath =
        preserveRelativePaths && relativePaths[i]
          ? cleanRelativeUploadPath(relativePaths[i], file.name)
          : cleanUploadFileName(file.name);
      let filePath = nextAvailableFilePath(uploadDir, requestedPath);
      let saved = false;
      for (let attempt = 0; attempt < 1_000; attempt += 1) {
        const filePathRelative = assertedRelativeUploadPath(uploadDir, filePath);
        await mkdir(dirname(filePath), { recursive: true });
        try {
          await writeFile(filePath, buffer, { flag: 'wx' });
          savedFiles.push(filePathRelative);
          saved = true;
          break;
        } catch (error) {
          if (!isDestinationCollision(error)) throw error;
          filePath = nextAvailableFilePath(uploadDir, requestedPath);
        }
      }
      if (!saved) {
        throw new InvalidUploadError('Could not reserve a unique upload filename');
      }
    }

    await writeDatasetImportMetadataBestEffort(uploadDir, sourceFolderPath);

    return NextResponse.json({
      message: 'Files uploaded successfully',
      files: savedFiles,
    });
  } catch (error: unknown) {
    console.error('Upload error:', error);
    const status =
      error && typeof error === 'object' && 'status' in error && typeof error.status === 'number'
        ? error.status
        : 500;
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Error uploading files' },
      { status },
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
