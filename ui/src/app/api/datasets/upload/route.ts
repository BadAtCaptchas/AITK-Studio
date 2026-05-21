// src/app/api/datasets/upload/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import { isAbsolute, join, resolve, relative, sep } from 'path';
import { getDatasetsRoot } from '@/server/settings';
import {
  isEncryptedDatasetFolder,
  resolveEncryptedObjectPath,
  validateEncryptedManifest,
  writeEncryptedManifest,
} from '@/server/encryptedDatasets';

export async function POST(request: NextRequest) {
  try {
    const datasetsPath = await getDatasetsRoot();
    if (!datasetsPath) {
      return NextResponse.json({ error: 'Datasets path not found' }, { status: 500 });
    }
    const formData = await request.formData();
    const files = formData.getAll('files');
    const datasetName = (formData.get('datasetName') as string)?.trim();
    const encrypted = formData.get('encrypted') === '1';

    if (!files || files.length === 0) {
      return NextResponse.json({ error: 'No files provided' }, { status: 400 });
    }

    if (!datasetName) {
      return NextResponse.json({ error: 'Dataset name is required' }, { status: 400 });
    }

    // Create upload directory if it doesn't exist
    const datasetsRoot = resolve(datasetsPath);
    const uploadDir = resolve(datasetsRoot, datasetName);
    const uploadDirRelative = relative(datasetsRoot, uploadDir);

    if (
      uploadDirRelative === '' ||
      uploadDirRelative === '.' ||
      uploadDirRelative.startsWith('..') ||
      uploadDirRelative.includes(`..${sep}`) ||
      isAbsolute(uploadDirRelative)
    ) {
      return NextResponse.json({ error: 'Invalid dataset name' }, { status: 400 });
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

      // Clean filename and ensure it's unique
      const fileName = file.name.replace(/[^a-zA-Z0-9.-]/g, '_');
      const filePath = join(uploadDir, fileName);

      await writeFile(filePath, buffer);
      savedFiles.push(fileName);
    }

    return NextResponse.json({
      message: 'Files uploaded successfully',
      files: savedFiles,
    });
  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json({ error: 'Error uploading files' }, { status: 500 });
  }
}

// Increase payload size limit (default is 4mb)
export const config = {
  api: {
    bodyParser: false,
    responseLimit: '50mb',
  },
};
