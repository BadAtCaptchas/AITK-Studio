/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder } from '@/server/settings';

function decodePath(value: string) {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function realpathIfExists(value: string) {
  if (!fs.existsSync(value)) return null;
  return fs.realpathSync(value);
}

function isInsideRoot(root: string, target: string) {
  const resolvedRoot = realpathIfExists(root);
  if (!resolvedRoot) return false;

  const relativePath = path.relative(resolvedRoot, target);
  return relativePath !== '' && !relativePath.startsWith('..') && !path.isAbsolute(relativePath);
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { filePath } = body;

    if (!filePath || typeof filePath !== 'string') {
      return new NextResponse('filePath is required', { status: 400 });
    }

    if (filePath.startsWith('remote://')) {
      return new NextResponse('Remote files cannot be deleted from this endpoint', { status: 400 });
    }

    const decodedFilePath = decodePath(filePath);
    const resolvedFilePath = realpathIfExists(decodedFilePath);

    if (!resolvedFilePath) {
      console.warn(`File not found: ${decodedFilePath}`);
      return new NextResponse('File not found', { status: 404 });
    }

    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const allowedDirs = [datasetRoot, trainingRoot].filter(Boolean);

    if (!allowedDirs.some(allowedDir => isInsideRoot(allowedDir, resolvedFilePath))) {
      console.warn(`Access denied: ${resolvedFilePath} not in ${allowedDirs.join(', ')}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    const stat = fs.statSync(resolvedFilePath);
    if (!stat.isFile()) {
      return new NextResponse('Not a file', { status: 400 });
    }

    fs.unlinkSync(resolvedFilePath);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error deleting file:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
