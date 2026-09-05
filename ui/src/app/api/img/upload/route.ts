import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { NextRequest, NextResponse } from 'next/server';
import { getDataRoot } from '@/server/settings';

import {
  cleanupStagedUpload,
  decodedUploadHeader,
  InvalidUploadError,
  moveStagedUploadNoReplace,
  streamRequestToStagingFile,
} from '@/server/streamedUpload';

export const runtime = 'nodejs';

const MAX_FILE_BYTES = 1024 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set([
  '.jpg',
  '.jpeg',
  '.png',
  '.gif',
  '.webp',
  '.jxl',
  '.bmp',
  '.mp4',
  '.avi',
  '.mov',
  '.mkv',
  '.wmv',
  '.m4v',
  '.flv',
  '.mp3',
  '.wav',
  '.flac',
  '.ogg',
]);
const VALIDATION_IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.gif', '.webp', '.jxl', '.bmp']);

function isDestinationCollision(error: unknown) {
  return error && typeof error === 'object' && 'code' in error && error.code === 'EEXIST';
}

export async function POST(request: NextRequest) {
  let stagingPath: string | null = null;
  try {
    const contentType = (request.headers.get('content-type') || '').split(';', 1)[0].trim().toLowerCase();
    if (contentType !== 'application/octet-stream') {
      return NextResponse.json({ error: 'Media uploads must use a streamed binary request' }, { status: 415 });
    }

    const originalFilename = decodedUploadHeader(request, 'x-aitk-file-name', 512);
    if (!originalFilename) {
      return NextResponse.json({ error: 'Upload filename is required' }, { status: 400 });
    }
    const extension = path.extname(path.basename(originalFilename.replace(/\\/g, '/'))).toLowerCase();
    if (!ALLOWED_EXTENSIONS.has(extension)) {
      return NextResponse.json({ error: 'Unsupported media type' }, { status: 415 });
    }

    const imageRoot = path.join(await getDataRoot(), 'images');
    const staged = await streamRequestToStagingFile(request, imageRoot, {
      maxBytes: MAX_FILE_BYTES,
      prefix: 'media-upload',
    });
    const pendingUploadPath = staged.stagingPath;
    stagingPath = pendingUploadPath;
    let filePath = '';
    for (let attempt = 0; attempt < 10; attempt += 1) {
      filePath = path.join(imageRoot, `${uuidv4()}${extension}`);
      try {
        await moveStagedUploadNoReplace(pendingUploadPath, filePath);
        stagingPath = null;
        break;
      } catch (error) {
        if (!isDestinationCollision(error)) throw error;
      }
    }
    if (stagingPath) {
      throw new InvalidUploadError('Could not reserve a unique media filename');
    }

    return NextResponse.json({
      message: 'File uploaded successfully',
      files: [filePath],
    });
  } catch (error) {
    await cleanupStagedUpload(stagingPath);
    console.error('Upload error:', error);
    const status =
      error && typeof error === 'object' && 'status' in error && typeof error.status === 'number' ? error.status : 500;
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Error uploading file' }, { status });
  }
}
