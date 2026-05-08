/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot, getTrainingFolder, getDataRoot } from '@/server/settings';

const contentTypeMap: { [key: string]: string } = {
  // Images
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.bmp': 'image/bmp',
  // Videos
  '.mp4': 'video/mp4',
  '.avi': 'video/x-msvideo',
  '.mov': 'video/quicktime',
  '.mkv': 'video/x-matroska',
  '.wmv': 'video/x-ms-wmv',
  '.m4v': 'video/x-m4v',
  '.flv': 'video/x-flv',
  // Audio
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
};

type ImageRouteParams = {
  imagePath: string | string[];
};

function getRequestedPath(request: NextRequest, imagePath: string | string[]) {
  const pathname = request.nextUrl?.pathname;
  const routePrefix = '/api/img/';
  const rawPath =
    pathname && pathname.startsWith(routePrefix)
      ? pathname.slice(routePrefix.length)
      : Array.isArray(imagePath)
        ? imagePath.join('/')
        : imagePath;

  return path.resolve(decodeURIComponent(rawPath));
}

async function resolveExistingDir(dir: string) {
  if (!dir) return null;
  return fs.promises.realpath(path.resolve(dir)).catch(() => null);
}

function isPathInsideRoot(root: string, filepath: string) {
  const relativePath = path.relative(root, filepath);
  return relativePath === '' || (!relativePath.startsWith('..') && !path.isAbsolute(relativePath));
}

export async function GET(request: NextRequest, { params }: { params: ImageRouteParams }) {
  const { imagePath } = await params;
  try {
    // Decode the path
    const filepath = getRequestedPath(request, imagePath);

    // Get allowed directories
    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const dataRoot = await getDataRoot();

    const allowedDirs = (
      await Promise.all([datasetRoot, trainingRoot, dataRoot].map(dir => resolveExistingDir(dir)))
    ).filter((dir): dir is string => dir !== null);

    // Security check: Ensure path is in allowed directory using canonical paths
    const canonicalPath = await fs.promises.realpath(filepath).catch(() => null);
    const isAllowed =
      canonicalPath !== null && allowedDirs.some(allowedDir => isPathInsideRoot(allowedDir, canonicalPath));

    if (!isAllowed) {
      console.warn(`Access denied: ${filepath} not in ${allowedDirs.join(', ')}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    // Stat file (async)
    const stat = await fs.promises.stat(canonicalPath).catch(() => null);
    if (!stat || !stat.isFile()) {
      return new NextResponse('File not found', { status: 404 });
    }

    const ext = path.extname(filepath).toLowerCase();
    const contentType = contentTypeMap[ext] || 'application/octet-stream';

    // Support range requests for video/audio seeking
    const rangeHeader = request.headers.get('range');
    if (rangeHeader) {
      const parts = rangeHeader.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
      const chunkSize = end - start + 1;

      const stream = fs.createReadStream(canonicalPath, { start, end });
      const readable = new ReadableStream({
        start(controller) {
          stream.on('data', chunk => controller.enqueue(chunk));
          stream.on('end', () => controller.close());
          stream.on('error', err => controller.error(err));
        },
        cancel() {
          stream.destroy();
        },
      });

      return new NextResponse(readable as any, {
        status: 206,
        headers: {
          'Content-Range': `bytes ${start}-${end}/${stat.size}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': String(chunkSize),
          'Content-Type': contentType,
          'Cache-Control': 'public, max-age=86400',
        },
      });
    }

    // Stream the file instead of buffering it entirely
    const stream = fs.createReadStream(canonicalPath);
    const readable = new ReadableStream({
      start(controller) {
        stream.on('data', chunk => controller.enqueue(chunk));
        stream.on('end', () => controller.close());
        stream.on('error', err => controller.error(err));
      },
      cancel() {
        stream.destroy();
      },
    });

    return new NextResponse(readable as any, {
      headers: {
        'Content-Type': contentType,
        'Content-Length': String(stat.size),
        'Cache-Control': 'public, max-age=86400',
        'Accept-Ranges': 'bytes',
      },
    });
  } catch (error) {
    console.error('Error serving image:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
