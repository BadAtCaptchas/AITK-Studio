/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';
import { getDatasetsRoot, getTrainingFolder, getDataRoot } from '@/server/settings';
import { findEncryptedDatasetRoot } from '@/server/encryptedDatasets';
import { getRemoteWorker, remoteProxyFetch } from '@/server/remoteClient';
import { parseRemoteDatasetAssetRef } from '@/utils/remoteDatasetRefs';

const contentTypeMap: { [key: string]: string } = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.bmp': 'image/bmp',
  '.mp4': 'video/mp4',
  '.avi': 'video/x-msvideo',
  '.mov': 'video/quicktime',
  '.mkv': 'video/x-matroska',
  '.wmv': 'video/x-ms-wmv',
  '.m4v': 'video/x-m4v',
  '.flv': 'video/x-flv',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
};

type ImageRouteParams = {
  imagePath: string | string[];
};

function getRequestedValue(request: NextRequest, imagePath: string | string[]) {
  const pathname = request.nextUrl?.pathname;
  const routePrefix = '/api/img/';
  const rawPath =
    pathname && pathname.startsWith(routePrefix)
      ? pathname.slice(routePrefix.length)
      : Array.isArray(imagePath)
        ? imagePath.join('/')
        : imagePath;

  return decodeURIComponent(rawPath);
}

function remoteAssetPath(remotePath: string) {
  return `/api/img/${encodeURIComponent(remotePath)}`;
}

function copyResponseHeaders(source: Response) {
  const headers = new Headers();
  for (const name of [
    'content-type',
    'content-length',
    'content-range',
    'accept-ranges',
    'cache-control',
    'content-disposition',
    'x-content-type-options',
    'etag',
  ]) {
    const value = source.headers.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
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
    const requestedValue = getRequestedValue(request, imagePath);
    const remoteAsset = parseRemoteDatasetAssetRef(requestedValue);
    if (remoteAsset) {
      const worker = await getRemoteWorker(remoteAsset.workerID);
      const remoteResponse = await remoteProxyFetch(worker, remoteAssetPath(remoteAsset.path), request.headers);
      return new NextResponse(remoteResponse.body, {
        status: remoteResponse.status,
        headers: copyResponseHeaders(remoteResponse),
      });
    }

    const filepath = path.resolve(requestedValue);

    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const dataRoot = await getDataRoot();

    const [canonicalDatasetRoot, canonicalTrainingRoot, canonicalDataRoot] = await Promise.all(
      [datasetRoot, trainingRoot, dataRoot].map(dir => resolveExistingDir(dir)),
    );
    const allowedDirs = [canonicalDatasetRoot, canonicalTrainingRoot, canonicalDataRoot].filter(
      (dir): dir is string => dir !== null,
    );

    const canonicalPath = await fs.promises.realpath(filepath).catch(() => null);
    if (!canonicalPath || !allowedDirs.some(allowedDir => isPathInsideRoot(allowedDir, canonicalPath))) {
      console.warn(`Access denied: ${filepath} not in ${allowedDirs.join(', ')}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    if (canonicalDatasetRoot && findEncryptedDatasetRoot(canonicalPath, canonicalDatasetRoot)) {
      return new NextResponse('Encrypted dataset objects are not served through this route', { status: 403 });
    }

    if (request.signal.aborted) {
      return new NextResponse(null, { status: 499 });
    }

    const stat = await fs.promises.stat(canonicalPath).catch(() => null);
    if (!stat || !stat.isFile()) {
      return new NextResponse('File not found', { status: 404 });
    }

    const ext = path.extname(canonicalPath).toLowerCase();
    const contentType = contentTypeMap[ext] || 'application/octet-stream';

    const etag = `W/"${stat.ino.toString(36)}-${stat.size.toString(36)}-${stat.mtimeMs.toString(36)}"`;
    const cacheControl = 'public, max-age=86400, immutable';

    const ifNoneMatch = request.headers.get('if-none-match');
    if (ifNoneMatch && ifNoneMatch === etag) {
      return new NextResponse(null, {
        status: 304,
        headers: {
          ETag: etag,
          'Cache-Control': cacheControl,
        },
      });
    }

    const buildBody = (start?: number, end?: number) => {
      const nodeStream =
        start !== undefined && end !== undefined
          ? fs.createReadStream(canonicalPath, { start, end })
          : fs.createReadStream(canonicalPath);

      const onAbort = () => nodeStream.destroy();
      if (request.signal.aborted) {
        nodeStream.destroy();
      } else {
        request.signal.addEventListener('abort', onAbort, { once: true });
      }
      nodeStream.once('close', () => request.signal.removeEventListener('abort', onAbort));

      return Readable.toWeb(nodeStream) as unknown as ReadableStream;
    };

    const rangeHeader = request.headers.get('range');
    if (rangeHeader) {
      const parts = rangeHeader.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
      const chunkSize = end - start + 1;

      return new NextResponse(buildBody(start, end) as any, {
        status: 206,
        headers: {
          'Content-Range': `bytes ${start}-${end}/${stat.size}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': String(chunkSize),
          'Content-Type': contentType,
          'Cache-Control': cacheControl,
          ETag: etag,
        },
      });
    }

    return new NextResponse(buildBody() as any, {
      headers: {
        'Content-Type': contentType,
        'Content-Length': String(stat.size),
        'Cache-Control': cacheControl,
        'Accept-Ranges': 'bytes',
        ETag: etag,
      },
    });
  } catch (error) {
    console.error('Error serving image:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
