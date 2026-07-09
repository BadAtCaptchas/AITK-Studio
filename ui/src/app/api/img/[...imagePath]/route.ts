/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';
import { getDatasetsRoot, getTrainingFolder, getDataRoot } from '@/server/settings';
import { isPathWithinRoot } from '@/server/pathContainment';
import { isRegisteredProjectPath } from '@/server/projectMediaSecurity';
import { findEncryptedDatasetRoot } from '@/server/encryptedDatasets';
import { getRemoteWorker, remoteProxyFetch } from '@/server/remoteClient';
import { isRemoteDatasetAssetRequestAuthorized } from '@/server/remoteDatasetAssetAccess';
import { parseRemoteDatasetAssetRef } from '@/utils/remoteDatasetRefs';

const contentTypeMap: { [key: string]: string } = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.jxl': 'image/jxl',
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

const privateMediaCacheControl = 'private, no-cache, must-revalidate';

type ImageRouteParams = {
  imagePath: string[];
};

function getRequestedValue(request: NextRequest, imagePath: string[]) {
  const pathname = request.nextUrl?.pathname;
  const routePrefix = '/api/img/';
  const rawPath =
    pathname && pathname.startsWith(routePrefix)
      ? pathname.slice(routePrefix.length)
      : imagePath.join('/');

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

function parseRange(value: string | null, size: number) {
  if (!value) return null;
  const match = /^bytes=(\d*)-(\d*)$/i.exec(value.trim());
  if (!match || (!match[1] && !match[2])) return 'invalid' as const;
  let start: number;
  let end: number;
  if (!match[1]) {
    const suffix = Number(match[2]);
    if (!Number.isSafeInteger(suffix) || suffix <= 0) return 'invalid' as const;
    start = Math.max(0, size - suffix);
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] ? Number(match[2]) : size - 1;
  }
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || start >= size) {
    return 'invalid' as const;
  }
  return { start, end: Math.min(end, size - 1) };
}

async function resolveExistingDir(dir: string) {
  if (!dir) return null;
  return fs.promises.realpath(path.resolve(dir)).catch(() => null);
}

export async function GET(request: NextRequest, { params }: { params: Promise<ImageRouteParams> }) {
  const { imagePath } = await params;
  try {
    const requestedValue = getRequestedValue(request, imagePath);
    const remoteAsset = parseRemoteDatasetAssetRef(requestedValue);
    if (remoteAsset) {
      if (
        !(await isRemoteDatasetAssetRequestAuthorized(
          request.headers,
          remoteAsset.workerID,
          remoteAsset.path,
          remoteAsset.expires,
          remoteAsset.signature,
        ))
      ) {
        return new NextResponse('Unauthorized', { status: 401 });
      }
      const worker = await getRemoteWorker(remoteAsset.workerID);
      const remoteResponse = await remoteProxyFetch(worker, remoteAssetPath(remoteAsset.path), request.headers);
      if ((remoteResponse.headers.get('content-type') || '').toLowerCase().includes('image/svg+xml')) {
        return new NextResponse('Unsupported media type', { status: 415 });
      }
      const headers = copyResponseHeaders(remoteResponse);
      headers.set('Cache-Control', privateMediaCacheControl);
      headers.set('X-Content-Type-Options', 'nosniff');
      return new NextResponse(remoteResponse.body, {
        status: remoteResponse.status,
        headers,
      });
    }

    const filepath = path.resolve(requestedValue);

    const datasetRoot = await getDatasetsRoot();
    const trainingRoot = await getTrainingFolder();
    const dataRoot = await getDataRoot();

    const [canonicalDatasetRoot, canonicalTrainingRoot, canonicalDataRoot] = await Promise.all(
      [datasetRoot, trainingRoot, dataRoot].map(dir => resolveExistingDir(dir)),
    );
    const generalAllowedDirs = [canonicalDatasetRoot, canonicalDataRoot].filter(
      (dir): dir is string => dir !== null,
    );

    const canonicalPath = await fs.promises.realpath(filepath).catch(() => null);
    if (!canonicalPath) {
      return new NextResponse('File not found', { status: 404 });
    }

    if (await isRegisteredProjectPath(canonicalPath)) {
      return new NextResponse('Project media requires a signed project-relative URL', { status: 403 });
    }

    const ext = path.extname(canonicalPath).toLowerCase();
    const contentType = contentTypeMap[ext];
    if (!contentType) {
      return new NextResponse('Unsupported media type', { status: 415 });
    }

    const trainingRelativePath =
      canonicalTrainingRoot && isPathWithinRoot(canonicalTrainingRoot, canonicalPath)
        ? path.relative(canonicalTrainingRoot, canonicalPath)
        : null;
    const isInGeneralAllowedDir =
      trainingRelativePath === null &&
      generalAllowedDirs.some(allowedDir => isPathWithinRoot(allowedDir, canonicalPath));
    const isInTrainingSamplesDir =
      trainingRelativePath !== null && trainingRelativePath.split(path.sep).includes('samples');

    if (!isInGeneralAllowedDir && !isInTrainingSamplesDir) {
      const allowedDirs = [...generalAllowedDirs, canonicalTrainingRoot].filter((dir): dir is string => dir !== null);
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

    const etag = `W/"${stat.ino.toString(36)}-${stat.size.toString(36)}-${stat.mtimeMs.toString(36)}"`;
    const cacheControl = privateMediaCacheControl;

    const ifNoneMatch = request.headers.get('if-none-match');
    if (ifNoneMatch && ifNoneMatch === etag) {
      return new NextResponse(null, {
        status: 304,
        headers: {
          ETag: etag,
          'Cache-Control': cacheControl,
          'X-Content-Type-Options': 'nosniff',
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

    const requestedRange = parseRange(request.headers.get('range'), stat.size);
    if (requestedRange === 'invalid') {
      return new NextResponse(null, {
        status: 416,
        headers: {
          'Content-Range': `bytes */${stat.size}`,
          'Cache-Control': cacheControl,
          'X-Content-Type-Options': 'nosniff',
        },
      });
    }
    if (requestedRange) {
      const chunkSize = requestedRange.end - requestedRange.start + 1;

      return new NextResponse(buildBody(requestedRange.start, requestedRange.end) as any, {
        status: 206,
        headers: {
          'Content-Range': `bytes ${requestedRange.start}-${requestedRange.end}/${stat.size}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': String(chunkSize),
          'Content-Type': contentType,
          'Cache-Control': cacheControl,
          'X-Content-Type-Options': 'nosniff',
          ETag: etag,
        },
      });
    }

    return new NextResponse(buildBody() as any, {
      headers: {
        'Content-Type': contentType,
        'Content-Length': String(stat.size),
        'Cache-Control': cacheControl,
        'X-Content-Type-Options': 'nosniff',
        'Accept-Ranges': 'bytes',
        ETag: etag,
      },
    });
  } catch (error) {
    console.error('Error serving image:', error);
    return new NextResponse('Internal Server Error', { status: 500 });
  }
}
