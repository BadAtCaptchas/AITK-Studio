import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';
import { getTrainingFolder } from '@/server/settings';
import { db } from '@/server/db';

const contentTypeMap: { [key: string]: string } = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.mp4': 'video/mp4',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.flac': 'audio/flac',
  '.ogg': 'audio/ogg',
};

type SampleRouteParams = {
  jobID: string;
  samplePath: string | string[];
};

function isPathInsideRoot(root: string, filepath: string) {
  const relativePath = path.relative(root, filepath);
  return relativePath === '' || (!relativePath.startsWith('..') && !path.isAbsolute(relativePath));
}

export async function GET(request: NextRequest, { params }: { params: SampleRouteParams }) {
  const { jobID, samplePath } = await params;
  const sampleSegments = Array.isArray(samplePath) ? samplePath : [samplePath];

  if (sampleSegments.length !== 1) {
    return new NextResponse('Invalid sample path', { status: 400 });
  }

  const ext = path.extname(sampleSegments[0]).toLowerCase();
  const contentType = contentTypeMap[ext];
  if (!contentType) {
    return new NextResponse('Unsupported media type', { status: 415 });
  }

  const job = await db.jobs.findById(jobID);
  if (!job) {
    return new NextResponse('Job not found', { status: 404 });
  }

  const trainingFolder = await getTrainingFolder();
  const canonicalTrainingFolder = await fs.promises.realpath(path.resolve(trainingFolder)).catch(() => null);
  const samplesFolder = path.resolve(trainingFolder, job.name, 'samples');
  const filepath = path.resolve(samplesFolder, sampleSegments[0]);
  const canonicalSamplesFolder = await fs.promises.realpath(samplesFolder).catch(() => null);
  const canonicalPath = await fs.promises.realpath(filepath).catch(() => null);

  if (
    !canonicalTrainingFolder ||
    !canonicalSamplesFolder ||
    !canonicalPath ||
    !isPathInsideRoot(canonicalTrainingFolder, canonicalSamplesFolder) ||
    !isPathInsideRoot(canonicalSamplesFolder, canonicalPath)
  ) {
    return new NextResponse('File not found', { status: 404 });
  }

  const stat = await fs.promises.stat(canonicalPath).catch(() => null);
  if (!stat || !stat.isFile()) {
    return new NextResponse('File not found', { status: 404 });
  }

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
}
