import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import { Readable } from 'stream';
import { db } from '@/server/db';
import { assertProjectJobEnabled } from '@/server/projects';
import { resolveJobSampleFile } from '@/server/jobSamples';

type SampleRouteParams = {
  jobID: string;
  samplePath: string[];
};

export async function GET(request: NextRequest, { params }: { params: Promise<SampleRouteParams> }) {
  const { jobID, samplePath } = await params;
  const sampleSegments = samplePath;

  if (sampleSegments.length !== 1) {
    return new NextResponse('Invalid sample path', { status: 400 });
  }

  const job = await db.jobs.findById(jobID);
  if (!job) {
    return new NextResponse('Job not found', { status: 404 });
  }
  try {
    await assertProjectJobEnabled(job);
  } catch (error: any) {
    return new NextResponse(error?.message || 'Project spaces are disabled', { status: error?.status || 403 });
  }

  const sampleFile = await resolveJobSampleFile(job, sampleSegments[0]);
  if (!sampleFile) {
    return new NextResponse('File not found', { status: 404 });
  }

  const { path: canonicalPath, stat, contentType } = sampleFile;
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
