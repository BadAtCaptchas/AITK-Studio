import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import { Readable } from 'stream';
import { db } from '@/server/db';

import { resolveJobSampleFile } from '@/server/jobSamples';
import path from 'path';
import { isRequestAuthenticated } from '@/utils/authSession';
import { isLocalWorker, getRemoteWorker, remoteJson } from '@/server/remoteClient';
import { normalizeStoragePathSetting } from '@/server/pathContainment';
import { removeSampleThumbnails } from '@/server/sampleThumbnails';
import { waveformArtwork } from '@/server/audioArtwork';
import { deleteSampleLayers } from '@/server/sampleLayers';

type SampleRouteParams = {
  jobID: string;
  samplePath: string[];
};

export async function DELETE(request: Request, { params }: { params: Promise<SampleRouteParams> }) {
  if (!await isRequestAuthenticated(request, process.env.AI_TOOLKIT_AUTH)) return Response.json({ error: 'Unauthorized' }, { status: 401 });
  const { jobID, samplePath } = await params;
  if (samplePath.length !== 1 || !samplePath[0] || /[\\/]/.test(samplePath[0])) return Response.json({ error: 'Invalid sample path' }, { status: 400 });
  const job = await db.jobs.findById(jobID);
  if (!job) return Response.json({ error: 'Job not found' }, { status: 404 });
  try {
    if (!isLocalWorker(job.worker_id)) {
      if (!job.remote_job_id) return Response.json({ error: 'Remote job not available' }, { status: 409 });
      return Response.json(await remoteJson<unknown>(await getRemoteWorker(job.worker_id), `/api/jobs/${encodeURIComponent(job.remote_job_id)}/samples/${encodeURIComponent(samplePath[0])}`, { method: 'DELETE' }));
    }
    const sample = await resolveJobSampleFile(job, samplePath[0]);
    if (!sample) return Response.json({ deleted: false });
    const folder = path.dirname(sample.path);
    await deleteSampleLayers(sample.path);
    await fs.promises.unlink(sample.path);
    await removeSampleThumbnails(folder, samplePath[0]);
    if (path.extname(sample.path) !== '.txt') {
      const caption = await normalizeStoragePathSetting(sample.path.slice(0, -path.extname(sample.path).length) + '.txt', folder);
      if (caption) await fs.promises.unlink(caption).catch(() => undefined);
    }
    return Response.json({ deleted: true });
  } catch {
    return Response.json({ error: 'Could not delete sample' }, { status: 500 });
  }
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

export async function GET(request: NextRequest, { params }: { params: Promise<SampleRouteParams> }) {
  if (!await isRequestAuthenticated(request, process.env.AI_TOOLKIT_AUTH)) return new NextResponse('Unauthorized', { status: 401 });
  const { jobID, samplePath } = await params;
  const sampleSegments = samplePath;

  if (sampleSegments.length !== 1) {
    return new NextResponse('Invalid sample path', { status: 400 });
  }

  const job = await db.jobs.findById(jobID);
  if (!job) {
    return new NextResponse('Job not found', { status: 404 });
  }

  const sampleFile = await resolveJobSampleFile(job, sampleSegments[0], {
    thumbnail: request.nextUrl.searchParams.get('thumb') === '1',
  });
  if (!sampleFile) {
    return new NextResponse('File not found', { status: 404 });
  }

  const { path: canonicalPath, stat, contentType } = sampleFile;
  if (request.nextUrl.searchParams.get('thumb') === '1' && contentType.startsWith('audio/')) {
    try {
      const artwork = await waveformArtwork(canonicalPath, stat.mtimeMs, stat.size);
      return new NextResponse(new Uint8Array(artwork), { headers: { 'Content-Type': 'image/png', 'Cache-Control': 'private, no-cache, must-revalidate', 'X-Content-Type-Options': 'nosniff' } });
    } catch { return new NextResponse('Artwork unavailable', { status: 503 }); }
  }
  const etag = `W/"${stat.ino.toString(36)}-${stat.size.toString(36)}-${stat.mtimeMs.toString(36)}"`;
  const cacheControl = 'private, no-cache, must-revalidate';

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
}
