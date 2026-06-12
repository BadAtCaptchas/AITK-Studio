import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/server/db';
import { getRemoteWorker, isLocalWorker, remoteProxyFetch } from '@/server/remoteClient';
import type { RemoteAssetType } from '@/server/remoteAssets';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function remoteAssetPath(type: RemoteAssetType, remotePath: string) {
  if (remotePath.startsWith('/api/jobs/')) return remotePath;
  const encoded = encodeURIComponent(remotePath);
  if (type === 'file') return `/api/files/${encoded}`;
  if (type === 'audio-art') return `/api/audio/art/${encoded}`;
  return `/api/img/${encoded}`;
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
  ]) {
    const value = source.headers.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}

export async function GET(request: NextRequest) {
  const jobID = request.nextUrl.searchParams.get('job_id') || '';
  const remotePath = request.nextUrl.searchParams.get('path') || '';
  const type = (request.nextUrl.searchParams.get('type') || 'img') as RemoteAssetType;

  if (!jobID || !remotePath) {
    return new NextResponse('Missing remote asset parameters', { status: 400 });
  }

  const job = await db.jobs.findById(jobID);
  if (!job || isLocalWorker(job.worker_id)) {
    return new NextResponse('Remote job not found', { status: 404 });
  }

  try {
    const worker = await getRemoteWorker(job.worker_id);
    const remoteResponse = await remoteProxyFetch(worker, remoteAssetPath(type, remotePath), request.headers);
    return new NextResponse(remoteResponse.body, {
      status: remoteResponse.status,
      headers: copyResponseHeaders(remoteResponse),
    });
  } catch (error) {
    console.error('Remote asset proxy failed:', error);
    return new NextResponse('Remote asset unavailable', { status: 502 });
  }
}
