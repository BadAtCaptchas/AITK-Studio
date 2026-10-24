import { db } from '@/server/db';
import { getRemoteWorker, isLocalWorker, remoteProxyFetch } from '@/server/remoteClient';
import { remoteAssetProxyPath, type RemoteAssetType } from '@/server/remoteAssets';


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

export async function GET(request: Request) {
  const jobID = new URL(request.url).searchParams.get('job_id') || '';
  const remotePath = new URL(request.url).searchParams.get('path') || '';
  const type = (new URL(request.url).searchParams.get('type') || 'img') as RemoteAssetType;
  const thumbnail = new URL(request.url).searchParams.get('thumb') === '1';

  if (!jobID || !remotePath) {
    return new Response('Missing remote asset parameters', { status: 400 });
  }

  const job = await db.jobs.findById(jobID);
  if (!job || isLocalWorker(job.worker_id)) {
    return new Response('Remote job not found', { status: 404 });
  }

  try {
    const worker = await getRemoteWorker(job.worker_id);
    const remoteResponse = await remoteProxyFetch(
      worker,
      remoteAssetProxyPath(type, remotePath, job.remote_job_id, { thumbnail }),
      request.headers,
      request.method === 'HEAD' ? 'HEAD' : 'GET',
      request.signal,
    );
    return new Response(remoteResponse.body, {
      status: remoteResponse.status,
      headers: copyResponseHeaders(remoteResponse),
    });
  } catch (error) {
    console.error('Remote asset proxy failed:', error);
    return new Response('Remote asset unavailable', { status: 502 });
  }
}
