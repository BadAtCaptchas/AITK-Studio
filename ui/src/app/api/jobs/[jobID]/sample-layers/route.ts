import fs from 'fs';
import { Readable } from 'stream';
import { NextRequest } from 'next/server';
import { db } from '@/server/db';
import { resolveJobSampleFile } from '@/server/jobSamples';
import { readSampleLayers } from '@/server/sampleLayers';
import { getRemoteWorker, isLocalWorker, remoteJson, remoteProxyFetch } from '@/server/remoteClient';
import { isRequestAuthenticated } from '@/utils/authSession';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  if (!(await isRequestAuthenticated(request, process.env.AI_TOOLKIT_AUTH)))
    return Response.json({ error: 'Unauthorized' }, { status: 401 });
  const { jobID } = await params;
  const name = request.nextUrl.searchParams.get('sample') || '';
  const rawLayer = request.nextUrl.searchParams.get('layer');
  if (
    !name ||
    /[\\/\x00-\x1f]/.test(name) ||
    !name.endsWith('.png') ||
    (rawLayer !== null && !/^(?:0|[1-9]\d?)$/.test(rawLayer))
  )
    return Response.json({ error: 'Invalid sample layer request' }, { status: 400 });
  const job = await db.jobs.findById(jobID);
  if (!job) return Response.json({ error: 'Job not found' }, { status: 404 });
  const url = `/api/jobs/${encodeURIComponent(jobID)}/sample-layers?sample=${encodeURIComponent(name)}`;
  try {
    if (!isLocalWorker(job.worker_id)) {
      if (!job.remote_job_id) return Response.json({ layers: [] });
      const worker = await getRemoteWorker(job.worker_id);
      const remoteUrl = `/api/jobs/${encodeURIComponent(job.remote_job_id)}/sample-layers?sample=${encodeURIComponent(name)}`;
      if (rawLayer !== null) {
        const response = await remoteProxyFetch(
          worker,
          `${remoteUrl}&layer=${rawLayer}`,
          request.headers,
          'GET',
          request.signal,
        );
        const headers = new Headers();
        for (const key of [
          'content-type',
          'content-length',
          'content-disposition',
          'cache-control',
          'x-content-type-options',
        ]) {
          const value = response.headers.get(key);
          if (value) headers.set(key, value);
        }
        return new Response(response.body, { status: response.status, headers });
      }
      const data: unknown = await remoteJson<unknown>(worker, remoteUrl);
      if (
        !data ||
        typeof data !== 'object' ||
        !('layers' in data) ||
        !Array.isArray(data.layers) ||
        data.layers.length > 32
      )
        throw new Error('Invalid remote sample layer response');
      return Response.json({ order: 'bottom-to-top', layers: data.layers.map((_, index) => `${url}&layer=${index}`) });
    }
    const sample = await resolveJobSampleFile(job, name);
    if (!sample) return Response.json({ error: 'Sample not found' }, { status: 404 });
    const group = await readSampleLayers(sample.path);
    if (rawLayer === null)
      return Response.json({
        order: 'bottom-to-top',
        layers: group?.layers.map((_, index) => `${url}&layer=${index}`) || [],
      });
    const target = group?.paths[Number(rawLayer)];
    if (!target) return Response.json({ error: 'Layer not found' }, { status: 404 });
    const stream = fs.createReadStream(target);
    const abort = () => stream.destroy();
    request.signal.addEventListener('abort', abort, { once: true });
    if (request.signal.aborted) abort();
    stream.once('close', () => request.signal.removeEventListener('abort', abort));
    return new Response(Readable.toWeb(stream) as ReadableStream<Uint8Array>, {
      headers: {
        'Content-Type': 'image/png',
        'Content-Length': String((await fs.promises.stat(target)).size),
        'Content-Disposition': `inline; filename="layer-${Number(rawLayer) + 1}.png"`,
        'Cache-Control': 'private, no-cache',
        'X-Content-Type-Options': 'nosniff',
      },
    });
  } catch (error) {
    return Response.json(
      { error: error instanceof Error ? error.message : 'Sample layers unavailable' },
      { status: 400 },
    );
  }
}
