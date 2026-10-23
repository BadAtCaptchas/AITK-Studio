import { NextResponse } from 'next/server';
import { db } from '@/server/db';

import {
  getRemoteWorker,
  isLocalWorker,
  isRemoteJobMissingError,
  markRemoteJobMissing,
  remotePollingJson,
  RemotePollUnavailableError,
} from '@/server/remoteClient';
import { makeRemoteAssetRef } from '@/server/remoteAssets';
import { listJobSampleUrls, listJobSampleMetadata } from '@/server/jobSamples';
import { parseSampleMetadataMap } from '@/utils/sampleMetadata';

export async function GET(request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const includeMetadata = new URL(request.url).searchParams.get('metadata') === '1';

  const job = await db.jobs.findById(jobID);

  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      return NextResponse.json({ samples: [] });
    }
    try {
      const worker = await getRemoteWorker(job.worker_id);
      const data = await remotePollingJson(
        worker,
        `/api/jobs/${encodeURIComponent(job.remote_job_id)}/samples${includeMetadata ? '?metadata=1' : ''}`,
        'samples',
        { signal: request.signal },
      );
      if (!data || typeof data !== 'object' || !('samples' in data) || !Array.isArray(data.samples) ||
          !data.samples.every((sample: unknown): sample is string => typeof sample === 'string')) throw new Error('Invalid remote samples');
      const samples = data.samples.map(sample => makeRemoteAssetRef(job.id, 'img', sample));
      if (!includeMetadata) return NextResponse.json({ samples });
      const metadata = parseSampleMetadataMap('metadata' in data ? data.metadata : undefined, data.samples);
      return NextResponse.json({ samples,
        metadata: Object.fromEntries(Object.entries(metadata).map(([sample, value]) => [makeRemoteAssetRef(job.id, 'img', sample), value])) });
    } catch (error) {
      if (request.signal.aborted) return new Response(null, { status: 499 });
      if (isRemoteJobMissingError(error)) {
        await markRemoteJobMissing(job);
        return NextResponse.json({ samples: [] });
      }
      if (!(error instanceof RemotePollUnavailableError)) {
        console.error('Error reading remote samples:', error instanceof Error ? error.message : 'Unknown error');
      }
      return NextResponse.json({ error: 'Error reading remote samples' }, { status: 502 });
    }
  }

  const samples = await listJobSampleUrls(job);

  return NextResponse.json(includeMetadata ? { samples, metadata: await listJobSampleMetadata(job, samples) } : { samples });
}
