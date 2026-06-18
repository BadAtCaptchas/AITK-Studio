import { NextResponse } from 'next/server';
import { db } from '@/server/db';
import { collectModelReferences, setConfigPathValue } from '@/server/trainingJobTransfer';
import { normalizeModelReferenceValue, prefetchModelReferences } from '@/server/hfModelPrefetch';
import { getRemoteWorker, isLocalWorker, remoteJson, syncRemoteJob } from '@/server/remoteClient';
import { assertProjectJobEnabled } from '@/server/projects';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type ModelPrefetchResponse = {
  handledValues: string[];
  downloads?: Array<{ value: string; path: string; kind: string; cached?: boolean }>;
  warnings: string[];
  updatedConfig: boolean;
  job?: unknown;
};

function isValidJobId(jobID: string) {
  return /^[a-zA-Z0-9_-]+$/.test(jobID);
}

function getPrefetchedFilePathByValue(downloads: ModelPrefetchResponse['downloads']) {
  return new Map<string, string>(
    (downloads || [])
      .filter(download => download.kind === 'file' && download.value && download.path)
      .map(download => [normalizeModelReferenceValue(download.value), download.path] as const),
  );
}

export async function POST(_request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  if (!isValidJobId(jobID)) {
    return NextResponse.json({ error: 'Invalid job ID' }, { status: 400 });
  }

  const job = await db.jobs.findById(jobID);
  if (!job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }
  try {
    await assertProjectJobEnabled(job);
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Project spaces are disabled' }, { status: error?.status || 403 });
  }
  if (job.job_type !== 'train') {
    return NextResponse.json({ error: 'Only training jobs can download model references' }, { status: 400 });
  }

  if (!isLocalWorker(job.worker_id)) {
    if (!job.remote_job_id) {
      return NextResponse.json({ error: 'Remote job has not been uploaded yet' }, { status: 409 });
    }

    try {
      const worker = await getRemoteWorker(job.worker_id);
      const remoteResult = await remoteJson<ModelPrefetchResponse>(
        worker,
        `/api/jobs/${encodeURIComponent(job.remote_job_id)}/prefetch-models`,
        { method: 'POST' },
      );
      const syncedJob = await syncRemoteJob(job);
      return NextResponse.json({ ...remoteResult, job: syncedJob });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to download model references on remote worker';
      return NextResponse.json({ error: message }, { status: 502 });
    }
  }

  let jobConfig: any;
  try {
    jobConfig = JSON.parse(job.job_config);
  } catch {
    return NextResponse.json({ error: 'Invalid job config' }, { status: 400 });
  }

  const modelReferences = collectModelReferences(jobConfig);
  const result = await prefetchModelReferences(modelReferences);
  const warnings = [...(result.warnings || [])];
  if (!result.handledValues.length) {
    warnings.push('No downloadable Hugging Face model references were found for this job.');
  }

  const prefetchedFilePathByValue = getPrefetchedFilePathByValue(result.downloads);
  let updatedConfig = false;
  if (!job.project_id) {
    for (const reference of modelReferences) {
      const localModelPath = prefetchedFilePathByValue.get(normalizeModelReferenceValue(reference.value));
      if (!localModelPath) continue;
      setConfigPathValue(jobConfig, reference.configPath, localModelPath);
      updatedConfig = true;
    }
  } else if (prefetchedFilePathByValue.size > 0) {
    warnings.push('Project run model references were kept as shared Hugging Face/global model references.');
  }

  const updatedJob = updatedConfig
    ? await db.jobs.update(job.id, { job_config: JSON.stringify(jobConfig) })
    : job;

  return NextResponse.json({
    handledValues: result.handledValues,
    downloads: result.downloads || [],
    warnings,
    updatedConfig,
    job: updatedJob,
  });
}
