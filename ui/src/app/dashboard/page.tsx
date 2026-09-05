'use client';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowRight, Database, Play, Box, Images, Monitor, Loader2 } from 'lucide-react';
import DashboardHardware from '@/components/DashboardHardware';
import useGPUInfo from '@/hooks/useGPUInfo';
import useMonitorStream from '@/hooks/useMonitorStream';
import useJobsList from '@/hooks/useJobsList';
import useDatasetList from '@/hooks/useDatasetList';
import { apiClient } from '@/utils/api';
import { getMediaUrl } from '@/utils/media';
import { nextHomeAction } from '@/utils/homeJourney';

export default function Dashboard() {
  const telemetry = useGPUInfo();
  const monitor = useMonitorStream();
  const { jobs, status: jobsStatus, refreshJobs } = useJobsList({ job_type: 'train', reloadInterval: 5000 });
  const {
    datasets,
    status: datasetsStatus,
    errors: datasetErrors,
    refreshDatasets,
  } = useDatasetList({ includeRemote: true });
  const [hasLoaded, setHasLoaded] = useState(false);
  useEffect(() => {
    if (jobsStatus === 'success' && datasetsStatus === 'success') setHasLoaded(true);
  }, [jobsStatus, datasetsStatus]);
  const [previews, setPreviews] = useState<string[]>([]);
  const action = nextHomeAction(jobs, datasets);
  const dataset = action.dataset;
  const initialLoading =
    !hasLoaded &&
    ((!jobs.length && ['idle', 'loading'].includes(jobsStatus)) || ['idle', 'loading'].includes(datasetsStatus));
  const unavailable =
    jobsStatus === 'error' || datasetsStatus === 'error' || (datasets.length === 0 && datasetErrors.length > 0);
  useEffect(() => {
    setPreviews([]);
    if (!dataset || dataset.encrypted) return;
    const controller = new AbortController();
    apiClient
      .post(
        '/api/datasets/listImages',
        { datasetName: dataset.name, worker_id: dataset.worker_id || 'local' },
        { signal: controller.signal },
      )
      .then(response => {
        const data: unknown = response.data;
        if (!data || typeof data !== 'object' || !('images' in data) || !Array.isArray(data.images)) return;
        const paths = data.images
          .flatMap((item: unknown) => {
            if (!item || typeof item !== 'object' || !('img_path' in item) || typeof item.img_path !== 'string')
              return [];
            return /\.(png|jpg|jpeg|webp|gif|avif)(?:$|\?)/i.test(item.img_path) ? [item.img_path] : [];
          })
          .slice(0, 3);
        if (!controller.signal.aborted) setPreviews(paths);
      })
      .catch(() => {
        /* Previews are optional; never imply that the dataset itself failed. */
      });
    return () => controller.abort();
  }, [dataset?.name, dataset?.worker_id, dataset?.encrypted]);
  return (
    <div className="journey-page">
      <header className="journey-heading">
        <div>
          <h1 className="journey-title">Your next training run</h1>
          <p className="mt-2 text-lg text-gray-400">Pick up where you left off.</p>
        </div>
        <a href="#hardware" className="journey-hardware-chip">
          <Monitor size={17} aria-hidden="true" />
          {telemetry.status === 'error'
            ? 'Hardware unavailable'
            : !telemetry.isGPUInfoLoaded
              ? 'Checking hardware…'
              : telemetry.gpuList[0]?.name || 'No GPU detected'}
        </a>
      </header>
      {datasetErrors.length > 0 && datasets.length > 0 && (
        <p role="status" className="mb-4 text-sm text-amber-400">
          Some remote datasets are unavailable. The next action uses the data that could be loaded.{' '}
          <button type="button" className="underline" onClick={() => refreshDatasets()}>
            Retry remote datasets
          </button>
        </p>
      )}
      <nav aria-label="Training journey" className="journey-steps">
        {[
          {
            id: 'prepare',
            label: 'Prepare data',
            detail: 'Review media and captions',
            href: '/datasets',
            icon: Database,
          },
          { id: 'train', label: 'Train', detail: 'Set up and monitor a run', href: '/jobs', icon: Play },
          { id: 'review', label: 'Review', detail: 'Explore your models', href: '/models', icon: Box },
        ].map(step => (
          <Link
            key={step.id}
            href={step.href}
            className={`journey-step ${!initialLoading && !unavailable && action.stage === step.id ? 'journey-step-current' : ''}`}
          >
            <span className="journey-step-icon">
              <step.icon aria-hidden="true" size={23} />
            </span>
            <span>
              <strong>{step.label}</strong>
              <small>{step.detail}</small>
            </span>
          </Link>
        ))}
      </nav>
      <section className="journey-card journey-next" aria-label="Next action" aria-busy={initialLoading}>
        {unavailable ? (
          <div role="alert">
            <h2 className="text-2xl font-semibold">Couldn’t load your workspace</h2>
            <p className="mt-3 text-gray-400">
              Run or dataset information is unavailable. Retry to see your next action.
            </p>
            <button
              className="studio-primary mt-5"
              onClick={() => {
                void refreshJobs();
                refreshDatasets();
              }}
            >
              Retry
            </button>
          </div>
        ) : initialLoading ? (
          <p role="status" className="flex items-center gap-3 text-gray-400">
            <Loader2 className="animate-spin" /> Loading your runs and datasets…
          </p>
        ) : (
          <>
            <div className="journey-preview" aria-label={previews.length ? 'Dataset preview' : undefined}>
              {previews.length ? (
                previews.map(src => <img key={src} src={getMediaUrl(src)} alt="" />)
              ) : (
                <div className="journey-preview-empty">
                  <Images size={48} strokeWidth={1} aria-hidden="true" />
                  <span>
                    {dataset?.encrypted
                      ? 'Encrypted media'
                      : action.job
                        ? `Run ${action.job.status}`
                        : 'Your data starts here'}
                  </span>
                </div>
              )}
            </div>
            <div className="min-w-0">
              <h2 className="break-words text-3xl font-semibold tracking-tight">{action.title}</h2>
              {dataset && (
                <p className="mt-3 text-gray-300">
                  {dataset.itemCount == null ? 'Media count unavailable' : `${dataset.itemCount} media items`}
                  {dataset.missingCaptionCount != null && (
                    <span className="text-[var(--studio-accent)]">
                      {' '}
                      · {dataset.missingCaptionCount} missing captions
                    </span>
                  )}
                </p>
              )}
              {action.job && (
                <p className="mt-3 capitalize text-[var(--studio-accent)]">
                  {action.job.status} · {action.job.step.toLocaleString()} steps
                </p>
              )}
              <p className="mt-4 text-gray-400">{action.description}</p>
              <div className="mt-7 flex flex-wrap items-center gap-5">
                <Link className="studio-primary" href={action.href}>
                  {action.label}
                  <ArrowRight size={19} aria-hidden="true" />
                </Link>
                {dataset && (
                  <Link className="studio-text-link" href="/datasets">
                    View datasets <ArrowRight size={17} />
                  </Link>
                )}
              </div>
            </div>
          </>
        )}
      </section>
      <section className="mt-10" aria-labelledby="recent-training-title">
        <div className="mb-4 flex items-center justify-between gap-4">
          <h2 id="recent-training-title" className="text-lg font-semibold">
            Recent training
          </h2>
          <Link className="studio-text-link text-sm" href="/jobs">
            All training <ArrowRight size={16} />
          </Link>
        </div>
        <div className="journey-card overflow-hidden">
          {jobs.length ? (
            <ul>
              {jobs.slice(0, 5).map(job => (
                <li className="journey-run" key={job.id}>
                  <div className="flex min-w-0 items-center gap-4">
                    <Box size={23} className="shrink-0 text-[var(--studio-accent)]" aria-hidden="true" />
                    <div className="min-w-0">
                      <Link href={`/jobs/${job.id}`} className="block truncate font-medium">
                        {job.name}
                      </Link>
                      <p className="mt-1 text-sm text-gray-400">{job.step.toLocaleString()} steps</p>
                    </div>
                  </div>
                  <span className="capitalize">{job.status}</span>
                  <time className="text-sm text-gray-400" dateTime={new Date(job.created_at).toISOString()}>
                    {new Date(job.created_at).toLocaleDateString()}
                  </time>
                  <Link className="studio-text-link text-sm" href={`/jobs/${job.id}`}>
                    {job.status === 'completed' ? 'View results' : 'View run'} <ArrowRight size={16} />
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <p className="p-6 text-gray-400">
              {jobsStatus === 'error'
                ? 'Run history is unavailable.'
                : initialLoading
                  ? 'Loading run history…'
                  : 'Your training runs will appear here.'}
            </p>
          )}
        </div>
      </section>
      <details id="hardware" className="journey-disclosure mt-7">
        <summary>Hardware and advanced controls</summary>
        <div className="mt-4 grid gap-5 lg:grid-cols-2">
          <DashboardHardware telemetry={telemetry} connected={monitor.connected} lastUpdated={monitor.lastUpdated} />
          <div className="p-4">
            <p className="mb-4 text-gray-400">
              Hardware availability does not guarantee that a model will fit. Review compute and storage settings before
              training.
            </p>
            <Link href="/settings" className="studio-text-link">
              Open settings <ArrowRight size={16} />
            </Link>
          </div>
        </div>
      </details>
    </div>
  );
}
