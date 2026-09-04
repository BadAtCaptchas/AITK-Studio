'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import classNames from 'classnames';
import {
  ArrowRight,
  Boxes,
  Check,
  CheckCircle2,
  Clock3,
  Database,
  FileText,
  Image as ImageIcon,
  Loader2,
  LockKeyhole,
  Play,
  Plus,
  RefreshCcw,
  Sparkles,
  TriangleAlert,
} from 'lucide-react';
import ProjectWorkspaceShell, { formatBytes, formatProjectTime } from '@/components/project/ProjectWorkspaceShell';
import { useProjectWorkspace } from '@/components/project/ProjectContext';
import { artifactPreviewUrl, loadProjectOverview } from '@/components/project/data';
import { PageNotice, ProgressBar, StatusBadge } from '@/components/OperatorPrimitives';
import useJobsList from '@/hooks/useJobsList';
import { getTotalSteps, startJob } from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import type { ProjectArtifact, ProjectOverview } from '@/components/project/types';
import type { Job } from '@/types';

function activeStatus(status: string) {
  return ['queued', 'starting', 'running', 'stopping'].includes(status);
}

function canResumeRun(job: Job | null) {
  if (!job) return false;
  if (['stopped', 'error'].includes(job.status)) return true;
  if (job.status !== 'completed' || job.job_type !== 'train') return false;
  try {
    const totalSteps = getTotalSteps(job);
    return totalSteps !== null && totalSteps > job.step;
  } catch {
    return false;
  }
}

function runProgress(job: Job | null) {
  if (!job || job.job_type !== 'train') return 0;
  const total = getTotalSteps(job);
  return total ? Math.max(0, Math.min(100, (job.step / total) * 100)) : 0;
}

function ArtifactPreview({ artifact }: { artifact: ProjectArtifact }) {
  const source = artifactPreviewUrl(artifact);
  return (
    <div className="relative aspect-square overflow-hidden rounded-sm border border-gray-800 bg-gray-900">
      {source && artifact.mediaKind === 'image' ? (
        <img
          src={source}
          alt={artifact.name}
          loading="lazy"
          className="h-full w-full object-cover"
          onError={event => {
            event.currentTarget.style.display = 'none';
          }}
        />
      ) : source && artifact.mediaKind === 'video' ? (
        <video src={source} muted preload="metadata" className="h-full w-full object-cover" />
      ) : (
        <div className="flex h-full items-center justify-center">
          <ImageIcon className="h-6 w-6 text-gray-700" />
        </div>
      )}
      <div className="absolute inset-x-0 bottom-0 truncate bg-gradient-to-t from-gray-950 via-gray-950/75 to-transparent px-2 pb-1.5 pt-5 text-[10px] text-gray-300">
        {artifact.name}
      </div>
    </div>
  );
}

function StageHeader({
  number,
  title,
  label,
  state,
}: {
  number: number;
  title: string;
  label: string;
  state: 'complete' | 'active' | 'pending' | 'warning';
}) {
  return (
    <div className="flex min-w-0 items-start gap-3">
      <span
        className={classNames(
          'flex h-8 w-8 flex-none items-center justify-center rounded-full border text-xs font-semibold',
          state === 'complete'
            ? 'border-emerald-700 bg-emerald-950/60 text-emerald-200'
            : state === 'active'
              ? 'border-brand-700 bg-brand-950/60 text-brand-200'
              : state === 'warning'
                ? 'border-amber-700 bg-amber-950/60 text-amber-200'
                : 'border-gray-700 bg-gray-900 text-gray-500',
        )}
      >
        {state === 'complete' ? <Check className="h-4 w-4" /> : number}
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-600">{label}</div>
        <h2 className="mt-0.5 text-sm font-semibold text-gray-100">{title}</h2>
      </div>
    </div>
  );
}

export default function ProjectWorkspacePage() {
  const { projectID, identity, archived } = useProjectWorkspace();
  const basePath = `/projects/${encodeURIComponent(projectID)}`;
  const [overview, setOverview] = useState<ProjectOverview | null>(null);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [resumeStatus, setResumeStatus] = useState<'idle' | 'resuming' | 'error'>('idle');
  const [resumeError, setResumeError] = useState('');
  const { jobs, status: jobsStatus, refreshJobs } = useJobsList({ projectID, reloadInterval: 5000 });

  const refreshOverview = () => {
    setStatus(current => (overview ? current : 'loading'));
    loadProjectOverview(projectID)
      .then(next => {
        setOverview(next);
        setStatus('success');
      })
      .catch(error => {
        console.error('Failed to load project overview:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    refreshOverview();
  }, [projectID]);

  const sortedJobs = useMemo(
    () => [...jobs].sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()),
    [jobs],
  );
  const activeJob = sortedJobs.find(job => activeStatus(job.status)) || null;
  const latestJob = activeJob || sortedJobs[0] || null;
  const resumableRun = !archived && !activeJob && canResumeRun(latestJob) ? latestJob : null;
  const datasets = overview?.datasets || [];
  const itemCount = datasets.reduce((total, dataset) => total + (dataset.itemCount || 0), 0);
  const missingCaptions = datasets.reduce((total, dataset) => total + (dataset.missingCaptionCount || 0), 0);
  const captionedCount = datasets.reduce((total, dataset) => total + (dataset.captionedItemCount || 0), 0);
  const captionCoverage = itemCount > 0 ? Math.round((captionedCount / itemCount) * 100) : 0;
  const datasetReady = datasets.length > 0 && missingCaptions === 0;
  const runFailed = latestJob && ['error', 'failed'].includes(latestJob.status);
  const outputReady = (overview?.recentOutputs.length || 0) > 0;
  const modelReady = (overview?.counts.models || 0) > 0;
  const projectName = identity?.project.name || overview?.project.name || 'Project';

  const resumeLatestRun = async () => {
    if (!resumableRun || resumeStatus === 'resuming') return;
    setResumeStatus('resuming');
    setResumeError('');
    try {
      await startJob(resumableRun.id, undefined, { background: resumableRun.worker_id !== 'local' });
      await startQueue(resumableRun.gpu_ids, resumableRun.worker_id);
      refreshJobs();
      refreshOverview();
      setResumeStatus('idle');
    } catch (error) {
      setResumeError(error instanceof Error ? error.message : 'Failed to resume this run.');
      setResumeStatus('error');
    }
  };

  return (
    <ProjectWorkspaceShell projectID={projectID} active="workspace" showHeader={false}>
      <div className="h-full overflow-auto">
        <div className="mx-auto max-w-[1680px] space-y-5 p-3 sm:p-5 lg:p-6">
          <section className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="min-w-0">
              <div className="text-[10px] font-semibold uppercase tracking-[0.22em] text-brand-400">
                Project overview
              </div>
              <h1 className="mt-3 text-3xl font-semibold tracking-[-0.04em] text-gray-100 sm:text-4xl">
                Move {projectName} through the studio
              </h1>
              <p className="mt-1.5 max-w-2xl text-sm text-gray-500">
                Readiness is advisory: you can enter any stage whenever the project needs it.
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                refreshOverview();
                refreshJobs();
              }}
              disabled={status === 'loading' || jobsStatus === 'loading'}
              className="operator-button h-9 self-start lg:self-auto"
            >
              <RefreshCcw
                className={classNames(
                  'h-4 w-4',
                  status === 'loading' || jobsStatus === 'loading' ? 'animate-spin' : '',
                )}
              />
              Refresh
            </button>
          </section>

          {archived ? (
            <PageNotice tone="warning" title="This project is archived">
              You can browse and download its datasets, runs, outputs, models, and files. Restore it from Settings
              before creating or changing content.
            </PageNotice>
          ) : null}

          {identity?.project.operation_error ? (
            <PageNotice tone="warning" title="Project setup needs attention">
              {identity.project.operation_error} The project identity is safe and usable; retry the import or add files
              from the project workspace.
            </PageNotice>
          ) : null}

          {status === 'error' ? (
            <PageNotice tone="danger" title="Overview data could not be loaded">
              The project identity is available, but its workspace summary or artifacts could not be read.
            </PageNotice>
          ) : null}

          {status === 'loading' && !overview ? (
            <div className="flex min-h-96 items-center justify-center rounded-md border border-gray-800 bg-gray-900/30 text-sm text-gray-500">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Preparing project overview
            </div>
          ) : (
            <>
              <section className="relative grid overflow-hidden rounded-md border border-gray-800 bg-gray-950 lg:grid-cols-3">
                <article className="relative flex min-h-[370px] min-w-0 flex-col border-b border-gray-800 p-4 lg:border-b-0 lg:border-r">
                  <StageHeader
                    number={1}
                    label="Prepare"
                    title="Prepare Dataset"
                    state={datasetReady ? 'complete' : datasets.length > 0 ? 'active' : 'pending'}
                  />
                  <div className="mt-5 grid grid-cols-3 gap-2">
                    <div className="rounded-sm border border-gray-800 bg-gray-900/50 p-2.5">
                      <Database className="h-4 w-4 text-brand-400" />
                      <div className="mt-2 text-lg font-semibold text-gray-100">{datasets.length}</div>
                      <div className="text-[10px] uppercase tracking-wide text-gray-600">Datasets</div>
                    </div>
                    <div className="rounded-sm border border-gray-800 bg-gray-900/50 p-2.5">
                      <ImageIcon className="h-4 w-4 text-gray-500" />
                      <div className="mt-2 text-lg font-semibold text-gray-100">{itemCount}</div>
                      <div className="text-[10px] uppercase tracking-wide text-gray-600">Media</div>
                    </div>
                    <div className="rounded-sm border border-gray-800 bg-gray-900/50 p-2.5">
                      {missingCaptions > 0 ? (
                        <TriangleAlert className="h-4 w-4 text-amber-400" />
                      ) : (
                        <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                      )}
                      <div className="mt-2 text-lg font-semibold text-gray-100">{captionCoverage}%</div>
                      <div className="text-[10px] uppercase tracking-wide text-gray-600">Captioned</div>
                    </div>
                  </div>
                  <div className="mt-4 space-y-2">
                    {datasets.slice(0, 3).map(dataset => (
                      <Link
                        key={dataset.name}
                        href={`${basePath}/datasets/${encodeURIComponent(dataset.name)}`}
                        className="flex min-w-0 items-center gap-2 rounded-sm border border-gray-800 bg-gray-900/25 px-2.5 py-2 hover:bg-gray-900/70"
                      >
                        {dataset.encrypted ? (
                          <LockKeyhole className="h-3.5 w-3.5 flex-none text-violet-300" />
                        ) : (
                          <Database className="h-3.5 w-3.5 flex-none text-brand-300" />
                        )}
                        <span className="min-w-0 flex-1 truncate text-xs text-gray-300">{dataset.name}</span>
                        <span
                          className={classNames(
                            'text-[10px]',
                            (dataset.missingCaptionCount || 0) > 0 ? 'text-amber-300' : 'text-emerald-300',
                          )}
                        >
                          {(dataset.missingCaptionCount || 0) > 0 ? `${dataset.missingCaptionCount} missing` : 'Ready'}
                        </span>
                      </Link>
                    ))}
                    {datasets.length === 0 ? (
                      <div className="rounded-sm border border-dashed border-gray-800 p-4 text-center text-xs text-gray-600">
                        Add or import a dataset to begin.
                      </div>
                    ) : null}
                  </div>
                  <Link
                    href={`${basePath}/datasets`}
                    className="operator-button mt-auto h-9 w-full border-brand-900 bg-brand-950/30 text-brand-100 hover:bg-brand-900/40"
                  >
                    {datasets.length > 0 ? 'Open datasets' : 'Add dataset'} <ArrowRight className="h-4 w-4" />
                  </Link>
                </article>

                <article className="relative flex min-h-[370px] min-w-0 flex-col border-b border-gray-800 p-4 lg:border-b-0 lg:border-r">
                  <StageHeader
                    number={2}
                    label="Train"
                    title="Train Model"
                    state={
                      runFailed ? 'warning' : activeJob ? 'active' : sortedJobs.length > 0 ? 'complete' : 'pending'
                    }
                  />
                  {latestJob ? (
                    <div className="mt-5 flex flex-1 flex-col">
                      <div className="rounded-sm border border-gray-800 bg-gray-900/45 p-3">
                        <div className="flex min-w-0 items-start justify-between gap-3">
                          <div className="min-w-0">
                            <div className="truncate text-sm font-semibold text-gray-100">{latestJob.name}</div>
                            <div className="mt-1 flex min-w-0 items-center gap-1.5 text-xs text-gray-500">
                              <span className="capitalize">{latestJob.job_type}</span>
                              <span>/</span>
                              <span className="truncate">
                                {latestJob.worker_id === 'local' ? 'Local worker' : latestJob.worker_id}
                              </span>
                            </div>
                          </div>
                          <StatusBadge status={latestJob.status} />
                        </div>
                        {latestJob.job_type === 'train' ? (
                          <div className="mt-4">
                            <div className="mb-1.5 flex items-center justify-between text-[10px] text-gray-500">
                              <span>Step {latestJob.step.toLocaleString()}</span>
                              <span>{Math.round(runProgress(latestJob))}%</span>
                            </div>
                            <ProgressBar
                              value={runProgress(latestJob)}
                              tone={runFailed ? 'danger' : activeJob ? 'info' : 'success'}
                            />
                          </div>
                        ) : null}
                        <div className="mt-3 line-clamp-2 min-h-8 text-xs leading-relaxed text-gray-500">
                          {latestJob.info ||
                            (activeJob
                              ? 'Waiting for the next worker update.'
                              : `Updated ${formatProjectTime(latestJob.updated_at)}`)}
                        </div>
                      </div>
                      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                        <div className="rounded-sm border border-gray-800 bg-gray-900/25 p-2.5 text-gray-500">
                          <Play className="mb-1.5 h-4 w-4 text-emerald-400" />
                          <strong className="block text-sm font-semibold text-gray-200">
                            {sortedJobs.length}
                          </strong>{' '}
                          total runs
                        </div>
                        <div className="rounded-sm border border-gray-800 bg-gray-900/25 p-2.5 text-gray-500">
                          <Clock3 className="mb-1.5 h-4 w-4 text-brand-400" />
                          <strong className="block truncate text-sm font-semibold text-gray-200">
                            {latestJob.speed_string || '—'}
                          </strong>{' '}
                          current speed
                        </div>
                      </div>
                      {resumeError ? (
                        <div className="mt-3 rounded-sm border border-rose-900 bg-rose-950/30 px-3 py-2 text-xs text-rose-200">
                          {resumeError}
                        </div>
                      ) : null}
                      {activeJob ? (
                        <Link
                          href={`${basePath}/runs/${encodeURIComponent(activeJob.id)}`}
                          className="operator-button mt-auto h-9 w-full border-emerald-900 bg-emerald-950/30 text-emerald-100 hover:bg-emerald-900/40"
                        >
                          Open active run <ArrowRight className="h-4 w-4" />
                        </Link>
                      ) : resumableRun ? (
                        <button
                          type="button"
                          onClick={() => void resumeLatestRun()}
                          disabled={resumeStatus === 'resuming'}
                          className="operator-button mt-auto h-9 w-full border-emerald-800 bg-emerald-950/50 text-emerald-100 hover:bg-emerald-900/50 disabled:cursor-wait disabled:opacity-70"
                        >
                          {resumeStatus === 'resuming' ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Play className="h-4 w-4" />
                          )}
                          {resumeStatus === 'resuming' ? 'Resuming run…' : 'Resume run'}
                        </button>
                      ) : archived && latestJob ? (
                        <Link
                          href={`${basePath}/runs/${encodeURIComponent(latestJob.id)}`}
                          className="operator-button mt-auto h-9 w-full"
                        >
                          View run <ArrowRight className="h-4 w-4" />
                        </Link>
                      ) : (
                        <Link
                          href={`${basePath}/runs/new`}
                          className="operator-button mt-auto h-9 w-full border-emerald-900 bg-emerald-950/30 text-emerald-100 hover:bg-emerald-900/40"
                        >
                          Start another run <ArrowRight className="h-4 w-4" />
                        </Link>
                      )}
                    </div>
                  ) : (
                    <div className="mt-5 flex flex-1 flex-col items-center justify-center rounded-sm border border-dashed border-gray-800 px-5 text-center">
                      <Play className="h-9 w-9 text-emerald-900" />
                      <div className="mt-3 text-sm font-semibold text-gray-300">No runs yet</div>
                      <p className="mt-1 text-xs leading-relaxed text-gray-600">
                        Choose a training configuration when your inputs are ready.
                      </p>
                      {!archived ? (
                        <Link
                          href={`${basePath}/runs/new`}
                          className="operator-button mt-4 h-9 border-emerald-900 bg-emerald-950/30 text-emerald-100"
                        >
                          <Plus className="h-4 w-4" /> New run
                        </Link>
                      ) : null}
                    </div>
                  )}
                </article>

                <article className="relative flex min-h-[370px] min-w-0 flex-col p-4">
                  <StageHeader
                    number={3}
                    label="Review"
                    title="Review Output"
                    state={outputReady || modelReady ? 'complete' : activeJob ? 'active' : 'pending'}
                  />
                  <div className="mt-5 grid grid-cols-2 gap-2">
                    {(overview?.recentOutputs || []).slice(0, 4).map(artifact => (
                      <Link key={artifact.id} href={`${basePath}/outputs?selected=${encodeURIComponent(artifact.id)}`}>
                        <ArtifactPreview artifact={artifact} />
                      </Link>
                    ))}
                    {!outputReady ? (
                      <div className="col-span-2 flex min-h-40 items-center justify-center rounded-sm border border-dashed border-gray-800 px-5 text-center">
                        <div>
                          <Sparkles className="mx-auto h-8 w-8 text-violet-900" />
                          <div className="mt-2 text-sm font-semibold text-gray-300">Outputs collect here</div>
                          <p className="mt-1 text-xs text-gray-600">
                            Training samples and generated media appear automatically.
                          </p>
                        </div>
                      </div>
                    ) : null}
                  </div>
                  <div className="mt-3 flex items-center justify-between rounded-sm border border-gray-800 bg-gray-900/30 px-3 py-2 text-xs">
                    <span className="inline-flex items-center gap-2 text-gray-400">
                      <Boxes className="h-4 w-4 text-violet-300" /> Model readiness
                    </span>
                    <span className={modelReady ? 'text-emerald-300' : 'text-gray-600'}>
                      {modelReady ? `${overview?.counts.models} available` : 'No model yet'}
                    </span>
                  </div>
                  <Link
                    href={`${basePath}/outputs`}
                    className="operator-button mt-auto h-9 w-full border-violet-900 bg-violet-950/30 text-violet-100 hover:bg-violet-900/40"
                  >
                    Review outputs <ArrowRight className="h-4 w-4" />
                  </Link>
                </article>
              </section>

              <section className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(300px,0.7fr)]">
                <div className="min-w-0 rounded-md border border-gray-800 bg-gray-900/30">
                  <div className="flex items-center justify-between gap-3 border-b border-gray-800 px-3 py-2.5 sm:px-4">
                    <div className="flex min-w-0 items-center gap-2">
                      <Play className="h-4 w-4 text-emerald-300" />
                      <h2 className="text-sm font-semibold text-gray-200">Recent runs</h2>
                    </div>
                    <Link href={`${basePath}/runs`} className="text-xs font-medium text-brand-400 hover:text-brand-300">
                      View all
                    </Link>
                  </div>
                  <div className="divide-y divide-gray-800/80">
                    {sortedJobs.slice(0, 5).map(job => (
                      <Link
                        key={job.id}
                        href={`${basePath}/runs/${encodeURIComponent(job.id)}`}
                        className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] gap-3 px-3 py-3 hover:bg-gray-900/70 sm:grid-cols-[minmax(0,1fr)_110px_120px_auto] sm:px-4"
                      >
                        <div className="min-w-0">
                          <div className="truncate text-sm font-medium text-gray-200">{job.name}</div>
                          <div className="mt-0.5 truncate text-xs text-gray-600">{job.info || job.job_type}</div>
                        </div>
                        <div className="hidden self-center text-xs text-gray-500 sm:block">
                          {job.worker_id === 'local' ? 'Local' : 'Remote'}
                        </div>
                        <div className="hidden self-center text-xs text-gray-500 sm:block">
                          {formatProjectTime(job.updated_at)}
                        </div>
                        <StatusBadge status={job.status} className="self-center" />
                      </Link>
                    ))}
                    {sortedJobs.length === 0 ? (
                      <div className="px-4 py-8 text-center text-xs text-gray-600">Project runs will appear here.</div>
                    ) : null}
                  </div>
                </div>

                <div className="min-w-0 rounded-md border border-gray-800 bg-gray-900/30">
                  <div className="flex items-center gap-2 border-b border-gray-800 px-3 py-2.5 sm:px-4">
                    <FileText className="h-4 w-4 text-brand-300" />
                    <h2 className="text-sm font-semibold text-gray-200">Recent activity</h2>
                  </div>
                  <div className="divide-y divide-gray-800/80">
                    {(overview?.recentActivity || []).slice(0, 6).map(item => (
                      <div key={item.id} className="flex min-w-0 gap-3 px-3 py-3 sm:px-4">
                        <span className="mt-1.5 h-1.5 w-1.5 flex-none rounded-full bg-brand-600" />
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-xs font-medium text-gray-300">{item.label}</div>
                          <div className="mt-0.5 truncate text-[11px] text-gray-600">{item.detail}</div>
                        </div>
                        <span className="flex-none text-[10px] text-gray-700">{formatProjectTime(item.updatedAt)}</span>
                      </div>
                    ))}
                    {(overview?.recentActivity.length || 0) === 0 ? (
                      <div className="px-4 py-8 text-center text-xs text-gray-600">
                        Activity will appear as the project changes.
                      </div>
                    ) : null}
                  </div>
                </div>
              </section>

              <section className="flex flex-col gap-3 rounded-md border border-gray-800 bg-gradient-to-r from-brand-950/20 via-gray-950 to-violet-950/20 p-4 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <div className="text-sm font-semibold text-gray-200">Continue with this project</div>
                  <p className="mt-1 text-xs text-gray-500">
                    Generate a test image from a ready model, or inspect every project-owned file.
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Link href={`${basePath}/models`} className="operator-button h-9">
                    <Boxes className="h-4 w-4" /> Models
                  </Link>
                  {!archived ? (
                    <Link
                      href={`${basePath}/generate`}
                      className="operator-button h-9 border-violet-900 bg-violet-950/40 text-violet-100"
                    >
                      <Sparkles className="h-4 w-4" /> Generate
                    </Link>
                  ) : null}
                </div>
              </section>
            </>
          )}
        </div>
      </div>
    </ProjectWorkspaceShell>
  );
}
