'use client';

import { use, useEffect, useState } from 'react';
import Link from 'next/link';
import { Database, FileText, Folder, Image as ImageIcon, Loader2, Play, Plus, Sparkles } from 'lucide-react';
import ProjectWorkspaceShell, { formatBytes, formatProjectTime } from '@/components/project/ProjectWorkspaceShell';
import { PageNotice, ProgressBar, StatusBadge } from '@/components/OperatorPrimitives';
import { apiClient } from '@/utils/api';
import type { ProjectSummary } from '@/components/project/types';

function jobProgress(summary: ProjectSummary | null) {
  const job = summary?.activeJob;
  if (!job?.total_steps) return 0;
  return Math.max(0, Math.min(100, (job.step / job.total_steps) * 100));
}

export default function ProjectWorkspacePage({ params }: { params: Promise<{ projectID: string }> }) {
  const { projectID: rawProjectID } = use(params);
  const projectID = decodeURIComponent(rawProjectID);
  const projectPath = `/projects/${encodeURIComponent(projectID)}`;
  const [summary, setSummary] = useState<ProjectSummary | null>(null);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      apiClient
        .get(`/api/projects/${encodeURIComponent(projectID)}/summary`)
        .then(res => {
          if (cancelled) return;
          setSummary(res.data);
          setStatus('success');
        })
        .catch(error => {
          if (cancelled) return;
          console.error('Failed to load project summary:', error);
          setStatus('error');
        });
    };
    setStatus('loading');
    load();
    const interval = window.setInterval(load, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [projectID]);

  const activeJob = summary?.activeJob || null;

  return (
    <ProjectWorkspaceShell projectID={projectID} active="workspace" description="Inputs, active run, and outputs for this project.">
      <div className="h-full overflow-auto p-4">
        <div className="mx-auto grid max-w-[1800px] gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
          <section className="min-w-0 space-y-4">
            {status === 'loading' && !summary && (
              <div className="flex h-64 items-center justify-center border border-gray-800 bg-gray-900/40 text-sm text-gray-400">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading Mission Control
              </div>
            )}

            {status === 'error' && (
              <PageNotice tone="danger" title="Project could not be loaded">
                The project may have been deleted, or the project database is unavailable.
              </PageNotice>
            )}

            <div className="grid min-h-[440px] grid-cols-1 border border-gray-800 bg-gray-950 lg:grid-cols-3">
              <section className="min-w-0 border-b border-gray-800 lg:border-b-0 lg:border-r">
                <div className="flex h-12 items-center justify-between gap-2 border-b border-gray-800 px-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <Database className="h-4 w-4 text-cyan-300" />
                    <h2 className="truncate text-sm font-semibold">Inputs</h2>
                  </div>
                  <Link href={`${projectPath}/datasets`} className="operator-button h-8 py-1 text-xs">
                    Open
                  </Link>
                </div>
                <div className="divide-y divide-gray-800">
                  {(summary?.datasets || []).slice(0, 8).map(dataset => (
                    <Link
                      key={dataset.name}
                      href={`${projectPath}/datasets/${encodeURIComponent(dataset.name)}`}
                      className="flex min-w-0 items-center gap-3 px-3 py-3 hover:bg-gray-900/70"
                    >
                      <span className="flex h-8 w-8 flex-none items-center justify-center rounded-sm border border-gray-800 bg-gray-900">
                        <Database className="h-4 w-4 text-cyan-200" />
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-medium text-gray-100">{dataset.name}</div>
                        <div className="text-xs text-gray-500">
                          {dataset.encrypted ? 'Encrypted' : `${dataset.itemCount ?? 0} media`} - captions{' '}
                          {dataset.missingCaptionCount ? `${dataset.missingCaptionCount} missing` : 'ready'}
                        </div>
                      </div>
                    </Link>
                  ))}
                  {summary && summary.datasets.length === 0 && (
                    <div className="flex min-h-[300px] items-center justify-center px-6 text-center">
                      <div>
                        <Database className="mx-auto h-9 w-9 text-cyan-300" />
                        <div className="mt-3 text-sm font-semibold text-gray-200">No project datasets yet</div>
                        <div className="mt-1 text-sm text-gray-500">Create or upload a dataset inside this project.</div>
                        <Link href={`${projectPath}/datasets`} className="operator-button mt-4 h-8 py-1 text-xs">
                          <Plus className="h-4 w-4" />
                          Add Dataset
                        </Link>
                      </div>
                    </div>
                  )}
                </div>
              </section>

              <section className="min-w-0 border-b border-gray-800 lg:border-b-0 lg:border-r">
                <div className="flex h-12 items-center justify-between gap-2 border-b border-gray-800 px-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <Play className="h-4 w-4 text-emerald-300" />
                    <h2 className="truncate text-sm font-semibold">Active Run</h2>
                  </div>
                  <Link href={`${projectPath}/runs`} className="operator-button h-8 py-1 text-xs">
                    Runs
                  </Link>
                </div>
                {activeJob ? (
                  <div className="space-y-3 p-3">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="truncate text-sm font-semibold text-gray-100">{activeJob.name}</div>
                        <div className="text-xs text-gray-500">
                          Step {activeJob.step} {activeJob.total_steps ? `/ ${activeJob.total_steps}` : ''}
                        </div>
                      </div>
                      <StatusBadge status={activeJob.status} />
                    </div>
                    <ProgressBar value={jobProgress(summary)} tone="info" />
                    <div className="border border-gray-800 bg-gray-900/60 px-3 py-2 text-xs text-gray-400">
                      {activeJob.info || 'Run is waiting for the next worker update.'}
                    </div>
                    <Link href={`${projectPath}/runs/${encodeURIComponent(activeJob.id)}`} className="operator-button h-8 py-1 text-xs">
                      Open Run
                    </Link>
                  </div>
                ) : (
                  <div className="flex min-h-[300px] items-center justify-center px-6 text-center">
                    <div>
                      <Play className="mx-auto h-9 w-9 text-emerald-300" />
                      <div className="mt-3 text-sm font-semibold text-gray-200">No active run</div>
                      <div className="mt-1 text-sm text-gray-500">Start training or generation scoped to this project.</div>
                      <Link href={`/jobs/new?project_id=${encodeURIComponent(projectID)}`} className="operator-button mt-4 h-8 py-1 text-xs">
                        <Plus className="h-4 w-4" />
                        New Run
                      </Link>
                    </div>
                  </div>
                )}
              </section>

              <section className="min-w-0">
                <div className="flex h-12 items-center justify-between gap-2 border-b border-gray-800 px-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <ImageIcon className="h-4 w-4 text-violet-300" />
                    <h2 className="truncate text-sm font-semibold">Outputs</h2>
                  </div>
                  <Link href={`${projectPath}/generate`} className="operator-button h-8 py-1 text-xs">
                    Generate
                  </Link>
                </div>
                <div className="divide-y divide-gray-800">
                  {(summary?.zones.outputs.recent || []).slice(0, 8).map(item => (
                    <Link key={item.path} href={`${projectPath}/files?path=${encodeURIComponent(item.path)}`} className="flex min-w-0 items-center gap-3 px-3 py-3 hover:bg-gray-900/70">
                      <span className="flex h-8 w-8 flex-none items-center justify-center rounded-sm border border-gray-800 bg-gray-900">
                        {item.kind === 'folder' ? <Folder className="h-4 w-4 text-violet-200" /> : <ImageIcon className="h-4 w-4 text-violet-200" />}
                      </span>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm text-gray-100">{item.name}</div>
                        <div className="text-xs text-gray-500">
                          {item.kind} - {formatBytes(item.size)} - {formatProjectTime(item.updatedAt)}
                        </div>
                      </div>
                    </Link>
                  ))}
                  {summary && summary.zones.outputs.recent.length === 0 && (
                    <div className="flex min-h-[300px] items-center justify-center px-6 text-center">
                      <div>
                        <Sparkles className="mx-auto h-9 w-9 text-violet-300" />
                        <div className="mt-3 text-sm font-semibold text-gray-200">No outputs yet</div>
                        <div className="mt-1 text-sm text-gray-500">Samples, checkpoints, and generated files collect here.</div>
                        <Link href={`${projectPath}/generate`} className="operator-button mt-4 h-8 py-1 text-xs">
                          Generate
                        </Link>
                      </div>
                    </div>
                  )}
                </div>
              </section>
            </div>

            <section className="border border-gray-800 bg-gray-950">
              <div className="flex h-12 items-center gap-2 border-b border-gray-800 px-3">
                <FileText className="h-4 w-4 text-cyan-300" />
                <h2 className="text-sm font-semibold">Timeline</h2>
              </div>
              <div className="divide-y divide-gray-800">
                {(summary?.recentActivity || []).slice(0, 8).map(item => (
                  <div key={item.id} className="grid grid-cols-[112px_minmax(0,1fr)] gap-3 px-3 py-2 text-sm">
                    <div className="text-xs text-gray-500">{formatProjectTime(item.updatedAt)}</div>
                    <div className="min-w-0">
                      <div className="truncate text-gray-200">{item.label}</div>
                      <div className="truncate text-xs text-gray-500">{item.detail}</div>
                    </div>
                  </div>
                ))}
                {summary && summary.recentActivity.length === 0 && (
                  <div className="px-3 py-6 text-sm text-gray-500">Project activity will appear here as you add inputs and run jobs.</div>
                )}
              </div>
            </section>
          </section>

          <aside className="min-w-0 border border-gray-800 bg-gray-950 xl:sticky xl:top-4 xl:h-[calc(100vh-6rem)]">
            <div className="flex h-12 items-center justify-between gap-2 border-b border-gray-800 px-3">
              <div className="flex min-w-0 items-center gap-2">
                <Folder className="h-4 w-4 text-cyan-300" />
                <h2 className="truncate text-sm font-semibold">Project Files</h2>
              </div>
              <Link href={`${projectPath}/files`} className="text-xs text-cyan-300 hover:text-cyan-200">
                Browse
              </Link>
            </div>
            <div className="divide-y divide-gray-900">
              {(summary?.fileTree || []).slice(0, 120).map(item => (
                <Link key={item.relativePath} href={`${projectPath}/files?path=${encodeURIComponent(item.path)}`} className="flex min-w-0 items-center gap-2 px-3 py-2 text-xs hover:bg-gray-900/70">
                  {item.kind === 'folder' ? <Folder className="h-3.5 w-3.5 flex-none text-cyan-300" /> : <FileText className="h-3.5 w-3.5 flex-none text-gray-500" />}
                  <span className="min-w-0 flex-1 truncate text-gray-300">{item.relativePath}</span>
                  {item.kind === 'file' && <span className="text-gray-600">{formatBytes(item.size)}</span>}
                </Link>
              ))}
              {summary && summary.fileTree.length === 0 && (
                <div className="px-3 py-8 text-sm text-gray-500">Project files will appear here.</div>
              )}
            </div>
          </aside>
        </div>
      </div>
    </ProjectWorkspaceShell>
  );
}
