'use client';

import { use, useEffect, useMemo, useState } from 'react';
import { BarChart3, Code2, Image as ImageIcon, LayoutDashboard, Loader2 } from 'lucide-react';
import ProjectWorkspaceShell from '@/components/project/ProjectWorkspaceShell';
import JobActionBar from '@/components/JobActionBar';
import JobConfigViewer from '@/components/JobConfigViewer';
import JobLossGraph from '@/components/JobLossGraph';
import JobOverview from '@/components/JobOverview';
import SampleImages, { SampleImagesMenu } from '@/components/SampleImages';
import { PageNotice, StatusBadge } from '@/components/OperatorPrimitives';
import useJob from '@/hooks/useJob';

type TabKey = 'overview' | 'samples' | 'loss' | 'config';

const tabs: Array<{ key: TabKey; label: string; icon: typeof LayoutDashboard; jobTypes?: string[] }> = [
  { key: 'overview', label: 'Overview', icon: LayoutDashboard },
  { key: 'samples', label: 'Samples', icon: ImageIcon, jobTypes: ['train', 'generate'] },
  { key: 'loss', label: 'Training Monitor', icon: BarChart3, jobTypes: ['train'] },
  { key: 'config', label: 'Config', icon: Code2 },
];

function shouldPoll(status?: string) {
  return !status || status === 'queued' || status === 'running' || status === 'stopping';
}

export default function ProjectRunDetailPage({
  params,
}: {
  params: Promise<{ projectID: string; jobID: string }>;
}) {
  const { projectID: rawProjectID, jobID: rawJobID } = use(params);
  const projectID = decodeURIComponent(rawProjectID);
  const jobID = decodeURIComponent(rawJobID);
  const [reloadInterval, setReloadInterval] = useState<number | null>(5000);
  const [activeTab, setActiveTab] = useState<TabKey>('overview');
  const { job, status, refreshJob } = useJob(jobID, reloadInterval);

  useEffect(() => {
    setReloadInterval(current => {
      const next = shouldPoll(job?.status) ? 5000 : null;
      return current === next ? current : next;
    });
  }, [job?.status]);

  const visibleTabs = useMemo(
    () => tabs.filter(tab => !tab.jobTypes || (job?.job_type && tab.jobTypes.includes(job.job_type))),
    [job?.job_type],
  );

  useEffect(() => {
    if (!visibleTabs.some(tab => tab.key === activeTab)) {
      setActiveTab('overview');
    }
  }, [activeTab, visibleTabs]);

  const title = job ? `${job.job_type === 'generate' ? 'Generate' : job.job_type === 'caption' ? 'Caption' : 'Train'} - ${job.name}` : 'Run';

  return (
    <ProjectWorkspaceShell projectID={projectID} active="runs" title={title} description="Project run detail and outputs.">
      <div className="h-full overflow-auto p-4">
        <div className="mx-auto max-w-[1500px] space-y-4">
          {status === 'loading' && !job && (
            <div className="flex h-64 items-center justify-center border border-gray-800 bg-gray-900/40 text-sm text-gray-400">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Loading run
            </div>
          )}

          {status === 'error' && (
            <PageNotice tone="danger" title="Run could not be loaded">
              The job may have been deleted, or the project database is unavailable.
            </PageNotice>
          )}

          {job && job.project_id !== projectID && (
            <PageNotice tone="warning" title="Run is not attached to this project">
              This job does not match the current project. Open it from the global Jobs page if needed.
            </PageNotice>
          )}

          {job && (
            <>
              <section className="border border-gray-800 bg-gray-950">
                <div className="flex flex-col gap-3 border-b border-gray-800 px-3 py-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0">
                    <div className="flex min-w-0 items-center gap-2">
                      <h1 className="truncate text-base font-semibold text-gray-100">{title}</h1>
                      <StatusBadge status={job.status} />
                    </div>
                    <div className="mt-1 truncate text-xs text-gray-500">{job.info || job.job_ref || job.id}</div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <SampleImagesMenu job={job} />
                    <JobActionBar job={job} onRefresh={refreshJob} autoStartQueue={false} />
                  </div>
                </div>
                <div className="flex flex-wrap gap-1 border-b border-gray-800 px-3 py-2">
                  {visibleTabs.map(tab => {
                    const Icon = tab.icon;
                    const active = activeTab === tab.key;
                    return (
                      <button
                        key={tab.key}
                        type="button"
                        onClick={() => setActiveTab(tab.key)}
                        className={`inline-flex h-9 items-center gap-2 border px-3 text-sm ${
                          active
                            ? 'border-cyan-800 bg-cyan-950/40 text-cyan-100'
                            : 'border-gray-800 bg-gray-950 text-gray-400 hover:text-gray-100'
                        }`}
                      >
                        <Icon className="h-4 w-4" />
                        {tab.label}
                      </button>
                    );
                  })}
                </div>
              </section>

              <section className="min-h-[520px] border border-gray-800 bg-gray-950">
                {activeTab === 'overview' && <JobOverview job={job} />}
                {activeTab === 'samples' && <SampleImages job={job} />}
                {activeTab === 'loss' && <JobLossGraph job={job} />}
                {activeTab === 'config' && <JobConfigViewer job={job} />}
              </section>
            </>
          )}
        </div>
      </div>
    </ProjectWorkspaceShell>
  );
}
