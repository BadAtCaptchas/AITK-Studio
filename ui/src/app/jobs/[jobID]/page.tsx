'use client';

import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { FaChevronLeft } from 'react-icons/fa';
import { MdDashboard, MdImage, MdShowChart, MdCode } from 'react-icons/md';
import { Button } from '@headlessui/react';
import { TopBar, MainContent } from '@/components/layout';
import useJob from '@/hooks/useJob';
import { isActiveJobStatus } from '@/utils/jobStatus';
import SampleImages, { SampleImagesMenu } from '@/components/SampleImages';
import JobNotes from '@/components/JobNotes';
import JobOverview from '@/components/JobOverview';
import JobActionBar from '@/components/JobActionBar';
import JobConfigViewer from '@/components/JobConfigViewer';
import JobLossGraph from '@/components/JobLossGraph';
import type { Job } from '@/types';
import { PageNotice, StatusBadge } from '@/components/OperatorPrimitives';
import { useRouter, useSearchParams } from '@/navigation';
import Link from '@/navigation/Link';

type PageKey = 'overview' | 'samples' | 'config' | 'loss_log' | 'notes';

interface Page {
  name: string;
  value: PageKey;
  icon: React.ComponentType<{ className?: string }>;
  component: React.ComponentType<{ job: Job }>;
  menuItem?: React.ComponentType<{ job?: Job | null }> | null;
  mainCss?: string;
  jobTypes?: string[]; // if specified, only show this page for these job types
}

const pages: Page[] = [
  {
    name: 'Overview',
    value: 'overview',
    icon: MdDashboard,
    component: JobOverview,
  },
  {
    name: 'Samples',
    value: 'samples',
    icon: MdImage,
    component: SampleImages,
    menuItem: SampleImagesMenu,
    jobTypes: ['train', 'generate'],
  },
  {
    name: 'Metrics',
    value: 'loss_log',
    icon: MdShowChart,
    component: JobLossGraph,
    mainCss: 'pb-4',
    jobTypes: ['train'],
  },
  {
    name: 'Configuration',
    value: 'config',
    icon: MdCode,
    component: JobConfigViewer,
    mainCss: 'px-0 pb-0',
  },
  { name: 'Notes', value: 'notes', icon: MdCode, component: JobNotes },
];

function shouldPollJob(job: Job | null) {
  if (!job) return true;
  return isActiveJobStatus(job.status);
}

export default function JobPage() {
  const { jobID = '' } = useParams<'jobID'>();
  const router = useRouter();
  const searchParams = useSearchParams();

  const [jobReloadInterval, setJobReloadInterval] = useState<number | null>(5000);
  const { job, status, refreshJob } = useJob(jobID, jobReloadInterval);
  const [visitedNotesJobID, setVisitedNotesJobID] = useState<string | null>(null);

  const nextInterval = shouldPollJob(job) ? 5000 : null;
  if (jobReloadInterval !== nextInterval) setJobReloadInterval(nextInterval);

  const jobType = job?.job_type || 'unknown';
  const availablePages = pages.filter(page => !page.jobTypes || page.jobTypes.includes(jobType));
  const requestedTab = searchParams.get('tab');
  const normalizedTab = requestedTab === 'metrics' ? 'loss_log' : requestedTab === 'logs' ? 'overview' : requestedTab;
  const pageKey = availablePages.find(page => page.value === normalizedTab)?.value || 'overview';
  const page = availablePages.find(page => page.value === pageKey);
  const selectPage = (value: PageKey) => {
    const url = new URL(window.location.href);
    url.searchParams.set('tab', value);
    router.replace(url.pathname + url.search + url.hash, { scroll: false });
  };
  if (pageKey === 'notes' && visitedNotesJobID !== jobID) setVisitedNotesJobID(jobID);

  let title = `Job: ${job?.name || (status === 'success' ? 'Not found' : 'Loading...')}`;
  if (jobType === 'caption') {
    title = `Captioning: ${job?.job_ref || 'Loading...'}`;
  } else if (jobType === 'generate') {
    title = `Generate: ${job?.name || 'Loading...'}`;
  }

  return (
    <>
      {/* Fixed top bar */}
      <TopBar>
        <div>
          <Link className="operator-icon-button" href="/jobs" title="Back to queue" aria-label="Back to queue">
            <FaChevronLeft />
          </Link>
        </div>
        <div className="min-w-0">
          <h1 className="truncate text-base font-semibold">{title}</h1>
        </div>
        {job && <StatusBadge status={job.status} />}
        <div className="flex-1"></div>
        {job && (
          <JobActionBar
            job={job}
            onRefresh={refreshJob}
            hideView
            afterDelete={() => {
              router.push('/jobs');
            }}
            autoStartQueue={true}
          />
        )}
      </TopBar>
      <MainContent
        className={job ? `!pt-[calc(var(--studio-page-toolbar-height,48px)+3rem)] ${page?.mainCss || ''}` : undefined}
      >
        {status === 'loading' && job == null && (
          <PageNotice tone="neutral" title="Loading job">
            Fetching job details, status, logs, and metrics.
          </PageNotice>
        )}
        {status === 'error' && job == null && (
          <PageNotice tone="danger" title="Could not load job">
            Check whether this job still exists, or{' '}
            <button type="button" className="underline" onClick={refreshJob} aria-label="Retry job refresh">
              retry loading it
            </button>
            .
          </PageNotice>
        )}
        {status === 'error' && job != null && (
          <PageNotice tone="warning" title="Job details could not be refreshed">
            Showing the last loaded status.{' '}
            <button type="button" className="underline" onClick={refreshJob} aria-label="Retry job refresh">
              Retry
            </button>
          </PageNotice>
        )}
        {status === 'success' && job == null && (
          <PageNotice tone="warning" title="Job not found">
            This job no longer exists or has not been imported into the local queue.
          </PageNotice>
        )}
        {job && (
          <>
            {availablePages.map(candidate => {
              const Component = candidate.component;
              const active = candidate.value === pageKey;
              if (!active && (candidate.value !== 'notes' || visitedNotesJobID !== jobID)) return null;
              return (
                <div
                  key={`${jobID}:${candidate.value}`}
                  role="tabpanel"
                  id={`job-panel-${candidate.value}`}
                  aria-labelledby={`job-tab-${candidate.value}`}
                  className={candidate.value === 'config' ? 'h-full min-h-80' : undefined}
                  hidden={!active}
                  tabIndex={0}
                >
                  <Component job={job} />
                </div>
              );
            })}
          </>
        )}
      </MainContent>
      {job && (
        <div className="operator-scrollbar-none absolute left-0 top-[var(--studio-page-toolbar-height,48px)] flex h-9 w-full items-center overflow-x-auto border-b border-gray-800 bg-gray-900 px-3 text-sm">
          <div
            role="tablist"
            aria-label="Job details"
            className="flex items-center"
            onKeyDown={event => {
              if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
              const tabs = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
              const index = tabs.indexOf(event.target as HTMLButtonElement);
              if (index < 0) return;
              event.preventDefault();
              const next =
                event.key === 'Home'
                  ? 0
                  : event.key === 'End'
                    ? tabs.length - 1
                    : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
              tabs[next]?.focus();
            }}
          >
            {availablePages.map(page => {
              return (
                <Button
                  key={page.value}
                  role="tab"
                  id={`job-tab-${page.value}`}
                  aria-controls={`job-panel-${page.value}`}
                  aria-selected={page.value === pageKey}
                  tabIndex={page.value === pageKey ? 0 : -1}
                  onClick={() => selectPage(page.value)}
                  className={`flex h-8 items-center gap-1.5 border-b-2 px-3 py-1 ${
                    page.value === pageKey
                      ? 'border-brand-400 text-brand-100'
                      : 'border-transparent text-gray-400 hover:text-gray-100'
                  }`}
                >
                  <page.icon className="text-sm" />
                  {page.name}
                </Button>
              );
            })}
          </div>
          {page?.menuItem && (
            <>
              <div className="flex-grow"></div>
              <page.menuItem job={job} />
            </>
          )}
        </div>
      )}
    </>
  );
}
