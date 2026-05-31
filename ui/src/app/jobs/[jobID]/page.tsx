'use client';

import { useEffect, useState, use } from 'react';
import { FaChevronLeft } from 'react-icons/fa';
import { MdDashboard, MdImage, MdShowChart, MdCode } from 'react-icons/md';
import { Button } from '@headlessui/react';
import { TopBar, MainContent } from '@/components/layout';
import useJob from '@/hooks/useJob';
import SampleImages, { SampleImagesMenu } from '@/components/SampleImages';
import JobOverview from '@/components/JobOverview';
import JobActionBar from '@/components/JobActionBar';
import JobConfigViewer from '@/components/JobConfigViewer';
import JobLossGraph from '@/components/JobLossGraph';
import type { Job } from '@/types';
import { PageNotice, StatusBadge } from '@/components/OperatorPrimitives';
import { useRouter } from 'next/navigation';

type PageKey = 'overview' | 'samples' | 'config' | 'loss_log';

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
    mainCss: 'pt-24',
  },
  {
    name: 'Samples',
    value: 'samples',
    icon: MdImage,
    component: SampleImages,
    menuItem: SampleImagesMenu,
    mainCss: 'pt-24',
    jobTypes: ['train', 'generate'],
  },
  {
    name: 'Training Monitor',
    value: 'loss_log',
    icon: MdShowChart,
    component: JobLossGraph,
    mainCss: 'pt-24 pb-4',
    jobTypes: ['train'],
  },
  {
    name: 'Config File',
    value: 'config',
    icon: MdCode,
    component: JobConfigViewer,
    mainCss: 'pt-[88px] px-0 pb-0',
  },
];

function shouldPollJob(job: Job | null) {
  if (!job) return true;
  return job.status === 'queued' || job.status === 'running' || job.status === 'stopping';
}

export default function JobPage({ params }: { params: Promise<{ jobID: string }> }) {
  const usableParams = use(params);
  const router = useRouter();
  const jobID = usableParams.jobID;
  const [jobReloadInterval, setJobReloadInterval] = useState<number | null>(5000);
  const { job, status, refreshJob } = useJob(jobID, jobReloadInterval);
  const [pageKey, setPageKey] = useState<PageKey>('overview');

  useEffect(() => {
    const nextInterval = shouldPollJob(job) ? 5000 : null;
    setJobReloadInterval(currentInterval => (currentInterval === nextInterval ? currentInterval : nextInterval));
  }, [job?.status]);

  const page = pages.find(p => p.value === pageKey);

  const jobType = job?.job_type || 'unknown';

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
          <Button className="operator-icon-button" onClick={() => router.push('/jobs')} title="Back to queue">
            <FaChevronLeft />
          </Button>
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
      <MainContent className={job ? pages.find(page => page.value === pageKey)?.mainCss : undefined}>
        {status === 'loading' && job == null && (
          <PageNotice tone="neutral" title="Loading job">
            Fetching job details, status, logs, and metrics.
          </PageNotice>
        )}
        {status === 'error' && job == null && (
          <PageNotice tone="danger" title="Could not load job">
            Refresh the queue or check whether this job still exists.
          </PageNotice>
        )}
        {status === 'success' && job == null && (
          <PageNotice tone="warning" title="Job not found">
            This job no longer exists or has not been imported into the local queue.
          </PageNotice>
        )}
        {job && (
          <>
            {pages.map(page => {
              const Component = page.component;
              return page.value === pageKey ? <Component key={page.value} job={job} /> : null;
            })}
          </>
        )}
      </MainContent>
      {job && (
      <div className="operator-scrollbar-none absolute left-0 top-12 flex h-9 w-full items-center overflow-x-auto border-b border-gray-800 bg-gray-900 px-2 text-sm">
        {pages.map(page => {
          if (page.jobTypes && !page.jobTypes.includes(jobType)) {
            return null;
          }
          return (
            <Button
              key={page.value}
              onClick={() => setPageKey(page.value)}
              className={`flex h-8 items-center gap-1.5 border-b-2 px-3 py-1 ${
                page.value === pageKey
                  ? 'border-cyan-400 text-cyan-100'
                  : 'border-transparent text-gray-400 hover:text-gray-100'
              }`}
            >
              <page.icon className="text-sm" />
              {page.name}
            </Button>
          );
        })}
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
