'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowRight, ChevronRight, Monitor, Play } from 'lucide-react';
import JobsTable, { type JobsViewStatus } from '@/components/JobsTable';
import DashboardHardware from '@/components/DashboardHardware';
import TensorBoardLink from '@/components/TensorBoardLink';
import useGPUInfo from '@/hooks/useGPUInfo';
import useMonitorStream from '@/hooks/useMonitorStream';
import useSettings from '@/hooks/useSettings';
import type { Job } from '@/types';

export default function Dashboard() {
  const telemetry = useGPUInfo();
  const monitor = useMonitorStream();
  const { settings } = useSettings();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobsStatus, setJobsStatus] = useState<JobsViewStatus>('loading');
  const projectsEnabled = settings.PROJECTS_ENABLED !== 'false';
  const gpuReady = telemetry.isGPUInfoLoaded && telemetry.status !== 'error' && telemetry.gpuList.length > 0;
  const gpuOnline = gpuReady && monitor.connected;

  return (
    <div className="studio-dashboard">
      <section className="studio-hero" aria-labelledby="dashboard-title">
        <div className="studio-hero-art" aria-hidden="true" />
        <header className="studio-topbar">
          <span className="font-semibold">Dashboard</span>
          <ChevronRight size={16} className="studio-muted" aria-hidden="true" />
          <span className="studio-workspace-label">
            <Monitor size={18} aria-hidden="true" /> Local workspace
          </span>
          <div className="ml-auto">
            <TensorBoardLink />
          </div>
        </header>
        <div className="studio-hero-content">
          <h1 id="dashboard-title">
            Make your
            <br />
            next model.
          </h1>
          <p>Your workspace is ready for a new training run.</p>
          <div className="studio-hero-actions">
            <Link href="/jobs/new" className="studio-primary">
              <Play size={18} fill="currentColor" aria-hidden="true" /> Start training
            </Link>
            <Link href="/generate" className="studio-text-link">
              Generate images <ArrowRight size={20} aria-hidden="true" />
            </Link>
          </div>
          <div className="studio-shortcuts">
            <Link href="/datasets">
              Datasets <ChevronRight size={17} aria-hidden="true" />
            </Link>
            {projectsEnabled && (
              <Link href="/projects">
                Projects <ChevronRight size={17} aria-hidden="true" />
              </Link>
            )}
          </div>
        </div>
      </section>
      <div className="studio-workspace-grid">
        <section className="studio-queue" aria-label="Training queue">
          <JobsTable
            variant="dashboard"
            onlyActive
            includeProjectActive
            onJobsChange={setJobs}
            onStatusChange={setJobsStatus}
          />
        </section>
        <DashboardHardware telemetry={telemetry} connected={monitor.connected} lastUpdated={monitor.lastUpdated} />
      </div>
      <footer className="studio-statusbar">
        <span>
          <span className={`studio-status-dot ${gpuOnline ? '' : 'studio-status-dot-muted'}`} />
          {gpuOnline
            ? `${telemetry.gpuList.length} GPU${telemetry.gpuList.length === 1 ? '' : 's'} online`
            : gpuReady
              ? 'GPU connection interrupted'
              : telemetry.isGPUInfoLoaded
                ? 'GPU telemetry unavailable'
                : 'Checking hardware'}
        </span>
        <span className="studio-muted">
          {jobsStatus === 'loading'
            ? 'Loading jobs'
            : jobsStatus === 'error'
              ? 'Job data incomplete'
              : `${jobs.length} active job${jobs.length === 1 ? '' : 's'}`}
        </span>
      </footer>
    </div>
  );
}
