'use client';

import GpuMonitor from '@/components/GPUMonitor';
import JobsTable from '@/components/JobsTable';
import TensorBoardLink from '@/components/TensorBoardLink';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';
import { Activity, ListOrdered, Plus } from 'lucide-react';

export default function Dashboard() {
  return (
    <>
      <TopBar>
        <div className="flex shrink-0 items-center gap-2">
          <Activity className="h-4 w-4 text-cyan-300" />
          <h1 className="text-base font-semibold">Dashboard</h1>
        </div>
        <div className="flex-1"></div>
        <Link href="/jobs/new" className="operator-button py-1 text-xs">
          <Plus className="h-3.5 w-3.5" />
          New Job
        </Link>
        <TensorBoardLink />
      </TopBar>
      <MainContent>
        <GpuMonitor />
        <div className="mt-5 w-full">
          <div className="mb-2 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <ListOrdered className="h-4 w-4 text-gray-500" />
              <h2 className="text-sm font-semibold uppercase tracking-wide text-gray-400">Active queues</h2>
            </div>
            <Link href="/jobs" className="text-xs text-cyan-300 hover:text-cyan-200">
              View all
            </Link>
          </div>
          <JobsTable onlyActive />
        </div>
      </MainContent>
    </>
  );
}
