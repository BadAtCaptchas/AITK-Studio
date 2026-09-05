'use client';
import { Suspense } from 'react';
import type { ReactNode } from 'react';
import { usePathname } from 'next/navigation';
import classNames from 'classnames';
import Sidebar from '@/components/Sidebar';
import WorkflowFeedback from '@/components/WorkflowFeedback';
export default function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const isDatasetStudio = pathname?.startsWith('/datasets/');
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-gray-950 md:flex-row">
      {!isDatasetStudio && <Sidebar />}
      <main
        className={classNames('relative min-h-0 min-w-0 flex-1 bg-gray-950 text-gray-100', {
          'overflow-hidden': isDatasetStudio,
          'overflow-auto': !isDatasetStudio,
        })}
      >
        <div className="fixed bottom-4 right-4 z-50 w-96 max-w-[calc(100vw-2rem)]">
          <WorkflowFeedback />
        </div>
        <Suspense>{children}</Suspense>
      </main>
    </div>
  );
}
