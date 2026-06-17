'use client';

import { Suspense } from 'react';
import type { ReactNode } from 'react';
import { usePathname } from 'next/navigation';
import classNames from 'classnames';
import Sidebar from '@/components/Sidebar';

export default function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const isDatasetStudio = pathname?.startsWith('/datasets/');
  const isProjectWorkspace = pathname?.startsWith('/projects/');

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-gray-950 md:flex-row">
      {!isDatasetStudio && !isProjectWorkspace && <Sidebar />}
      <main
        className={classNames('relative min-h-0 flex-1 bg-gray-950 text-gray-100', {
          'overflow-hidden': isDatasetStudio || isProjectWorkspace,
          'overflow-auto': !isDatasetStudio && !isProjectWorkspace,
        })}
      >
        <Suspense>{children}</Suspense>
      </main>
    </div>
  );
}
