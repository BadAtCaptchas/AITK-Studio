'use client';
import { Suspense, useCallback, useState } from 'react';
import type { ReactNode } from 'react';
import { usePathname } from 'next/navigation';
import classNames from 'classnames';
import Sidebar from '@/components/Sidebar';
import WorkflowFeedback from '@/components/WorkflowFeedback';
import CommandPalette from '@/components/CommandPalette';
export default function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const isDatasetStudio = pathname?.startsWith('/datasets/');
  const [commandsOpen, setCommandsOpen] = useState(false);
  const openCommands = useCallback(() => setCommandsOpen(true), []);
  const closeCommands = useCallback(() => setCommandsOpen(false), []);
  return (
    <div className="studio-app-shell flex h-dvh flex-col overflow-hidden bg-gray-950">
      <Sidebar onSearch={openCommands} />
      <CommandPalette
        key={commandsOpen ? 'open' : 'closed'}
        open={commandsOpen}
        onOpen={openCommands}
        onClose={closeCommands}
      />
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
