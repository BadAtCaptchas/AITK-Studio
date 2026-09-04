'use client';

import type { ReactNode } from 'react';

export type ProjectSection =
  | 'workspace'
  | 'datasets'
  | 'runs'
  | 'outputs'
  | 'models'
  | 'files'
  | 'generate'
  | 'settings';

export function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value >= 10 || index === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[index]}`;
}

export function formatProjectTime(value: string | Date) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Unknown';
  return date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}

export default function ProjectWorkspaceShell({
  title,
  description,
  actions,
  children,
  showHeader = true,
}: {
  projectID: string;
  active: ProjectSection;
  title?: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
  showHeader?: boolean;
}) {
  return (
    <div className="flex h-full min-h-0 flex-col bg-gray-950 text-gray-100">
      {showHeader && (title || description || actions) ? (
        <div className="flex min-h-[72px] flex-none items-center gap-4 border-b border-gray-800 bg-gray-950 px-5 py-3 sm:px-8">
          <div className="min-w-0 flex-1">
            {title ? <h2 className="truncate text-sm font-semibold text-gray-100 sm:text-base">{title}</h2> : null}
            {description ? <p className="mt-0.5 truncate text-xs text-gray-500">{description}</p> : null}
          </div>
          {actions ? <div className="flex flex-none items-center gap-2">{actions}</div> : null}
        </div>
      ) : null}
      <div className="relative min-h-0 flex-1 overflow-hidden">{children}</div>
    </div>
  );
}
