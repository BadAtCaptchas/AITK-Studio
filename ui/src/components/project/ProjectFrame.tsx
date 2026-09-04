'use client';

import type { ReactNode } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import classNames from 'classnames';
import {
  AlertTriangle,
  Archive,
  Boxes,
  Database,
  FolderKanban,
  FolderOpen,
  HardDrive,
  Images,
  LayoutDashboard,
  Loader2,
  Play,
  Plus,
  Radio,
  Settings,
  Sparkles,
} from 'lucide-react';
import Sidebar from '@/components/Sidebar';
import { PageNotice } from '@/components/OperatorPrimitives';
import { useProjectWorkspace } from './ProjectContext';

const navItems = [
  { label: 'Overview', suffix: '', icon: LayoutDashboard },
  { label: 'Datasets', suffix: '/datasets', icon: Database },
  { label: 'Runs', suffix: '/runs', icon: Play },
  { label: 'Outputs', suffix: '/outputs', icon: Images },
  { label: 'Models', suffix: '/models', icon: Boxes },
  { label: 'Files', suffix: '/files', icon: FolderOpen },
  { label: 'Settings', suffix: '/settings', icon: Settings },
] as const;

function actionForPath(pathname: string, basePath: string) {
  if (pathname.includes('/outputs') || pathname.includes('/models')) {
    return { label: 'Generate', href: `${basePath}/generate`, icon: Sparkles };
  }
  if (pathname.includes('/datasets')) {
    return { label: 'New run', href: `${basePath}/runs/new`, icon: Plus };
  }
  return { label: 'New run', href: `${basePath}/runs/new`, icon: Plus };
}

function navItemIsActive(pathname: string, basePath: string, suffix: string) {
  if (!suffix) return pathname === basePath || pathname === `${basePath}/`;
  return pathname === `${basePath}${suffix}` || pathname.startsWith(`${basePath}${suffix}/`);
}

function isProjectDatasetDetailPath(pathname: string, basePath: string) {
  const datasetPrefix = `${basePath}/datasets/`;
  if (!pathname.startsWith(datasetPrefix)) return false;

  const datasetPath = pathname.slice(datasetPrefix.length).replace(/\/$/, '');
  return datasetPath.length > 0 && !datasetPath.includes('/');
}

export default function ProjectFrame({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { identity, status, error, archived, projectID } = useProjectWorkspace();
  const basePath = `/projects/${encodeURIComponent(projectID)}`;
  const action = actionForPath(pathname, basePath);
  const ActionIcon = action.icon;
  const project = identity?.project;
  const replicas = identity?.replicas || [];
  const unhealthyReplicas = replicas.filter(replica => !['ready', 'syncing'].includes(replica.status));
  const homeLabel =
    project?.home_instance_name || replicas.find(replica => replica.role === 'home')?.instanceName || 'Local home';

  const workspaceContent =
    status === 'disabled' ? (
      <div className="h-full overflow-auto p-4 sm:p-6">
        <div className="mx-auto max-w-2xl">
          <PageNotice
            tone="warning"
            title="Project spaces are disabled"
            action={
              <Link href="/settings" className="operator-button h-8 text-xs">
                Open Settings
              </Link>
            }
          >
            Existing project data is preserved. Re-enable Projects to open this workspace.
          </PageNotice>
        </div>
      </div>
    ) : status === 'error' ? (
      <div className="h-full overflow-auto p-4 sm:p-6">
        <div className="mx-auto max-w-2xl">
          <PageNotice tone="danger" title="Project could not be loaded">
            {error || 'The project may no longer exist, or its storage is unavailable.'}
          </PageNotice>
          <Link href="/projects" className="operator-button mt-3 h-9">
            Back to Projects
          </Link>
        </div>
      </div>
    ) : status === 'loading' && !identity ? (
      <div className="flex h-full items-center justify-center text-sm text-gray-500">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading project workspace
      </div>
    ) : (
      children
    );

  if (isProjectDatasetDetailPath(pathname, basePath)) {
    return (
      <section className="flex h-full min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-gray-950 text-gray-100">
        <div className="relative min-h-0 flex-1 overflow-hidden">{workspaceContent}</div>
      </section>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-gray-950 md:flex-row">
      <Sidebar />
      <section className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden text-gray-100">
        <header className="flex flex-none flex-col border-b border-gray-800 bg-gray-950/95">
          <div className="flex min-h-[84px] items-center gap-4 px-5 py-3 sm:px-8">
            <Link
              href="/projects"
              aria-label="Back to projects"
              className="flex h-10 w-10 flex-none items-center justify-center overflow-hidden rounded-md border border-gray-800 bg-gray-900"
            >
              {project?.badge_asset ? (
                <img src={project.badge_asset} alt="" className="h-full w-full object-cover" />
              ) : (
                <FolderKanban className="h-5 w-5 text-brand-300" />
              )}
            </Link>
            <div className="min-w-0 flex-1">
              <div className="flex min-w-0 flex-wrap items-center gap-2">
                <h1 className="truncate text-sm font-semibold text-gray-100 sm:text-base">
                  {project?.name || (status === 'loading' ? 'Loading project' : 'Project')}
                </h1>
                {archived ? (
                  <span className="inline-flex items-center gap-1 rounded-full border border-amber-800 bg-amber-950/40 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-200">
                    <Archive className="h-3 w-3" /> Archived
                  </span>
                ) : null}
                {status === 'loading' ? <Loader2 className="h-3.5 w-3.5 animate-spin text-gray-500" /> : null}
              </div>
              <div className="mt-0.5 flex min-w-0 items-center gap-2 text-[11px] text-gray-500 sm:text-xs">
                <span className="inline-flex min-w-0 items-center gap-1 truncate">
                  <HardDrive className="h-3 w-3 flex-none" />
                  <span className="truncate">{homeLabel}</span>
                </span>
                <span aria-hidden="true">/</span>
                <span className="truncate font-mono">{project?.slug || projectID}</span>
                {replicas.length > 1 ? (
                  <span className="hidden items-center gap-1 sm:inline-flex">
                    <Radio className="h-3 w-3" /> {replicas.length - 1} replica{replicas.length === 2 ? '' : 's'}
                  </span>
                ) : null}
                {unhealthyReplicas.length > 0 ? (
                  <span
                    className="inline-flex items-center gap-1 text-amber-300"
                    title="A project replica needs attention"
                  >
                    <AlertTriangle className="h-3 w-3" /> {unhealthyReplicas.length}
                  </span>
                ) : null}
              </div>
            </div>
            {!archived && status === 'ready' ? (
              <Link
                href={action.href}
                className="operator-button-primary h-10 flex-none"
              >
                <ActionIcon className="h-4 w-4" />
                <span className="hidden sm:inline">{action.label}</span>
              </Link>
            ) : null}
          </div>

          <nav
            className="operator-scrollbar-none flex min-w-0 gap-2 overflow-x-auto border-t border-gray-800 px-4 sm:px-7"
            aria-label="Project sections"
          >
            {navItems.map(item => {
              const href = `${basePath}${item.suffix}`;
              const active = navItemIsActive(pathname, basePath, item.suffix);
              const Icon = item.icon;
              return (
                <Link
                  key={item.label}
                  href={href}
                  aria-current={active ? 'page' : undefined}
                  className={classNames(
                    'group relative inline-flex h-11 flex-none items-center gap-2 px-2.5 text-xs font-medium transition-colors sm:px-3 sm:text-sm',
                    active ? 'text-brand-100' : 'text-gray-500 hover:text-gray-200',
                  )}
                >
                  <Icon
                    className={classNames(
                      'h-4 w-4',
                      active ? 'text-brand-300' : 'text-gray-600 group-hover:text-gray-400',
                    )}
                  />
                  {item.label}
                  <span
                    className={classNames(
                      'absolute inset-x-2 bottom-0 h-0.5 rounded-full transition-colors',
                      active ? 'bg-brand-400' : 'bg-transparent',
                    )}
                  />
                </Link>
              );
            })}
          </nav>
        </header>
        <div className="relative min-h-0 flex-1 overflow-hidden">{workspaceContent}</div>
      </section>
    </div>
  );
}
