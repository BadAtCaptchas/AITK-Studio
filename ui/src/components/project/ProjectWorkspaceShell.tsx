'use client';

import { type ReactNode, useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import classNames from 'classnames';
import {
  ArrowLeftRight,
  Boxes,
  Database,
  FolderOpen,
  Loader2,
  Play,
  Plus,
  Settings,
  Sparkles,
  Wand2,
} from 'lucide-react';
import { apiClient } from '@/utils/api';
import type { ProjectSummary } from './types';
import useSettings from '@/hooks/useSettings';

type ProjectSection = 'workspace' | 'files' | 'datasets' | 'runs' | 'generate' | 'settings';

const navItems: Array<{ section: ProjectSection; label: string; hrefSuffix: string; icon: typeof Boxes }> = [
  { section: 'workspace', label: 'Workspace', hrefSuffix: '', icon: Boxes },
  { section: 'files', label: 'Files', hrefSuffix: '/files', icon: FolderOpen },
  { section: 'datasets', label: 'Datasets', hrefSuffix: '/datasets', icon: Database },
  { section: 'runs', label: 'Runs', hrefSuffix: '/runs', icon: Play },
  { section: 'generate', label: 'Generate', hrefSuffix: '/generate', icon: Wand2 },
  { section: 'settings', label: 'Settings', hrefSuffix: '/settings', icon: Settings },
];

function ProjectBadge({ summary }: { summary: ProjectSummary | null }) {
  const src = summary?.project.badge_asset || '/assets/projects/project-badge-default.png';
  return (
    <span className="flex h-10 w-10 flex-none overflow-hidden rounded-sm border border-cyan-500/25 bg-gray-900">
      <img src={src} alt="" className="h-full w-full object-cover" />
    </span>
  );
}

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
  projectID,
  active,
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
  const pathname = usePathname();
  const { settings, isSettingsLoaded } = useSettings();
  const [summary, setSummary] = useState<ProjectSummary | null>(null);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const basePath = `/projects/${encodeURIComponent(projectID)}`;
  const projectsEnabled = settings.PROJECTS_ENABLED !== 'false';

  useEffect(() => {
    if (!isSettingsLoaded) return;
    if (!projectsEnabled) {
      setSummary(null);
      setStatus('success');
      return;
    }
    let cancelled = false;
    setStatus(current => (current === 'success' ? current : 'loading'));
    apiClient
      .get(`/api/projects/${encodeURIComponent(projectID)}/summary`)
      .then(res => {
        if (cancelled) return;
        setSummary(res.data);
        setStatus('success');
      })
      .catch(error => {
        if (cancelled) return;
        console.error('Failed to load project summary:', error);
        setStatus('error');
      });
    return () => {
      cancelled = true;
    };
  }, [projectID, pathname, isSettingsLoaded, projectsEnabled]);

  const projectTitle = summary?.project.name || 'Project Mission Control';
  const headerTitle = title || (active === 'workspace' ? 'Workspace' : navItems.find(item => item.section === active)?.label);

  if (isSettingsLoaded && !projectsEnabled) {
    return (
      <div className="flex h-full min-h-0 bg-gray-950 text-gray-100">
        <div className="flex min-w-0 flex-1 flex-col">
          <header className="flex h-16 flex-none items-center gap-3 border-b border-gray-900 bg-gray-950 px-4">
            <FolderOpen className="h-5 w-5 text-amber-300" />
            <div className="min-w-0">
              <h1 className="truncate text-base font-semibold">Project spaces are disabled</h1>
              <p className="truncate text-xs text-gray-500">Existing project data is preserved and hidden until re-enabled.</p>
            </div>
          </header>
          <main className="min-h-0 flex-1 overflow-auto p-4">
            <div className="max-w-2xl border border-amber-800/60 bg-amber-950/10 p-4">
              <div className="text-sm font-semibold text-amber-100">Workspaces are currently blocked</div>
              <p className="mt-1 text-sm text-gray-400">
                Re-enable Project spaces in Settings to open this workspace, project files, datasets, runs, or generation tools.
              </p>
              <Link
                href="/settings"
                className="mt-4 inline-flex h-9 items-center border border-cyan-800 bg-cyan-950/40 px-3 text-sm font-semibold text-cyan-100 hover:bg-cyan-900/40"
              >
                Open Settings
              </Link>
            </div>
          </main>
        </div>
      </div>
    );
  }

  return (
    <div id="workspace" className="flex h-full min-h-0 bg-gray-950 text-gray-100">
      <aside className="flex w-[78px] flex-none flex-col border-r border-gray-900 bg-gray-950">
        <div className="flex h-16 items-center justify-center border-b border-gray-900">
          <ProjectBadge summary={summary} />
        </div>
        <nav className="flex flex-1 flex-col items-center gap-1 px-2 py-4">
          {navItems.map(item => {
            const Icon = item.icon;
            const isActive = active === item.section;
            return (
              <Link
                key={item.section}
                href={`${basePath}${item.hrefSuffix}`}
                title={item.label}
                className={classNames(
                  'group flex h-[58px] w-full flex-col items-center justify-center gap-1 rounded-sm border transition-colors',
                  isActive
                    ? 'border-cyan-900/60 bg-cyan-950/20 text-cyan-100'
                    : 'border-transparent text-gray-500 hover:border-gray-800 hover:bg-gray-900 hover:text-gray-200',
                )}
              >
                <Icon className={classNames('h-5 w-5', isActive ? 'text-cyan-200' : 'group-hover:text-cyan-200')} />
                <span className="w-full truncate text-center text-[10px]">{item.label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="border-t border-gray-900 p-2">
          <Link
            href="/projects"
            title="Switch project"
            className="flex h-[58px] flex-col items-center justify-center gap-1 rounded-sm border border-gray-800 bg-gray-900/50 text-gray-300 hover:text-cyan-200"
          >
            <ArrowLeftRight className="h-5 w-5" />
            <span className="text-[10px]">Switch</span>
          </Link>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        {showHeader && (
          <header className="flex h-16 flex-none items-center gap-3 border-b border-gray-900 bg-gray-950 px-4">
            <div className="min-w-0 flex-1">
              <div className="flex min-w-0 items-center gap-2">
                <h1 className="truncate text-base font-semibold">{projectTitle}</h1>
                <span className="rounded-sm border border-cyan-700/50 bg-cyan-950/30 px-1.5 py-0.5 text-[10px] uppercase text-cyan-200">
                  isolated workspace
                </span>
                {status === 'loading' && <Loader2 className="h-3.5 w-3.5 animate-spin text-gray-500" />}
              </div>
              <div className="truncate text-xs text-gray-500">
                {summary ? `projects/${summary.project.slug}` : 'Loading project'}
                {summary?.roots.root ? ` - ${summary.roots.root}` : ''}
              </div>
            </div>
            <div className="hidden min-w-0 flex-1 lg:block">
              <div className="truncate text-right text-xs text-gray-500">{description || headerTitle}</div>
            </div>
            {actions ? <div className="flex flex-none items-center gap-2">{actions}</div> : null}
            <Link
              href={`${basePath}/runs/new`}
              className="operator-button h-9 border-emerald-800 bg-emerald-950/70 text-emerald-100 hover:bg-emerald-900"
            >
              <Plus className="h-4 w-4" />
              <span className="hidden sm:inline">New Run</span>
            </Link>
            <Link href={`${basePath}/generate`} className="operator-button h-9">
              <Sparkles className="h-4 w-4 text-cyan-200" />
              <span className="hidden sm:inline">Generate</span>
            </Link>
          </header>
        )}

        <main className="relative min-h-0 flex-1 overflow-hidden">
          {status === 'error' && showHeader ? (
            <div className="m-4 border border-rose-900 bg-rose-950/30 px-4 py-3 text-sm text-rose-200">
              Project could not be loaded.
            </div>
          ) : null}
          {children}
        </main>
      </div>
    </div>
  );
}
