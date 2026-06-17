'use client';

import { useEffect, useMemo, useState } from 'react';
import type React from 'react';
import Link from 'next/link';
import { Database, FileText, FolderKanban, Globe2, Image as ImageIcon, Loader2, Play, Plus, Search, ShieldCheck } from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';
import { PageNotice } from '@/components/OperatorPrimitives';
import { apiClient } from '@/utils/api';
import type { Project } from '@/types';

type ProjectWithRoots = Project & {
  roots?: {
    root: string;
    datasets: string;
    runs: string;
    outputs: string;
    models: string;
    assets: string;
    notes: string;
    configs: string;
    cache: string;
  };
};

function formatDate(value: string | Date) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Unknown';
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function ProjectBadge({ project, size = 'h-10 w-10' }: { project?: Partial<Project>; size?: string }) {
  const badge = project?.badge_asset || '/assets/projects/project-badge-default.png';
  return (
    <span className={`${size} flex-none overflow-hidden rounded-sm border border-cyan-500/25 bg-gray-900`}>
      <img src={badge} alt="" className="h-full w-full object-cover" />
    </span>
  );
}

const sandboxFolders = [
  { label: 'datasets', icon: Database, tone: 'text-cyan-300' },
  { label: 'runs', icon: Play, tone: 'text-emerald-300' },
  { label: 'outputs', icon: ImageIcon, tone: 'text-violet-300' },
  { label: 'notes', icon: FileText, tone: 'text-gray-400' },
];

function SandboxFolderPreview() {
  return (
    <div className="mt-auto border border-gray-800 bg-gray-950">
      <div className="grid grid-cols-2 gap-px bg-gray-800">
        {sandboxFolders.map(item => {
          const Icon = item.icon;
          return (
            <div key={item.label} className="flex items-center gap-2 bg-gray-950 px-3 py-2">
              <Icon className={`h-4 w-4 flex-none ${item.tone}`} />
              <span className="truncate font-mono text-xs text-gray-400">{item.label}/</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function ProjectsPage() {
  const [projects, setProjects] = useState<ProjectWithRoots[]>([]);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [query, setQuery] = useState('');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [createStatus, setCreateStatus] = useState<'idle' | 'saving' | 'error'>('idle');
  const [createError, setCreateError] = useState('');

  const refreshProjects = () => {
    setStatus('loading');
    apiClient
      .get('/api/projects')
      .then(res => {
        setProjects(Array.isArray(res.data?.projects) ? res.data.projects : []);
        setStatus('success');
      })
      .catch(error => {
        console.error('Failed to load projects:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    refreshProjects();
  }, []);

  const filteredProjects = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return projects;
    return projects.filter(project =>
      [project.name, project.slug, project.description, project.roots?.root]
        .filter(Boolean)
        .some(value => `${value}`.toLowerCase().includes(normalized)),
    );
  }, [projects, query]);

  const createProject = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!name.trim() || createStatus === 'saving') return;
    setCreateStatus('saving');
    setCreateError('');
    try {
      const res = await apiClient.post('/api/projects', {
        name,
        description,
        badge_asset: '/assets/projects/project-badge-default.png',
      });
      const project = res.data?.project as ProjectWithRoots;
      setProjects(current => [project, ...current.filter(item => item.id !== project.id)]);
      setName('');
      setDescription('');
      setCreateStatus('idle');
    } catch (error: any) {
      setCreateStatus('error');
      setCreateError(error?.response?.data?.error || 'Failed to create project.');
    }
  };

  return (
    <>
      <TopBar>
        <div className="flex shrink-0 items-center gap-2">
          <FolderKanban className="h-4 w-4 text-cyan-300" />
          <h1 className="text-base font-semibold">Projects</h1>
        </div>
        <div className="flex-1"></div>
        <label className="relative hidden w-full max-w-sm md:block">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
          <input
            value={query}
            onChange={event => setQuery(event.target.value)}
            placeholder="Find project..."
            className="h-9 w-full border border-gray-800 bg-gray-950 pl-9 pr-3 text-sm text-gray-100 placeholder:text-gray-500 outline-none focus:border-cyan-700"
          />
        </label>
      </TopBar>

      <MainContent className="bg-gray-950 px-0 pt-12 sm:px-0">
        <div className="mx-auto flex max-w-[1380px] flex-col gap-5 px-3 py-5 sm:px-4 lg:px-5">
          <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_380px]">
            <div className="min-w-0 border border-gray-800 bg-gray-900/40">
              <div className="border-b border-gray-800 px-4 py-3">
                <div className="flex min-w-0 items-center gap-2 text-sm font-semibold text-gray-100">
                  <ShieldCheck className="h-4 w-4 text-cyan-300" />
                  Project Workspaces
                </div>
                <p className="mt-1 text-xs text-gray-500">
                  Open a bounded workspace where datasets, runs, outputs, notes, and assets live together.
                </p>
              </div>

              <div className="divide-y divide-gray-800">
                <Link href="/dashboard" className="flex items-center gap-3 px-4 py-3 transition-colors hover:bg-gray-900">
                  <span className="flex h-10 w-10 flex-none items-center justify-center rounded-sm border border-gray-700 bg-gray-950">
                    <Globe2 className="h-5 w-5 text-gray-400" />
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium text-gray-100">Global workspace</div>
                    <div className="truncate text-xs text-gray-500">Existing datasets, jobs, output, and settings remain here.</div>
                  </div>
                  <span className="hidden text-xs text-gray-500 sm:inline">Open global</span>
                </Link>

                {status === 'loading' && (
                  <div className="flex items-center gap-2 px-4 py-6 text-sm text-gray-400">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading projects
                  </div>
                )}

                {status === 'error' && (
                  <div className="p-4">
                    <PageNotice tone="danger" title="Projects could not be loaded">
                      Check the local database connection and try again.
                    </PageNotice>
                  </div>
                )}

                {status === 'success' && filteredProjects.length === 0 && (
                  <div className="px-4 py-8">
                    <PageNotice tone="neutral" title={query ? 'No projects match the filter' : 'No projects yet'}>
                      Create a project to get an isolated Mission Control workspace.
                    </PageNotice>
                  </div>
                )}

                {filteredProjects.map(project => (
                  <Link
                    key={project.id}
                    href={`/projects/${encodeURIComponent(project.slug)}`}
                    className="group flex items-center gap-3 px-4 py-3 transition-colors hover:bg-gray-900"
                  >
                    <ProjectBadge project={project} />
                    <div className="min-w-0 flex-1">
                      <div className="flex min-w-0 items-center gap-2">
                        <div className="truncate text-sm font-medium text-gray-100">{project.name}</div>
                        <span className="hidden rounded-sm border border-gray-800 bg-gray-950 px-1.5 py-0.5 text-[10px] text-cyan-200 sm:inline">
                          isolated
                        </span>
                      </div>
                      <div className="mt-0.5 truncate text-xs text-gray-500">
                        {project.description || project.roots?.root || `projects/${project.slug}`}
                      </div>
                    </div>
                    <div className="hidden text-right text-xs text-gray-500 md:block">
                      <div>{formatDate(project.updated_at)}</div>
                      <div className="mt-0.5 opacity-70">{project.slug}</div>
                    </div>
                  </Link>
                ))}
              </div>
            </div>

            <aside className="border border-gray-800 bg-gray-900/40">
              <form onSubmit={createProject} className="flex h-full flex-col">
                <div className="border-b border-gray-800 px-4 py-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-gray-100">
                    <Plus className="h-4 w-4 text-emerald-300" />
                    New Project
                  </div>
                  <p className="mt-1 text-xs text-gray-500">A project creates its own datasets, runs, outputs, and notes folders.</p>
                </div>
                <div className="flex flex-1 flex-col gap-3 p-4">
                  <label className="block">
                    <span className="mb-1 block text-xs font-medium text-gray-400">Name</span>
                    <input
                      value={name}
                      onChange={event => setName(event.target.value)}
                      placeholder="Flux Portrait Set"
                      className="h-9 w-full border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 placeholder:text-gray-600 outline-none focus:border-cyan-700"
                    />
                  </label>
                  <label className="block">
                    <span className="mb-1 block text-xs font-medium text-gray-400">Description</span>
                    <textarea
                      value={description}
                      onChange={event => setDescription(event.target.value)}
                      placeholder="Training workspace for portrait LoRAs..."
                      rows={4}
                      className="w-full resize-none border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-100 placeholder:text-gray-600 outline-none focus:border-cyan-700"
                    />
                  </label>
                  <SandboxFolderPreview />
                  {createError && <div className="text-xs text-rose-300">{createError}</div>}
                  <button
                    type="submit"
                    disabled={!name.trim() || createStatus === 'saving'}
                    className="operator-button h-9 border-emerald-800 bg-emerald-950/70 text-emerald-100 hover:bg-emerald-900 disabled:opacity-50"
                  >
                    {createStatus === 'saving' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                    Create Project
                  </button>
                </div>
              </form>
            </aside>
          </section>
        </div>
      </MainContent>
    </>
  );
}
