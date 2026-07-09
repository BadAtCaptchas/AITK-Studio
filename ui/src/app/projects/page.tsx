'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import classNames from 'classnames';
import {
  Activity,
  Archive,
  ArrowRight,
  Boxes,
  Database,
  FolderKanban,
  Globe2,
  HardDrive,
  Image as ImageIcon,
  Loader2,
  Play,
  Plus,
  Search,
  ServerOff,
  SlidersHorizontal,
} from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';
import { PageNotice } from '@/components/OperatorPrimitives';
import useSettings from '@/hooks/useSettings';
import { artifactPreviewUrl, loadProjectCards, projectIsArchived } from '@/components/project/data';
import { formatBytes, formatProjectTime } from '@/components/project/ProjectWorkspaceShell';
import QuickCreateProjectModal from '@/components/project/QuickCreateProjectModal';
import type { ProjectCardSummary, ProjectRecord, ProjectWorkflowState } from '@/components/project/types';

const workflowLabels: Record<ProjectWorkflowState, { label: string; className: string }> = {
  empty: { label: 'Ready to set up', className: 'border-gray-700 bg-gray-900 text-gray-300' },
  preparing: { label: 'Preparing dataset', className: 'border-cyan-800 bg-cyan-950/40 text-cyan-200' },
  training: { label: 'Training', className: 'border-emerald-800 bg-emerald-950/40 text-emerald-200' },
  review: { label: 'Review outputs', className: 'border-violet-800 bg-violet-950/40 text-violet-200' },
  ready: { label: 'Ready to train', className: 'border-blue-800 bg-blue-950/40 text-blue-200' },
  attention: { label: 'Needs attention', className: 'border-amber-800 bg-amber-950/40 text-amber-200' },
};

function ProjectPreview({ card }: { card: ProjectCardSummary }) {
  const preview = card.latestOutput ? artifactPreviewUrl(card.latestOutput) : null;
  return (
    <div className="relative aspect-[16/8] overflow-hidden border-b border-gray-800 bg-[radial-gradient(circle_at_top_right,rgba(8,145,178,0.16),transparent_45%),linear-gradient(135deg,rgba(17,24,39,0.9),rgba(3,7,18,1))]">
      {preview && card.latestOutput?.mediaKind === 'image' ? (
        <img
          src={preview}
          alt=""
          loading="lazy"
          className="h-full w-full object-cover opacity-90 transition-transform duration-300 group-hover:scale-[1.02]"
          onError={event => {
            event.currentTarget.style.display = 'none';
          }}
        />
      ) : (
        <div className="flex h-full items-center justify-center">
          <FolderKanban className="h-10 w-10 text-cyan-900" />
        </div>
      )}
      <div className="absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-gray-950/90 to-transparent" />
      <span
        className={classNames(
          'absolute left-3 top-3 rounded-full border px-2 py-1 text-[10px] font-semibold uppercase tracking-wide',
          workflowLabels[card.workflowState].className,
        )}
      >
        {workflowLabels[card.workflowState].label}
      </span>
      {card.replicaWarnings > 0 ? (
        <span className="absolute right-3 top-3 inline-flex items-center gap-1 rounded-full border border-amber-800 bg-amber-950/80 px-2 py-1 text-[10px] text-amber-200">
          <ServerOff className="h-3 w-3" /> {card.replicaWarnings}
        </span>
      ) : null}
    </div>
  );
}

function ProjectCard({ card }: { card: ProjectCardSummary }) {
  const project = card.project;
  return (
    <Link
      href={`/projects/${encodeURIComponent(project.id)}`}
      className="group flex min-w-0 flex-col overflow-hidden rounded-md border border-gray-800 bg-gray-900/45 shadow-sm transition hover:-translate-y-0.5 hover:border-gray-700 hover:bg-gray-900/70"
    >
      <ProjectPreview card={card} />
      <div className="flex min-w-0 flex-1 flex-col p-3.5">
        <div className="flex min-w-0 items-start gap-3">
          <div className="min-w-0 flex-1">
            <h2 className="truncate text-sm font-semibold text-gray-100">{project.name}</h2>
            <p className="mt-1 line-clamp-2 min-h-8 text-xs leading-relaxed text-gray-500">
              {project.description || 'A focused workspace for datasets, training, and generated results.'}
            </p>
          </div>
          <ArrowRight className="mt-0.5 h-4 w-4 flex-none text-gray-600 transition group-hover:translate-x-0.5 group-hover:text-cyan-300" />
        </div>
        <div className="mt-4 grid grid-cols-4 gap-1 rounded-sm border border-gray-800 bg-gray-950/70 p-1">
          {[
            { icon: Database, label: 'Data', value: card.counts.datasets },
            { icon: Play, label: 'Runs', value: card.counts.jobs },
            { icon: ImageIcon, label: 'Output', value: card.counts.outputs },
            { icon: Boxes, label: 'Models', value: card.counts.models },
          ].map(metric => (
            <div key={metric.label} className="min-w-0 px-1 py-1.5 text-center">
              <metric.icon className="mx-auto h-3.5 w-3.5 text-gray-600" />
              <div className="mt-1 text-xs font-semibold text-gray-200">{metric.value}</div>
              <div className="truncate text-[9px] uppercase tracking-wide text-gray-600">{metric.label}</div>
            </div>
          ))}
        </div>
        <div className="mt-3 flex min-w-0 items-center justify-between gap-2 text-[11px] text-gray-600">
          <span className="inline-flex min-w-0 items-center gap-1 truncate">
            <HardDrive className="h-3 w-3 flex-none" /> {formatBytes(card.totalBytes)}
          </span>
          <span className="flex-none">Updated {formatProjectTime(project.updated_at)}</span>
        </div>
      </div>
    </Link>
  );
}

export default function ProjectsPage() {
  const router = useRouter();
  const { settings, isSettingsLoaded } = useSettings();
  const [cards, setCards] = useState<ProjectCardSummary[]>([]);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [query, setQuery] = useState('');
  const [view, setView] = useState<'active' | 'archived'>('active');
  const [sort, setSort] = useState<'updated' | 'name' | 'activity'>('updated');
  const [createOpen, setCreateOpen] = useState(false);
  const projectsEnabled = settings.PROJECTS_ENABLED !== 'false';

  const refresh = () => {
    setStatus('loading');
    loadProjectCards()
      .then(nextCards => {
        setCards(nextCards);
        setStatus('success');
      })
      .catch(error => {
        console.error('Failed to load projects:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    if (!isSettingsLoaded) return;
    if (!projectsEnabled) {
      setCards([]);
      setStatus('success');
      return;
    }
    refresh();
  }, [isSettingsLoaded, projectsEnabled]);

  const archivedCount = cards.filter(card => projectIsArchived(card.project)).length;
  const activeCount = cards.length - archivedCount;
  const visibleCards = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const next = cards.filter(card => {
      if (projectIsArchived(card.project) !== (view === 'archived')) return false;
      if (!normalizedQuery) return true;
      return [card.project.name, card.project.slug, card.project.description]
        .filter(Boolean)
        .some(value => value.toLowerCase().includes(normalizedQuery));
    });
    return next.sort((a, b) => {
      if (sort === 'name') return a.project.name.localeCompare(b.project.name);
      if (sort === 'activity' && a.counts.activeJobs !== b.counts.activeJobs)
        return b.counts.activeJobs - a.counts.activeJobs;
      return new Date(b.project.updated_at).getTime() - new Date(a.project.updated_at).getTime();
    });
  }, [cards, query, sort, view]);

  const activeRuns = cards.reduce((total, card) => total + card.counts.activeJobs, 0);
  const readyModels = cards.reduce((total, card) => total + card.counts.models, 0);

  const handleCreated = (project: ProjectRecord) => {
    setCreateOpen(false);
    router.push(`/projects/${encodeURIComponent(project.id || project.slug)}`);
  };

  return (
    <>
      <TopBar>
        <div className="flex shrink-0 items-center gap-2">
          <FolderKanban className="h-4 w-4 text-cyan-300" />
          <h1 className="text-base font-semibold">Projects</h1>
        </div>
        <div className="flex-1" />
        {projectsEnabled ? (
          <button
            type="button"
            onClick={() => setCreateOpen(true)}
            className="operator-button h-9 border-cyan-800 bg-cyan-950/60 text-cyan-100 hover:bg-cyan-900/60"
          >
            <Plus className="h-4 w-4" /> New project
          </button>
        ) : null}
      </TopBar>

      <MainContent className="bg-gray-950 px-0 pt-12 sm:px-0">
        <div className="mx-auto max-w-[1480px] px-3 py-5 sm:px-5 lg:px-7">
          {isSettingsLoaded && !projectsEnabled ? (
            <div className="max-w-2xl">
              <PageNotice
                tone="warning"
                title="Project spaces are disabled"
                action={
                  <Link href="/settings" className="operator-button h-8 text-xs">
                    Open Settings
                  </Link>
                }
              >
                Existing project folders and records are preserved. Re-enable Projects when you want to use isolated
                workspaces.
              </PageNotice>
            </div>
          ) : (
            <div className="space-y-5">
              <section className="relative overflow-hidden rounded-md border border-gray-800 bg-[radial-gradient(circle_at_15%_0%,rgba(8,145,178,0.16),transparent_38%),linear-gradient(135deg,rgba(17,24,39,0.86),rgba(3,7,18,0.98))] p-5 sm:p-6">
                <div className="relative flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
                  <div className="max-w-2xl">
                    <div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">
                      Production workspaces
                    </div>
                    <h2 className="mt-2 text-xl font-semibold tracking-tight text-white sm:text-2xl">
                      From raw dataset to review-ready model
                    </h2>
                    <p className="mt-2 max-w-xl text-sm leading-relaxed text-gray-400">
                      Keep every dataset, run, output, model, and note together while the global workspace stays
                      available for quick tasks.
                    </p>
                  </div>
                  <div className="grid grid-cols-3 gap-2 sm:min-w-[390px]">
                    {[
                      { label: 'Active projects', value: activeCount, icon: FolderKanban },
                      { label: 'Running now', value: activeRuns, icon: Activity },
                      { label: 'Ready models', value: readyModels, icon: Boxes },
                    ].map(item => (
                      <div key={item.label} className="rounded-sm border border-gray-800 bg-gray-950/60 p-3">
                        <item.icon className="h-4 w-4 text-gray-600" />
                        <div className="mt-2 text-lg font-semibold text-gray-100">{item.value}</div>
                        <div className="text-[10px] uppercase tracking-wide text-gray-600">{item.label}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </section>

              <section className="grid items-start gap-4 xl:grid-cols-[300px_minmax(0,1fr)]">
                <Link
                  href="/dashboard"
                  className="group flex min-h-44 flex-col rounded-md border border-dashed border-gray-700 bg-gray-900/25 p-4 transition hover:border-cyan-800 hover:bg-cyan-950/10"
                >
                  <span className="flex h-10 w-10 items-center justify-center rounded-md border border-gray-700 bg-gray-950 text-gray-400 group-hover:text-cyan-300">
                    <Globe2 className="h-5 w-5" />
                  </span>
                  <div className="mt-4 text-sm font-semibold text-gray-200">Global Workspace</div>
                  <p className="mt-1 text-xs leading-relaxed text-gray-500">
                    Open shared datasets, queues, and generation tools outside any project.
                  </p>
                  <span className="mt-auto inline-flex items-center gap-1 pt-4 text-xs font-medium text-cyan-400">
                    Open dashboard <ArrowRight className="h-3.5 w-3.5 transition group-hover:translate-x-0.5" />
                  </span>
                </Link>

                <div className="min-w-0">
                  <div className="flex flex-col gap-3 border-b border-gray-800 pb-3 lg:flex-row lg:items-center">
                    <div className="flex min-w-0 items-center gap-1 rounded-sm border border-gray-800 bg-gray-900/50 p-1">
                      <button
                        type="button"
                        onClick={() => setView('active')}
                        className={classNames(
                          'h-8 rounded-sm px-3 text-xs font-medium',
                          view === 'active' ? 'bg-gray-700 text-gray-100' : 'text-gray-500 hover:text-gray-200',
                        )}
                      >
                        Active <span className="ml-1 text-gray-500">{activeCount}</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => setView('archived')}
                        className={classNames(
                          'inline-flex h-8 items-center gap-1.5 rounded-sm px-3 text-xs font-medium',
                          view === 'archived' ? 'bg-gray-700 text-gray-100' : 'text-gray-500 hover:text-gray-200',
                        )}
                      >
                        <Archive className="h-3.5 w-3.5" /> Archived{' '}
                        <span className="text-gray-500">{archivedCount}</span>
                      </button>
                    </div>
                    <label className="relative min-w-0 flex-1">
                      <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-600" />
                      <input
                        value={query}
                        onChange={event => setQuery(event.target.value)}
                        placeholder="Search projects"
                        className="h-10 w-full rounded-sm border border-gray-800 bg-gray-950 pl-9 pr-3 text-sm text-gray-100 outline-none placeholder:text-gray-600 focus:border-cyan-700"
                      />
                    </label>
                    <label className="relative flex-none">
                      <SlidersHorizontal className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-600" />
                      <select
                        value={sort}
                        onChange={event => setSort(event.target.value as typeof sort)}
                        className="h-10 rounded-sm border border-gray-800 bg-gray-950 pl-9 pr-8 text-xs text-gray-300 outline-none focus:border-cyan-700"
                      >
                        <option value="updated">Recently updated</option>
                        <option value="name">Project name</option>
                        <option value="activity">Active work first</option>
                      </select>
                    </label>
                  </div>

                  {status === 'loading' ? (
                    <div className="flex min-h-64 items-center justify-center text-sm text-gray-500">
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading studios
                    </div>
                  ) : status === 'error' ? (
                    <div className="py-4">
                      <PageNotice tone="danger" title="Projects could not be loaded">
                        Check the database connection, then refresh this page.
                      </PageNotice>
                    </div>
                  ) : visibleCards.length === 0 ? (
                    <div className="mt-4 flex min-h-64 items-center justify-center rounded-md border border-dashed border-gray-800 bg-gray-900/20 p-6 text-center">
                      <div>
                        <FolderKanban className="mx-auto h-8 w-8 text-gray-700" />
                        <h3 className="mt-3 text-sm font-semibold text-gray-300">
                          {query
                            ? 'No projects match your search'
                            : view === 'archived'
                              ? 'No archived projects'
                              : 'Create your first production workspace'}
                        </h3>
                        <p className="mt-1 text-xs text-gray-600">
                          {query
                            ? 'Try a different project name or description.'
                            : 'Prepare a dataset, train, and review the result in one place.'}
                        </p>
                        {!query && view === 'active' ? (
                          <button
                            type="button"
                            onClick={() => setCreateOpen(true)}
                            className="operator-button mt-4 h-9"
                          >
                            <Plus className="h-4 w-4" /> New project
                          </button>
                        ) : null}
                      </div>
                    </div>
                  ) : (
                    <div className="mt-4 grid gap-4 sm:grid-cols-2 2xl:grid-cols-3">
                      {visibleCards.map(card => (
                        <ProjectCard key={card.project.id} card={card} />
                      ))}
                    </div>
                  )}
                </div>
              </section>
            </div>
          )}
        </div>
      </MainContent>

      <QuickCreateProjectModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        projects={cards.filter(card => !projectIsArchived(card.project)).map(card => card.project)}
        onCreated={handleCreated}
      />
    </>
  );
}
