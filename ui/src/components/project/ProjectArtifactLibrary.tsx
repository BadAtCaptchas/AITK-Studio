'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useSearchParams } from 'next/navigation';
import classNames from 'classnames';
import {
  Boxes,
  Check,
  ChevronRight,
  Cloud,
  Download,
  ExternalLink,
  File,
  FileArchive,
  FileAudio,
  FileImage,
  FileVideo,
  Folder,
  Grid2X2,
  HardDrive,
  Image as ImageIcon,
  Info,
  List,
  Loader2,
  Play,
  RefreshCcw,
  Search,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  TriangleAlert,
  X,
} from 'lucide-react';
import { openConfirm } from '@/components/ConfirmModal';
import { PageNotice } from '@/components/OperatorPrimitives';
import { apiClient } from '@/utils/api';
import { artifactDownloadUrl, artifactPreviewUrl, loadProjectArtifacts } from './data';
import { useProjectWorkspace } from './ProjectContext';
import { formatBytes, formatProjectTime } from './ProjectWorkspaceShell';
import type { ProjectArtifact, ProjectArtifactAvailability, ProjectArtifactZone } from './types';

type LibraryMode = 'outputs' | 'models';
type ViewMode = 'grid' | 'list';

const availabilityMeta: Record<
  ProjectArtifactAvailability,
  { label: string; className: string; icon: typeof HardDrive }
> = {
  local: { label: 'Local', className: 'border-emerald-900 bg-emerald-950/50 text-emerald-200', icon: HardDrive },
  remote: { label: 'Remote', className: 'border-cyan-900 bg-cyan-950/50 text-cyan-200', icon: Cloud },
  both: { label: 'Synced', className: 'border-violet-900 bg-violet-950/50 text-violet-200', icon: Check },
  missing: { label: 'Unavailable', className: 'border-amber-900 bg-amber-950/50 text-amber-200', icon: TriangleAlert },
};

function artifactIcon(artifact: ProjectArtifact) {
  if (artifact.kind === 'folder') return Folder;
  if (artifact.mediaKind === 'image') return FileImage;
  if (artifact.mediaKind === 'video') return FileVideo;
  if (artifact.mediaKind === 'audio') return FileAudio;
  if (artifact.mediaKind === 'archive') return FileArchive;
  if (artifact.mediaKind === 'model') return Boxes;
  return File;
}

function metadataText(metadata: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    const value = metadata[key];
    if (typeof value === 'string' && value.trim()) return value;
    if (typeof value === 'number' && Number.isFinite(value)) return `${value}`;
    if (Array.isArray(value)) {
      const values = value.filter((item): item is string => typeof item === 'string' && !!item.trim());
      if (values.length) return values.join(', ');
    }
  }
  return '';
}

function modelType(artifact: ProjectArtifact) {
  const explicit = metadataText(artifact.metadata, ['modelType', 'model_type', 'network_type', 'type']).toLowerCase();
  if (explicit.includes('lora') || artifact.name.toLowerCase().includes('lora')) return 'LoRA';
  if (explicit.includes('adapter')) return 'Adapter';
  if (explicit.includes('checkpoint') || /\.ckpt$/i.test(artifact.name)) return 'Checkpoint';
  return artifact.name.split('.').pop()?.toUpperCase() || 'Model';
}

function ArtifactMedia({ artifact, contain = false }: { artifact: ProjectArtifact; contain?: boolean }) {
  const source = artifactPreviewUrl(artifact);
  const Icon = artifactIcon(artifact);
  if (source && artifact.mediaKind === 'image') {
    return (
      <img
        src={source}
        alt={artifact.name}
        loading="lazy"
        className={classNames('h-full w-full', contain ? 'object-contain' : 'object-cover')}
        onError={event => {
          event.currentTarget.style.display = 'none';
        }}
      />
    );
  }
  if (source && artifact.mediaKind === 'video') {
    return (
      <video
        src={source}
        controls={contain}
        muted={!contain}
        preload="metadata"
        className={classNames('h-full w-full', contain ? 'object-contain' : 'object-cover')}
      />
    );
  }
  if (source && artifact.mediaKind === 'audio') {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-3 p-4">
        <FileAudio className="h-10 w-10 text-violet-300" />
        <audio src={source} controls className="w-full max-w-md" />
      </div>
    );
  }
  return (
    <div className="flex h-full w-full flex-col items-center justify-center bg-[radial-gradient(circle_at_center,rgba(8,145,178,0.10),transparent_58%)]">
      <Icon className={classNames('text-gray-700', contain ? 'h-16 w-16' : 'h-9 w-9')} />
      {artifact.mediaKind === 'model' ? (
        <span className="mt-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-gray-600">
          {modelType(artifact)}
        </span>
      ) : null}
    </div>
  );
}

function AvailabilityBadge({ artifact }: { artifact: ProjectArtifact }) {
  const meta = availabilityMeta[artifact.availability];
  const Icon = meta.icon;
  return (
    <span
      className={classNames(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium',
        meta.className,
      )}
    >
      <Icon className="h-3 w-3" /> {meta.label}
    </span>
  );
}

function errorStatus(error: unknown) {
  if (!error || typeof error !== 'object' || !('response' in error)) return 0;
  return (error as { response?: { status?: number } }).response?.status || 0;
}

export default function ProjectArtifactLibrary({ mode }: { mode: LibraryMode }) {
  const searchParams = useSearchParams();
  const { projectID, apiBase, archived } = useProjectWorkspace();
  const zone: ProjectArtifactZone = mode;
  const basePath = `/projects/${encodeURIComponent(projectID)}`;
  const [artifacts, setArtifacts] = useState<ProjectArtifact[]>([]);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [query, setQuery] = useState('');
  const [kindFilter, setKindFilter] = useState('all');
  const [sourceFilter, setSourceFilter] = useState('all');
  const [availabilityFilter, setAvailabilityFilter] = useState('all');
  const [sort, setSort] = useState<'newest' | 'name' | 'size'>('newest');
  const [view, setView] = useState<ViewMode>('grid');
  const [selectedID, setSelectedID] = useState<string | null>(searchParams.get('selected'));
  const [deleteStatus, setDeleteStatus] = useState<string | null>(null);

  const refresh = () => {
    setStatus(current => (artifacts.length > 0 ? current : 'loading'));
    loadProjectArtifacts(projectID, zone)
      .then(next => {
        setArtifacts(next);
        setStatus('success');
      })
      .catch(error => {
        console.error(`Failed to load project ${mode}:`, error);
        setStatus('error');
      });
  };

  useEffect(() => {
    refresh();
  }, [projectID, zone]);

  const filteredArtifacts = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return artifacts
      .filter(artifact => {
        if (
          normalizedQuery &&
          ![artifact.name, artifact.relativePath, artifact.workerName || ''].some(value =>
            value.toLowerCase().includes(normalizedQuery),
          )
        )
          return false;
        if (kindFilter !== 'all') {
          if (mode === 'outputs' && artifact.mediaKind !== kindFilter) return false;
          if (mode === 'models' && modelType(artifact).toLowerCase() !== kindFilter) return false;
        }
        if (sourceFilter !== 'all' && artifact.sourceKind !== sourceFilter) return false;
        if (availabilityFilter !== 'all' && artifact.availability !== availabilityFilter) return false;
        return true;
      })
      .sort((a, b) => {
        if (sort === 'name') return a.name.localeCompare(b.name);
        if (sort === 'size') return b.size - a.size;
        return new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
      });
  }, [artifacts, availabilityFilter, kindFilter, mode, query, sort, sourceFilter]);

  const selected = artifacts.find(artifact => artifact.id === selectedID) || null;
  const totalBytes = artifacts.reduce((total, artifact) => total + artifact.size, 0);
  const remoteCount = artifacts.filter(artifact => artifact.availability === 'remote').length;

  const removeArtifact = (artifact: ProjectArtifact) => {
    if (archived || artifact.kind !== 'file') return;
    openConfirm({
      title: `Delete ${artifact.name}?`,
      message: 'This removes the project output from storage. Runs and models that reference it are not deleted.',
      confirmText: 'Delete output',
      type: 'danger',
      onConfirm: async () => {
        setDeleteStatus(artifact.id);
        try {
          try {
            await apiClient.delete(`${apiBase}/artifacts/${encodeURIComponent(artifact.id)}`, {
              data: { revision: artifact.metadata.revision },
            });
          } catch (error: unknown) {
            if (![404, 405].includes(errorStatus(error))) throw error;
            await apiClient.delete(`${apiBase}/files`, { data: { path: artifact.path || artifact.relativePath } });
          }
          setArtifacts(current => current.filter(item => item.id !== artifact.id));
          setSelectedID(current => (current === artifact.id ? null : current));
        } catch (error) {
          console.error('Failed to delete project output:', error);
        } finally {
          setDeleteStatus(null);
        }
      },
    });
  };

  const kindOptions =
    mode === 'outputs'
      ? [
          ['all', 'All media'],
          ['image', 'Images'],
          ['video', 'Video'],
          ['audio', 'Audio'],
          ['other', 'Other'],
        ]
      : [
          ['all', 'All models'],
          ['lora', 'LoRA'],
          ['checkpoint', 'Checkpoint'],
          ['adapter', 'Adapter'],
        ];

  return (
    <div className="relative flex h-full min-h-0 flex-col overflow-hidden">
      <div className="flex-none border-b border-gray-900 bg-gray-950 px-3 py-3 sm:px-4">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-center">
          <label className="relative min-w-0 flex-1 xl:max-w-md">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-600" />
            <input
              value={query}
              onChange={event => setQuery(event.target.value)}
              placeholder={`Search ${mode}`}
              className="h-9 w-full rounded-sm border border-gray-800 bg-gray-950 pl-9 pr-3 text-sm text-gray-100 outline-none placeholder:text-gray-600 focus:border-cyan-700"
            />
          </label>
          <div className="operator-scrollbar-none flex min-w-0 items-center gap-2 overflow-x-auto">
            <select
              value={kindFilter}
              onChange={event => setKindFilter(event.target.value)}
              className="h-9 flex-none rounded-sm border border-gray-800 bg-gray-950 px-2.5 text-xs text-gray-300 outline-none focus:border-cyan-700"
            >
              {kindOptions.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
            <select
              value={sourceFilter}
              onChange={event => setSourceFilter(event.target.value)}
              className="h-9 flex-none rounded-sm border border-gray-800 bg-gray-950 px-2.5 text-xs text-gray-300 outline-none focus:border-cyan-700"
            >
              <option value="all">All sources</option>
              <option value="training">Training</option>
              <option value="generation">Generation</option>
              <option value="import">Imported</option>
            </select>
            <select
              value={availabilityFilter}
              onChange={event => setAvailabilityFilter(event.target.value)}
              className="h-9 flex-none rounded-sm border border-gray-800 bg-gray-950 px-2.5 text-xs text-gray-300 outline-none focus:border-cyan-700"
            >
              <option value="all">Anywhere</option>
              <option value="local">Local</option>
              <option value="remote">Remote</option>
              <option value="both">Synced</option>
              <option value="missing">Unavailable</option>
            </select>
            <label className="relative flex-none">
              <SlidersHorizontal className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-gray-600" />
              <select
                value={sort}
                onChange={event => setSort(event.target.value as typeof sort)}
                className="h-9 rounded-sm border border-gray-800 bg-gray-950 pl-8 pr-7 text-xs text-gray-300 outline-none focus:border-cyan-700"
              >
                <option value="newest">Newest</option>
                <option value="name">Name</option>
                <option value="size">Largest</option>
              </select>
            </label>
            <div className="flex flex-none rounded-sm border border-gray-800 bg-gray-950 p-0.5">
              <button
                type="button"
                onClick={() => setView('grid')}
                aria-label="Grid view"
                className={classNames(
                  'operator-icon-button h-8 w-8',
                  view === 'grid' ? 'bg-gray-800 text-cyan-200' : '',
                )}
              >
                <Grid2X2 className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={() => setView('list')}
                aria-label="List view"
                className={classNames(
                  'operator-icon-button h-8 w-8',
                  view === 'list' ? 'bg-gray-800 text-cyan-200' : '',
                )}
              >
                <List className="h-4 w-4" />
              </button>
            </div>
            <button type="button" onClick={refresh} className="operator-icon-button h-9 w-9 flex-none" title="Refresh">
              <RefreshCcw className={classNames('h-4 w-4', status === 'loading' ? 'animate-spin' : '')} />
            </button>
          </div>
        </div>
        <div className="mt-2 flex items-center gap-3 text-[11px] text-gray-600">
          <span>
            {filteredArtifacts.length} of {artifacts.length} {mode}
          </span>
          <span>/</span>
          <span>{formatBytes(totalBytes)}</span>
          {remoteCount > 0 ? (
            <>
              <span>/</span>
              <span>{remoteCount} remote</span>
            </>
          ) : null}
        </div>
      </div>

      <div className="relative min-h-0 flex-1 overflow-auto p-3 sm:p-4">
        {status === 'loading' && artifacts.length === 0 ? (
          <div className="flex min-h-72 items-center justify-center text-sm text-gray-500">
            <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Indexing project {mode}
          </div>
        ) : status === 'error' ? (
          <PageNotice tone="danger" title={`${mode === 'models' ? 'Models' : 'Outputs'} could not be loaded`}>
            The project artifact index is unavailable. Project files remain unchanged.
          </PageNotice>
        ) : filteredArtifacts.length === 0 ? (
          <div className="flex min-h-72 items-center justify-center rounded-md border border-dashed border-gray-800 bg-gray-900/20 p-6 text-center">
            <div>
              {mode === 'models' ? (
                <Boxes className="mx-auto h-9 w-9 text-gray-700" />
              ) : (
                <ImageIcon className="mx-auto h-9 w-9 text-gray-700" />
              )}
              <h3 className="mt-3 text-sm font-semibold text-gray-300">
                {query || kindFilter !== 'all' || sourceFilter !== 'all' || availabilityFilter !== 'all'
                  ? `No ${mode} match these filters`
                  : `No project ${mode} yet`}
              </h3>
              <p className="mt-1 text-xs text-gray-600">
                {mode === 'models'
                  ? 'Checkpoints and project model files will be indexed here.'
                  : 'Training samples and generated media will appear here.'}
              </p>
              {!archived ? (
                <Link
                  href={mode === 'models' ? `${basePath}/runs/new` : `${basePath}/generate`}
                  className="operator-button mt-4 h-9"
                >
                  {mode === 'models' ? <Play className="h-4 w-4" /> : <Sparkles className="h-4 w-4" />}
                  {mode === 'models' ? 'Start a run' : 'Generate'}
                </Link>
              ) : null}
            </div>
          </div>
        ) : view === 'grid' ? (
          <div
            className={classNames(
              'grid gap-3',
              selected
                ? 'sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4'
                : 'sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5',
            )}
          >
            {filteredArtifacts.map(artifact => {
              const selectedArtifact = selectedID === artifact.id;
              return (
                <button
                  key={artifact.id}
                  type="button"
                  onClick={() => setSelectedID(artifact.id)}
                  className={classNames(
                    'group min-w-0 overflow-hidden rounded-md border bg-gray-900/40 text-left transition hover:border-gray-600',
                    selectedArtifact ? 'border-cyan-700 ring-1 ring-cyan-900' : 'border-gray-800',
                  )}
                >
                  <div
                    className={classNames(
                      'relative overflow-hidden border-b border-gray-800 bg-gray-950',
                      mode === 'models' ? 'aspect-[4/3]' : 'aspect-square',
                    )}
                  >
                    <ArtifactMedia artifact={artifact} />
                    <div className="absolute left-2 top-2">
                      <AvailabilityBadge artifact={artifact} />
                    </div>
                    {artifact.kind === 'folder' ? (
                      <span className="absolute right-2 top-2 rounded-full border border-gray-700 bg-gray-950/80 px-2 py-0.5 text-[10px] text-gray-400">
                        Folder
                      </span>
                    ) : null}
                  </div>
                  <div className="p-3">
                    <div className="truncate text-sm font-medium text-gray-200">{artifact.name}</div>
                    <div className="mt-1 flex min-w-0 items-center justify-between gap-2 text-[11px] text-gray-600">
                      <span className="truncate">{mode === 'models' ? modelType(artifact) : artifact.sourceKind}</span>
                      <span className="flex-none">{formatBytes(artifact.size)}</span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        ) : (
          <div className="overflow-hidden rounded-md border border-gray-800 bg-gray-900/30">
            <div className="hidden grid-cols-[minmax(0,1fr)_110px_100px_130px_32px] gap-3 border-b border-gray-800 px-3 py-2 text-[10px] font-semibold uppercase tracking-wide text-gray-600 md:grid">
              <span>Name</span>
              <span>Source</span>
              <span>Size</span>
              <span>Updated</span>
              <span />
            </div>
            <div className="divide-y divide-gray-800/80">
              {filteredArtifacts.map(artifact => {
                const Icon = artifactIcon(artifact);
                return (
                  <button
                    key={artifact.id}
                    type="button"
                    onClick={() => setSelectedID(artifact.id)}
                    className={classNames(
                      'grid w-full min-w-0 grid-cols-[minmax(0,1fr)_32px] items-center gap-3 px-3 py-2.5 text-left hover:bg-gray-900/80 md:grid-cols-[minmax(0,1fr)_110px_100px_130px_32px]',
                      selectedID === artifact.id ? 'bg-cyan-950/20' : '',
                    )}
                  >
                    <span className="flex min-w-0 items-center gap-3">
                      <span className="flex h-9 w-9 flex-none items-center justify-center rounded-sm border border-gray-800 bg-gray-950">
                        <Icon className="h-4 w-4 text-gray-500" />
                      </span>
                      <span className="min-w-0">
                        <span className="block truncate text-sm font-medium text-gray-200">{artifact.name}</span>
                        <span className="block truncate text-[11px] text-gray-600">{artifact.relativePath}</span>
                      </span>
                    </span>
                    <span className="hidden text-xs capitalize text-gray-500 md:block">
                      {mode === 'models' ? modelType(artifact) : artifact.sourceKind}
                    </span>
                    <span className="hidden text-xs text-gray-500 md:block">{formatBytes(artifact.size)}</span>
                    <span className="hidden text-xs text-gray-500 md:block">
                      {formatProjectTime(artifact.updatedAt)}
                    </span>
                    <ChevronRight className="h-4 w-4 text-gray-600" />
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {selected ? (
        <div className="absolute inset-0 z-20 flex justify-end bg-gray-950/70 backdrop-blur-[1px] lg:left-auto lg:w-[420px] lg:border-l lg:border-gray-800 lg:bg-gray-950 lg:backdrop-blur-none">
          <aside className="flex h-full w-full min-w-0 flex-col bg-gray-950 sm:max-w-[520px] lg:max-w-none">
            <div className="flex h-14 flex-none items-center gap-3 border-b border-gray-800 px-3">
              <Info className="h-4 w-4 text-cyan-300" />
              <div className="min-w-0 flex-1 truncate text-sm font-semibold text-gray-200">{selected.name}</div>
              <button
                type="button"
                onClick={() => setSelectedID(null)}
                className="operator-icon-button"
                aria-label="Close inspector"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-auto">
              <div
                className={classNames(
                  'flex items-center justify-center overflow-hidden border-b border-gray-800 bg-black/20',
                  mode === 'models' ? 'h-56' : 'min-h-72 max-h-[52vh]',
                )}
              >
                <ArtifactMedia artifact={selected} contain />
              </div>
              <div className="space-y-4 p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <AvailabilityBadge artifact={selected} />
                  <span className="rounded-full border border-gray-800 bg-gray-900 px-2 py-0.5 text-[10px] capitalize text-gray-400">
                    {mode === 'models' ? modelType(selected) : selected.mediaKind}
                  </span>
                </div>
                <dl className="grid grid-cols-[100px_minmax(0,1fr)] gap-x-3 gap-y-2 text-xs">
                  <dt className="text-gray-600">Project path</dt>
                  <dd className="break-all font-mono text-gray-300">{selected.relativePath}</dd>
                  <dt className="text-gray-600">Size</dt>
                  <dd className="text-gray-300">{formatBytes(selected.size)}</dd>
                  <dt className="text-gray-600">Updated</dt>
                  <dd className="text-gray-300">{formatProjectTime(selected.updatedAt)}</dd>
                  <dt className="text-gray-600">Source</dt>
                  <dd className="capitalize text-gray-300">
                    {selected.sourceKind}
                    {selected.workerName ? ` / ${selected.workerName}` : ''}
                  </dd>
                  {mode === 'models' ? (
                    <>
                      <dt className="text-gray-600">Base model</dt>
                      <dd className="text-gray-300">
                        {metadataText(selected.metadata, ['baseModel', 'base_model', 'ss_base_model_version']) ||
                          'Not reported'}
                      </dd>
                      <dt className="text-gray-600">Trigger words</dt>
                      <dd className="break-words text-gray-300">
                        {metadataText(selected.metadata, ['triggerWords', 'trigger_words', 'ss_tag_frequency']) ||
                          'Not reported'}
                      </dd>
                    </>
                  ) : null}
                </dl>
                {selected.availability === 'missing' ? (
                  <PageNotice tone="warning" title="Artifact is unavailable">
                    The index knows about this artifact, but no reachable replica currently has its bytes.
                  </PageNotice>
                ) : null}
              </div>
            </div>
            <div className="flex flex-none flex-wrap gap-2 border-t border-gray-800 p-3">
              {artifactDownloadUrl(selected) && selected.availability !== 'missing' ? (
                <a href={artifactDownloadUrl(selected) || '#'} className="operator-button h-9">
                  <Download className="h-4 w-4" /> Download
                </a>
              ) : null}
              <Link
                href={`${basePath}/files?path=${encodeURIComponent(selected.relativePath)}`}
                className="operator-button h-9"
              >
                <ExternalLink className="h-4 w-4" /> Files
              </Link>
              {selected.sourceRunId ? (
                <Link
                  href={`${basePath}/runs/${encodeURIComponent(selected.sourceRunId)}`}
                  className="operator-button h-9"
                >
                  <Play className="h-4 w-4" /> Source run
                </Link>
              ) : null}
              {mode === 'models' && !archived && selected.availability !== 'missing' ? (
                <Link
                  href={`${basePath}/generate?model_ref=${encodeURIComponent(selected.portableRef || selected.relativePath)}`}
                  className="operator-button h-9 border-violet-900 bg-violet-950/40 text-violet-100"
                >
                  <Sparkles className="h-4 w-4" /> Generate
                </Link>
              ) : null}
              {mode === 'outputs' && !archived && selected.kind === 'file' ? (
                <button
                  type="button"
                  onClick={() => removeArtifact(selected)}
                  disabled={deleteStatus === selected.id}
                  className="operator-button h-9 border-rose-900 bg-rose-950/30 text-rose-200 hover:bg-rose-900/50"
                >
                  {deleteStatus === selected.id ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}{' '}
                  Delete
                </button>
              ) : null}
            </div>
          </aside>
        </div>
      ) : null}
    </div>
  );
}
