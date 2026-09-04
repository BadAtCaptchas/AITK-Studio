'use client';

import { use, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  ArrowLeft,
  ChevronRight,
  Copy,
  Download,
  FileAudio,
  FileText,
  FileVideo,
  Folder,
  Image as ImageIcon,
  Loader2,
  Pencil,
  Search,
  ShieldAlert,
  Trash2,
} from 'lucide-react';
import ProjectWorkspaceShell, { formatBytes, formatProjectTime } from '@/components/project/ProjectWorkspaceShell';
import { PageNotice } from '@/components/OperatorPrimitives';
import { Modal } from '@/components/Modal';
import { openConfirm } from '@/components/ConfirmModal';
import { apiClient } from '@/utils/api';
import type { ProjectFileTreeItem } from '@/components/project/types';

type MediaKind = 'image' | 'video' | 'audio';

type FilePreview = {
  item: ProjectFileTreeItem;
  children?: ProjectFileTreeItem[];
  content?: string;
  mediaUrl?: string;
  mediaKind?: MediaKind;
  downloadUrl?: string;
  truncated?: boolean;
};

const protectedZones = new Set(['datasets', 'runs', 'models']);

function itemIcon(item: Pick<ProjectFileTreeItem, 'kind' | 'name'>, className = 'h-4 w-4') {
  if (item.kind === 'folder') return <Folder className={`${className} text-brand-300`} />;
  const extension = item.name.split('.').pop()?.toLowerCase();
  if (['png', 'jpg', 'jpeg', 'webp', 'jxl', 'gif', 'bmp'].includes(extension || '')) {
    return <ImageIcon className={`${className} text-violet-300`} />;
  }
  if (['mp4', 'webm', 'mov', 'mkv'].includes(extension || '')) {
    return <FileVideo className={`${className} text-brand-300`} />;
  }
  if (['mp3', 'wav', 'flac', 'ogg', 'm4a'].includes(extension || '')) {
    return <FileAudio className={`${className} text-emerald-300`} />;
  }
  return <FileText className={`${className} text-gray-500`} />;
}

function isProtectedItem(item: ProjectFileTreeItem | undefined) {
  if (!item?.relativePath) return true;
  const segments = item.relativePath.split(/[\\/]+/).filter(Boolean);
  return segments.length === 1 || protectedZones.has((segments[0] || '').toLowerCase());
}

export default function ProjectFilesPage({ params }: { params: Promise<{ projectID: string }> }) {
  const { projectID: rawProjectID } = use(params);
  const projectID = decodeURIComponent(rawProjectID);
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedPath = searchParams.get('path') || '';
  const filesHref = `/projects/${encodeURIComponent(projectID)}/files`;
  const [rootPreview, setRootPreview] = useState<FilePreview | null>(null);
  const [preview, setPreview] = useState<FilePreview | null>(null);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [previewStatus, setPreviewStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [message, setMessage] = useState<{ tone: 'neutral' | 'danger' | 'success'; title: string; body: string } | null>(null);
  const [filterText, setFilterText] = useState('');
  const [searchResults, setSearchResults] = useState<ProjectFileTreeItem[] | null>(null);
  const [searchTruncated, setSearchTruncated] = useState(false);
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameValue, setRenameValue] = useState('');
  const [mutationPending, setMutationPending] = useState(false);
  const searchRequest = useRef<AbortController | null>(null);

  const selectPath = (path?: string) => {
    router.push(path ? `${filesHref}?path=${encodeURIComponent(path)}` : filesHref);
  };

  const loadRoot = async () => {
    setStatus('loading');
    try {
      const response = await apiClient.get(`/api/projects/${encodeURIComponent(projectID)}/files`);
      setRootPreview(response.data);
      setStatus('success');
      if (!selectedPath) setPreview(response.data);
    } catch (error) {
      console.error('Failed to load project file root:', error);
      setStatus('error');
    }
  };

  const loadPreview = async (path?: string) => {
    setPreviewStatus('loading');
    try {
      const response = await apiClient.get(`/api/projects/${encodeURIComponent(projectID)}/files`, {
        params: path ? { path } : {},
      });
      setPreview(response.data);
      setPreviewStatus('idle');
    } catch (error) {
      console.error('Failed to load project file preview:', error);
      setPreviewStatus('error');
    }
  };

  useEffect(() => {
    void loadRoot();
  }, [projectID]);

  useEffect(() => {
    void loadPreview(selectedPath || undefined);
  }, [projectID, selectedPath]);

  useEffect(() => {
    const query = filterText.trim();
    searchRequest.current?.abort();
    if (!query) {
      setSearchResults(null);
      setSearchTruncated(false);
      return;
    }
    const controller = new AbortController();
    searchRequest.current = controller;
    const timer = window.setTimeout(() => {
      apiClient
        .get(`/api/projects/${encodeURIComponent(projectID)}/files`, {
          params: { search: query },
          signal: controller.signal,
        })
        .then(response => {
          setSearchResults(Array.isArray(response.data?.children) ? response.data.children : []);
          setSearchTruncated(response.data?.truncated === true);
        })
        .catch(error => {
          if (controller.signal.aborted) return;
          console.error('Failed to search project files:', error);
          setMessage({ tone: 'danger', title: 'Search failed', body: 'Project files could not be searched.' });
        });
    }, 250);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [filterText, projectID]);

  const breadcrumbs = useMemo(() => {
    const relativePath = preview?.item.relativePath || '';
    const parts = relativePath.split(/[\\/]+/).filter(Boolean);
    return parts.map((label, index) => ({
      label,
      path: parts.slice(0, index + 1).join('/'),
    }));
  }, [preview?.item.relativePath]);

  const navigationItems = searchResults ?? rootPreview?.children ?? [];
  const selectedProtected = isProtectedItem(preview?.item);

  const copyPath = async () => {
    if (!preview?.item.path) return;
    const portablePath = preview.item.path.replace(/\\/g, '/').split('/').map(encodeURIComponent).join('/');
    await navigator.clipboard.writeText(`aitk-project://${projectID}/${portablePath}`);
    setMessage({ tone: 'success', title: 'Portable reference copied', body: 'The project-relative asset reference is on your clipboard.' });
  };

  const openRename = () => {
    if (!preview?.item.path || selectedProtected) return;
    setRenameValue(preview.item.name);
    setRenameOpen(true);
  };

  const renameSelected = async () => {
    if (!preview?.item.path || !renameValue.trim() || mutationPending) return;
    setMutationPending(true);
    setMessage(null);
    try {
      const response = await apiClient.patch(`/api/projects/${encodeURIComponent(projectID)}/files`, {
        path: preview.item.path,
        newName: renameValue,
      });
      setRenameOpen(false);
      await loadRoot();
      selectPath(response.data?.path);
      setMessage({ tone: 'success', title: 'Item renamed', body: `Renamed to ${renameValue.trim()}.` });
    } catch (error: unknown) {
      const candidate = error as { response?: { data?: { error?: string } } };
      setMessage({ tone: 'danger', title: 'Rename failed', body: candidate.response?.data?.error || 'The item could not be renamed.' });
    } finally {
      setMutationPending(false);
    }
  };

  const deleteSelected = () => {
    if (!preview?.item.path || selectedProtected || mutationPending) return;
    const item = preview.item;
    openConfirm({
      title: `Delete ${item.kind}`,
      message: `Permanently delete “${item.relativePath || item.name}” from this project? This cannot be undone.`,
      confirmText: 'Delete',
      type: 'danger',
      onConfirm: async () => {
        setMutationPending(true);
        setMessage(null);
        try {
          await apiClient.delete(`/api/projects/${encodeURIComponent(projectID)}/files`, {
            data: { path: item.path, recursive: true },
          });
          await loadRoot();
          selectPath();
          setMessage({ tone: 'success', title: 'Item deleted', body: `${item.name} was removed from the project.` });
        } catch (error: unknown) {
          const candidate = error as { response?: { data?: { error?: string } } };
          setMessage({ tone: 'danger', title: 'Delete failed', body: candidate.response?.data?.error || 'The item could not be deleted.' });
        } finally {
          setMutationPending(false);
        }
      },
    });
  };

  return (
    <ProjectWorkspaceShell projectID={projectID} active="files" title="Files" description="Browse project zones and inspect local or synchronized assets.">
      <div className="grid h-full min-h-0 grid-cols-1 overflow-auto lg:grid-cols-[330px_minmax(0,1fr)] lg:overflow-hidden">
        <section className="flex min-h-[260px] flex-col border-b border-gray-900 bg-gray-950 lg:min-h-0 lg:border-b-0 lg:border-r">
          <div className="border-b border-gray-800 p-3">
            <label className="relative block">
              <span className="sr-only">Search project files</span>
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
              <input
                value={filterText}
                onChange={event => setFilterText(event.target.value)}
                placeholder="Search every project zone"
                className="h-10 w-full border border-gray-800 bg-gray-950 pl-9 pr-3 text-sm text-gray-100 placeholder:text-gray-500 outline-none focus:border-brand-600 focus:ring-1 focus:ring-brand-800"
              />
            </label>
            {searchTruncated ? <p className="mt-2 text-xs text-amber-300">Showing the first 250 matches.</p> : null}
          </div>

          <div className="min-h-0 flex-1 overflow-auto">
            {status === 'loading' && !rootPreview ? (
              <div className="flex h-40 items-center justify-center text-sm text-gray-400">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading files
              </div>
            ) : null}
            {status === 'error' ? (
              <div className="p-3">
                <PageNotice tone="danger" title="Files could not be loaded" action={<button className="operator-button h-8 text-xs" onClick={() => void loadRoot()}>Retry</button>}>
                  The registered project root is missing or unavailable.
                </PageNotice>
              </div>
            ) : null}
            <div className="divide-y divide-gray-900">
              {navigationItems.map(item => (
                <button
                  key={item.path}
                  type="button"
                  onClick={() => selectPath(item.path)}
                  className="grid min-h-12 w-full grid-cols-[minmax(0,1fr)_72px] items-center gap-3 px-3 py-2 text-left text-xs hover:bg-gray-900/70 focus:bg-gray-900/70 focus:outline-none"
                >
                  <span className="flex min-w-0 items-center gap-2.5">
                    {itemIcon(item)}
                    <span className="min-w-0">
                      <span className="block truncate text-sm text-gray-200">{searchResults ? item.relativePath : item.name}</span>
                      {!searchResults && item.kind === 'folder' ? <span className="text-[11px] text-gray-600">Project zone</span> : null}
                    </span>
                  </span>
                  <span className="text-right text-gray-600">{item.kind === 'folder' ? 'folder' : formatBytes(item.size)}</span>
                </button>
              ))}
              {rootPreview && navigationItems.length === 0 ? (
                <div className="px-4 py-10 text-center text-sm text-gray-500">{searchResults ? 'No files match this search.' : 'This project has no visible files.'}</div>
              ) : null}
            </div>
          </div>
        </section>

        <section className="min-h-[520px] overflow-auto bg-gray-950 p-3 sm:p-4 lg:min-h-0">
          <div className="mx-auto max-w-6xl space-y-4">
            <div className="border border-gray-800 bg-gray-950">
              <div className="flex min-h-12 flex-wrap items-center gap-1 border-b border-gray-800 px-3 py-2 text-xs text-gray-500">
                <button type="button" onClick={() => selectPath()} className="rounded-sm px-2 py-1 text-gray-300 hover:bg-gray-800 hover:text-white">
                  Project
                </button>
                {breadcrumbs.map(crumb => (
                  <span key={crumb.path} className="flex min-w-0 items-center gap-1">
                    <ChevronRight className="h-3.5 w-3.5 flex-none" />
                    <button type="button" onClick={() => selectPath(crumb.path)} className="max-w-48 truncate rounded-sm px-2 py-1 text-gray-300 hover:bg-gray-800 hover:text-white">
                      {crumb.label}
                    </button>
                  </span>
                ))}
              </div>
              <div className="flex flex-col gap-3 p-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex min-w-0 items-center gap-3">
                  <span className="flex h-10 w-10 flex-none items-center justify-center border border-gray-800 bg-gray-900">
                    {preview?.item ? itemIcon(preview.item, 'h-5 w-5') : <Folder className="h-5 w-5 text-brand-300" />}
                  </span>
                  <div className="min-w-0">
                    <h2 className="truncate text-base font-semibold text-gray-100">{preview?.item.relativePath || preview?.item.name || 'Project root'}</h2>
                    <p className="mt-0.5 text-xs text-gray-500">
                      {preview?.item.kind || 'folder'} {preview?.item.size ? `· ${formatBytes(preview.item.size)}` : ''}{' '}
                      {preview?.item.updatedAt ? `· ${formatProjectTime(preview.item.updatedAt)}` : ''}
                    </p>
                  </div>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <button type="button" onClick={() => void copyPath()} disabled={!preview?.item.path} className="operator-button h-9 text-xs">
                    <Copy className="h-4 w-4" /> Copy Reference
                  </button>
                  {preview?.downloadUrl ? (
                    <a href={preview.downloadUrl} className="operator-button h-9 text-xs">
                      <Download className="h-4 w-4" /> Download
                    </a>
                  ) : null}
                  <button type="button" onClick={openRename} disabled={!preview?.item.path || selectedProtected || mutationPending} className="operator-button h-9 text-xs">
                    <Pencil className="h-4 w-4" /> Rename
                  </button>
                  <button type="button" onClick={deleteSelected} disabled={!preview?.item.path || selectedProtected || mutationPending} className="operator-button h-9 border-rose-900 bg-rose-950/40 text-xs text-rose-100">
                    <Trash2 className="h-4 w-4" /> Delete
                  </button>
                </div>
              </div>
              {selectedProtected && preview?.item.relativePath ? (
                <div className="flex items-center gap-2 border-t border-amber-900/60 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">
                  <ShieldAlert className="h-4 w-4 flex-none" /> Managed project content must be changed from its dedicated workspace.
                </div>
              ) : null}
            </div>

            {message ? (
              <PageNotice tone={message.tone === 'success' ? 'success' : message.tone} title={message.title}>{message.body}</PageNotice>
            ) : null}
            {previewStatus === 'error' ? (
              <PageNotice tone="danger" title="Preview could not be loaded" action={<button className="operator-button h-8 text-xs" onClick={() => void loadPreview(selectedPath || undefined)}>Retry</button>}>
                Select another item or retry this project path.
              </PageNotice>
            ) : null}

            {previewStatus === 'loading' ? (
              <div className="flex h-72 items-center justify-center border border-gray-800 bg-gray-950 text-sm text-gray-400">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading preview
              </div>
            ) : preview?.item.kind === 'folder' ? (
              <div className="border border-gray-800 bg-gray-950">
                <div className="border-b border-gray-800 px-3 py-2 text-xs text-gray-500">{preview.children?.length || 0} items</div>
                <div className="divide-y divide-gray-900">
                  {(preview.children || []).map(child => (
                    <button key={child.path} type="button" onClick={() => selectPath(child.path)} className="flex min-h-12 w-full min-w-0 items-center gap-3 px-3 py-2 text-left hover:bg-gray-900/70 focus:bg-gray-900/70 focus:outline-none">
                      {itemIcon(child)}
                      <span className="min-w-0 flex-1 truncate text-sm text-gray-300">{child.name}</span>
                      <span className="text-xs text-gray-600">{child.kind === 'folder' ? 'folder' : formatBytes(child.size)}</span>
                    </button>
                  ))}
                  {(preview.children || []).length === 0 ? <div className="px-4 py-12 text-center text-sm text-gray-500">This folder is empty.</div> : null}
                </div>
              </div>
            ) : preview?.mediaUrl && preview.mediaKind === 'image' ? (
              <div className="border border-gray-800 bg-gray-950 p-3">
                <div className="flex min-h-[420px] items-center justify-center bg-black">
                  <img src={preview.mediaUrl} alt={preview.item.name} className="max-h-[72vh] max-w-full object-contain" />
                </div>
              </div>
            ) : preview?.mediaUrl && preview.mediaKind === 'video' ? (
              <div className="border border-gray-800 bg-black p-3"><video src={preview.mediaUrl} controls className="max-h-[72vh] w-full" /></div>
            ) : preview?.mediaUrl && preview.mediaKind === 'audio' ? (
              <div className="flex min-h-48 items-center border border-gray-800 bg-gray-950 p-6"><audio src={preview.mediaUrl} controls className="w-full" /></div>
            ) : typeof preview?.content === 'string' ? (
              <pre className="max-h-[72vh] overflow-auto border border-gray-800 bg-gray-950 p-4 text-xs leading-5 text-gray-300">{preview.content}</pre>
            ) : (
              <div className="flex h-72 items-center justify-center border border-dashed border-gray-800 bg-gray-950 px-6 text-center text-sm text-gray-500">
                <div><ArrowLeft className="mx-auto mb-3 h-8 w-8 text-gray-600" />Choose a project zone or file to inspect it here.</div>
              </div>
            )}
          </div>
        </section>
      </div>

      <Modal isOpen={renameOpen} onClose={() => setRenameOpen(false)} title="Rename project item" size="sm" closeOnOverlayClick={!mutationPending}>
        <form
          onSubmit={event => {
            event.preventDefault();
            void renameSelected();
          }}
          className="space-y-4"
        >
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-gray-400">New name</span>
            <input autoFocus value={renameValue} onChange={event => setRenameValue(event.target.value)} className="h-10 w-full border border-gray-700 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-brand-600" />
          </label>
          <div className="flex justify-end gap-2">
            <button type="button" onClick={() => setRenameOpen(false)} disabled={mutationPending} className="operator-button h-9">Cancel</button>
            <button type="submit" disabled={!renameValue.trim() || mutationPending} className="operator-button h-9 border-brand-800 bg-brand-950/50 text-brand-100">
              {mutationPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Pencil className="h-4 w-4" />} Rename
            </button>
          </div>
        </form>
      </Modal>
    </ProjectWorkspaceShell>
  );
}
