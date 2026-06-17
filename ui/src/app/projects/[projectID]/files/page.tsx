'use client';

import { use, useEffect, useMemo, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Copy, Download, FileText, Folder, Image as ImageIcon, Loader2, Pencil, Search, Trash2 } from 'lucide-react';
import ProjectWorkspaceShell, { formatBytes, formatProjectTime } from '@/components/project/ProjectWorkspaceShell';
import { PageNotice } from '@/components/OperatorPrimitives';
import { apiClient } from '@/utils/api';
import type { ProjectFileTreeItem, ProjectSummary } from '@/components/project/types';

type FilePreview = {
  item: ProjectFileTreeItem;
  children?: ProjectFileTreeItem[];
  content?: string;
  mediaUrl?: string;
  downloadUrl?: string;
};

export default function ProjectFilesPage({ params }: { params: Promise<{ projectID: string }> }) {
  const { projectID: rawProjectID } = use(params);
  const projectID = decodeURIComponent(rawProjectID);
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedPath = searchParams.get('path') || '';
  const filesHref = `/projects/${encodeURIComponent(projectID)}/files`;
  const [summary, setSummary] = useState<ProjectSummary | null>(null);
  const [preview, setPreview] = useState<FilePreview | null>(null);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [previewStatus, setPreviewStatus] = useState<'idle' | 'loading' | 'error'>('idle');
  const [message, setMessage] = useState('');
  const [filterText, setFilterText] = useState('');

  const refreshSummary = () =>
    apiClient
      .get(`/api/projects/${encodeURIComponent(projectID)}/summary`)
      .then(res => {
        setSummary(res.data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Failed to load project files:', error);
        setStatus('error');
      });

  const loadPreview = (path?: string) => {
    setPreviewStatus('loading');
    apiClient
      .get(`/api/projects/${encodeURIComponent(projectID)}/files`, { params: path ? { path } : {} })
      .then(res => {
        setPreview(res.data);
        setPreviewStatus('idle');
      })
      .catch(error => {
        console.error('Failed to load project file preview:', error);
        setPreviewStatus('error');
      });
  };

  useEffect(() => {
    setStatus('loading');
    void refreshSummary();
  }, [projectID]);

  useEffect(() => {
    loadPreview(selectedPath || undefined);
  }, [projectID, selectedPath]);

  const selectPath = (path?: string) => {
    router.push(path ? `${filesHref}?path=${encodeURIComponent(path)}` : filesHref);
  };

  const items = useMemo(() => {
    const query = filterText.trim().toLowerCase();
    const files = summary?.fileTree || [];
    if (!query) return files;
    return files.filter(item => [item.name, item.relativePath, item.kind].some(value => value.toLowerCase().includes(query)));
  }, [filterText, summary?.fileTree]);

  const copyPath = async () => {
    if (!preview?.item.path) return;
    await navigator.clipboard.writeText(preview.item.path);
    setMessage('Path copied.');
  };

  const renameSelected = async () => {
    if (!preview?.item.path) return;
    const nextName = window.prompt('Rename selected item', preview.item.name)?.trim();
    if (!nextName) return;
    setMessage('');
    try {
      const res = await apiClient.patch(`/api/projects/${encodeURIComponent(projectID)}/files`, {
        path: preview.item.path,
        newName: nextName,
      });
      await refreshSummary();
      selectPath(res.data?.path);
      setMessage('Renamed.');
    } catch (error: any) {
      setMessage(error?.response?.data?.error || 'Rename failed.');
    }
  };

  const deleteSelected = async () => {
    if (!preview?.item.path) return;
    const confirmed = window.confirm(`Delete "${preview.item.relativePath || preview.item.name}" from this project?`);
    if (!confirmed) return;
    setMessage('');
    try {
      await apiClient.delete(`/api/projects/${encodeURIComponent(projectID)}/files`, { data: { path: preview.item.path } });
      await refreshSummary();
      selectPath();
      setMessage('Deleted.');
    } catch (error: any) {
      setMessage(error?.response?.data?.error || 'Delete failed.');
    }
  };

  return (
    <ProjectWorkspaceShell projectID={projectID} active="files" title="Files" description="Browse and manage project-owned files.">
      <div className="grid h-full min-h-0 grid-cols-1 overflow-hidden lg:grid-cols-[minmax(320px,430px)_minmax(0,1fr)]">
        <section className="flex min-h-0 flex-col border-r border-gray-900 bg-gray-950">
          <div className="border-b border-gray-800 p-3">
            <label className="relative block">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
              <input
                value={filterText}
                onChange={event => setFilterText(event.target.value)}
                placeholder="Search project files"
                className="h-9 w-full border border-gray-800 bg-gray-950 pl-9 pr-3 text-sm text-gray-100 placeholder:text-gray-500 outline-none focus:border-cyan-700"
              />
            </label>
          </div>
          <div className="min-h-0 flex-1 overflow-auto">
            {status === 'loading' && !summary ? (
              <div className="flex h-48 items-center justify-center text-sm text-gray-400">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading files
              </div>
            ) : null}
            {status === 'error' ? (
              <div className="p-3">
                <PageNotice tone="danger" title="Files could not be loaded">
                  The project folder could not be read.
                </PageNotice>
              </div>
            ) : null}
            <div className="divide-y divide-gray-900">
              {items.map(item => (
                <button
                  key={item.path}
                  type="button"
                  onClick={() => selectPath(item.path)}
                  className="grid w-full grid-cols-[minmax(0,1fr)_84px] gap-3 px-3 py-2 text-left text-xs hover:bg-gray-900/70"
                >
                  <span className="flex min-w-0 items-center gap-2">
                    {item.kind === 'folder' ? <Folder className="h-3.5 w-3.5 flex-none text-cyan-300" /> : <FileText className="h-3.5 w-3.5 flex-none text-gray-500" />}
                    <span className="truncate text-gray-300">{item.relativePath}</span>
                  </span>
                  <span className="text-right text-gray-600">{item.kind === 'folder' ? 'folder' : formatBytes(item.size)}</span>
                </button>
              ))}
              {summary && items.length === 0 && (
                <div className="px-3 py-8 text-sm text-gray-500">No files match this project search.</div>
              )}
            </div>
          </div>
        </section>

        <section className="min-h-0 overflow-auto bg-[#02060a] p-4">
          <div className="mx-auto max-w-5xl space-y-4">
            <div className="flex flex-col gap-3 border border-gray-800 bg-gray-950 p-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0">
                <h2 className="truncate text-sm font-semibold text-gray-100">{preview?.item.relativePath || preview?.item.name || 'Project Root'}</h2>
                <p className="mt-0.5 text-xs text-gray-500">
                  {preview?.item.kind || 'folder'} {preview?.item.size ? `- ${formatBytes(preview.item.size)}` : ''}{' '}
                  {preview?.item.updatedAt ? `- ${formatProjectTime(preview.item.updatedAt)}` : ''}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <button type="button" onClick={() => void copyPath()} disabled={!preview?.item.path} className="operator-button h-8 py-1 text-xs">
                  <Copy className="h-4 w-4" />
                  Copy Path
                </button>
                {preview?.downloadUrl && (
                  <a href={preview.downloadUrl} className="operator-button h-8 py-1 text-xs">
                    <Download className="h-4 w-4" />
                    Download
                  </a>
                )}
                <button type="button" onClick={() => void renameSelected()} disabled={!preview?.item.path} className="operator-button h-8 py-1 text-xs">
                  <Pencil className="h-4 w-4" />
                  Rename
                </button>
                <button type="button" onClick={() => void deleteSelected()} disabled={!preview?.item.path} className="operator-button h-8 border-rose-900 bg-rose-950/40 py-1 text-xs text-rose-100">
                  <Trash2 className="h-4 w-4" />
                  Delete
                </button>
              </div>
            </div>

            {message && <div className="border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-300">{message}</div>}
            {previewStatus === 'error' && (
              <PageNotice tone="danger" title="Preview could not be loaded">
                Select another file or refresh the project.
              </PageNotice>
            )}

            {previewStatus === 'loading' ? (
              <div className="flex h-64 items-center justify-center border border-gray-800 bg-gray-950 text-sm text-gray-400">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading preview
              </div>
            ) : preview?.item.kind === 'folder' ? (
              <div className="border border-gray-800 bg-gray-950">
                <div className="border-b border-gray-800 px-3 py-2 text-xs text-gray-500">{preview.children?.length || 0} items</div>
                <div className="divide-y divide-gray-900">
                  {(preview.children || []).map(child => (
                    <button key={child.path} type="button" onClick={() => selectPath(child.path)} className="flex w-full min-w-0 items-center gap-2 px-3 py-2 text-left text-xs hover:bg-gray-900/70">
                      {child.kind === 'folder' ? <Folder className="h-3.5 w-3.5 flex-none text-cyan-300" /> : <FileText className="h-3.5 w-3.5 flex-none text-gray-500" />}
                      <span className="min-w-0 flex-1 truncate text-gray-300">{child.name}</span>
                      <span className="text-gray-600">{child.kind === 'folder' ? 'folder' : formatBytes(child.size)}</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : preview?.mediaUrl ? (
              <div className="border border-gray-800 bg-gray-950 p-3">
                <div className="flex min-h-[420px] items-center justify-center bg-black">
                  <img src={preview.mediaUrl} alt={preview.item.name} className="max-h-[72vh] max-w-full object-contain" />
                </div>
              </div>
            ) : typeof preview?.content === 'string' ? (
              <pre className="max-h-[72vh] overflow-auto border border-gray-800 bg-gray-950 p-3 text-xs leading-5 text-gray-300">{preview.content}</pre>
            ) : (
              <div className="flex h-64 items-center justify-center border border-dashed border-gray-800 bg-gray-950 px-6 text-center text-sm text-gray-500">
                <div>
                  <ImageIcon className="mx-auto mb-3 h-8 w-8 text-gray-600" />
                  Select a text, image, audio, or video file to preview. Other file types can still be downloaded.
                </div>
              </div>
            )}
          </div>
        </section>
      </div>
    </ProjectWorkspaceShell>
  );
}
