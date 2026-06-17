'use client';

import { use, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { FolderCheck, Loader2, Save, Settings, Trash2 } from 'lucide-react';
import ProjectWorkspaceShell, { formatBytes } from '@/components/project/ProjectWorkspaceShell';
import { PageNotice } from '@/components/OperatorPrimitives';
import { apiClient } from '@/utils/api';
import type { ProjectSummary } from '@/components/project/types';

export default function ProjectSettingsPage({ params }: { params: Promise<{ projectID: string }> }) {
  const { projectID: rawProjectID } = use(params);
  const projectID = decodeURIComponent(rawProjectID);
  const router = useRouter();
  const [summary, setSummary] = useState<ProjectSummary | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');

  const loadProject = () =>
    apiClient
      .get(`/api/projects/${encodeURIComponent(projectID)}/summary`)
      .then(res => {
        setSummary(res.data);
        setName(res.data?.project?.name || '');
        setDescription(res.data?.project?.description || '');
        setStatus('success');
      })
      .catch(error => {
        console.error('Failed to load project settings:', error);
        setStatus('error');
      });

  useEffect(() => {
    setStatus('loading');
    void loadProject();
  }, [projectID]);

  const folderRows = useMemo(() => {
    if (!summary) return [];
    const zones = summary.zones;
    return [
      { key: 'datasets', path: summary.roots.datasets, count: zones.inputs.fileCount + zones.inputs.folderCount, bytes: zones.inputs.totalBytes },
      { key: 'runs', path: summary.roots.runs, count: zones.runs.fileCount + zones.runs.folderCount, bytes: zones.runs.totalBytes },
      { key: 'outputs', path: summary.roots.outputs, count: zones.outputs.fileCount + zones.outputs.folderCount, bytes: zones.outputs.totalBytes },
      { key: 'models', path: summary.roots.models, count: zones.models.fileCount + zones.models.folderCount, bytes: zones.models.totalBytes },
      { key: 'configs', path: summary.roots.configs, count: 0, bytes: 0 },
      { key: 'assets', path: summary.roots.assets, count: 0, bytes: 0 },
      { key: 'notes', path: summary.roots.notes, count: 0, bytes: 0 },
      { key: 'cache', path: summary.roots.cache, count: 0, bytes: 0 },
    ];
  }, [summary]);

  const saveProject = async () => {
    if (!name.trim() || saveStatus === 'saving') return;
    setSaveStatus('saving');
    setMessage('');
    try {
      await apiClient.patch(`/api/projects/${encodeURIComponent(projectID)}`, {
        name,
        description,
      });
      await loadProject();
      setSaveStatus('success');
      setMessage('Project settings saved.');
    } catch (error: any) {
      setSaveStatus('error');
      setMessage(error?.response?.data?.error || 'Failed to save project settings.');
    }
  };

  const deleteProject = async () => {
    if (!summary) return;
    const first = window.confirm(`Delete project "${summary.project.name}" from the project list? Project files on disk are left in place.`);
    if (!first) return;
    const typed = window.prompt(`Type ${summary.project.slug} to confirm deletion`);
    if (typed !== summary.project.slug) return;
    try {
      await apiClient.delete(`/api/projects/${encodeURIComponent(projectID)}`);
      router.push('/projects');
    } catch (error: any) {
      setMessage(error?.response?.data?.error || 'Failed to delete project.');
    }
  };

  return (
    <ProjectWorkspaceShell projectID={projectID} active="settings" title="Settings" description="Project identity and sandbox metadata.">
      <div className="h-full overflow-auto p-4">
        <div className="mx-auto grid max-w-[1400px] gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(360px,0.8fr)]">
          <section className="space-y-4">
            {status === 'loading' && !summary && (
              <div className="flex h-48 items-center justify-center border border-gray-800 bg-gray-900/40 text-sm text-gray-400">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading settings
              </div>
            )}
            {status === 'error' && (
              <PageNotice tone="danger" title="Project settings could not be loaded">
                The project may have been deleted, or the project database is unavailable.
              </PageNotice>
            )}
            {message && (
              <PageNotice tone={saveStatus === 'error' ? 'danger' : 'neutral'} title={saveStatus === 'error' ? 'Settings action failed' : 'Settings'}>
                {message}
              </PageNotice>
            )}

            <section className="border border-gray-800 bg-gray-950">
              <div className="flex h-12 items-center gap-2 border-b border-gray-800 px-3">
                <Settings className="h-4 w-4 text-cyan-300" />
                <h2 className="text-sm font-semibold text-gray-100">Project Identity</h2>
              </div>
              <div className="space-y-3 p-3">
                <label className="block">
                  <span className="mb-1 block text-xs font-medium text-gray-400">Name</span>
                  <input
                    value={name}
                    onChange={event => setName(event.target.value)}
                    className="h-9 w-full border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-700"
                  />
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-medium text-gray-400">Description</span>
                  <textarea
                    value={description}
                    onChange={event => setDescription(event.target.value)}
                    rows={5}
                    className="w-full resize-none border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-100 outline-none focus:border-cyan-700"
                  />
                </label>
                <button type="button" onClick={() => void saveProject()} disabled={!name.trim() || saveStatus === 'saving'} className="operator-button h-9">
                  {saveStatus === 'saving' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
                  Save Settings
                </button>
              </div>
            </section>

            <section className="border border-rose-950 bg-rose-950/10">
              <div className="flex h-12 items-center gap-2 border-b border-rose-950 px-3">
                <Trash2 className="h-4 w-4 text-rose-300" />
                <h2 className="text-sm font-semibold text-rose-100">Danger Zone</h2>
              </div>
              <div className="space-y-3 p-3 text-sm text-rose-100">
                <p className="text-rose-200/80">Remove this project from the project list. Files on disk are not deleted by this action.</p>
                <button type="button" onClick={() => void deleteProject()} className="operator-button h-9 border-rose-900 bg-rose-950/60 text-rose-100">
                  <Trash2 className="h-4 w-4" />
                  Delete Project
                </button>
              </div>
            </section>
          </section>

          <aside className="space-y-4">
            <section className="border border-gray-800 bg-gray-950">
              <div className="flex h-12 items-center gap-2 border-b border-gray-800 px-3">
                <FolderCheck className="h-4 w-4 text-cyan-300" />
                <h2 className="text-sm font-semibold text-gray-100">Sandbox Roots</h2>
              </div>
              <div className="divide-y divide-gray-800 text-xs">
                {summary && (
                  <div className="grid grid-cols-[84px_minmax(0,1fr)] gap-2 px-3 py-2">
                    <span className="text-gray-500">root</span>
                    <span className="truncate text-gray-300">{summary.roots.root}</span>
                  </div>
                )}
                {folderRows.map(row => (
                  <div key={row.key} className="grid grid-cols-[84px_minmax(0,1fr)] gap-2 px-3 py-2">
                    <span className="capitalize text-gray-500">{row.key}</span>
                    <span className="min-w-0">
                      <span className="block truncate text-gray-300">{row.path}</span>
                      <span className="text-gray-600">{row.count} items, {formatBytes(row.bytes)}</span>
                    </span>
                  </div>
                ))}
              </div>
            </section>
          </aside>
        </div>
      </div>
    </ProjectWorkspaceShell>
  );
}
