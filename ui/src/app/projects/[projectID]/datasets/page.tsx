'use client';

import { use, useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { Database, Loader2, Plus, Search, Upload } from 'lucide-react';
import ProjectWorkspaceShell from '@/components/project/ProjectWorkspaceShell';
import { PageNotice } from '@/components/OperatorPrimitives';
import { apiClient } from '@/utils/api';
import type { ProjectSummary } from '@/components/project/types';

function safeDatasetNameFromFile(file: File) {
  return (
    file.name
      .replace(/\.[^.]+$/, '')
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '_')
      .replace(/^_+|_+$/g, '')
      .slice(0, 60) || 'project_dataset'
  );
}

export default function ProjectDatasetsPage({ params }: { params: Promise<{ projectID: string }> }) {
  const { projectID: rawProjectID } = use(params);
  const projectID = decodeURIComponent(rawProjectID);
  const projectPath = `/projects/${encodeURIComponent(projectID)}`;
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const [summary, setSummary] = useState<ProjectSummary | null>(null);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [actionStatus, setActionStatus] = useState<'idle' | 'creating' | 'uploading'>('idle');
  const [actionError, setActionError] = useState('');
  const [filterText, setFilterText] = useState('');

  const refreshSummary = () => {
    apiClient
      .get(`/api/projects/${encodeURIComponent(projectID)}/summary`)
      .then(res => {
        setSummary(res.data);
        setStatus('success');
      })
      .catch(error => {
        console.error('Failed to load project datasets:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    setStatus('loading');
    refreshSummary();
  }, [projectID]);

  const filteredDatasets = useMemo(() => {
    const query = filterText.trim().toLowerCase();
    const datasets = summary?.datasets || [];
    if (!query) return datasets;
    return datasets.filter(dataset => [dataset.name, dataset.ref || '', dataset.encrypted ? 'encrypted' : 'local'].some(value => value.toLowerCase().includes(query)));
  }, [filterText, summary?.datasets]);

  const createDataset = async () => {
    if (actionStatus !== 'idle') return;
    const rawName = window.prompt('Dataset name for this project');
    const name = rawName?.trim();
    if (!name) return;
    setActionStatus('creating');
    setActionError('');
    try {
      const res = await apiClient.post('/api/datasets/create', { name, project_id: projectID });
      const createdName = res.data?.name || name;
      window.location.href = `${projectPath}/datasets/${encodeURIComponent(createdName)}`;
    } catch (error: any) {
      setActionError(error?.response?.data?.error || 'Failed to create dataset.');
      setActionStatus('idle');
    }
  };

  const uploadDatasetFiles = async (files: FileList | null) => {
    if (!files?.length || actionStatus !== 'idle') return;
    const suggested = safeDatasetNameFromFile(files[0]);
    const rawName = window.prompt('Dataset name for uploaded files', suggested);
    const datasetName = rawName?.trim();
    if (!datasetName) return;

    const form = new FormData();
    form.append('datasetName', datasetName);
    form.append('project_id', projectID);
    Array.from(files).forEach(file => form.append('files', file));

    setActionStatus('uploading');
    setActionError('');
    try {
      await apiClient.post('/api/datasets/upload', form);
      window.location.href = `${projectPath}/datasets/${encodeURIComponent(datasetName)}`;
    } catch (error: any) {
      setActionError(error?.response?.data?.error || 'Failed to upload dataset files.');
      setActionStatus('idle');
    } finally {
      if (uploadInputRef.current) uploadInputRef.current.value = '';
    }
  };

  return (
    <ProjectWorkspaceShell
      projectID={projectID}
      active="datasets"
      title="Datasets"
      description="Project-owned inputs and captions."
      actions={
        <>
          <button type="button" onClick={() => uploadInputRef.current?.click()} disabled={actionStatus !== 'idle'} className="operator-button h-9">
            {actionStatus === 'uploading' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
            <span className="hidden sm:inline">Upload</span>
          </button>
          <button type="button" onClick={() => void createDataset()} disabled={actionStatus !== 'idle'} className="operator-button h-9">
            {actionStatus === 'creating' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
            <span className="hidden sm:inline">Create</span>
          </button>
        </>
      }
    >
      <input
        ref={uploadInputRef}
        type="file"
        multiple
        className="hidden"
        onChange={event => void uploadDatasetFiles(event.currentTarget.files)}
      />
      <div className="h-full overflow-auto p-4">
        <div className="mx-auto max-w-[1500px] space-y-4">
          {actionError && (
            <PageNotice tone="danger" title="Dataset action failed">
              {actionError}
            </PageNotice>
          )}
          {status === 'error' && (
            <PageNotice tone="danger" title="Datasets could not be loaded">
              The project dataset folder could not be read.
            </PageNotice>
          )}

          <section className="border border-gray-800 bg-gray-950">
            <div className="flex flex-col gap-3 border-b border-gray-800 px-3 py-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-w-0">
                <h2 className="text-sm font-semibold text-gray-100">Project Datasets</h2>
                <p className="mt-0.5 text-xs text-gray-500">{filteredDatasets.length} of {summary?.datasets.length || 0} shown</p>
              </div>
              <label className="relative block w-full sm:w-80">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
                <input
                  value={filterText}
                  onChange={event => setFilterText(event.target.value)}
                  placeholder="Filter datasets"
                  className="h-9 w-full border border-gray-800 bg-gray-950 pl-9 pr-3 text-sm text-gray-100 placeholder:text-gray-500 outline-none focus:border-cyan-700"
                />
              </label>
            </div>

            {status === 'loading' && !summary ? (
              <div className="flex h-48 items-center justify-center text-sm text-gray-400">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading datasets
              </div>
            ) : (
              <div className="divide-y divide-gray-800">
                {filteredDatasets.map(dataset => (
                  <Link
                    key={dataset.name}
                    href={`${projectPath}/datasets/${encodeURIComponent(dataset.name)}`}
                    className="grid grid-cols-[minmax(0,1fr)_120px_120px] items-center gap-3 px-3 py-3 text-sm hover:bg-gray-900/70"
                  >
                    <div className="flex min-w-0 items-center gap-3">
                      <span className="flex h-9 w-9 flex-none items-center justify-center rounded-sm border border-gray-800 bg-gray-900">
                        <Database className="h-4 w-4 text-cyan-200" />
                      </span>
                      <div className="min-w-0">
                        <div className="truncate font-medium text-gray-100">{dataset.name}</div>
                        <div className="truncate text-xs text-gray-500">{dataset.path || dataset.ref || 'project dataset'}</div>
                      </div>
                    </div>
                    <span className="text-xs text-gray-500">{dataset.itemCount ?? 0} media</span>
                    <span className="text-xs text-gray-500">{dataset.encrypted ? 'Encrypted' : 'Local'}</span>
                  </Link>
                ))}
                {summary && filteredDatasets.length === 0 && (
                  <div className="flex min-h-[320px] items-center justify-center px-6 text-center">
                    <div>
                      <Database className="mx-auto h-10 w-10 text-cyan-300" />
                      <div className="mt-3 text-sm font-semibold text-gray-200">
                        {summary.datasets.length === 0 ? 'No project datasets yet' : 'No datasets match'}
                      </div>
                      <div className="mt-1 text-sm text-gray-500">
                        {summary.datasets.length === 0 ? 'Create or upload a dataset inside this project sandbox.' : 'Clear the filter to see all project datasets.'}
                      </div>
                      {summary.datasets.length === 0 && (
                        <div className="mt-4 flex justify-center gap-2">
                          <button type="button" onClick={() => uploadInputRef.current?.click()} className="operator-button h-8 py-1 text-xs">
                            <Upload className="h-4 w-4" />
                            Upload
                          </button>
                          <button type="button" onClick={() => void createDataset()} className="operator-button h-8 py-1 text-xs">
                            <Plus className="h-4 w-4" />
                            Create
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>
        </div>
      </div>
    </ProjectWorkspaceShell>
  );
}
