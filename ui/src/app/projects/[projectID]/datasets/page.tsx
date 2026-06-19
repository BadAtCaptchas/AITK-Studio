'use client';

import { use, useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Database, FolderInput, Loader2, Plus, Search, Trash2, Upload } from 'lucide-react';
import { Modal } from '@/components/Modal';
import { openConfirm } from '@/components/ConfirmModal';
import DatasetWatchFoldersButton from '@/components/DatasetWatchFoldersButton';
import ProjectWorkspaceShell from '@/components/project/ProjectWorkspaceShell';
import { PageNotice } from '@/components/OperatorPrimitives';
import { apiClient } from '@/utils/api';
import {
  createEmptyEncryptedManifest,
  getRememberedEncryptedDatasetKey,
  rememberEncryptedDatasetKey,
} from '@/utils/encryptedDatasets';
import type { ProjectSummary } from '@/components/project/types';
import type { DatasetSummary } from '@/types';

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

function validateDatasetName(name: string) {
  const trimmed = name.trim();
  if (!trimmed) return 'Dataset name is required.';
  if (trimmed === '.' || trimmed.includes('..') || /[\\/]/.test(trimmed)) {
    return 'Dataset name cannot contain path separators or "..".';
  }
  if (/[<>:"|?*\x00-\x1f]/.test(trimmed)) {
    return 'Dataset name contains invalid filename characters.';
  }
  return '';
}

export default function ProjectDatasetsPage({ params }: { params: Promise<{ projectID: string }> }) {
  const { projectID: rawProjectID } = use(params);
  const projectID = decodeURIComponent(rawProjectID);
  const router = useRouter();
  const projectPath = `/projects/${encodeURIComponent(projectID)}`;
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const [summary, setSummary] = useState<ProjectSummary | null>(null);
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [actionStatus, setActionStatus] = useState<'idle' | 'creating' | 'uploading' | 'importing' | 'deleting'>('idle');
  const [actionError, setActionError] = useState('');
  const [filterText, setFilterText] = useState('');
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [newDatasetName, setNewDatasetName] = useState('');
  const [createEncrypted, setCreateEncrypted] = useState(false);
  const [datasetPassword, setDatasetPassword] = useState('');
  const [datasetPasswordConfirm, setDatasetPasswordConfirm] = useState('');
  const [createError, setCreateError] = useState('');
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [uploadDatasetName, setUploadDatasetName] = useState('');
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploadError, setUploadError] = useState('');
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [globalDatasets, setGlobalDatasets] = useState<DatasetSummary[]>([]);
  const [importFilterText, setImportFilterText] = useState('');
  const [selectedImportDatasetName, setSelectedImportDatasetName] = useState('');
  const [importDatasetName, setImportDatasetName] = useState('');
  const [importStatus, setImportStatus] = useState<'idle' | 'loading'>('idle');
  const [importError, setImportError] = useState('');

  const refreshSummary = () => {
    return apiClient
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

  const importableGlobalDatasets = useMemo(
    () => globalDatasets.filter(dataset => dataset.source !== 'remote' && !!dataset.path),
    [globalDatasets],
  );
  const filteredGlobalDatasets = useMemo(() => {
    const query = importFilterText.trim().toLowerCase();
    if (!query) return importableGlobalDatasets;
    return importableGlobalDatasets.filter(dataset =>
      [dataset.name, dataset.path || '', dataset.encrypted ? 'encrypted' : 'local'].some(value =>
        value.toLowerCase().includes(query),
      ),
    );
  }, [importFilterText, importableGlobalDatasets]);
  const selectedImportDataset = useMemo(
    () => importableGlobalDatasets.find(dataset => dataset.name === selectedImportDatasetName) || null,
    [importableGlobalDatasets, selectedImportDatasetName],
  );

  const createNameError = validateDatasetName(newDatasetName);
  const encryptedPasswordError =
    createEncrypted && !datasetPassword
      ? 'Password is required for encrypted datasets.'
      : createEncrypted && datasetPassword !== datasetPasswordConfirm
        ? 'Passwords do not match.'
        : '';
  const canSubmitCreate = !createNameError && !encryptedPasswordError && actionStatus === 'idle';
  const uploadNameError = validateDatasetName(uploadDatasetName);
  const canSubmitUpload = !uploadNameError && uploadFiles.length > 0 && actionStatus === 'idle';
  const importNameError = validateDatasetName(importDatasetName);
  const canSubmitImport =
    !!selectedImportDataset?.path && !importNameError && actionStatus === 'idle' && importStatus !== 'loading';

  const suggestProjectDatasetName = (sourceName: string) => {
    const base = sourceName.trim() || 'imported_dataset';
    const existing = new Set((summary?.datasets || []).map(dataset => dataset.name.toLowerCase()));
    if (!existing.has(base.toLowerCase())) return base;
    let counter = 2;
    let candidate = `${base}_copy`;
    while (existing.has(candidate.toLowerCase())) {
      candidate = `${base}_copy_${counter}`;
      counter += 1;
    }
    return candidate;
  };

  const rememberedGlobalDatasetKey = (dataset: DatasetSummary) =>
    (dataset.ref ? getRememberedEncryptedDatasetKey(dataset.ref) : null) ||
    (dataset.path ? getRememberedEncryptedDatasetKey(dataset.path) : null) ||
    getRememberedEncryptedDatasetKey(dataset.name);

  const rememberProjectEncryptedKey = (datasetName: string, rawKeyB64: string, datasetPath?: string) => {
    rememberEncryptedDatasetKey(datasetName, rawKeyB64);
    rememberEncryptedDatasetKey(`project:${projectID}:${datasetName}`, rawKeyB64);
    if (datasetPath) rememberEncryptedDatasetKey(datasetPath, rawKeyB64);
    if (summary?.roots.datasets) {
      rememberEncryptedDatasetKey(`${summary.roots.datasets.replace(/[\\/]+$/, '')}\\${datasetName}`, rawKeyB64);
    }
  };

  const openCreateModal = () => {
    if (actionStatus !== 'idle') return;
    setNewDatasetName('');
    setCreateEncrypted(false);
    setDatasetPassword('');
    setDatasetPasswordConfirm('');
    setCreateError('');
    setActionError('');
    setCreateModalOpen(true);
  };

  const closeCreateModal = () => {
    if (actionStatus === 'creating') return;
    setCreateModalOpen(false);
  };

  const createDataset = async () => {
    if (actionStatus !== 'idle') return;
    const name = newDatasetName.trim();
    const validationError = validateDatasetName(name);
    const passwordError =
      createEncrypted && !datasetPassword
        ? 'Password is required for encrypted datasets.'
        : createEncrypted && datasetPassword !== datasetPasswordConfirm
          ? 'Passwords do not match.'
          : '';
    if (validationError || passwordError) {
      setCreateError(validationError || passwordError);
      return;
    }
    setActionStatus('creating');
    setActionError('');
    setCreateError('');
    try {
      let encryptedManifest = null;
      let rawKeyB64: string | null = null;
      if (createEncrypted) {
        const result = await createEmptyEncryptedManifest('password', datasetPassword);
        encryptedManifest = result.manifest;
        rawKeyB64 = result.rawKeyB64;
      }

      const res = await apiClient.post('/api/datasets/create', {
        name,
        project_id: projectID,
        encrypted: createEncrypted,
        encryptedManifest,
      });
      const createdName = res.data?.name || name;
      if (rawKeyB64) rememberProjectEncryptedKey(createdName, rawKeyB64);
      setCreateModalOpen(false);
      router.push(`${projectPath}/datasets/${encodeURIComponent(createdName)}`);
    } catch (error: any) {
      setCreateError(error?.response?.data?.error || error?.message || 'Failed to create dataset.');
      setActionStatus('idle');
    }
  };

  const openUploadNameModal = (files: FileList | null) => {
    if (!files?.length || actionStatus !== 'idle') return;
    const nextFiles = Array.from(files);
    setUploadFiles(nextFiles);
    setUploadDatasetName(safeDatasetNameFromFile(nextFiles[0]));
    setUploadError('');
    setActionError('');
    setUploadModalOpen(true);
  };

  const closeUploadModal = () => {
    if (actionStatus === 'uploading') return;
    setUploadModalOpen(false);
    setUploadFiles([]);
    setUploadDatasetName('');
    setUploadError('');
    if (uploadInputRef.current) uploadInputRef.current.value = '';
  };

  const selectImportDataset = (dataset: DatasetSummary) => {
    setSelectedImportDatasetName(dataset.name);
    setImportDatasetName(suggestProjectDatasetName(dataset.name));
    setImportError('');
  };

  const loadGlobalDatasetsForImport = async () => {
    setImportStatus('loading');
    setImportError('');
    try {
      const res = await apiClient.get('/api/datasets/list', { params: { worker_id: 'local' } });
      const datasets = Array.isArray(res.data) ? res.data : Array.isArray(res.data?.datasets) ? res.data.datasets : [];
      const importable = datasets.filter((dataset: DatasetSummary) => dataset.source !== 'remote' && !!dataset.path);
      setGlobalDatasets(importable);
      const nextSelection = importable[0] || null;
      setSelectedImportDatasetName(nextSelection?.name || '');
      setImportDatasetName(nextSelection ? suggestProjectDatasetName(nextSelection.name) : '');
    } catch (error: any) {
      setGlobalDatasets([]);
      setSelectedImportDatasetName('');
      setImportDatasetName('');
      setImportError(error?.response?.data?.error || error?.message || 'Failed to load global datasets.');
    } finally {
      setImportStatus('idle');
    }
  };

  const openImportModal = () => {
    if (actionStatus !== 'idle') return;
    setImportModalOpen(true);
    setImportFilterText('');
    setImportError('');
    setActionError('');
    void loadGlobalDatasetsForImport();
  };

  const closeImportModal = () => {
    if (actionStatus === 'importing') return;
    setImportModalOpen(false);
    setImportError('');
    setImportFilterText('');
  };

  const uploadDatasetFiles = async () => {
    if (actionStatus !== 'idle') return;
    const datasetName = uploadDatasetName.trim();
    const validationError = validateDatasetName(datasetName);
    if (validationError) {
      setUploadError(validationError);
      return;
    }
    if (uploadFiles.length === 0) {
      setUploadError('Choose at least one file to upload.');
      return;
    }

    const form = new FormData();
    form.append('datasetName', datasetName);
    form.append('project_id', projectID);
    uploadFiles.forEach(file => form.append('files', file));

    setActionStatus('uploading');
    setActionError('');
    setUploadError('');
    try {
      await apiClient.post('/api/datasets/upload', form);
      setUploadModalOpen(false);
      router.push(`${projectPath}/datasets/${encodeURIComponent(datasetName)}`);
    } catch (error: any) {
      setUploadError(error?.response?.data?.error || 'Failed to upload dataset files.');
      setActionStatus('idle');
    } finally {
      if (uploadInputRef.current) uploadInputRef.current.value = '';
    }
  };

  const importGlobalDataset = async () => {
    if (actionStatus !== 'idle' || !selectedImportDataset?.path) return;
    const name = importDatasetName.trim();
    const validationError = validateDatasetName(name);
    if (validationError) {
      setImportError(validationError);
      return;
    }

    setActionStatus('importing');
    setActionError('');
    setImportError('');
    try {
      const res = await apiClient.post('/api/datasets/copy', {
        datasetPath: selectedImportDataset.path,
        name,
        project_id: projectID,
        source_project_id: null,
      });
      const createdName = res.data?.name || name;
      const createdPath = res.data?.path;
      const rememberedKey = selectedImportDataset.encrypted ? rememberedGlobalDatasetKey(selectedImportDataset) : null;
      if (rememberedKey) rememberProjectEncryptedKey(createdName, rememberedKey, createdPath);
      setImportModalOpen(false);
      router.push(`${projectPath}/datasets/${encodeURIComponent(createdName)}`);
    } catch (error: any) {
      setImportError(error?.response?.data?.error || error?.message || 'Failed to import dataset.');
      setActionStatus('idle');
    }
  };

  const handleDeleteDataset = (dataset: DatasetSummary) => {
    if (actionStatus !== 'idle') return;
    openConfirm({
      title: 'Delete Dataset',
      message: `Are you sure you want to delete the dataset "${dataset.name}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: async () => {
        setActionStatus('deleting');
        setActionError('');
        try {
          await apiClient.post('/api/datasets/delete', {
            name: dataset.name,
            project_id: projectID,
          });
          await refreshSummary();
        } catch (error: any) {
          setActionError(error?.response?.data?.error || error?.message || 'Failed to delete dataset.');
        } finally {
          setActionStatus('idle');
        }
      },
    });
  };

  return (
    <ProjectWorkspaceShell
      projectID={projectID}
      active="datasets"
      title="Datasets"
      description="Project-owned inputs and captions."
      actions={
        <>
          <button type="button" onClick={openImportModal} disabled={actionStatus !== 'idle'} className="operator-button h-9">
            {actionStatus === 'importing' ? <Loader2 className="h-4 w-4 animate-spin" /> : <FolderInput className="h-4 w-4" />}
            <span className="hidden sm:inline">Import</span>
          </button>
          <button type="button" onClick={() => uploadInputRef.current?.click()} disabled={actionStatus !== 'idle'} className="operator-button h-9">
            {actionStatus === 'uploading' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
            <span className="hidden sm:inline">Upload</span>
          </button>
          <button type="button" onClick={openCreateModal} disabled={actionStatus !== 'idle'} className="operator-button h-9">
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
        onChange={event => openUploadNameModal(event.currentTarget.files)}
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
                  <div
                    key={dataset.name}
                    className="grid grid-cols-[minmax(0,1fr)_72px_72px_70px] items-center gap-2 px-3 py-3 text-sm hover:bg-gray-900/70 sm:grid-cols-[minmax(0,1fr)_120px_120px_76px] sm:gap-3"
                  >
                    <Link
                      href={`${projectPath}/datasets/${encodeURIComponent(dataset.name)}`}
                      className="min-w-0"
                    >
                      <span className="flex min-w-0 items-center gap-3">
                        <span className="flex h-9 w-9 flex-none items-center justify-center rounded-sm border border-gray-800 bg-gray-900">
                          <Database className="h-4 w-4 text-cyan-200" />
                        </span>
                        <span className="min-w-0">
                          <span className="block truncate font-medium text-gray-100">{dataset.name}</span>
                          <span className="block truncate text-xs text-gray-500">{dataset.path || dataset.ref || 'project dataset'}</span>
                        </span>
                      </span>
                    </Link>
                    <span className="truncate text-xs text-gray-500">{dataset.itemCount ?? 0} media</span>
                    <span className="truncate text-xs text-gray-500">{dataset.encrypted ? 'Encrypted' : 'Local'}</span>
                    <span className="flex items-center justify-end gap-1">
                      {!dataset.encrypted && (
                        <DatasetWatchFoldersButton
                          datasetName={dataset.name}
                          projectID={projectID}
                          workerID="local"
                          defaultSourcePath={dataset.importSourcePath}
                          label={`Watch folders for ${dataset.name}`}
                          icon="eye"
                          iconOnly
                          className="inline-flex h-7 w-7 items-center justify-center rounded-sm text-gray-400 transition-colors hover:bg-cyan-700 hover:text-white sm:h-8 sm:w-8"
                          onRefresh={() => {
                            void refreshSummary();
                          }}
                        />
                      )}
                      <button
                        type="button"
                        onClick={() => handleDeleteDataset(dataset)}
                        disabled={actionStatus !== 'idle'}
                        className="inline-flex h-7 w-7 items-center justify-center rounded-sm text-gray-400 transition-colors hover:bg-red-600 hover:text-white disabled:cursor-not-allowed disabled:opacity-50 sm:h-8 sm:w-8"
                        title="Delete dataset"
                        aria-label={`Delete ${dataset.name}`}
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </span>
                  </div>
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
                          <button type="button" onClick={openImportModal} className="operator-button h-8 py-1 text-xs">
                            <FolderInput className="h-4 w-4" />
                            Import
                          </button>
                          <button type="button" onClick={() => uploadInputRef.current?.click()} className="operator-button h-8 py-1 text-xs">
                            <Upload className="h-4 w-4" />
                            Upload
                          </button>
                          <button type="button" onClick={openCreateModal} className="operator-button h-8 py-1 text-xs">
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
      <Modal isOpen={importModalOpen} onClose={closeImportModal} title="Import Existing Dataset" size="lg" closeOnOverlayClick={actionStatus !== 'importing'}>
        <form
          className="space-y-4 text-gray-200"
          onSubmit={event => {
            event.preventDefault();
            void importGlobalDataset();
          }}
        >
          <label className="relative block">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-500" />
            <input
              value={importFilterText}
              onChange={event => setImportFilterText(event.target.value)}
              placeholder="Filter global datasets"
              className="h-9 w-full border border-gray-800 bg-gray-950 pl-9 pr-3 text-sm text-gray-100 placeholder:text-gray-500 outline-none focus:border-cyan-700"
            />
          </label>

          <div className="max-h-72 overflow-auto border border-gray-800 bg-gray-950">
            {importStatus === 'loading' ? (
              <div className="flex h-28 items-center justify-center text-sm text-gray-400">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading datasets
              </div>
            ) : filteredGlobalDatasets.length > 0 ? (
              <div className="divide-y divide-gray-800">
                {filteredGlobalDatasets.map(dataset => {
                  const selected = dataset.name === selectedImportDatasetName;
                  return (
                    <button
                      key={dataset.name}
                      type="button"
                      onClick={() => selectImportDataset(dataset)}
                      className={`grid w-full grid-cols-[minmax(0,1fr)_90px_90px] items-center gap-3 px-3 py-3 text-left text-sm hover:bg-gray-900/70 ${
                        selected ? 'bg-cyan-950/20 outline outline-1 outline-cyan-900/70' : ''
                      }`}
                    >
                      <span className="flex min-w-0 items-center gap-3">
                        <span className="flex h-9 w-9 flex-none items-center justify-center rounded-sm border border-gray-800 bg-gray-900">
                          <Database className="h-4 w-4 text-cyan-200" />
                        </span>
                        <span className="min-w-0">
                          <span className="block truncate font-medium text-gray-100">{dataset.name}</span>
                          <span className="block truncate text-xs text-gray-500">{dataset.path}</span>
                        </span>
                      </span>
                      <span className="text-xs text-gray-500">{dataset.itemCount ?? 0} media</span>
                      <span className="text-xs text-gray-500">{dataset.encrypted ? 'Encrypted' : 'Local'}</span>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="flex h-28 items-center justify-center px-4 text-center text-sm text-gray-500">
                {globalDatasets.length === 0 ? 'No global datasets found.' : 'No datasets match.'}
              </div>
            )}
          </div>

          <label className="block">
            <span className="mb-1 block text-xs font-medium text-gray-400">Project Dataset Name</span>
            <input
              value={importDatasetName}
              onChange={event => {
                setImportDatasetName(event.target.value);
                setImportError('');
              }}
              disabled={!selectedImportDataset}
              className="h-9 w-full border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-700 disabled:opacity-50"
            />
            {importDatasetName.trim() && importNameError ? <span className="mt-1 block text-xs text-rose-300">{importNameError}</span> : null}
          </label>

          {importError && <div className="text-xs text-rose-300">{importError}</div>}

          <div className="flex justify-end gap-2">
            <button type="button" onClick={closeImportModal} disabled={actionStatus === 'importing'} className="operator-button h-9">
              Cancel
            </button>
            <button
              type="submit"
              disabled={!canSubmitImport}
              className="operator-button h-9 border-emerald-800 bg-emerald-950/70 text-emerald-100 hover:bg-emerald-900 disabled:opacity-50"
            >
              {actionStatus === 'importing' ? <Loader2 className="h-4 w-4 animate-spin" /> : <FolderInput className="h-4 w-4" />}
              Import
            </button>
          </div>
        </form>
      </Modal>

      <Modal isOpen={createModalOpen} onClose={closeCreateModal} title="Create Project Dataset" size="md" closeOnOverlayClick={actionStatus !== 'creating'}>
        <form
          className="space-y-4 text-gray-200"
          onSubmit={event => {
            event.preventDefault();
            void createDataset();
          }}
        >
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-gray-400">Dataset Name</span>
            <input
              value={newDatasetName}
              onChange={event => {
                setNewDatasetName(event.target.value);
                setCreateError('');
              }}
              autoFocus
              placeholder="portrait_inputs"
              className="h-9 w-full border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 placeholder:text-gray-600 outline-none focus:border-cyan-700"
            />
            {newDatasetName.trim() && createNameError ? <span className="mt-1 block text-xs text-rose-300">{createNameError}</span> : null}
          </label>

          <label className="flex cursor-pointer items-start gap-3 border border-gray-800 bg-gray-950 p-3">
            <input
              type="checkbox"
              checked={createEncrypted}
              onChange={event => {
                setCreateEncrypted(event.target.checked);
                setCreateError('');
              }}
              className="mt-1 h-4 w-4 rounded border-gray-700 bg-gray-950 text-cyan-500"
            />
            <span>
              <span className="block text-sm font-medium text-gray-100">Encrypted dataset</span>
              <span className="mt-0.5 block text-xs text-gray-500">Store image and caption objects in an encrypted project dataset.</span>
            </span>
          </label>

          {createEncrypted && (
            <div className="space-y-3 border border-gray-800 bg-gray-950 p-3">
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-gray-400">Password</span>
                <input
                  type="password"
                  value={datasetPassword}
                  onChange={event => {
                    setDatasetPassword(event.target.value);
                    setCreateError('');
                  }}
                  className="h-9 w-full border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-700"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-gray-400">Confirm Password</span>
                <input
                  type="password"
                  value={datasetPasswordConfirm}
                  onChange={event => {
                    setDatasetPasswordConfirm(event.target.value);
                    setCreateError('');
                  }}
                  className="h-9 w-full border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-700"
                />
              </label>
              {encryptedPasswordError ? <div className="text-xs text-rose-300">{encryptedPasswordError}</div> : null}
            </div>
          )}

          {(createError || actionError) && <div className="text-xs text-rose-300">{createError || actionError}</div>}

          <div className="flex justify-end gap-2">
            <button type="button" onClick={closeCreateModal} disabled={actionStatus === 'creating'} className="operator-button h-9">
              Cancel
            </button>
            <button
              type="submit"
              disabled={!canSubmitCreate}
              className="operator-button h-9 border-emerald-800 bg-emerald-950/70 text-emerald-100 hover:bg-emerald-900 disabled:opacity-50"
            >
              {actionStatus === 'creating' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
              Create
            </button>
          </div>
        </form>
      </Modal>

      <Modal isOpen={uploadModalOpen} onClose={closeUploadModal} title="Upload Project Dataset" size="md" closeOnOverlayClick={actionStatus !== 'uploading'}>
        <form
          className="space-y-4 text-gray-200"
          onSubmit={event => {
            event.preventDefault();
            void uploadDatasetFiles();
          }}
        >
          <div className="border border-gray-800 bg-gray-950 px-3 py-2 text-xs text-gray-400">
            {uploadFiles.length} file{uploadFiles.length === 1 ? '' : 's'} selected
          </div>
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-gray-400">Dataset Name</span>
            <input
              value={uploadDatasetName}
              onChange={event => {
                setUploadDatasetName(event.target.value);
                setUploadError('');
              }}
              autoFocus
              className="h-9 w-full border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-700"
            />
            {uploadDatasetName.trim() && uploadNameError ? <span className="mt-1 block text-xs text-rose-300">{uploadNameError}</span> : null}
          </label>
          {uploadError && <div className="text-xs text-rose-300">{uploadError}</div>}
          <div className="flex justify-end gap-2">
            <button type="button" onClick={closeUploadModal} disabled={actionStatus === 'uploading'} className="operator-button h-9">
              Cancel
            </button>
            <button
              type="submit"
              disabled={!canSubmitUpload}
              className="operator-button h-9 border-emerald-800 bg-emerald-950/70 text-emerald-100 hover:bg-emerald-900 disabled:opacity-50"
            >
              {actionStatus === 'uploading' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
              Upload
            </button>
          </div>
        </form>
      </Modal>
    </ProjectWorkspaceShell>
  );
}
