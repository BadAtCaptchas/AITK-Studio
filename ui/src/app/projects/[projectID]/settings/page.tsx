'use client';

import { use, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Archive,
  Cloud,
  FolderCheck,
  HardDrive,
  Link2,
  Loader2,
  RadioTower,
  RefreshCcw,
  RotateCcw,
  Save,
  Server,
  Settings,
  Trash2,
  Unplug,
} from 'lucide-react';
import ProjectWorkspaceShell, { formatBytes } from '@/components/project/ProjectWorkspaceShell';
import { PageNotice } from '@/components/OperatorPrimitives';
import { Modal } from '@/components/Modal';
import { openConfirm } from '@/components/ConfirmModal';
import { useProjectWorkspace } from '@/components/project/ProjectContext';
import useWorkers from '@/hooks/useWorkers';
import { apiClient } from '@/utils/api';
import type { ProjectSummary } from '@/components/project/types';
import type { ProjectSyncOperation as ProjectSyncOperationRecord, ProjectSyncProfile } from '@/types';

type LifecycleProject = ProjectSummary['project'] & {
  lifecycle_state?: 'creating' | 'active' | 'archived' | 'relocating' | 'purging';
  storage_root_path?: string;
  archived_at?: string | Date | null;
  revision?: number;
  home_instance_id?: string | null;
  operation_error?: string | null;
  home_worker_id?: string | null;
};

type SyncConflict = {
  path: string;
  resolution?: 'keep-home' | 'keep-worker' | 'keep-both';
};

type PurgePreview = {
  project_id: string;
  slug: string;
  revision: number;
  root_path: string;
  job_count: number;
  file_count: number;
  total_bytes: number;
  blockers: Array<{ code: string; message: string }>;
  confirmation_text: string;
  can_purge: boolean;
};

export default function ProjectSettingsPage({ params }: { params: Promise<{ projectID: string }> }) {
  const { projectID: rawProjectID } = use(params);
  const projectID = decodeURIComponent(rawProjectID);
  const router = useRouter();
  const { identity, refreshIdentity } = useProjectWorkspace();
  const { workers } = useWorkers();
  const [summary, setSummary] = useState<ProjectSummary | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [actionStatus, setActionStatus] = useState<'idle' | 'saving' | 'archiving' | 'restoring' | 'previewing' | 'purging' | 'relocating'>('idle');
  const [notice, setNotice] = useState<{ tone: 'neutral' | 'success' | 'danger' | 'warning'; title: string; message: string } | null>(null);
  const [purgePreview, setPurgePreview] = useState<PurgePreview | null>(null);
  const [relocateOpen, setRelocateOpen] = useState(false);
  const [destinationRoot, setDestinationRoot] = useState('');
  const [relocateMode, setRelocateMode] = useState<'copy' | 'move'>('copy');
  const [selectedWorkerID, setSelectedWorkerID] = useState('');
  const [remoteAction, setRemoteAction] = useState<string | null>(null);
  const [syncOperations, setSyncOperations] = useState<ProjectSyncOperationRecord[]>([]);

  const loadProject = async () => {
    try {
      const response = await apiClient.get(`/api/projects/${encodeURIComponent(projectID)}/summary`);
      setSummary(response.data);
      setName(response.data?.project?.name || '');
      setDescription(response.data?.project?.description || '');
      setStatus('success');
    } catch (error) {
      console.error('Failed to load project settings:', error);
      setStatus('error');
    }
  };

  useEffect(() => {
    setStatus('loading');
    void loadProject();
  }, [projectID]);

  useEffect(() => {
    let cancelled = false;
    apiClient
      .get(`/api/projects/${encodeURIComponent(projectID)}/sync`)
      .then(response => {
        if (!cancelled) setSyncOperations(Array.isArray(response.data?.operations) ? response.data.operations : []);
      })
      .catch(() => {
        if (!cancelled) setSyncOperations([]);
      });
    return () => {
      cancelled = true;
    };
  }, [projectID]);

  const project = summary?.project as LifecycleProject | undefined;
  const lifecycleState = project?.lifecycle_state || 'active';
  const revision = project?.revision || 1;
  const isArchived = lifecycleState === 'archived';
  const busy = actionStatus !== 'idle' || remoteAction !== null;
  const replicas = identity?.replicas || [];
  const isLocalHome = !project?.home_worker_id || project.home_worker_id === 'local';

  const compatibleWorkers = useMemo(
    () => workers.filter(worker => worker.enabled && worker.capabilities.includes('project-sync-v1')),
    [workers],
  );
  const linkedWorkerIDs = useMemo(() => new Set(replicas.map(replica => replica.instanceId)), [replicas]);
  const availableWorkers = useMemo(
    () => compatibleWorkers.filter(worker => !linkedWorkerIDs.has(worker.id)),
    [compatibleWorkers, linkedWorkerIDs],
  );

  useEffect(() => {
    if (selectedWorkerID && availableWorkers.some(worker => worker.id === selectedWorkerID)) return;
    setSelectedWorkerID(availableWorkers[0]?.id || '');
  }, [availableWorkers, selectedWorkerID]);

  const folderRows = useMemo(() => {
    if (!summary) return [];
    const zones = summary.zones;
    return [
      { key: 'datasets', path: summary.roots.datasets, count: zones.inputs.fileCount + zones.inputs.folderCount, bytes: zones.inputs.totalBytes },
      { key: 'runs', path: summary.roots.runs, count: zones.runs.fileCount + zones.runs.folderCount, bytes: zones.runs.totalBytes },
      { key: 'outputs', path: summary.roots.outputs, count: zones.outputs.fileCount + zones.outputs.folderCount, bytes: zones.outputs.totalBytes },
      { key: 'models', path: summary.roots.models, count: zones.models.fileCount + zones.models.folderCount, bytes: zones.models.totalBytes },
      { key: 'configs', path: summary.roots.configs },
      { key: 'assets', path: summary.roots.assets },
      { key: 'notes', path: summary.roots.notes },
      { key: 'cache', path: summary.roots.cache },
    ];
  }, [summary]);

  const errorMessage = (error: unknown, fallback: string) => {
    const candidate = error as { response?: { data?: { error?: string } }; message?: string };
    return candidate.response?.data?.error || candidate.message || fallback;
  };

  const refreshRemoteState = async () => {
    const [operationsResponse] = await Promise.all([
      apiClient.get(`/api/projects/${encodeURIComponent(projectID)}/sync`),
      refreshIdentity(),
      loadProject(),
    ]);
    setSyncOperations(Array.isArray(operationsResponse.data?.operations) ? operationsResponse.data.operations : []);
  };

  const runRemoteAction = async (key: string, action: () => Promise<void>, successTitle: string, successMessage: string) => {
    if (remoteAction) return;
    setRemoteAction(key);
    setNotice(null);
    try {
      await action();
      await refreshRemoteState();
      setNotice({ tone: 'success', title: successTitle, message: successMessage });
    } catch (error) {
      setNotice({ tone: 'danger', title: 'Remote operation failed', message: errorMessage(error, 'The project was not changed.') });
    } finally {
      setRemoteAction(null);
    }
  };

  const linkReplica = async () => {
    if (!selectedWorkerID || busy || isArchived || !isLocalHome) return;
    await runRemoteAction(
      `link:${selectedWorkerID}`,
      async () => {
        await apiClient.post(`/api/projects/${encodeURIComponent(projectID)}/replicas`, { worker_id: selectedWorkerID });
      },
      'Replica linked',
      'The worker is ready for an initial full synchronization.',
    );
  };

  const startSync = async (workerID: string, profile: ProjectSyncProfile) => {
    if (busy || isArchived || !isLocalHome) return;
    await runRemoteAction(
      `sync:${workerID}:${profile}`,
      async () => {
        await apiClient.post(`/api/projects/${encodeURIComponent(projectID)}/sync`, {
          worker_id: workerID,
          profile,
          run_now: true,
        });
      },
      `${profile[0].toUpperCase()}${profile.slice(1)} sync finished`,
      profile === 'results'
        ? 'Remote results are now available from the project home.'
        : 'The replica manifest and transferred files were verified.',
    );
  };

  const removeReplica = (workerID: string, workerName: string) => {
    if (busy || isArchived || !isLocalHome) return;
    openConfirm({
      title: `Remove ${workerName}`,
      message: 'The remote project copy will be removed only if the worker is reachable and has no active project runs.',
      confirmText: 'Remove Replica',
      type: 'warning',
      onConfirm: async () => {
        await runRemoteAction(
          `remove:${workerID}`,
          async () => {
            await apiClient.delete(
              `/api/projects/${encodeURIComponent(projectID)}/replicas/${encodeURIComponent(workerID)}`,
            );
          },
          'Replica removed',
          'The worker is no longer linked to this project.',
        );
      },
    });
  };

  const rehomeToReplica = (workerID: string, workerName: string) => {
    if (busy || isArchived || !isLocalHome) return;
    openConfirm({
      title: `Make ${workerName} the project home`,
      message: 'This requires no active jobs and performs a verified full sync before ownership changes. This instance becomes read-only afterward.',
      confirmText: 'Rehome Project',
      type: 'warning',
      onConfirm: async () => {
        await runRemoteAction(
          `rehome:${workerID}`,
          async () => {
            await apiClient.post(`/api/projects/${encodeURIComponent(projectID)}/rehome`, {
              worker_id: workerID,
              expected_revision: revision,
            });
          },
          'Project rehomed',
          `${workerName} is now the authoritative project home.`,
        );
      },
    });
  };

  const parseConflicts = (operation: ProjectSyncOperationRecord): SyncConflict[] => {
    try {
      const value: unknown = JSON.parse(operation.conflicts || '[]');
      return Array.isArray(value)
        ? value.filter(
            (item): item is SyncConflict =>
              !!item && typeof item === 'object' && 'path' in item && typeof item.path === 'string',
          )
        : [];
    } catch {
      return [];
    }
  };

  const resolveConflicts = async (
    operation: ProjectSyncOperationRecord,
    resolution: 'keep-home' | 'keep-worker' | 'keep-both',
  ) => {
    const conflicts = parseConflicts(operation);
    if (conflicts.length === 0 || busy || isArchived || !isLocalHome) return;
    await runRemoteAction(
      `resolve:${operation.id}:${resolution}`,
      async () => {
        await apiClient.post(
          `/api/projects/${encodeURIComponent(projectID)}/sync/${encodeURIComponent(operation.id)}/resolve`,
          {
            resolutions: Object.fromEntries(conflicts.map(conflict => [conflict.path, resolution])),
            run_now: true,
          },
        );
      },
      'Conflicts resolved',
      'The synchronization resumed with the selected deterministic policy.',
    );
  };

  const saveProject = async () => {
    if (!name.trim() || busy || isArchived) return;
    setActionStatus('saving');
    setNotice(null);
    try {
      await apiClient.patch(`/api/projects/${encodeURIComponent(projectID)}`, { name, description });
      await loadProject();
      setNotice({ tone: 'success', title: 'Project updated', message: 'Identity settings were saved.' });
    } catch (error) {
      setNotice({ tone: 'danger', title: 'Save failed', message: errorMessage(error, 'Project settings could not be saved.') });
    } finally {
      setActionStatus('idle');
    }
  };

  const archiveProject = () => {
    if (!project || busy || isArchived) return;
    openConfirm({
      title: `Archive ${project.name}`,
      message: 'Archived projects remain browsable, but datasets, runs, generation, and file changes become read-only until restored.',
      confirmText: 'Archive Project',
      type: 'warning',
      onConfirm: async () => {
        setActionStatus('archiving');
        setNotice(null);
        try {
          await apiClient.post(`/api/projects/${encodeURIComponent(projectID)}/archive`, { expected_revision: revision });
          await loadProject();
          setNotice({ tone: 'success', title: 'Project archived', message: 'Files and history are preserved.' });
        } catch (error) {
          setNotice({ tone: 'danger', title: 'Archive blocked', message: errorMessage(error, 'Stop active project work before archiving.') });
        } finally {
          setActionStatus('idle');
        }
      },
    });
  };

  const restoreProject = async () => {
    if (!project || busy || !isArchived) return;
    setActionStatus('restoring');
    setNotice(null);
    try {
      await apiClient.post(`/api/projects/${encodeURIComponent(projectID)}/restore`, { expected_revision: revision });
      await loadProject();
      setNotice({ tone: 'success', title: 'Project restored', message: 'The workspace is writable again.' });
    } catch (error) {
      setNotice({ tone: 'danger', title: 'Restore failed', message: errorMessage(error, 'The registered project root could not be restored.') });
    } finally {
      setActionStatus('idle');
    }
  };

  const previewPurge = async () => {
    if (!project || busy || !isArchived) return;
    setActionStatus('previewing');
    setNotice(null);
    try {
      const response = await apiClient.get(`/api/projects/${encodeURIComponent(projectID)}/purge-preview`);
      setPurgePreview(response.data);
    } catch (error) {
      setNotice({ tone: 'danger', title: 'Purge preview failed', message: errorMessage(error, 'Project data could not be inspected safely.') });
    } finally {
      setActionStatus('idle');
    }
  };

  const requestPurge = () => {
    if (!purgePreview?.can_purge || busy) return;
    openConfirm({
      title: 'Permanently purge project',
      message: `This removes ${purgePreview.job_count} jobs and ${formatBytes(purgePreview.total_bytes)} of registered project data, including synchronized replicas. Type ${purgePreview.confirmation_text} to continue.`,
      confirmText: 'Purge Everything',
      inputTitle: purgePreview.confirmation_text,
      type: 'danger',
      onConfirm: async value => {
        if (value !== purgePreview.confirmation_text) {
          setNotice({ tone: 'danger', title: 'Confirmation did not match', message: 'The project was not changed.' });
          return;
        }
        setActionStatus('purging');
        try {
          await apiClient.post(`/api/projects/${encodeURIComponent(projectID)}/purge`, {
            expected_revision: purgePreview.revision,
            confirmation: value,
            scope: 'project_and_all_data',
          });
          router.push('/projects?state=archived');
        } catch (error) {
          setNotice({ tone: 'danger', title: 'Purge failed', message: errorMessage(error, 'No unverified data was deleted.') });
          setActionStatus('idle');
        }
      },
    });
  };

  const relocateProject = async () => {
    if (!destinationRoot.trim() || busy || !isArchived) return;
    setActionStatus('relocating');
    setNotice(null);
    try {
      await apiClient.post(`/api/projects/${encodeURIComponent(projectID)}/relocate`, {
        destination_storage_root: destinationRoot,
        mode: relocateMode,
        expected_revision: revision,
        confirmation: relocateMode === 'move' ? project?.slug : undefined,
      });
      setRelocateOpen(false);
      await loadProject();
      setNotice({ tone: 'success', title: 'Workspace relocated', message: 'The verified destination is now registered for this project.' });
    } catch (error) {
      setNotice({ tone: 'danger', title: 'Relocation failed', message: errorMessage(error, 'The original project root remains authoritative.') });
    } finally {
      setActionStatus('idle');
    }
  };

  return (
    <ProjectWorkspaceShell projectID={projectID} active="settings" title="Settings" description="Project identity, storage, home worker, and lifecycle.">
      <div className="h-full overflow-auto p-3 sm:p-4">
        <div className="mx-auto grid max-w-[1460px] gap-4 xl:grid-cols-[minmax(0,1fr)_440px]">
          <section className="space-y-4">
            {status === 'loading' && !summary ? (
              <div className="flex h-48 items-center justify-center border border-gray-800 bg-gray-900/40 text-sm text-gray-400">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Loading settings
              </div>
            ) : null}
            {status === 'error' ? (
              <PageNotice tone="danger" title="Project settings could not be loaded" action={<button className="operator-button h-8 text-xs" onClick={() => void loadProject()}>Retry</button>}>
                The project may be unavailable or its home worker may be offline.
              </PageNotice>
            ) : null}
            {notice ? <PageNotice tone={notice.tone} title={notice.title}>{notice.message}</PageNotice> : null}
            {project?.operation_error ? <PageNotice tone="warning" title="Previous project operation needs attention">{project.operation_error}</PageNotice> : null}

            <section className="border border-gray-800 bg-gray-950">
              <div className="flex h-12 items-center gap-2 border-b border-gray-800 px-3">
                <Settings className="h-4 w-4 text-cyan-300" />
                <h2 className="text-sm font-semibold text-gray-100">Project Identity</h2>
              </div>
              <div className="space-y-4 p-4">
                <label className="block">
                  <span className="mb-1 block text-xs font-medium text-gray-400">Name</span>
                  <input value={name} onChange={event => setName(event.target.value)} disabled={isArchived} className="h-10 w-full border border-gray-800 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600 disabled:opacity-60" />
                </label>
                <label className="block">
                  <span className="mb-1 block text-xs font-medium text-gray-400">Description</span>
                  <textarea value={description} onChange={event => setDescription(event.target.value)} disabled={isArchived} rows={5} className="w-full resize-y border border-gray-800 bg-gray-950 px-3 py-2 text-sm text-gray-100 outline-none focus:border-cyan-600 disabled:opacity-60" />
                </label>
                <div className="grid gap-3 text-xs sm:grid-cols-2">
                  <div className="border border-gray-800 bg-gray-900/40 p-3"><div className="text-gray-500">Project ID</div><div className="mt-1 break-all font-mono text-gray-300">{project?.id || '—'}</div></div>
                  <div className="border border-gray-800 bg-gray-900/40 p-3"><div className="text-gray-500">Slug</div><div className="mt-1 font-mono text-gray-300">{project?.slug || '—'}</div></div>
                </div>
                <button type="button" onClick={() => void saveProject()} disabled={!name.trim() || busy || isArchived} className="operator-button h-10 border-cyan-800 bg-cyan-950/40 text-cyan-100">
                  {actionStatus === 'saving' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />} Save Settings
                </button>
              </div>
            </section>

            <section className="border border-gray-800 bg-gray-950">
              <div className="flex h-12 items-center gap-2 border-b border-gray-800 px-3">
                <Server className="h-4 w-4 text-cyan-300" />
                <h2 className="text-sm font-semibold text-gray-100">Workspace Home</h2>
              </div>
              <div className="grid gap-3 p-4 text-sm sm:grid-cols-2">
                <div className="border border-gray-800 bg-gray-900/40 p-3"><div className="text-xs text-gray-500">Authoritative instance</div><div className="mt-1 break-all font-mono text-xs text-gray-300">{project?.home_instance_id || 'Local instance'}</div></div>
                <div className="border border-gray-800 bg-gray-900/40 p-3"><div className="text-xs text-gray-500">Lifecycle state</div><div className="mt-1 capitalize text-gray-200">{lifecycleState}</div></div>
              </div>
            </section>

            <section className="border border-gray-800 bg-gray-950">
              <div className="flex min-h-12 flex-wrap items-center justify-between gap-2 border-b border-gray-800 px-3 py-2">
                <div className="flex items-center gap-2">
                  <RadioTower className="h-4 w-4 text-cyan-300" />
                  <h2 className="text-sm font-semibold text-gray-100">Execution Replicas</h2>
                </div>
                <span className="text-[11px] text-gray-500">project-sync-v1 only</span>
              </div>
              <div className="space-y-3 p-4">
                {!isLocalHome ? (
                  <PageNotice tone="warning" title="Remote home is authoritative">
                    Replica settings are read-only here. Manage synchronization from the project home worker.
                  </PageNotice>
                ) : null}

                {replicas.map(replica => {
                  const worker = workers.find(candidate => candidate.id === replica.instanceId);
                  const workerName = worker?.name || replica.instanceName || replica.instanceId;
                  return (
                    <article key={replica.id} className="border border-gray-800 bg-gray-900/30 p-3">
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <Cloud className="h-4 w-4 text-cyan-300" />
                            <span className="truncate text-sm font-semibold text-gray-200">{workerName}</span>
                            <span className={`rounded-sm border px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${replica.status === 'ready' ? 'border-emerald-900 text-emerald-300' : replica.status === 'syncing' ? 'border-cyan-900 text-cyan-300' : 'border-amber-900 text-amber-300'}`}>
                              {replica.status}
                            </span>
                          </div>
                          <div className="mt-1 text-[11px] text-gray-500">
                            {replica.lastSyncedAt
                              ? `Last verified ${new Date(replica.lastSyncedAt).toLocaleString()}`
                              : 'Initial synchronization required'}
                          </div>
                          {replica.error ? <div className="mt-1 text-xs text-rose-300">{replica.error}</div> : null}
                        </div>
                        <button
                          type="button"
                          onClick={() => removeReplica(replica.instanceId, workerName)}
                          disabled={busy || isArchived || !isLocalHome}
                          className="operator-button h-8 border-rose-950 px-2 text-xs text-rose-200"
                        >
                          {remoteAction === `remove:${replica.instanceId}` ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Unplug className="h-3.5 w-3.5" />}
                          Remove
                        </button>
                      </div>
                      <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
                        {(['full', 'launch', 'results'] as const).map(profile => (
                          <button
                            key={profile}
                            type="button"
                            onClick={() => void startSync(replica.instanceId, profile)}
                            disabled={busy || isArchived || !isLocalHome}
                            className="operator-button h-8 px-2 text-xs capitalize"
                          >
                            {remoteAction === `sync:${replica.instanceId}:${profile}` ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCcw className="h-3.5 w-3.5" />}
                            {profile}
                          </button>
                        ))}
                        <button
                          type="button"
                          onClick={() => rehomeToReplica(replica.instanceId, workerName)}
                          disabled={busy || isArchived || !isLocalHome}
                          className="operator-button h-8 border-amber-900 bg-amber-950/20 px-2 text-xs text-amber-100"
                        >
                          {remoteAction === `rehome:${replica.instanceId}` ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Server className="h-3.5 w-3.5" />}
                          Make home
                        </button>
                      </div>
                    </article>
                  );
                })}

                {replicas.length === 0 ? (
                  <div className="border border-dashed border-gray-800 px-4 py-6 text-center">
                    <Cloud className="mx-auto h-7 w-7 text-gray-700" />
                    <div className="mt-2 text-sm font-medium text-gray-300">No execution replicas</div>
                    <p className="mt-1 text-xs text-gray-600">Link a compatible worker to train or generate with project-scoped files.</p>
                  </div>
                ) : null}

                {isLocalHome && !isArchived ? (
                  <div className="flex flex-col gap-2 border-t border-gray-800 pt-3 sm:flex-row">
                    <select
                      value={selectedWorkerID}
                      onChange={event => setSelectedWorkerID(event.target.value)}
                      disabled={busy || availableWorkers.length === 0}
                      aria-label="Worker to link as a project replica"
                      className="h-9 min-w-0 flex-1 border border-gray-800 bg-gray-950 px-2 text-xs text-gray-200 outline-none focus:border-cyan-700 disabled:opacity-60"
                    >
                      {availableWorkers.length === 0 ? <option value="">No compatible unlinked workers</option> : null}
                      {availableWorkers.map(worker => <option key={worker.id} value={worker.id}>{worker.name}</option>)}
                    </select>
                    <button
                      type="button"
                      onClick={() => void linkReplica()}
                      disabled={!selectedWorkerID || busy}
                      className="operator-button h-9 border-cyan-900 bg-cyan-950/30 text-xs text-cyan-100"
                    >
                      {remoteAction === `link:${selectedWorkerID}` ? <Loader2 className="h-4 w-4 animate-spin" /> : <Link2 className="h-4 w-4" />}
                      Link Worker
                    </button>
                  </div>
                ) : null}

                {syncOperations.filter(operation => operation.status === 'conflict').map(operation => {
                  const conflicts = parseConflicts(operation);
                  return (
                    <div key={operation.id} className="border border-amber-900/70 bg-amber-950/10 p-3">
                      <div className="text-sm font-semibold text-amber-100">Sync conflict needs a decision</div>
                      <p className="mt-1 text-xs text-amber-200/70">
                        {conflicts.length} path{conflicts.length === 1 ? '' : 's'} changed on both the home and worker. Apply one policy to this operation.
                      </p>
                      <div className="mt-3 flex flex-wrap gap-2">
                        <button type="button" onClick={() => void resolveConflicts(operation, 'keep-home')} disabled={busy || conflicts.length === 0} className="operator-button h-8 text-xs">Keep home</button>
                        <button type="button" onClick={() => void resolveConflicts(operation, 'keep-worker')} disabled={busy || conflicts.length === 0} className="operator-button h-8 text-xs">Keep worker</button>
                        <button type="button" onClick={() => void resolveConflicts(operation, 'keep-both')} disabled={busy || conflicts.length === 0} className="operator-button h-8 text-xs">Keep both</button>
                      </div>
                    </div>
                  );
                })}

                {syncOperations.length > 0 ? (
                  <div className="border-t border-gray-800 pt-3">
                    <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-gray-600">Recent sync activity</div>
                    <div className="space-y-1.5">
                      {syncOperations.slice(0, 4).map(operation => (
                        <div key={operation.id} className="grid grid-cols-[70px_minmax(0,1fr)_auto] items-center gap-2 text-xs">
                          <span className="capitalize text-gray-400">{operation.profile}</span>
                          <span className="truncate text-gray-600">{operation.error || operation.phase}</span>
                          <span className={operation.status === 'completed' ? 'text-emerald-300' : operation.status === 'failed' ? 'text-rose-300' : 'text-amber-300'}>{operation.status.replaceAll('_', ' ')}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>
            </section>

            <section className={`border ${isArchived ? 'border-amber-800/70 bg-amber-950/10' : 'border-gray-800 bg-gray-950'}`}>
              <div className="flex h-12 items-center gap-2 border-b border-inherit px-3">
                <Archive className="h-4 w-4 text-amber-300" />
                <h2 className="text-sm font-semibold text-gray-100">Archive</h2>
              </div>
              <div className="space-y-3 p-4 text-sm text-gray-400">
                <p>{isArchived ? 'This project is read-only. Restore it to create datasets, runs, outputs, or file changes.' : 'Archive preserves every file and job while removing the workspace from the active project board.'}</p>
                {isArchived ? (
                  <button type="button" onClick={() => void restoreProject()} disabled={busy} className="operator-button h-10 border-emerald-800 bg-emerald-950/40 text-emerald-100">
                    {actionStatus === 'restoring' ? <Loader2 className="h-4 w-4 animate-spin" /> : <RotateCcw className="h-4 w-4" />} Restore Project
                  </button>
                ) : (
                  <button type="button" onClick={archiveProject} disabled={busy} className="operator-button h-10 border-amber-800 bg-amber-950/40 text-amber-100">
                    {actionStatus === 'archiving' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Archive className="h-4 w-4" />} Archive Project
                  </button>
                )}
              </div>
            </section>

            {isArchived ? (
              <section className="border border-rose-950 bg-rose-950/10">
                <div className="flex h-12 items-center gap-2 border-b border-rose-950 px-3"><Trash2 className="h-4 w-4 text-rose-300" /><h2 className="text-sm font-semibold text-rose-100">Permanent Purge</h2></div>
                <div className="space-y-3 p-4 text-sm text-rose-100">
                  <p className="text-rose-200/80">Purge removes this project, its jobs, registered files, secrets, metrics, and reachable replicas. Preview blockers and exact size before confirmation.</p>
                  <button type="button" onClick={() => void previewPurge()} disabled={busy} className="operator-button h-10 border-rose-900 bg-rose-950/60 text-rose-100">
                    {actionStatus === 'previewing' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />} Preview Purge
                  </button>
                  {purgePreview ? (
                    <div className="border border-rose-900/70 bg-black/20 p-3 text-xs">
                      <div>{purgePreview.file_count} files · {formatBytes(purgePreview.total_bytes)} · {purgePreview.job_count} jobs</div>
                      {purgePreview.blockers.map(blocker => <div key={`${blocker.code}:${blocker.message}`} className="mt-2 text-amber-200">{blocker.message}</div>)}
                      <button type="button" onClick={requestPurge} disabled={!purgePreview.can_purge || busy} className="operator-button mt-3 h-9 border-rose-800 bg-rose-900/50 text-rose-50">Purge Project and Data</button>
                    </div>
                  ) : null}
                </div>
              </section>
            ) : null}
          </section>

          <aside className="space-y-4">
            <section className="border border-gray-800 bg-gray-950">
              <div className="flex h-12 items-center gap-2 border-b border-gray-800 px-3"><FolderCheck className="h-4 w-4 text-cyan-300" /><h2 className="text-sm font-semibold text-gray-100">Registered Storage</h2></div>
              <div className="divide-y divide-gray-800 text-xs">
                {summary ? <div className="px-3 py-3"><div className="text-gray-500">Project root</div><div className="mt-1 break-all font-mono text-gray-300">{summary.roots.root}</div></div> : null}
                {project?.storage_root_path ? <div className="px-3 py-3"><div className="text-gray-500">Storage boundary</div><div className="mt-1 break-all font-mono text-gray-300">{project.storage_root_path}</div></div> : null}
                {folderRows.map(row => (
                  <div key={row.key} className="grid grid-cols-[72px_minmax(0,1fr)] gap-2 px-3 py-2.5">
                    <span className="capitalize text-gray-500">{row.key}</span>
                    <span className="min-w-0"><span className="block truncate font-mono text-gray-300">{row.path}</span>{'count' in row ? <span className="text-gray-600">{row.count} items · {formatBytes(row.bytes || 0)}</span> : null}</span>
                  </div>
                ))}
              </div>
              {isArchived ? (
                <div className="border-t border-gray-800 p-3"><button type="button" onClick={() => { setDestinationRoot(project?.storage_root_path || ''); setRelocateOpen(true); }} className="operator-button h-9"><HardDrive className="h-4 w-4" /> Relocate Workspace</button></div>
              ) : null}
            </section>
          </aside>
        </div>
      </div>

      <Modal isOpen={relocateOpen} onClose={() => setRelocateOpen(false)} title="Relocate archived workspace" size="md" closeOnOverlayClick={!busy}>
        <form onSubmit={event => { event.preventDefault(); void relocateProject(); }} className="space-y-4">
          <PageNotice tone="warning" title="Verified copy first">The current root remains authoritative until the destination inventory and hashes match.</PageNotice>
          <label className="block"><span className="mb-1 block text-xs font-medium text-gray-400">Destination storage root</span><input value={destinationRoot} onChange={event => setDestinationRoot(event.target.value)} className="h-10 w-full border border-gray-700 bg-gray-950 px-3 font-mono text-sm text-gray-100 outline-none focus:border-cyan-600" /></label>
          <fieldset><legend className="mb-2 text-xs font-medium text-gray-400">After verification</legend><div className="grid gap-2 sm:grid-cols-2">
            <label className={`cursor-pointer border p-3 text-sm ${relocateMode === 'copy' ? 'border-cyan-700 bg-cyan-950/20 text-cyan-100' : 'border-gray-800 text-gray-400'}`}><input type="radio" className="sr-only" checked={relocateMode === 'copy'} onChange={() => setRelocateMode('copy')} /><span className="block font-medium">Keep old backup</span><span className="mt-1 block text-xs opacity-75">Register the new copy and leave the original unmanaged.</span></label>
            <label className={`cursor-pointer border p-3 text-sm ${relocateMode === 'move' ? 'border-amber-700 bg-amber-950/20 text-amber-100' : 'border-gray-800 text-gray-400'}`}><input type="radio" className="sr-only" checked={relocateMode === 'move'} onChange={() => setRelocateMode('move')} /><span className="block font-medium">Move workspace</span><span className="mt-1 block text-xs opacity-75">Delete the old root only after the new root is registered.</span></label>
          </div></fieldset>
          <div className="flex justify-end gap-2"><button type="button" onClick={() => setRelocateOpen(false)} disabled={busy} className="operator-button h-9">Cancel</button><button type="submit" disabled={!destinationRoot.trim() || busy} className="operator-button h-9 border-cyan-800 bg-cyan-950/40 text-cyan-100">{actionStatus === 'relocating' ? <Loader2 className="h-4 w-4 animate-spin" /> : <HardDrive className="h-4 w-4" />} Relocate</button></div>
        </form>
      </Modal>
    </ProjectWorkspaceShell>
  );
}
