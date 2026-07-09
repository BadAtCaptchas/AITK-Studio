'use client';

import { useEffect, useMemo, useState } from 'react';
import { Copy, FolderInput, Loader2, Plus, Server, Sparkles } from 'lucide-react';
import classNames from 'classnames';
import { Modal } from '@/components/Modal';
import useWorkers from '@/hooks/useWorkers';
import { apiClient } from '@/utils/api';
import type { ProjectRecord } from './types';

type SetupMode = 'blank' | 'import' | 'clone';

const setupOptions = [
  {
    id: 'blank' as const,
    label: 'Start blank',
    detail: 'Create an empty studio and add inputs when you are ready.',
    icon: Sparkles,
  },
  {
    id: 'import' as const,
    label: 'Import workspace',
    detail: 'Register content from an existing workspace folder.',
    icon: FolderInput,
  },
  {
    id: 'clone' as const,
    label: 'Clone project',
    detail: 'Copy a project without active state or cache files.',
    icon: Copy,
  },
];

function messageFromError(error: unknown) {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { data?: { error?: string } } }).response;
    if (response?.data?.error) return response.data.error;
  }
  return error instanceof Error ? error.message : 'Project could not be created.';
}

export default function QuickCreateProjectModal({
  open,
  onClose,
  projects,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  projects: ProjectRecord[];
  onCreated: (project: ProjectRecord) => void;
}) {
  const { workers } = useWorkers();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [setupMode, setSetupMode] = useState<SetupMode>('blank');
  const [sourcePath, setSourcePath] = useState('');
  const [sourceProjectID, setSourceProjectID] = useState('');
  const [homeWorkerID, setHomeWorkerID] = useState('local');
  const [status, setStatus] = useState<'idle' | 'saving' | 'error'>('idle');
  const [error, setError] = useState('');

  const compatibleWorkers = useMemo(
    () =>
      workers.filter(worker => {
        if (!worker.enabled) return false;
        if (!worker.capabilities) return true;
        return worker.capabilities.includes('project-sync-v1');
      }),
    [workers],
  );

  useEffect(() => {
    if (!open) return;
    setStatus('idle');
    setError('');
  }, [open]);

  const createProject = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!name.trim() || status === 'saving') return;
    if (setupMode === 'import' && !sourcePath.trim()) {
      setError('Choose the existing workspace folder to import.');
      return;
    }
    if (setupMode === 'clone' && !sourceProjectID) {
      setError('Choose a project to clone.');
      return;
    }
    setStatus('saving');
    setError('');
    try {
      const response = await apiClient.post('/api/projects', {
        name: name.trim(),
        description: description.trim(),
        badge_asset: '/assets/projects/project-badge-default.png',
        setup_mode: setupMode,
        ...(setupMode === 'import' ? { import_root: sourcePath.trim() } : {}),
        ...(setupMode === 'clone' ? { clone_from_project_id: sourceProjectID } : {}),
        home_worker_id: homeWorkerID,
      });
      const project = (response.data?.project || response.data) as ProjectRecord;
      const setupError = response.data?.setup?.status === 'failed' ? response.data.setup.error : null;
      if (typeof setupError === 'string' && setupError && !project.operation_error) {
        project.operation_error = setupError;
      }
      onCreated(project);
      setName('');
      setDescription('');
      setSetupMode('blank');
      setSourcePath('');
      setSourceProjectID('');
      setHomeWorkerID('local');
      setStatus('idle');
    } catch (createError: unknown) {
      setStatus('error');
      setError(messageFromError(createError));
    }
  };

  return (
    <Modal isOpen={open} onClose={onClose} title="Create a project" size="lg">
      <form onSubmit={createProject} className="space-y-5">
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="block">
            <span className="mb-1.5 block text-xs font-medium text-gray-300">Project name</span>
            <input
              value={name}
              onChange={event => setName(event.target.value)}
              placeholder="Campaign character LoRA"
              autoFocus
              className="h-10 w-full rounded-sm border border-gray-700 bg-gray-950 px-3 text-sm text-gray-100 outline-none placeholder:text-gray-600 focus:border-cyan-600"
            />
          </label>
          <label className="block">
            <span className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-gray-300">
              <Server className="h-3.5 w-3.5" /> Home instance
            </span>
            <select
              value={homeWorkerID}
              onChange={event => setHomeWorkerID(event.target.value)}
              className="h-10 w-full rounded-sm border border-gray-700 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
            >
              <option value="local">This studio (local)</option>
              {compatibleWorkers.map(worker => (
                <option key={worker.id} value={worker.id}>
                  {worker.name}
                </option>
              ))}
            </select>
          </label>
        </div>

        <label className="block">
          <span className="mb-1.5 block text-xs font-medium text-gray-300">Description (optional)</span>
          <textarea
            value={description}
            onChange={event => setDescription(event.target.value)}
            placeholder="What are you preparing or training?"
            rows={2}
            className="w-full resize-none rounded-sm border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 outline-none placeholder:text-gray-600 focus:border-cyan-600"
          />
        </label>

        <fieldset>
          <legend className="mb-2 text-xs font-medium text-gray-300">Set up the workspace</legend>
          <div className="grid gap-2 sm:grid-cols-3">
            {setupOptions.map(option => {
              const Icon = option.icon;
              const selected = setupMode === option.id;
              return (
                <button
                  key={option.id}
                  type="button"
                  onClick={() => {
                    setSetupMode(option.id);
                    setError('');
                  }}
                  className={classNames(
                    'min-h-28 rounded-sm border p-3 text-left transition-colors',
                    selected
                      ? 'border-cyan-700 bg-cyan-950/30 text-gray-100'
                      : 'border-gray-800 bg-gray-950/60 text-gray-400 hover:border-gray-700 hover:text-gray-200',
                  )}
                >
                  <Icon className={classNames('h-5 w-5', selected ? 'text-cyan-300' : 'text-gray-500')} />
                  <span className="mt-2 block text-sm font-semibold">{option.label}</span>
                  <span className="mt-1 block text-xs leading-relaxed text-gray-500">{option.detail}</span>
                </button>
              );
            })}
          </div>
        </fieldset>

        {setupMode === 'import' ? (
          <label className="block">
            <span className="mb-1.5 block text-xs font-medium text-gray-300">Existing workspace folder</span>
            <input
              value={sourcePath}
              onChange={event => setSourcePath(event.target.value)}
              placeholder="Path on the home instance"
              className="h-10 w-full rounded-sm border border-gray-700 bg-gray-950 px-3 font-mono text-xs text-gray-100 outline-none placeholder:text-gray-600 focus:border-cyan-600"
            />
          </label>
        ) : null}

        {setupMode === 'clone' ? (
          <label className="block">
            <span className="mb-1.5 block text-xs font-medium text-gray-300">Source project</span>
            <select
              value={sourceProjectID}
              onChange={event => setSourceProjectID(event.target.value)}
              className="h-10 w-full rounded-sm border border-gray-700 bg-gray-950 px-3 text-sm text-gray-100 outline-none focus:border-cyan-600"
            >
              <option value="">Choose a project</option>
              {projects.map(project => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        {error ? (
          <div className="rounded-sm border border-rose-900 bg-rose-950/30 px-3 py-2 text-xs text-rose-200">
            {error}
          </div>
        ) : null}

        <div className="flex flex-col-reverse gap-2 border-t border-gray-800 pt-4 sm:flex-row sm:justify-end">
          <button type="button" onClick={onClose} className="operator-button h-9">
            Cancel
          </button>
          <button
            type="submit"
            disabled={!name.trim() || status === 'saving'}
            className="operator-button h-9 border-cyan-800 bg-cyan-950/60 text-cyan-100 hover:bg-cyan-900/60"
          >
            {status === 'saving' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
            Create and open
          </button>
        </div>
      </form>
    </Modal>
  );
}
