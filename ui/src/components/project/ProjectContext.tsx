'use client';

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import useSettings from '@/hooks/useSettings';
import { loadProjectIdentity, projectIsArchived } from './data';
import type { ProjectIdentity } from './types';

type ProjectLoadStatus = 'loading' | 'ready' | 'disabled' | 'error';

type ProjectWorkspaceContextValue = {
  requestedProjectID: string;
  projectID: string;
  identity: ProjectIdentity | null;
  status: ProjectLoadStatus;
  error: string | null;
  archived: boolean;
  apiBase: string;
  refreshIdentity: () => Promise<void>;
};

const ProjectWorkspaceContext = createContext<ProjectWorkspaceContextValue | null>(null);

function errorMessage(error: unknown) {
  if (error && typeof error === 'object' && 'response' in error) {
    const response = (error as { response?: { data?: { error?: string } } }).response;
    if (response?.data?.error) return response.data.error;
  }
  return error instanceof Error ? error.message : 'Project could not be loaded.';
}

export function ProjectProvider({ projectID, children }: { projectID: string; children: ReactNode }) {
  const { settings, isSettingsLoaded } = useSettings();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [identity, setIdentity] = useState<ProjectIdentity | null>(null);
  const [status, setStatus] = useState<ProjectLoadStatus>('loading');
  const [error, setError] = useState<string | null>(null);
  const projectsEnabled = settings.PROJECTS_ENABLED !== 'false';

  const refreshIdentity = useCallback(async () => {
    if (!isSettingsLoaded) return;
    if (!projectsEnabled) {
      setIdentity(null);
      setError(null);
      setStatus('disabled');
      return;
    }
    setStatus(current => (current === 'ready' ? current : 'loading'));
    try {
      const nextIdentity = await loadProjectIdentity(projectID);
      setIdentity(nextIdentity);
      setError(null);
      setStatus('ready');
    } catch (loadError: unknown) {
      setIdentity(null);
      setError(errorMessage(loadError));
      setStatus('error');
    }
  }, [isSettingsLoaded, projectID, projectsEnabled]);

  useEffect(() => {
    void refreshIdentity();
  }, [refreshIdentity]);

  useEffect(() => {
    const canonicalID = identity?.project.id;
    if (!canonicalID || canonicalID === projectID || !pathname) return;
    const segments = pathname.split('/');
    if (segments[1] !== 'projects' || !segments[2]) return;
    segments[2] = encodeURIComponent(canonicalID);
    const query = searchParams.toString();
    router.replace(`${segments.join('/')}${query ? `?${query}` : ''}`);
  }, [identity?.project.id, pathname, projectID, router, searchParams]);

  const canonicalProjectID = identity?.project.id || projectID;
  const value = useMemo<ProjectWorkspaceContextValue>(
    () => ({
      requestedProjectID: projectID,
      projectID: canonicalProjectID,
      identity,
      status,
      error,
      archived: projectIsArchived(identity?.project),
      apiBase: `/api/projects/${encodeURIComponent(canonicalProjectID)}`,
      refreshIdentity,
    }),
    [canonicalProjectID, error, identity, projectID, refreshIdentity, status],
  );

  return <ProjectWorkspaceContext.Provider value={value}>{children}</ProjectWorkspaceContext.Provider>;
}

export function useProjectWorkspace() {
  const context = useContext(ProjectWorkspaceContext);
  if (!context) throw new Error('useProjectWorkspace must be used inside ProjectProvider');
  return context;
}
