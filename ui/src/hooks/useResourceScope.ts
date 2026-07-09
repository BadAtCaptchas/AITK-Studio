'use client';

import { useCallback, useEffect, useState } from 'react';
import type { Project } from '@/types';
import { apiClient } from '@/utils/api';

export const RESOURCE_SCOPES = ['global', 'all', 'project'] as const;

export type ResourceScope = (typeof RESOURCE_SCOPES)[number];

export type ProjectScopeOption = Pick<Project, 'id' | 'slug' | 'name' | 'lifecycle_state'>;

function isResourceScope(value: string | null): value is ResourceScope {
  return value === 'global' || value === 'all' || value === 'project';
}

function readLocationScope(): { scope: ResourceScope; projectID: string | null } {
  if (typeof window === 'undefined') return { scope: 'global', projectID: null };
  const params = new URLSearchParams(window.location.search);
  const projectID = params.get('project_id')?.trim() || null;
  const rawScope = params.get('scope');
  const scope = isResourceScope(rawScope) ? rawScope : projectID ? 'project' : 'global';
  return { scope, projectID };
}

function writeLocationScope(scope: ResourceScope, projectID: string | null) {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  if (scope === 'global') url.searchParams.delete('scope');
  else url.searchParams.set('scope', scope);
  if (scope === 'project' && projectID) url.searchParams.set('project_id', projectID);
  else url.searchParams.delete('project_id');
  window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`);
}

export function useProjectScopeOptions() {
  const [projects, setProjects] = useState<ProjectScopeOption[]>([]);
  const [projectsEnabled, setProjectsEnabled] = useState(false);
  const [status, setStatus] = useState<'loading' | 'success' | 'disabled' | 'error'>('loading');

  useEffect(() => {
    let cancelled = false;
    apiClient
      .get('/api/projects')
      .then(response => {
        if (cancelled) return;
        const rawProjects = Array.isArray(response.data?.projects) ? response.data.projects : [];
        const nextProjects = rawProjects
          .filter(
            (project: unknown): project is ProjectScopeOption =>
              Boolean(
                project &&
                  typeof project === 'object' &&
                  'id' in project &&
                  typeof project.id === 'string' &&
                  'slug' in project &&
                  typeof project.slug === 'string' &&
                  'name' in project &&
                  typeof project.name === 'string' &&
                  'lifecycle_state' in project &&
                  typeof project.lifecycle_state === 'string',
              ),
          )
          .sort((left: ProjectScopeOption, right: ProjectScopeOption) => left.name.localeCompare(right.name));
        setProjects(nextProjects);
        setProjectsEnabled(true);
        setStatus('success');
      })
      .catch(error => {
        if (cancelled) return;
        const responseStatus = error && typeof error === 'object' && 'response' in error
          ? (error as { response?: { status?: number } }).response?.status
          : undefined;
        setProjects([]);
        setProjectsEnabled(false);
        setStatus(responseStatus === 403 ? 'disabled' : 'error');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { projects, projectsEnabled, status };
}

export default function useResourceScope() {
  const [scope, setScopeState] = useState<ResourceScope>('global');
  const [projectID, setProjectIDState] = useState<string | null>(null);
  const { projects, projectsEnabled, status } = useProjectScopeOptions();

  useEffect(() => {
    const syncFromLocation = () => {
      const next = readLocationScope();
      setScopeState(next.scope);
      setProjectIDState(next.projectID);
    };
    syncFromLocation();
    window.addEventListener('popstate', syncFromLocation);
    return () => window.removeEventListener('popstate', syncFromLocation);
  }, []);

  useEffect(() => {
    if (status === 'loading') return;
    if (status === 'disabled' && scope !== 'global') {
      setScopeState('global');
      setProjectIDState(null);
      writeLocationScope('global', null);
      return;
    }
    if (status === 'error') return;
    if (scope !== 'project') return;
    const resolved = projects.find(project => project.id === projectID || project.slug === projectID);
    const nextProjectID = resolved?.id || projects[0]?.id || null;
    if (!nextProjectID) {
      setScopeState('global');
      setProjectIDState(null);
      writeLocationScope('global', null);
      return;
    }
    if (projectID !== nextProjectID) {
      setProjectIDState(nextProjectID);
      writeLocationScope('project', nextProjectID);
    }
  }, [projectID, projects, projectsEnabled, scope, status]);

  const setScope = useCallback(
    (nextScope: ResourceScope) => {
      const nextProjectID = nextScope === 'project' ? projectID || projects[0]?.id || null : null;
      if (nextScope === 'project' && !nextProjectID) return;
      setScopeState(nextScope);
      setProjectIDState(nextProjectID);
      writeLocationScope(nextScope, nextProjectID);
    },
    [projectID, projects],
  );

  const setProjectID = useCallback((nextProjectID: string) => {
    setScopeState('project');
    setProjectIDState(nextProjectID);
    writeLocationScope('project', nextProjectID);
  }, []);

  const selectedProject = projects.find(project => project.id === projectID) || null;

  return {
    scope,
    projectID: scope === 'project' ? projectID : null,
    projects,
    projectsEnabled,
    projectsStatus: status,
    selectedProject,
    setScope,
    setProjectID,
  };
}
