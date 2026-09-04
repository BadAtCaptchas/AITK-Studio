'use client';

import { Boxes, FolderKanban, Globe2 } from 'lucide-react';
import type { ProjectScopeOption, ResourceScope } from '@/hooks/useResourceScope';

type ResourceScopeFilterProps = {
  scope: ResourceScope;
  projectID: string | null;
  projects: ProjectScopeOption[];
  projectsEnabled: boolean;
  onScopeChange: (scope: ResourceScope) => void;
  onProjectChange: (projectID: string) => void;
  compact?: boolean;
};

const SCOPE_OPTIONS = [
  { value: 'global', label: 'Global', icon: Globe2 },
  { value: 'all', label: 'All', icon: Boxes },
  { value: 'project', label: 'Project', icon: FolderKanban },
] as const;

export default function ResourceScopeFilter({
  scope,
  projectID,
  projects,
  projectsEnabled,
  onScopeChange,
  onProjectChange,
  compact = false,
}: ResourceScopeFilterProps) {
  return (
    <div className="flex min-w-0 items-center gap-2" aria-label="Workspace scope">
      <div className="inline-flex h-8 shrink-0 rounded-sm border border-gray-800 bg-gray-950 p-0.5">
        {SCOPE_OPTIONS.map(option => {
          const Icon = option.icon;
          const disabled = option.value !== 'global' && (!projectsEnabled || projects.length === 0);
          return (
            <button
              key={option.value}
              type="button"
              disabled={disabled}
              onClick={() => onScopeChange(option.value)}
              className={`inline-flex items-center justify-center gap-1.5 rounded-sm px-2 text-xs font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-35 ${
                scope === option.value
                  ? 'bg-gray-800 text-gray-100'
                  : 'text-gray-400 hover:bg-gray-900 hover:text-gray-200'
              }`}
              aria-pressed={scope === option.value}
              title={`${option.label} workspace`}
            >
              <Icon className="h-3.5 w-3.5" />
              {!compact || scope === option.value ? option.label : <span className="sr-only">{option.label}</span>}
            </button>
          );
        })}
      </div>
      {scope === 'project' && projects.length > 0 ? (
        <select
          value={projectID || projects[0].id}
          onChange={event => onProjectChange(event.target.value)}
          className="h-8 min-w-0 max-w-52 rounded-sm border border-brand-800/70 bg-gray-950 px-2 text-xs text-gray-100 outline-none focus:border-brand-500 [color-scheme:dark]"
          aria-label="Project"
        >
          {projects.map(project => (
            <option key={project.id} value={project.id}>
              {project.name}{project.lifecycle_state === 'archived' ? ' (Archived)' : ''}
            </option>
          ))}
        </select>
      ) : null}
    </div>
  );
}

export function ProjectResourceBadge({
  projectID,
  projectName,
}: {
  projectID?: string | null;
  projectName?: string | null;
}) {
  if (!projectID) {
    return (
      <span className="inline-flex max-w-full items-center gap-1 rounded-sm border border-gray-700 bg-gray-950 px-1.5 py-0.5 text-[10px] font-medium text-gray-400">
        <Globe2 className="h-3 w-3 flex-none" />
        Global
      </span>
    );
  }
  return (
    <span
      className="inline-flex max-w-full items-center gap-1 rounded-sm border border-brand-800/70 bg-brand-950/35 px-1.5 py-0.5 text-[10px] font-medium text-brand-200"
      title={projectName || projectID}
    >
      <FolderKanban className="h-3 w-3 flex-none" />
      <span className="truncate">{projectName || projectID}</span>
    </span>
  );
}
