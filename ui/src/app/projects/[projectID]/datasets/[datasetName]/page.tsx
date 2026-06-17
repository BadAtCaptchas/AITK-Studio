'use client';

import { use, useEffect, useState } from 'react';
import DatasetEditorPage from '@/components/DatasetEditorPage';
import ProjectWorkspaceShell from '@/components/project/ProjectWorkspaceShell';
import { apiClient } from '@/utils/api';
import type { ProjectSummary } from '@/components/project/types';

export default function ProjectDatasetEditorRoute({
  params,
}: {
  params: Promise<{ projectID: string; datasetName: string }>;
}) {
  const { projectID: rawProjectID, datasetName: rawDatasetName } = use(params);
  const projectID = decodeURIComponent(rawProjectID);
  const datasetName = decodeURIComponent(rawDatasetName);
  const [summary, setSummary] = useState<ProjectSummary | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiClient
      .get(`/api/projects/${encodeURIComponent(projectID)}/summary`)
      .then(res => {
        if (!cancelled) setSummary(res.data);
      })
      .catch(error => {
        console.error('Failed to load project for dataset editor:', error);
      });
    return () => {
      cancelled = true;
    };
  }, [projectID]);

  return (
    <ProjectWorkspaceShell projectID={projectID} active="datasets" showHeader={false}>
      <DatasetEditorPage
        datasetName={datasetName}
        projectID={projectID}
        datasetRoot={summary?.roots.datasets || null}
        projectName={summary?.project.name || null}
        returnHref={`/projects/${encodeURIComponent(projectID)}/datasets`}
      />
    </ProjectWorkspaceShell>
  );
}
