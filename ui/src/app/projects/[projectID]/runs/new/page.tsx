'use client';

import { use } from 'react';
import ProjectWorkspaceShell from '@/components/project/ProjectWorkspaceShell';
import { TrainingFormContent } from '@/app/jobs/new/TrainingFormContent';

export default function ProjectNewRunPage({ params }: { params: Promise<{ projectID: string }> }) {
  const { projectID: rawProjectID } = use(params);
  const projectID = decodeURIComponent(rawProjectID);

  return (
    <ProjectWorkspaceShell projectID={projectID} active="runs" showHeader={false}>
      <TrainingFormContent projectIDOverride={projectID} />
    </ProjectWorkspaceShell>
  );
}
