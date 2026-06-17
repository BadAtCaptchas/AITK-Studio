'use client';

import { use } from 'react';
import GeneratePage from '@/app/generate/page';
import ProjectWorkspaceShell from '@/components/project/ProjectWorkspaceShell';

export default function ProjectGeneratePage({ params }: { params: Promise<{ projectID: string }> }) {
  const { projectID: rawProjectID } = use(params);
  const projectID = decodeURIComponent(rawProjectID);

  return (
    <ProjectWorkspaceShell projectID={projectID} active="generate" showHeader={false}>
      <GeneratePage projectIDOverride={projectID} />
    </ProjectWorkspaceShell>
  );
}
