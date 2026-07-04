'use client';

import { use } from 'react';
import { GeneratePageContent } from '@/app/generate/GeneratePageContent';
import ProjectWorkspaceShell from '@/components/project/ProjectWorkspaceShell';

export default function ProjectGeneratePage({ params }: { params: Promise<{ projectID: string }> }) {
  const { projectID: rawProjectID } = use(params);
  const projectID = decodeURIComponent(rawProjectID);

  return (
    <ProjectWorkspaceShell projectID={projectID} active="generate" showHeader={false}>
      <GeneratePageContent projectIDOverride={projectID} />
    </ProjectWorkspaceShell>
  );
}
