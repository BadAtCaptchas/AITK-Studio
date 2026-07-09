'use client';

import { use } from 'react';
import type { ReactNode } from 'react';
import ProjectFrame from '@/components/project/ProjectFrame';
import { ProjectProvider } from '@/components/project/ProjectContext';

export default function ProjectLayout({
  children,
  params,
}: {
  children: ReactNode;
  params: Promise<{ projectID: string }>;
}) {
  const { projectID: rawProjectID } = use(params);
  const projectID = decodeURIComponent(rawProjectID);
  return (
    <ProjectProvider projectID={projectID}>
      <ProjectFrame>{children}</ProjectFrame>
    </ProjectProvider>
  );
}
