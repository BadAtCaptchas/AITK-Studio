'use client';

import ProjectArtifactLibrary from '@/components/project/ProjectArtifactLibrary';
import ProjectWorkspaceShell from '@/components/project/ProjectWorkspaceShell';
import { useProjectWorkspace } from '@/components/project/ProjectContext';

export default function ProjectOutputsPage() {
  const { projectID } = useProjectWorkspace();
  return (
    <ProjectWorkspaceShell
      projectID={projectID}
      active="outputs"
      title="Outputs"
      description="Review generated media and training samples from every project run."
    >
      <ProjectArtifactLibrary mode="outputs" />
    </ProjectWorkspaceShell>
  );
}
