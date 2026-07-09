'use client';

import ProjectArtifactLibrary from '@/components/project/ProjectArtifactLibrary';
import ProjectWorkspaceShell from '@/components/project/ProjectWorkspaceShell';
import { useProjectWorkspace } from '@/components/project/ProjectContext';

export default function ProjectModelsPage() {
  const { projectID } = useProjectWorkspace();
  return (
    <ProjectWorkspaceShell
      projectID={projectID}
      active="models"
      title="Models"
      description="Project checkpoints and model files, ready to download or use for generation."
    >
      <ProjectArtifactLibrary mode="models" />
    </ProjectWorkspaceShell>
  );
}
