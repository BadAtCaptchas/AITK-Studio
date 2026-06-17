'use client';

import { use } from 'react';
import Link from 'next/link';
import { Plus } from 'lucide-react';
import JobsTable from '@/components/JobsTable';
import ProjectWorkspaceShell from '@/components/project/ProjectWorkspaceShell';

export default function ProjectRunsPage({ params }: { params: Promise<{ projectID: string }> }) {
  const { projectID: rawProjectID } = use(params);
  const projectID = decodeURIComponent(rawProjectID);

  return (
    <ProjectWorkspaceShell
      projectID={projectID}
      active="runs"
      title="Runs"
      description="Training, captioning, and generation jobs scoped to this project."
      actions={
        <Link href={`/jobs/new?project_id=${encodeURIComponent(projectID)}`} className="operator-button h-9">
          <Plus className="h-4 w-4" />
          New Train
        </Link>
      }
    >
      <div className="h-full overflow-auto p-4">
        <div className="mx-auto max-w-[1500px]">
          <JobsTable projectID={projectID} />
        </div>
      </div>
    </ProjectWorkspaceShell>
  );
}
