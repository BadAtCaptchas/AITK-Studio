import { NextResponse } from 'next/server';
import { archiveProject } from '@/server/projects';
import { areProjectsEnabled, PROJECT_SPACES_DISABLED_MESSAGE } from '@/server/settings';
import { ensureProjectApiAccess, projectApiError, readJsonObject } from '../../projectApi';

export async function POST(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = ensureProjectApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }
  try {
    const { projectID } = await params;
    const body = await readJsonObject(request);
    const project = await archiveProject(decodeURIComponent(projectID), body.expected_revision);
    return NextResponse.json({ project, archived: true, canonical_project_id: project.id });
  } catch (error) {
    return projectApiError(error, 'Failed to archive project');
  }
}
