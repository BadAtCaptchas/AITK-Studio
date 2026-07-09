import { NextResponse } from 'next/server';
import { restoreProject } from '@/server/projects';
import { areProjectsEnabled, PROJECT_SPACES_DISABLED_MESSAGE } from '@/server/settings';
import { ensureProjectApiAccess, projectApiError, readJsonObject } from '../../projectApi';

export async function POST(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = await ensureProjectApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }
  try {
    const { projectID } = await params;
    const body = await readJsonObject(request);
    const project = await restoreProject(decodeURIComponent(projectID), body.expected_revision);
    return NextResponse.json({ project, restored: true, canonical_project_id: project.id });
  } catch (error) {
    return projectApiError(error, 'Failed to restore project');
  }
}
