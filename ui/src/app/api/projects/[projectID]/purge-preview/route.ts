import { NextResponse } from 'next/server';
import { getProjectPurgePreview } from '@/server/projects';
import { areProjectsEnabled, PROJECT_SPACES_DISABLED_MESSAGE } from '@/server/settings';
import { ensureProjectApiAccess, projectApiError } from '../../projectApi';

export async function GET(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = ensureProjectApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }
  try {
    const { projectID } = await params;
    return NextResponse.json(await getProjectPurgePreview(decodeURIComponent(projectID)));
  } catch (error) {
    return projectApiError(error, 'Failed to preview project purge');
  }
}
