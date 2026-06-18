import { NextResponse } from 'next/server';
import { UniqueConstraintError, db } from '@/server/db';
import { createProject, ensureProjectFolders } from '@/server/projects';
import { areProjectsEnabled, PROJECT_SPACES_DISABLED_MESSAGE } from '@/server/settings';

function ensureApiAccess(request: Request): NextResponse | null {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH;
  if (!tokenToUse) return null;

  const token = request.headers.get('authorization')?.split(' ')[1];
  if (token !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  return null;
}

export async function GET(request: Request) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const projects = await Promise.all(
      (await db.projects.list()).map(async project => ({
        ...project,
        roots: await ensureProjectFolders(project),
      })),
    );
    return NextResponse.json({ projects });
  } catch (error) {
    console.error('Failed to list projects:', error);
    return NextResponse.json({ error: 'Failed to list projects' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const body = await request.json();
    const project = await createProject(body || {});
    return NextResponse.json({ project, roots: await ensureProjectFolders(project) });
  } catch (error: any) {
    if (error instanceof UniqueConstraintError || error?.code === 'P2002') {
      return NextResponse.json({ error: 'Project slug already exists' }, { status: 409 });
    }
    return NextResponse.json(
      { error: error?.message || 'Failed to create project' },
      { status: typeof error?.status === 'number' ? error.status : 500 },
    );
  }
}
