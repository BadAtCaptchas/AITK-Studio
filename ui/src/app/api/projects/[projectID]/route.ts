import { NextResponse } from 'next/server';
import { UniqueConstraintError, db } from '@/server/db';
import { cleanProjectSlug, ensureProjectFolders, resolveProject, safeProjectName } from '@/server/projects';
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

export async function GET(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const { projectID } = await params;
    const project = await resolveProject(decodeURIComponent(projectID));
    return NextResponse.json({ project, roots: await ensureProjectFolders(project) });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Project not found' }, { status: 404 });
  }
}

export async function PATCH(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const { projectID } = await params;
    const existing = await resolveProject(decodeURIComponent(projectID));
    const body = await request.json();
    const patch: Record<string, string | null> = {};

    if ('name' in body) {
      const name = safeProjectName(body.name);
      if (!name) return NextResponse.json({ error: 'Project name is required' }, { status: 400 });
      patch.name = name;
    }
    if ('description' in body) {
      patch.description = typeof body.description === 'string' ? body.description.trim().slice(0, 500) : '';
    }
    if ('badge_asset' in body) {
      patch.badge_asset = typeof body.badge_asset === 'string' && body.badge_asset.trim() ? body.badge_asset.trim() : null;
    }
    if ('slug' in body && typeof body.slug === 'string' && body.slug.trim()) {
      const slug = cleanProjectSlug(body.slug);
      if (!slug) return NextResponse.json({ error: 'Invalid project slug' }, { status: 400 });
      patch.slug = slug;
    }

    const project = await db.projects.update(existing.id, patch);
    return NextResponse.json({ project, roots: await ensureProjectFolders(project) });
  } catch (error: any) {
    if (error instanceof UniqueConstraintError || error?.code === 'P2002') {
      return NextResponse.json({ error: 'Project slug already exists' }, { status: 409 });
    }
    return NextResponse.json({ error: error?.message || 'Failed to update project' }, { status: 500 });
  }
}

export async function DELETE(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = ensureApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const { projectID } = await params;
    const project = await resolveProject(decodeURIComponent(projectID));
    await db.projects.delete(project.id);
    return NextResponse.json({ success: true, deletedProject: project });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to delete project' }, { status: 500 });
  }
}
