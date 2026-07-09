import { NextResponse } from 'next/server';
import { UniqueConstraintError, db } from '@/server/db';
import { archiveProject, cleanProjectSlug, getProjectRoots, resolveProject, safeProjectName } from '@/server/projects';
import { areProjectsEnabled, PROJECT_SPACES_DISABLED_MESSAGE } from '@/server/settings';
import { ensureProjectApiAccess, projectApiError, readJsonObject } from '../projectApi';

export async function GET(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = await ensureProjectApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const { projectID } = await params;
    const requestedIdentifier = decodeURIComponent(projectID);
    const project = await resolveProject(requestedIdentifier, { intent: 'read' });
    const [replicas, syncOperations, workers] = await Promise.all([
      db.projectReplicas.listByProject(project.id),
      db.projectSyncOperations.list({ project_id: project.id }),
      db.workerNodes.list(),
    ]);
    const workerNames = new Map(workers.map(worker => [worker.id, worker.name]));
    return NextResponse.json({
      project,
      roots: await getProjectRoots(project),
      replicas: replicas.map(replica => ({
        ...replica,
        worker_name: workerNames.get(replica.worker_id) || replica.worker_id,
        status: replica.state,
      })),
      sync_operations: syncOperations.slice(0, 20),
      canonical_project_id: project.id,
      requested_by_slug: requestedIdentifier !== project.id,
    });
  } catch (error) {
    return projectApiError(error, 'Project not found');
  }
}

export async function PATCH(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = await ensureProjectApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const { projectID } = await params;
    const existing = await resolveProject(decodeURIComponent(projectID), { intent: 'write' });
    const body = await readJsonObject(request);
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
    return NextResponse.json({ project, roots: await getProjectRoots(project), canonical_project_id: project.id });
  } catch (error: any) {
    if (error instanceof UniqueConstraintError || error?.code === 'P2002') {
      return NextResponse.json({ error: 'Project slug already exists' }, { status: 409 });
    }
    return projectApiError(error, 'Failed to update project');
  }
}

export async function DELETE(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  const accessResponse = await ensureProjectApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const { projectID } = await params;
    const existing = await resolveProject(decodeURIComponent(projectID), { intent: 'lifecycle' });
    const project = await archiveProject(existing.id, existing.revision);
    return NextResponse.json({
      success: true,
      archived: true,
      deprecated: 'DELETE archives a project; use POST /archive. Permanent deletion requires /purge.',
      deletedProject: project,
      project,
    });
  } catch (error) {
    return projectApiError(error, 'Failed to archive project');
  }
}
