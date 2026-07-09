import { NextResponse } from 'next/server';
import { assertExecutionReplica, projectSyncWorkerError } from '@/server/projectSyncWorker';
import {
  buildProjectSyncManifest,
  commitProjectSyncPlan,
  assertProjectSyncQuota,
  parseProjectManifestEntry,
  PROJECT_SYNC_PROFILES,
  ProjectSyncProtocolError,
  type ProjectCommitPlan,
  type ProjectSyncProfileName,
} from '@/server/projectSyncProtocol';
import { getProjectRoots } from '@/server/projects';

export const runtime = 'nodejs';

export async function POST(request: Request, { params }: { params: Promise<{ projectID: string }> }) {
  try {
    const { projectID } = await params;
    const body: unknown = await request.json();
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
      throw new ProjectSyncProtocolError('Request body must be an object');
    }
    const value = body as Record<string, unknown>;
    const profileValue = value.profile;
    if (typeof profileValue !== 'string' || !(PROJECT_SYNC_PROFILES as readonly string[]).includes(profileValue)) {
      throw new ProjectSyncProtocolError('Unsupported project sync profile');
    }
    if (typeof value.operation_id !== 'string' || !Array.isArray(value.files)) {
      throw new ProjectSyncProtocolError('operation_id and files are required');
    }
    const project = await assertExecutionReplica(
      decodeURIComponent(projectID),
      typeof value.home_instance_id === 'string' ? value.home_instance_id : null,
    );
    const preservePaths = Array.isArray(value.preserve_paths)
      ? value.preserve_paths.map(item => {
          if (!item || typeof item !== 'object' || Array.isArray(item)) {
            throw new ProjectSyncProtocolError('Invalid preserve_paths entry');
          }
          const preserve = item as Record<string, unknown>;
          if (typeof preserve.path !== 'string' || typeof preserve.preserve_as !== 'string') {
            throw new ProjectSyncProtocolError('Invalid preserve_paths entry');
          }
          return { path: preserve.path, preserve_as: preserve.preserve_as };
        })
      : [];
    const files = value.files.map(entry => parseProjectManifestEntry(entry, profileValue as ProjectSyncProfileName));
    assertProjectSyncQuota(files);
    const plan: ProjectCommitPlan = {
      project_id: project.id,
      operation_id: value.operation_id,
      profile: profileValue as ProjectSyncProfileName,
      files,
      delete_paths: Array.isArray(value.delete_paths)
        ? value.delete_paths.filter((item): item is string => typeof item === 'string')
        : [],
      preserve_paths: preservePaths,
    };
    const roots = await getProjectRoots(project);
    await commitProjectSyncPlan(roots.root, plan);
    const manifest = await buildProjectSyncManifest(roots.root, project.id, plan.profile);
    return NextResponse.json({ committed: true, manifest });
  } catch (error) {
    const response = projectSyncWorkerError(error, 'Failed to commit project sync');
    return NextResponse.json(response.body, { status: response.status });
  }
}
