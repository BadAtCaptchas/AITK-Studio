import { NextResponse } from 'next/server';
import { UniqueConstraintError, db } from '@/server/db';
import { createProject, getProjectRoots } from '@/server/projects';
import {
  setupProjectWorkspace,
  validateProjectSetupRequest,
  type ProjectSetupMode,
  type ProjectSetupRequest,
} from '@/server/projectSetup';
import { rehomeProject } from '@/server/projectSync';
import { listDatasetSummaries } from '@/server/encryptedDatasets';
import { indexProjectArtifacts } from '@/server/projectArtifactIndex';
import { areProjectsEnabled, PROJECT_SPACES_DISABLED_MESSAGE } from '@/server/settings';
import { ensureProjectApiAccess, projectApiError, readJsonObject } from './projectApi';
import type { Project } from '@/types';

const ACTIVE_JOB_STATUSES = new Set(['queued', 'starting', 'running', 'stopping']);

async function buildProjectCard(project: Project) {
  const roots = await getProjectRoots(project);
  const [datasets, jobs, artifacts, replicas] = await Promise.all([
    listDatasetSummaries(roots.datasets, { createIfMissing: false }),
    db.jobs.list({ project_id: project.id }),
    indexProjectArtifacts(project, { kind: 'all', maxEntries: 3_000 }),
    db.projectReplicas.listByProject(project.id),
  ]);
  const activeJobs = jobs.filter(job => ACTIVE_JOB_STATUSES.has(job.status));
  const outputArtifacts = artifacts.filter(artifact => ['image', 'video', 'audio'].includes(artifact.kind));
  const modelArtifacts = artifacts.filter(artifact => artifact.kind === 'model');
  const workflowStage =
    activeJobs.length > 0 ? 'train' : outputArtifacts.length > 0 || modelArtifacts.length > 0 ? 'review' : 'prepare';

  return {
    project,
    roots,
    workflow_stage: workflowStage,
    dataset_count: datasets.length,
    run_count: jobs.length,
    output_count: outputArtifacts.length,
    model_count: modelArtifacts.length,
    total_bytes: artifacts.reduce((total, artifact) => total + artifact.size, 0),
    active_job:
      activeJobs.sort((left, right) => new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime())[0] ||
      null,
    latest_output: outputArtifacts[0] || null,
    replica_warnings: replicas.filter(replica => !['in_sync', 'syncing'].includes(replica.state)).length,
    counts: {
      datasets: datasets.length,
      jobs: jobs.length,
      activeJobs: activeJobs.length,
      outputs: outputArtifacts.length,
      models: modelArtifacts.length,
    },
  };
}

export async function GET(request: Request) {
  const accessResponse = ensureProjectApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const records = await db.projects.list();
    const cards = await Promise.all(records.map(buildProjectCard));
    const projects = cards.map(card => ({
      ...card.project,
      roots: card.roots,
      canonical_project_id: card.project.id,
    }));
    return NextResponse.json({ projects, cards });
  } catch (error) {
    console.error('Failed to list projects:', error);
    return NextResponse.json({ error: 'Failed to list projects' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  const accessResponse = ensureProjectApiAccess(request);
  if (accessResponse) return accessResponse;
  if (!(await areProjectsEnabled())) {
    return NextResponse.json({ error: PROJECT_SPACES_DISABLED_MESSAGE }, { status: 403 });
  }

  try {
    const body = await readJsonObject(request);
    const setupMode = body.setup_mode === undefined ? 'blank' : body.setup_mode;
    if (!['blank', 'import', 'clone'].includes(String(setupMode))) {
      return NextResponse.json(
        { error: 'setup_mode must be blank, import, or clone', code: 'PROJECT_INVALID_INPUT' },
        { status: 400 },
      );
    }
    const setupRequest: ProjectSetupRequest = {
      mode: setupMode as ProjectSetupMode,
      importRoot: typeof body.import_root === 'string' ? body.import_root.trim() : undefined,
      cloneFromProjectID:
        typeof body.clone_from_project_id === 'string' ? body.clone_from_project_id.trim() : undefined,
    };
    await validateProjectSetupRequest(setupRequest);
    const project = await createProject({
      name: body.name,
      slug: body.slug,
      description: body.description,
      badge_asset: body.badge_asset,
    });
    const setup = await setupProjectWorkspace(project, setupRequest);
    let currentProject = (await db.projects.findById(project.id)) || project;
    const requestedHomeWorkerID =
      typeof body.home_worker_id === 'string' && body.home_worker_id.trim() ? body.home_worker_id.trim() : 'local';
    let homeSetup: { status: 'local' | 'completed' | 'failed'; worker_id: string; error: string | null } = {
      status: 'local',
      worker_id: 'local',
      error: null,
    };
    if (requestedHomeWorkerID !== 'local' && setup.status === 'completed') {
      try {
        const rehome = await rehomeProject(currentProject.id, requestedHomeWorkerID, currentProject.revision);
        currentProject = rehome.project;
        homeSetup = { status: 'completed', worker_id: requestedHomeWorkerID, error: null };
      } catch (homeError) {
        const message = homeError instanceof Error ? homeError.message.slice(0, 500) : 'Remote home setup failed';
        currentProject = await db.projects.update(currentProject.id, { operation_error: message });
        homeSetup = { status: 'failed', worker_id: requestedHomeWorkerID, error: message };
      }
    }
    return NextResponse.json(
      {
        project: currentProject,
        roots: await getProjectRoots(currentProject),
        canonical_project_id: currentProject.id,
        setup,
        home_setup: homeSetup,
      },
      { status: 201 },
    );
  } catch (error: unknown) {
    const errorCode = error && typeof error === 'object' && 'code' in error ? error.code : null;
    if (error instanceof UniqueConstraintError || errorCode === 'P2002') {
      return NextResponse.json({ error: 'Project slug already exists' }, { status: 409 });
    }
    return projectApiError(error, 'Failed to create project');
  }
}
