import { NextResponse } from 'next/server';
import { db } from '@/server/db';
import { listDatasetSummaries } from '@/server/encryptedDatasets';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import type { DatasetSummary } from '@/types';
import { makeRemoteDatasetRef } from '@/utils/remoteDatasetRefs';
import { assertProjectScopeEnabled, rejectRemoteProjectScope, resolveDatasetScope } from '@/server/datasetScope';
import { assertProjectsEnabled } from '@/server/settings';

type ScopeProject = { id: string; name: string; slug: string; lifecycle_state: DatasetSummary['project_lifecycle_state'] };

function decorateRemoteDatasets(worker: { id: string; name: string }, datasets: DatasetSummary[]) {
  return datasets.map(dataset => ({
    ...dataset,
    source: 'remote' as const,
    worker_id: worker.id,
    worker_name: worker.name,
    ref: makeRemoteDatasetRef(worker.id, dataset.name),
    path: undefined,
    project_id: null,
    project_name: null,
    project_slug: null,
    project_lifecycle_state: null,
  }));
}

function decorateScopedDatasets(datasets: DatasetSummary[], project: ScopeProject | null) {
  return datasets.map(dataset => ({
    ...dataset,
    source: dataset.source || ('local' as const),
    worker_id: dataset.worker_id || 'local',
    worker_name: dataset.worker_name || 'Local',
    ref: project
      ? `aitk-dataset://project/${encodeURIComponent(project.id)}/${encodeURIComponent(dataset.name)}`
      : dataset.ref || `aitk-dataset://local/${encodeURIComponent(dataset.name)}`,
    project_id: project?.id || null,
    project_name: project?.name || null,
    project_slug: project?.slug || null,
    project_lifecycle_state: project?.lifecycle_state || null,
  }));
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const workerID = searchParams.get('worker_id') || 'local';
    const includeRemote = searchParams.get('include_remote') === '1';
    const projectID = searchParams.get('project_id');
    const rawScope = searchParams.get('scope');
    if (searchParams.has('project_id') && !projectID?.trim()) {
      return NextResponse.json(
        { error: 'project_id cannot be blank', code: 'PROJECT_INVALID_SCOPE' },
        { status: 400 },
      );
    }
    if (rawScope && !['global', 'all', 'project'].includes(rawScope)) {
      return NextResponse.json({ error: 'scope must be global, all, or project', code: 'INVALID_SCOPE' }, { status: 400 });
    }
    const scope = (rawScope || (projectID ? 'project' : 'global')) as 'global' | 'all' | 'project';
    if (scope === 'project' && !projectID) {
      return NextResponse.json({ error: 'project_id is required for project scope', code: 'PROJECT_ID_REQUIRED' }, { status: 400 });
    }
    if (rawScope && scope !== 'project' && projectID) {
      return NextResponse.json({ error: 'project_id is only valid for project scope', code: 'INVALID_SCOPE' }, { status: 400 });
    }
    if (scope === 'all') await assertProjectsEnabled();
    await assertProjectScopeEnabled(projectID);
    rejectRemoteProjectScope(workerID, scope === 'global' ? null : projectID || '__all__');

    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      const remoteDatasets = await remoteJson<DatasetSummary[]>(worker, '/api/datasets/list');
      return NextResponse.json({
        datasets: decorateRemoteDatasets(worker, Array.isArray(remoteDatasets) ? remoteDatasets : []),
        errors: [],
      });
    }

    const { datasetsRoot, project } = await resolveDatasetScope(scope === 'project' ? projectID : null, { intent: 'read' });
    const localDatasets = decorateScopedDatasets(
      await listDatasetSummaries(datasetsRoot, { createIfMissing: false }),
      project
        ? { id: project.id, name: project.name, slug: project.slug, lifecycle_state: project.lifecycle_state }
        : null,
    );
    if (scope === 'project') return NextResponse.json(localDatasets);

    const errors: Array<{ worker_id: string; worker_name: string; error: string }> = [];
    const remoteResults = includeRemote
      ? await Promise.all(
          (await db.workerNodes.list({ enabled: true })).map(async workerRecord => {
            try {
              const worker = await getRemoteWorker(workerRecord.id);
              const remoteDatasets = await remoteJson<DatasetSummary[]>(worker, '/api/datasets/list');
              return decorateRemoteDatasets(worker, Array.isArray(remoteDatasets) ? remoteDatasets : []);
            } catch (error) {
              errors.push({
                worker_id: workerRecord.id,
                worker_name: workerRecord.name,
                error: error instanceof Error ? error.message : 'Failed to fetch remote datasets',
              });
              return [];
            }
          }),
        )
      : [];

    const projectResults = scope === 'all'
      ? await Promise.all(
          (await db.projects.list()).map(async projectRecord => {
            try {
              const projectScope = await resolveDatasetScope(projectRecord.id, { intent: 'read' });
              return decorateScopedDatasets(
                await listDatasetSummaries(projectScope.datasetsRoot, { createIfMissing: false }),
                {
                  id: projectRecord.id,
                  name: projectRecord.name,
                  slug: projectRecord.slug,
                  lifecycle_state: projectRecord.lifecycle_state,
                },
              );
            } catch (error) {
              errors.push({
                worker_id: `project:${projectRecord.id}`,
                worker_name: projectRecord.name,
                error: error instanceof Error ? error.message : 'Failed to fetch project datasets',
              });
              return [];
            }
          }),
        )
      : [];

    if (!includeRemote && scope === 'global') return NextResponse.json(localDatasets);

    return NextResponse.json({
      datasets: [...localDatasets, ...projectResults.flat(), ...remoteResults.flat()],
      errors,
      scope,
      project_id: null,
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        error: error?.message || 'Failed to fetch datasets',
        ...(typeof error?.code === 'string' ? { code: error.code } : {}),
        ...(error?.details === undefined ? {} : { details: error.details }),
      },
      { status: typeof error?.status === 'number' ? error.status : 500 },
    );
  }
}
