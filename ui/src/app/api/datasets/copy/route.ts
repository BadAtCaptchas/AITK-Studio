import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { NextResponse } from 'next/server';
import { DatasetScopeError, resolveDatasetScope } from '@/server/datasetScope';
import { copyDatasetBetweenRoots } from '@/server/datasetCopy';

async function postCommand(request: Request) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    const datasetPath = typeof body?.datasetPath === 'string' ? body.datasetPath : '';
    const destinationScope = await resolveDatasetScope();
    const sourceScope = destinationScope;
    const destination = await copyDatasetBetweenRoots({
      datasetPath,
      sourceDatasetsRoot: sourceScope.datasetsRoot,
      destinationDatasetsRoot: destinationScope.datasetsRoot,
      requestedName: typeof body?.name === 'string' ? body.name : undefined,
      suffix: typeof body?.suffix === 'string' ? body.suffix : 'copy',
    });

    return NextResponse.json(destination);
  } catch (error) {
    const status = error instanceof DatasetScopeError ? error.status : 400;
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Failed to copy dataset' }, { status });
  }
}

export const POST = withCommandBoundary(postCommand);
