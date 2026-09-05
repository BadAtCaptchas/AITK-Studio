import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { NextResponse } from 'next/server';
import { DatasetScopeError, resolveDatasetScope } from '@/server/datasetScope';
import { copyDatasetBetweenRoots } from '@/server/datasetCopy';

export async function POST(request: Request) {
  try {
    const body = assertGlobalPayload(await request.json());
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
