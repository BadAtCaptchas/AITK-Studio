import { NextResponse } from 'next/server';
import { resolveDatasetScope } from '@/server/datasetScope';
import { copyDatasetBetweenRoots } from '@/server/datasetCopy';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const datasetPath = typeof body?.datasetPath === 'string' ? body.datasetPath : '';
    const hasSourceProject = Object.prototype.hasOwnProperty.call(body || {}, 'source_project_id');
    const destinationScope = await resolveDatasetScope(body?.project_id);
    const sourceScope = hasSourceProject
      ? await resolveDatasetScope(body?.source_project_id)
      : destinationScope;
    const destination = await copyDatasetBetweenRoots({
      datasetPath,
      sourceDatasetsRoot: sourceScope.datasetsRoot,
      destinationDatasetsRoot: destinationScope.datasetsRoot,
      requestedName: typeof body?.name === 'string' ? body.name : undefined,
      suffix: typeof body?.suffix === 'string' ? body.suffix : 'copy',
    });

    return NextResponse.json(destination);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to copy dataset' },
      { status: 400 },
    );
  }
}
