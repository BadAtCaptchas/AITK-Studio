import { NextRequest, NextResponse } from 'next/server';
import { resolveDatasetFolder } from '@/server/encryptedDatasets';
import { readDatasetRootCaption } from '@/server/datasetRootCaption';
import { assertProjectScopeEnabled, DatasetScopeError, resolveDatasetScope } from '@/server/datasetScope';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const datasetName = typeof body?.datasetName === 'string' ? body.datasetName : '';
    const projectID = body?.project_id;
    await assertProjectScopeEnabled(projectID);
    const { datasetsRoot } = await resolveDatasetScope(projectID);
    const datasetFolder = resolveDatasetFolder(datasetsRoot, datasetName);
    return NextResponse.json(await readDatasetRootCaption(datasetFolder));
  } catch (error) {
    const status = error instanceof DatasetScopeError ? error.status : 400;
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to read dataset root caption' },
      { status },
    );
  }
}
