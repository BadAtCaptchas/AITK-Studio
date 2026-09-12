import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { NextRequest, NextResponse } from 'next/server';
import { resolveDatasetFolder } from '@/server/encryptedDatasets';
import { readDatasetRootCaption } from '@/server/datasetRootCaption';
import { DatasetScopeError, resolveDatasetScope } from '@/server/datasetScope';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

async function postCommand(request: NextRequest) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    const datasetName = typeof body?.datasetName === 'string' ? body.datasetName : '';

    const { datasetsRoot } = await resolveDatasetScope();
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

export const POST = withCommandBoundary(postCommand);
