import { withPlainDatasetCommand } from '@/server/plainDatasetCommand';
import { readJsonCommand } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { resolveDatasetFolder } from '@/server/encryptedDatasets';
import { readDatasetRootCaption } from '@/server/datasetRootCaption';
import { DatasetScopeError, resolveDatasetScope } from '@/server/datasetScope';


async function postCommand(request: Request) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    const datasetName = typeof body?.datasetName === 'string' ? body.datasetName : '';

    const { datasetsRoot } = await resolveDatasetScope();
    const datasetFolder = resolveDatasetFolder(datasetsRoot, datasetName);
    return Response.json(await readDatasetRootCaption(datasetFolder));
  } catch (error) {
    const status = error instanceof DatasetScopeError ? error.status : 400;
    return Response.json(
      { error: error instanceof Error ? error.message : 'Failed to read dataset root caption' },
      { status },
    );
  }
}

export const POST = withPlainDatasetCommand(postCommand);
