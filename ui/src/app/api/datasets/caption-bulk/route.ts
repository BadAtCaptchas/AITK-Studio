import { withPlainDatasetCommand } from '@/server/plainDatasetCommand';
import { readJsonCommand } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import {
  decodeRemoteCaptionBulkPaths,
  isDatasetCaptionBulkError,
  mapRemoteCaptionBulkResult,
  performPlainDatasetCaptionBulkAction,
  type DatasetCaptionBulkRequest,
  type DatasetCaptionBulkResult,
} from '@/server/datasetCaptionBulk';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { DatasetScopeError, resolveDatasetScope } from '@/server/datasetScope';


async function postCommand(request: Request) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request)) as DatasetCaptionBulkRequest & {
      worker_id?: string;
    };
    const workerID = typeof body?.worker_id === 'string' ? body.worker_id : 'local';

    if (!isLocalWorker(workerID)) {
      const worker = await getRemoteWorker(workerID);
      const imgPaths = Array.isArray(body.imgPaths) ? body.imgPaths : [];
      const decoded = decodeRemoteCaptionBulkPaths(imgPaths, workerID);
      const remoteResult = await remoteJson<DatasetCaptionBulkResult>(worker, '/api/datasets/caption-bulk', {
        method: 'POST',
        body: JSON.stringify({
          ...body,
          worker_id: 'local',
          imgPaths: decoded.remotePaths,
        }),
      });
      return Response.json(mapRemoteCaptionBulkResult(remoteResult, decoded.refByRemotePath));
    }

    const { datasetsRoot } = await resolveDatasetScope();
    const result = await performPlainDatasetCaptionBulkAction(datasetsRoot, body);
    return Response.json(result);
  } catch (error) {
    if (isDatasetCaptionBulkError(error)) {
      return Response.json({ error: error.message }, { status: error.status });
    }
    if (error instanceof DatasetScopeError) {
      return Response.json({ error: error.message }, { status: error.status });
    }
    return Response.json(
      { error: error instanceof Error ? error.message : 'Failed to run bulk caption action' },
      { status: 500 },
    );
  }
}

export const POST = withPlainDatasetCommand(postCommand);
