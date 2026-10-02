import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import { NextResponse } from 'next/server';
import { getRemoteWorker, remoteJson } from '@/server/remoteClient';
import { DatasetCaptionWriteError, writePlainDatasetCaption } from '@/server/datasetCaptionWrite';
import { parseRemoteDatasetAssetRef } from '@/utils/remoteDatasetRefs';
import { DatasetScopeError, resolveDatasetScope } from '@/server/datasetScope';

async function postCommand(request: Request) {
  try {
    const body = assertGlobalPayload(await readJsonCommand(request));
    const { imgPath, caption } = body;
    if (typeof imgPath !== 'string' || !imgPath || typeof caption !== 'string') {
      return NextResponse.json({ error: 'imgPath and caption must be strings' }, { status: 400 });
    }
    const remoteAsset = parseRemoteDatasetAssetRef(imgPath);
    if (remoteAsset) {
      const worker = await getRemoteWorker(remoteAsset.workerID);
      return NextResponse.json(
        await remoteJson(worker, '/api/img/caption', {
          method: 'POST',
          body: JSON.stringify({ imgPath: remoteAsset.path, caption }),
        }),
      );
    }

    const { datasetsRoot } = await resolveDatasetScope();
    return NextResponse.json(await writePlainDatasetCaption(datasetsRoot, imgPath, caption));
  } catch (error) {
    if (error instanceof DatasetScopeError || error instanceof DatasetCaptionWriteError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    return NextResponse.json({ error: 'Failed to save caption' }, { status: 500 });
  }
}

export const POST = withCommandBoundary(postCommand);
