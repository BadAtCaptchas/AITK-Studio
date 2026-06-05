import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';
import { findEncryptedDatasetRoot } from '@/server/encryptedDatasets';
import { getRemoteWorker, remoteJson } from '@/server/remoteClient';
import { resolveCaptionWritePath } from '@/server/captionFiles';
import { parseRemoteDatasetAssetRef } from '@/utils/remoteDatasetRefs';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { imgPath, caption } = body;
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

    const datasetsPath = await getDatasetsRoot();
    const datasetsRoot = path.resolve(datasetsPath);
    const resolvedImagePath = path.resolve(imgPath);
    const relativeImagePath = path.relative(datasetsRoot, resolvedImagePath);

    // make sure the resolved image path is in the dataset path
    if (relativeImagePath.startsWith('..') || path.isAbsolute(relativeImagePath)) {
      return NextResponse.json({ error: 'Invalid image path' }, { status: 400 });
    }

    if (findEncryptedDatasetRoot(resolvedImagePath, datasetsRoot)) {
      return NextResponse.json({ error: 'Encrypted captions must be saved through the encrypted dataset API' }, { status: 403 });
    }

    // if img doesnt exist, ignore
    if (!fs.existsSync(resolvedImagePath)) {
      return NextResponse.json({ error: 'Image does not exist' }, { status: 404 });
    }

    const captionText = typeof caption === 'string' ? caption : String(caption ?? '');
    const captionPath = resolveCaptionWritePath(resolvedImagePath, captionText);
    // save caption to file
    fs.writeFileSync(captionPath, captionText);

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
