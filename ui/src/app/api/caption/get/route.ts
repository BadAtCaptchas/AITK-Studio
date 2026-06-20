/* eslint-disable */
import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import { findEncryptedDatasetRoot } from '@/server/encryptedDatasets';
import { getRemoteWorker, remoteFetch } from '@/server/remoteClient';
import { readCaptionSidecar } from '@/server/captionFiles';
import { parseRemoteDatasetAssetRef } from '@/utils/remoteDatasetRefs';
import { DatasetScopeError, resolveDatasetScope } from '@/server/datasetScope';
import { sanitizeCaptionText } from '@/utils/captionQuality';

export async function POST(request: NextRequest) {
  let body;
  try {
    body = await request.json();
  } catch {
    // Client aborted the request before body was fully sent
    return new NextResponse(null, { status: 499 });
  }

  if (request.signal.aborted) {
    return new NextResponse(null, { status: 499 });
  }

  const { imgPath } = body;
  try {
    const remoteAsset = parseRemoteDatasetAssetRef(imgPath);
    if (remoteAsset) {
      const worker = await getRemoteWorker(remoteAsset.workerID);
      const remoteResponse = await remoteFetch(worker, '/api/caption/get', {
        method: 'POST',
        body: JSON.stringify({ imgPath: remoteAsset.path }),
        headers: { 'Content-Type': 'application/json' },
      });
      if (remoteResponse.ok) {
        return new NextResponse(sanitizeCaptionText(await remoteResponse.text()), {
          status: remoteResponse.status,
          headers: {
            'Content-Type': remoteResponse.headers.get('content-type') || 'text/plain; charset=utf-8',
            'Cache-Control': 'no-store',
          },
        });
      }
      return new NextResponse(remoteResponse.body, {
        status: remoteResponse.status,
        headers: {
          'Content-Type': remoteResponse.headers.get('content-type') || 'text/plain; charset=utf-8',
          'Cache-Control': 'no-store',
        },
      });
    }

    // Decode the path
    const filepath = imgPath;

    // Get allowed directories
    const { datasetsRoot: allowedDir } = await resolveDatasetScope(body?.project_id);

    const resolvedFilePath = path.resolve(filepath);
    const allowedRoot = path.resolve(allowedDir);
    const relativeFilePath = path.relative(allowedRoot, resolvedFilePath);
    const isAllowed = relativeFilePath !== '' && !relativeFilePath.startsWith('..') && !path.isAbsolute(relativeFilePath);

    if (!isAllowed) {
      console.warn(`Access denied: ${filepath} not in ${allowedDir}`);
      return new NextResponse('Access denied', { status: 403 });
    }

    if (findEncryptedDatasetRoot(resolvedFilePath, allowedRoot)) {
      return new NextResponse('Encrypted captions are not served through this route', { status: 403 });
    }

    return new NextResponse(readCaptionSidecar(resolvedFilePath));
  } catch (error) {
    console.error('Error getting caption:', error);
    if (error instanceof DatasetScopeError) {
      return new NextResponse(error.message, { status: error.status });
    }
    return new NextResponse('Error getting caption', { status: 500 });
  }
}
