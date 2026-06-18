import { NextResponse } from 'next/server';
import {
  deletePlainImagePaths,
  isImageDeleteError,
  type ImageDeleteBulkResult,
  type ImageDeleteItemResult,
} from '@/server/imageDelete';
import { getRemoteWorker, remoteJson } from '@/server/remoteClient';
import { parseRemoteDatasetAssetRef } from '@/utils/remoteDatasetRefs';
import { DatasetScopeError, resolveDatasetScope } from '@/server/datasetScope';

type RemoteGroup = {
  workerID: string;
  paths: string[];
  refByRemotePath: Record<string, string>;
};

function requireAuth(request: Request) {
  const tokenToUse = process.env.AI_TOOLKIT_AUTH || null;
  if (!tokenToUse) return null;
  const token = request.headers.get('Authorization')?.split(' ')[1];
  if (!token || token !== tokenToUse) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  return null;
}

function emptyBulkResult(): ImageDeleteBulkResult {
  return {
    success: true,
    requested: 0,
    deleted: 0,
    skipped: 0,
    failed: 0,
    removedPaths: [],
    results: [],
  };
}

function combineBulkResults(results: ImageDeleteBulkResult[]) {
  const combined = emptyBulkResult();
  for (const result of results) {
    combined.requested += result.requested;
    combined.deleted += result.deleted;
    combined.skipped += result.skipped;
    combined.failed += result.failed;
    combined.removedPaths.push(...result.removedPaths);
    combined.results.push(...result.results);
  }
  combined.success = combined.failed === 0;
  return combined;
}

function mapRemoteResult(result: ImageDeleteBulkResult, refByRemotePath: Record<string, string>): ImageDeleteBulkResult {
  const mapPath = (remotePath: string) => refByRemotePath[remotePath] || remotePath;
  return {
    ...result,
    removedPaths: result.removedPaths.map(mapPath),
    results: result.results.map(
      (item): ImageDeleteItemResult => ({
        ...item,
        imgPath: mapPath(item.imgPath),
      }),
    ),
  };
}

function splitRemoteAndLocalPaths(imgPaths: unknown) {
  if (!Array.isArray(imgPaths)) {
    return { localPaths: null, remoteGroups: null, error: 'imgPaths must be an array' };
  }

  const localPaths: string[] = [];
  const remoteGroups = new Map<string, RemoteGroup>();

  for (const value of imgPaths) {
    if (typeof value !== 'string' || !value.trim()) continue;
    const remoteAsset = parseRemoteDatasetAssetRef(value);
    if (!remoteAsset) {
      localPaths.push(value);
      continue;
    }
    if (remoteAsset.type !== 'img') {
      return { localPaths: null, remoteGroups: null, error: 'Invalid remote image path' };
    }
    const group =
      remoteGroups.get(remoteAsset.workerID) ||
      {
        workerID: remoteAsset.workerID,
        paths: [],
        refByRemotePath: {},
      };
    group.paths.push(remoteAsset.path);
    group.refByRemotePath[remoteAsset.path] = value;
    remoteGroups.set(remoteAsset.workerID, group);
  }

  if (localPaths.length === 0 && remoteGroups.size === 0) {
    return { localPaths: null, remoteGroups: null, error: 'No images were selected' };
  }

  return { localPaths, remoteGroups: Array.from(remoteGroups.values()), error: null };
}

export async function POST(request: Request) {
  try {
    const authError = requireAuth(request);
    if (authError) return authError;

    const body = await request.json();
    const { localPaths, remoteGroups, error } = splitRemoteAndLocalPaths(body?.imgPaths);
    if (error || !localPaths || !remoteGroups) {
      return NextResponse.json({ error: error || 'Invalid image paths' }, { status: 400 });
    }

    const results: ImageDeleteBulkResult[] = [];

    if (localPaths.length > 0) {
      const { datasetsRoot, trainingRoot } = await resolveDatasetScope(body?.project_id);
      results.push(await deletePlainImagePaths(localPaths, datasetsRoot, trainingRoot));
    }

    for (const group of remoteGroups) {
      const worker = await getRemoteWorker(group.workerID);
      const remoteResult = await remoteJson<ImageDeleteBulkResult>(worker, '/api/img/delete-bulk', {
        method: 'POST',
        body: JSON.stringify({ imgPaths: group.paths }),
      });
      results.push(mapRemoteResult(remoteResult, group.refByRemotePath));
    }

    return NextResponse.json(combineBulkResults(results));
  } catch (error) {
    if (error instanceof DatasetScopeError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    if (isImageDeleteError(error)) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    return NextResponse.json({ error: 'Failed to delete images' }, { status: 500 });
  }
}
