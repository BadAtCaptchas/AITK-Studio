import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { rejectRemoteProjectScope, resolveDatasetScope } from '@/server/datasetScope';

function resolveWithinRoot(root: string, target: unknown) {
  if (typeof target !== 'string' || target.trim().length === 0) {
    return null;
  }

  const resolvedRoot = path.resolve(root);
  const resolvedPath = path.resolve(resolvedRoot, target);
  const relativePath = path.relative(resolvedRoot, resolvedPath);

  if (relativePath === '' || relativePath.startsWith('..') || path.isAbsolute(relativePath)) {
    return null;
  }

  return resolvedPath;
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { name } = body;
    const workerID = typeof body?.worker_id === 'string' ? body.worker_id : 'local';
    const projectID = body?.project_id;
    if (!isLocalWorker(workerID)) {
      rejectRemoteProjectScope(workerID, projectID);
      const worker = await getRemoteWorker(workerID);
      return NextResponse.json(
        await remoteJson(worker, '/api/datasets/delete', {
          method: 'POST',
          body: JSON.stringify({ name }),
        }),
      );
    }

    const { datasetsRoot: datasetsPath } = await resolveDatasetScope(projectID);
    const datasetPath = resolveWithinRoot(datasetsPath, name);

    if (!datasetPath) {
      return NextResponse.json({ error: 'Invalid dataset path' }, { status: 400 });
    }

    // if folder doesnt exist, ignore
    if (!fs.existsSync(datasetPath)) {
      return NextResponse.json({ success: true });
    }

    // delete it and return success
    fs.rmSync(datasetPath, { recursive: true, force: true });
    return NextResponse.json({ success: true });
  } catch (error: any) {
    return NextResponse.json({ error: error?.message || 'Failed to delete dataset' }, { status: error?.status || 500 });
  }
}
