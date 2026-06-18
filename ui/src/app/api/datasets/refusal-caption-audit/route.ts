import { NextResponse } from 'next/server';
import fs from 'fs';
import { auditDatasetRefusalCaptions } from '@/server/datasetRefusalCaptionAudit';
import {
  isEncryptedDatasetFolder,
  resolveDatasetFolder,
} from '@/server/encryptedDatasets';
import { getRemoteWorker, isLocalWorker, remoteJson } from '@/server/remoteClient';
import { makeSignedRemoteDatasetAssetRef } from '@/server/remoteDatasetAssetAccess';
import {
  assertProjectScopeEnabled,
  DatasetScopeError,
  rejectRemoteProjectScope,
  resolveDatasetScope,
} from '@/server/datasetScope';
import { parseRemoteDatasetAssetRef } from '@/utils/remoteDatasetRefs';

type AuditResponse = Awaited<ReturnType<typeof auditDatasetRefusalCaptions>>;

function remoteRefMap(itemPaths: unknown, workerID: string) {
  const refByRemotePath: Record<string, string> = {};
  if (!Array.isArray(itemPaths)) return refByRemotePath;
  for (const value of itemPaths) {
    if (typeof value !== 'string') continue;
    const parsed = parseRemoteDatasetAssetRef(value);
    if (parsed?.workerID === workerID) {
      refByRemotePath[parsed.path] = value;
    }
  }
  return refByRemotePath;
}

function mapRemoteAuditResponse(data: AuditResponse, workerID: string, refByRemotePath: Record<string, string>) {
  const refusals: Record<string, string> = {};
  Object.entries(data.refusals || {}).forEach(([remotePath, caption]) => {
    refusals[refByRemotePath[remotePath] || makeSignedRemoteDatasetAssetRef(workerID, 'img', remotePath)] = caption;
  });
  return { ...data, refusals };
}

export async function POST(request: Request) {
  const body = await request.json();
  const datasetName = body?.datasetName;
  const workerID = typeof body?.worker_id === 'string' ? body.worker_id : 'local';
  const projectID = body?.project_id;

  try {
    await assertProjectScopeEnabled(projectID);
  } catch (error: any) {
    return NextResponse.json({ error: error.message }, { status: error.status || 400 });
  }

  if (!isLocalWorker(workerID)) {
    try {
      rejectRemoteProjectScope(workerID, projectID);
      const worker = await getRemoteWorker(workerID);
      const refByRemotePath = remoteRefMap(body?.itemPaths, workerID);
      const data = await remoteJson<AuditResponse>(worker, '/api/datasets/refusal-caption-audit', {
        method: 'POST',
        body: JSON.stringify({ datasetName }),
      });
      return NextResponse.json(mapRemoteAuditResponse(data, workerID, refByRemotePath));
    } catch (error: any) {
      return NextResponse.json(
        { error: error?.message || 'Failed to audit remote captions.' },
        { status: error?.status || 500 },
      );
    }
  }

  let datasetFolder: string;
  try {
    const scope = await resolveDatasetScope(projectID);
    datasetFolder = resolveDatasetFolder(scope.datasetsRoot, datasetName);
  } catch (error: any) {
    if (error instanceof DatasetScopeError) {
      return NextResponse.json({ error: error.message }, { status: error.status });
    }
    return NextResponse.json({ error: error?.message || 'Invalid dataset name' }, { status: 400 });
  }

  try {
    if (!fs.existsSync(datasetFolder)) {
      return NextResponse.json({ error: `Folder '${datasetName}' not found` }, { status: 404 });
    }
    if (isEncryptedDatasetFolder(datasetFolder)) {
      return NextResponse.json({
        datasetFingerprint: '',
        scanned: 0,
        refusalCount: 0,
        refusals: {},
        cached: false,
        encrypted: true,
      });
    }

    return NextResponse.json(await auditDatasetRefusalCaptions(datasetFolder));
  } catch (error) {
    console.error('Error auditing dataset refusal captions:', error);
    return NextResponse.json({ error: 'Failed to audit captions' }, { status: 500 });
  }
}

