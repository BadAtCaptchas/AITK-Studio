import fs from 'fs';
import { NextRequest, NextResponse } from 'next/server';
import {
  isEncryptedDatasetFolder,
  resolveDatasetFolder,
  resolveEncryptedObjectPath,
} from '@/server/encryptedDatasets';
import { getRemoteWorker, isLocalWorker, remoteFetch } from '@/server/remoteClient';
import { rejectRemoteProjectScope, resolveDatasetScope } from '@/server/datasetScope';

function copyResponseHeaders(source: Response) {
  const headers = new Headers();
  for (const name of ['content-type', 'content-length', 'cache-control']) {
    const value = source.headers.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}

export async function POST(request: NextRequest) {
  try {
    const { datasetName, objectPath, worker_id, project_id } = await request.json();
    if (typeof datasetName !== 'string' || typeof objectPath !== 'string') {
      return NextResponse.json({ error: 'Invalid encrypted object request' }, { status: 400 });
    }

    if (!isLocalWorker(worker_id)) {
      rejectRemoteProjectScope(worker_id, project_id);
      const worker = await getRemoteWorker(worker_id);
      const remoteResponse = await remoteFetch(worker, '/api/datasets/encrypted/object', {
        method: 'POST',
        body: JSON.stringify({ datasetName, objectPath }),
        headers: { 'Content-Type': 'application/json' },
      });
      return new NextResponse(remoteResponse.body, {
        status: remoteResponse.status,
        headers: copyResponseHeaders(remoteResponse),
      });
    }

    const { datasetsRoot } = await resolveDatasetScope(project_id);
    const datasetFolder = resolveDatasetFolder(datasetsRoot, datasetName);
    if (!isEncryptedDatasetFolder(datasetFolder)) {
      return NextResponse.json({ error: 'Encrypted dataset not found' }, { status: 404 });
    }

    const resolvedObjectPath = resolveEncryptedObjectPath(datasetFolder, objectPath);
    if (!fs.existsSync(resolvedObjectPath)) {
      return NextResponse.json({ error: 'Encrypted object not found' }, { status: 404 });
    }

    const stat = fs.statSync(resolvedObjectPath);
    const stream = fs.createReadStream(resolvedObjectPath);
    const readable = new ReadableStream({
      start(controller) {
        stream.on('data', chunk => controller.enqueue(chunk));
        stream.on('end', () => controller.close());
        stream.on('error', err => controller.error(err));
      },
      cancel() {
        stream.destroy();
      },
    });

    return new NextResponse(readable as any, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Length': String(stat.size),
        'Cache-Control': 'no-store',
      },
    });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Failed to read encrypted object' }, { status: 400 });
  }
}
