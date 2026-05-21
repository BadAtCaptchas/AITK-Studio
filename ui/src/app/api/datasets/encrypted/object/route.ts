import fs from 'fs';
import { NextRequest, NextResponse } from 'next/server';
import { getDatasetsRoot } from '@/server/settings';
import {
  isEncryptedDatasetFolder,
  resolveDatasetFolder,
  resolveEncryptedObjectPath,
} from '@/server/encryptedDatasets';

export async function POST(request: NextRequest) {
  try {
    const { datasetName, objectPath } = await request.json();
    if (typeof datasetName !== 'string' || typeof objectPath !== 'string') {
      return NextResponse.json({ error: 'Invalid encrypted object request' }, { status: 400 });
    }

    const datasetsRoot = await getDatasetsRoot();
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
