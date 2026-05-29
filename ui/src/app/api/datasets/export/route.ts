import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { Readable } from 'stream';
import { NextRequest, NextResponse } from 'next/server';
import { getDatasetsRoot } from '@/server/settings';
import { resolveDatasetFolder } from '@/server/encryptedDatasets';
import { createDatasetExportArchive, datasetExportFileName } from '@/server/datasetTransfer';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  try {
    const { datasetName } = await request.json();
    if (typeof datasetName !== 'string') {
      return NextResponse.json({ error: 'Dataset name is required' }, { status: 400 });
    }

    const datasetsRoot = await getDatasetsRoot();
    const datasetFolder = resolveDatasetFolder(datasetsRoot, datasetName);
    if (!fs.existsSync(datasetFolder) || !fs.statSync(datasetFolder).isDirectory()) {
      return NextResponse.json({ error: 'Dataset not found' }, { status: 404 });
    }

    const exportRoot = path.join(datasetsRoot, '.aitk-dataset-exports');
    const zipPath = path.join(exportRoot, datasetExportFileName(datasetName));
    await createDatasetExportArchive(datasetName, datasetFolder, zipPath);

    const stat = await fsp.stat(zipPath);
    const nodeStream = fs.createReadStream(zipPath);
    nodeStream.on('close', () => {
      void fsp.rm(zipPath, { force: true }).catch(() => undefined);
    });

    return new NextResponse(Readable.toWeb(nodeStream) as unknown as ReadableStream, {
      headers: {
        'Content-Type': 'application/zip',
        'Content-Length': String(stat.size),
        'Content-Disposition': `attachment; filename="${path.basename(zipPath).replace(/"/g, '_')}"`,
        'Cache-Control': 'no-store',
      },
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to export dataset' },
      { status: 500 },
    );
  }
}
