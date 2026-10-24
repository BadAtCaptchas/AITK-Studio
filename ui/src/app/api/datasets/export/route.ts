import { readJsonCommand, withCommandBoundary } from '@/server/commandInput';
import { assertGlobalPayload } from '@/utils/obsoleteWorkspaceGuard';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { Readable } from 'stream';
import { resolveDatasetFolder } from '@/server/encryptedDatasets';
import { createDatasetExportArchive, datasetExportFileName } from '@/server/datasetTransfer';
import { DatasetScopeError, resolveDatasetScope } from '@/server/datasetScope';


async function postCommand(request: Request) {
  try {
    const { datasetName } = assertGlobalPayload(await readJsonCommand(request));
    if (typeof datasetName !== 'string') {
      return Response.json({ error: 'Dataset name is required' }, { status: 400 });
    }

    const { datasetsRoot } = await resolveDatasetScope();
    const datasetFolder = resolveDatasetFolder(datasetsRoot, datasetName);
    if (!fs.existsSync(datasetFolder) || !fs.statSync(datasetFolder).isDirectory()) {
      return Response.json({ error: 'Dataset not found' }, { status: 404 });
    }

    const exportRoot = path.join(datasetsRoot, '.aitk-dataset-exports');
    const zipPath = path.join(exportRoot, datasetExportFileName(datasetName));
    await createDatasetExportArchive(datasetName, datasetFolder, zipPath);

    const stat = await fsp.stat(zipPath);
    const nodeStream = fs.createReadStream(zipPath);
    nodeStream.on('close', () => {
      void fsp.rm(zipPath, { force: true }).catch(() => undefined);
    });

    return new Response(Readable.toWeb(nodeStream) as unknown as ReadableStream, {
      headers: {
        'Content-Type': 'application/zip',
        'Content-Length': String(stat.size),
        'Content-Disposition': `attachment; filename="${path.basename(zipPath).replace(/"/g, '_')}"`,
        'Cache-Control': 'no-store',
      },
    });
  } catch (error) {
    const status = error instanceof DatasetScopeError ? error.status : 500;
    return Response.json(
      { error: error instanceof Error ? error.message : 'Failed to export dataset' },
      { status },
    );
  }
}

export const POST = withCommandBoundary(postCommand);
