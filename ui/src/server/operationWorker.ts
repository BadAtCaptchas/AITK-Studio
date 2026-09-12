import fs from 'fs/promises';
import path from 'path';
import { executeArchiveImport, importRoot } from './archiveImports';
import { runPendingOperations, type Operation } from './operations';
import { setUploadState, ensureUploadDirectory, cleanupExpiredUploads } from './archiveUploadManifest';
import { executeRemoteStart } from './remoteStartOperation';
import { db } from './db';
import { getDatasetsRoot, getTrainingFolder } from './settings';
import { isPathWithinRoot } from './pathContainment';

async function execute(operation: Operation): Promise<unknown> {
  if (operation.input.kind === 'remote-start') return executeRemoteStart(operation);
  return executeArchiveImport(operation);
}
async function cleanup(operation: Operation): Promise<void> {
  const input = operation.input;
  if (input.kind === 'remote-start') {
    if (operation.state !== 'completed')
      await db.jobs.updateIf(
        input.jobID,
        { attempt_id: operation.id, status: 'remote-starting' },
        {
          status: 'error',
          info: operation.error || `Remote start ${operation.state}`,
          remote_error: operation.error,
        },
      );
    const roots = [
      path.join(await getTrainingFolder(), '.aitk-remote-bundles'),
      path.join(await getDatasetsRoot(), '.aitk-remote-dataset-sync'),
      path.join(await getDatasetsRoot(), '.aitk-remote-caption-bundles'),
    ].map(root => path.resolve(root));
    for (const [key, value] of Object.entries(operation.checkpoint)) {
      if (
        !(key === 'bundleZipPath' || key.startsWith('datasetZip:') || key === 'captionZip') ||
        typeof value !== 'string' ||
        path.extname(value) !== '.zip'
      )
        continue;
      const canonical = await fs.realpath(value).catch(() => null);
      if (canonical && roots.some(root => isPathWithinRoot(root, canonical))) await fs.rm(canonical, { force: true });
    }
    return;
  }
  const root = await importRoot(input.kind);
  if (path.resolve(input.root) !== root || !/^[a-zA-Z0-9_-]{8,120}$/.test(input.uploadID)) return;
  if (input.chunked)
    await setUploadState(root, input.uploadID, operation.state === 'completed' ? 'completed' : 'failed');
  const directory = await ensureUploadDirectory(root, input.uploadID);
  await fs.rm(directory, { recursive: true, force: true });
}
export async function processOperations(): Promise<void> {
  await runPendingOperations(execute, cleanup);
  for (const kind of ['dataset-import', 'job-import'] as const) await cleanupExpiredUploads(await importRoot(kind));
}
