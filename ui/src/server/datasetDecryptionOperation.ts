import fs from 'fs/promises';
import path from 'path';
import { decryptDataset } from './datasetDecryption';
import { fetchOperationKeys } from './operationKeys';
import { beginOperationCommit, checkpointOperation, updateOperation, type Operation } from './operations';
import { getDatasetsRoot } from './settings';
import { isPathWithinRoot } from './pathContainment';
import { withProcessLease } from './processLease';

export async function executeDatasetDecryption(operation: Operation): Promise<unknown> {
  const input = operation.input;
  if (input.kind !== 'dataset-decrypt') throw new Error('Not a decryption operation');
  if (path.resolve(input.root) !== path.resolve(await getDatasetsRoot()))
    throw new Error('Dataset root changed. Start migration again.');
  await cleanupDatasetDecryption(operation);
  const keys = await fetchOperationKeys(operation.id);
  const key = keys.find(value => value.datasetPath === input.sourceName);
  if (!key) throw Object.assign(new Error('Unlock the dataset again'), { code: 'OPERATION_KEYS_REQUIRED' });
  return withProcessLease(`dataset-publish:${input.root}:${input.outputName.toLowerCase()}`, async lease =>
    decryptDataset({
      ...input,
      keyB64: key.keyB64,
      operationID: operation.id,
      progress: async (completed, total) => {
        await lease.assertOwned();
        await checkpointOperation(operation.id, 'decrypting');
        await updateOperation(operation.id, { progress: { completed, total } });
      },
      beforeCommit: async () => {
        await lease.assertOwned();
        await beginOperationCommit(operation.id, 'publishing');
      },
    }),
  );
}

export async function cleanupDatasetDecryption(operation: Operation): Promise<void> {
  if (operation.input.kind !== 'dataset-decrypt') return;
  const root = await fs.realpath(await getDatasetsRoot());
  if (root !== (await fs.realpath(operation.input.root).catch(() => null))) return;
  for (const name of await fs.readdir(root)) {
    if (!name.startsWith(`.aitk-decrypt-${operation.id}-`)) continue;
    const stage = path.join(root, name);
    const entry = await fs.lstat(stage).catch(() => null);
    if (!entry?.isDirectory() || entry.isSymbolicLink() || !isPathWithinRoot(root, await fs.realpath(stage))) continue;
    if ((await fs.readFile(path.join(stage, '.aitk-migration-owner'), 'utf8').catch(() => '')) !== operation.id)
      continue;
    await fs.rm(stage, { recursive: true, force: true });
  }
}
