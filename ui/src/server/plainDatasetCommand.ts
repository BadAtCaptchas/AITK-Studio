import fs from 'fs/promises';
import path from 'path';
import { CommandInputError, isRecord, readJsonCommand, withCommandBoundary } from './commandInput';
import { findEncryptedDatasetRoot } from './encryptedDatasets';
import { resolveDatasetScope } from './datasetScope';
import { DATASET_MIGRATION_MESSAGE, DATASET_MIGRATION_REQUIRED } from './datasetDecryption';

export function migrationRequired(): never {
  throw new CommandInputError(DATASET_MIGRATION_MESSAGE, 410, DATASET_MIGRATION_REQUIRED);
}
export async function assertPlainDatasetPath(target: string, root: string): Promise<void> {
  const resolved = path.resolve(root, target);
  if (findEncryptedDatasetRoot(resolved, root)) migrationRequired();
  const [canonical, canonicalRoot] = await Promise.all([
    fs.realpath(resolved).catch(() => null),
    fs.realpath(root).catch(() => null),
  ]);
  if (canonical && canonicalRoot && findEncryptedDatasetRoot(canonical, canonicalRoot)) migrationRequired();
}
const pathFields = new Set([
  'name',
  'datasetName',
  'datasetPath',
  'folderPath',
  'folder_path',
  'dataset_path',
  'path',
  'imgPath',
  'imgPaths',
  'imagePath',
  'sourceDataset',
  'sourceDatasets',
]);
const retiredFields = new Set([
  'encrypted',
  'encryptedManifest',
  'outputEncrypted',
  'outputEncryptedManifest',
  'encryptedDatasetKeyB64',
  'encryptedDatasetKeys',
  'outputKeyB64',
]);
export function withPlainDatasetCommand<R extends Request, A extends unknown[]>(
  handler: (request: R, ...args: A) => Promise<Response>,
) {
  return withCommandBoundary(async (request: R, ...args: A) => {
    const body = ['GET', 'HEAD'].includes(request.method)
      ? Object.fromEntries(new URL(request.url).searchParams)
      : await readJsonCommand(request);
    const { datasetsRoot } = await resolveDatasetScope();
    async function inspect(value: unknown, key = ''): Promise<void> {
      if (retiredFields.has(key) && value && (!Array.isArray(value) || value.length)) migrationRequired();
      if (typeof value === 'string' && pathFields.has(key)) await assertPlainDatasetPath(value, datasetsRoot);
      else if (Array.isArray(value)) for (const entry of value) await inspect(entry, key);
      else if (isRecord(value)) for (const [name, entry] of Object.entries(value)) await inspect(entry, name);
    }
    await inspect(body);
    return handler(request, ...args);
  });
}
