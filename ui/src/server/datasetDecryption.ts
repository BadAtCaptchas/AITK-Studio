import crypto from 'crypto';
import { publishDatasetDirectory } from './publishDatasetDirectory';
import fs from 'fs/promises';
import path from 'path';
import { isRecord } from './commandInput';
import {
  cleanDatasetName,
  readEncryptedManifest,
  resolveEncryptedObjectPath,
  validateEncryptedCatalogKey,
} from './encryptedDatasets';
import { resolveDatasetMutationPath } from './datasetMutationPaths';
import { isPathWithinRoot } from './pathContainment';
import { ROOT_CAPTION_FILE_NAME } from './datasetRootCaption';

export const DATASET_MIGRATION_REQUIRED = 'ENCRYPTED_DATASET_MIGRATION_REQUIRED';
export const DATASET_MIGRATION_MESSAGE =
  'Encrypted datasets have been retired. Decrypt this dataset to a plain copy before using it.';

export type DecryptionResult = {
  sourceName: string;
  datasetName: string;
  itemCount: number;
  captionCount: number;
  renamed: Array<{ original: string; name: string }>;
};
type MigrationItem = { name: string; objectPath: string; captionObjectPath?: string };

function migrationItems(value: unknown): { items: MigrationItem[]; rootCaption?: string } {
  if (!isRecord(value) || value.version !== 1 || !Array.isArray(value.items))
    throw new Error('Invalid encrypted catalog');
  const items = value.items.map((item: unknown) => {
    if (
      !isRecord(item) ||
      typeof item.name !== 'string' ||
      !item.name ||
      typeof item.objectPath !== 'string' ||
      (item.captionObjectPath !== undefined && typeof item.captionObjectPath !== 'string')
    )
      throw new Error('Invalid encrypted catalog item');
    if (
      item.name !== path.basename(item.name) ||
      /[\\/\x00-\x1f<>:"|?*]/.test(item.name) ||
      item.name.startsWith('.') ||
      /[. ]$/.test(item.name) ||
      /^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)/i.test(item.name)
    )
      throw new Error('Unsafe encrypted item name');
    return {
      name: item.name,
      objectPath: item.objectPath,
      captionObjectPath: item.captionObjectPath as string | undefined,
    };
  });
  if (value.rootCaption !== undefined && typeof value.rootCaption !== 'string') throw new Error('Invalid root caption');
  return { items, rootCaption: value.rootCaption as string | undefined };
}

async function decryptObject(folder: string, objectPath: string, key: Buffer): Promise<Buffer> {
  const candidate = resolveEncryptedObjectPath(folder, objectPath);
  const canonical = await fs.realpath(candidate);
  if (!isPathWithinRoot(folder, canonical)) throw new Error('Encrypted object escapes its dataset');
  const payload: unknown = JSON.parse(await fs.readFile(canonical, 'utf8'));
  if (!isRecord(payload) || typeof payload.nonce !== 'string' || typeof payload.data !== 'string')
    throw new Error('Invalid encrypted object');
  const nonce = Buffer.from(payload.nonce, 'base64'),
    data = Buffer.from(payload.data, 'base64');
  if (nonce.length !== 12 || data.length < 16) throw new Error('Invalid encrypted object');
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, nonce);
  decipher.setAAD(Buffer.from(`aitk-encrypted-object:${objectPath.replace(/\\/g, '/')}`));
  decipher.setAuthTag(data.subarray(-16));
  try {
    return Buffer.concat([decipher.update(data.subarray(0, -16)), decipher.final()]);
  } catch {
    throw new Error('Unable to authenticate encrypted object. Check the credential and dataset integrity.');
  }
}

export async function decryptDataset(options: {
  root: string;
  sourceName: string;
  outputName: string;
  keyB64: string;
  operationID: string;
  progress: (completed: number, total: number) => Promise<void>;
  beforeCommit: () => Promise<void>;
}): Promise<DecryptionResult> {
  const { root, sourceName, outputName, operationID } = options;
  if (!/^[a-f0-9]{64}$/.test(operationID)) throw new Error('Invalid migration operation');
  if (
    !sourceName ||
    cleanDatasetName(sourceName) !== sourceName ||
    !outputName ||
    cleanDatasetName(outputName) !== outputName ||
    sourceName.toLowerCase() === outputName.toLowerCase()
  )
    throw new Error('Choose a different valid output dataset name');
  const source = await resolveDatasetMutationPath(root, sourceName);
  const destination = await resolveDatasetMutationPath(root, outputName, { allowMissingLeaf: true });
  const metadataName = '.aitk_dataset_metadata.json';
  const previous: unknown = await fs
    .readFile(path.join(destination.canonicalPath, metadataName), 'utf8')
    .then(JSON.parse)
    .catch(() => null);
  if (
    isRecord(previous) &&
    isRecord(previous.decryption) &&
    previous.decryption.operationID === operationID &&
    isRecord(previous.decryption.result)
  ) {
    const result = previous.decryption.result;
    if (
      result.sourceName === sourceName &&
      result.datasetName === outputName &&
      typeof result.itemCount === 'number' &&
      typeof result.captionCount === 'number' &&
      Array.isArray(result.renamed)
    )
      return result as DecryptionResult;
  }
  if (
    await fs.lstat(destination.entryPath).then(
      () => true,
      () => false,
    )
  )
    throw new Error('Output dataset already exists');
  const manifest = await readEncryptedManifest(source.canonicalPath);
  const fingerprint = JSON.stringify(manifest);
  const key = Buffer.from(options.keyB64, 'base64');
  let staging: string | undefined;
  try {
    if (key.length !== 32) throw new Error('Invalid dataset key');
    validateEncryptedCatalogKey(manifest, options.keyB64);
    const catalogData = Buffer.from(manifest.catalog.data, 'base64');
    const catalogDecipher = crypto.createDecipheriv('aes-256-gcm', key, Buffer.from(manifest.catalog.nonce, 'base64'));
    catalogDecipher.setAAD(Buffer.from('aitk-encrypted-catalog:v1'));
    catalogDecipher.setAuthTag(catalogData.subarray(-16));
    const catalog = migrationItems(
      JSON.parse(
        Buffer.concat([catalogDecipher.update(catalogData.subarray(0, -16)), catalogDecipher.final()]).toString('utf8'),
      ),
    );
    staging = await fs.mkdtemp(path.join(destination.canonicalRoot, `.aitk-decrypt-${operationID}-`));
    const result: DecryptionResult = {
      sourceName,
      datasetName: outputName,
      itemCount: 0,
      captionCount: 0,
      renamed: [],
    };
    const used = new Set([ROOT_CAPTION_FILE_NAME.toLowerCase(), metadataName]);
    await fs.writeFile(path.join(staging, '.aitk-migration-owner'), operationID, { flag: 'wx' });
    for (const item of catalog.items) {
      await options.progress(result.itemCount, catalog.items.length);
      const parsed = path.parse(item.name);
      let name = item.name,
        suffix = 2;
      while (used.has(name.toLowerCase()) || used.has(`${path.parse(name).name}.txt`.toLowerCase()))
        name = `${parsed.name}_${suffix++}${parsed.ext}`;
      const captionName = `${path.parse(name).name}.txt`;
      used.add(name.toLowerCase());
      used.add(captionName.toLowerCase());
      if (name !== item.name) result.renamed.push({ original: item.name, name });
      await fs.writeFile(path.join(staging, name), await decryptObject(source.canonicalPath, item.objectPath, key), {
        flag: 'wx',
      });
      if (item.captionObjectPath) {
        await fs.writeFile(
          path.join(staging, captionName),
          await decryptObject(source.canonicalPath, item.captionObjectPath, key),
          { flag: 'wx' },
        );
        result.captionCount++;
      }
      result.itemCount++;
    }
    if (catalog.rootCaption !== undefined)
      await fs.writeFile(path.join(staging, ROOT_CAPTION_FILE_NAME), catalog.rootCaption, { flag: 'wx' });
    if (JSON.stringify(await readEncryptedManifest(source.canonicalPath)) !== fingerprint)
      throw new Error('Source dataset changed during migration');
    await fs.writeFile(
      path.join(staging, metadataName),
      JSON.stringify({ version: 1, decryption: { operationID, result } }),
      { flag: 'wx' },
    );
    await options.progress(result.itemCount, catalog.items.length);
    await options.beforeCommit();
    // Revalidate the parent and destination immediately before publication.
    const final = await resolveDatasetMutationPath(root, outputName, { allowMissingLeaf: true });
    if (
      final.entryPath !== destination.entryPath ||
      (await fs.lstat(final.entryPath).then(
        () => true,
        () => false,
      ))
    )
      throw new Error('Output dataset already exists or changed');
    await publishDatasetDirectory(staging, final.entryPath);
    return result;
  } finally {
    key.fill(0);
    if (staging) {
      const entry = await fs.lstat(staging).catch(() => null);
      if (entry && !entry.isSymbolicLink() && isPathWithinRoot(destination.canonicalRoot, await fs.realpath(staging)))
        await fs.rm(staging, { recursive: true, force: true });
    }
  }
}
