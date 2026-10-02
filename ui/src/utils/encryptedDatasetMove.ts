import type { EncryptedDatasetCatalog, EncryptedDatasetItem, EncryptedDatasetManifest } from '../types';
import { apiClient } from './api';
import { encryptCatalog } from './encryptedDatasets';
import { buildEncryptedObjectRequestBody } from './encryptedObjectMediaCache';
import { uploadDatasetFile } from './streamedUploads';

export async function moveEncryptedDatasetItems(options: {
  datasetName: string;
  destinationName: string;
  workerID: string;
  key: CryptoKey;
  manifest: EncryptedDatasetManifest;
  catalog: EncryptedDatasetCatalog;
  items: EncryptedDatasetItem[];
}): Promise<{ createdName: string; nextManifest: EncryptedDatasetManifest; nextCatalog: EncryptedDatasetCatalog }> {
  const { datasetName, destinationName, workerID, key, manifest, catalog, items } = options;
  const { manifest: emptyManifest } = await encryptCatalog({ version: 1, items: [] }, key, manifest);
  const response = await apiClient.post('/api/datasets/create', {
    name: destinationName,
    worker_id: workerID,
    encrypted: true,
    encryptedManifest: emptyManifest,
  });
  const created: unknown = response.data;
  if (!created || typeof created !== 'object' || !('name' in created) || typeof created.name !== 'string' || !created.name) {
    throw new Error('The destination dataset could not be created.');
  }
  const createdName = created.name;
  for (const item of items) {
    for (const objectPath of [item.objectPath, item.captionObjectPath]) {
      if (!objectPath) continue;
      const response = await apiClient.post(
        '/api/datasets/encrypted/object',
        buildEncryptedObjectRequestBody({ datasetName, workerID, objectPath }),
        { responseType: 'blob' },
      );
      const blob: unknown = response.data;
      if (!(blob instanceof Blob)) throw new Error('The encrypted dataset object could not be read.');
      // Transfer each ciphertext as a binary body, without base64 JSON or an
      // in-memory batch that grows with the number of selected images.
      await uploadDatasetFile(blob, {
        datasetName: createdName,
        filename: objectPath.replace(/^objects\//, ''),
        workerID,
        encryptedObjectPath: objectPath,
      });
    }
  }
  const now = new Date().toISOString();
  const { manifest: targetManifest } = await encryptCatalog(
    { version: 1, items: items.map(item => ({ ...item, updatedAt: now })) },
    key,
    emptyManifest,
  );
  await apiClient.post('/api/datasets/encrypted/update', {
    datasetName: createdName,
    worker_id: workerID,
    manifest: targetManifest,
  });

  // The destination must have all objects and a committed catalog before the
  // source loses either its catalog entries or its ciphertext objects.
  const movedIDs = new Set(items.map(item => item.id));
  const nextCatalog: EncryptedDatasetCatalog = {
    ...catalog,
    items: catalog.items.filter(item => !movedIDs.has(item.id)),
  };
  const { manifest: nextManifest } = await encryptCatalog(nextCatalog, key, manifest);
  await apiClient.post('/api/datasets/encrypted/update', {
    datasetName,
    worker_id: workerID,
    manifest: nextManifest,
    deleteObjects: items.flatMap(item =>
      [item.objectPath, item.captionObjectPath].filter((value): value is string => Boolean(value)),
    ),
  });
  return { createdName, nextManifest, nextCatalog };
}
