import type { EncryptedDatasetItem } from '@/types';
import { apiClient } from '@/utils/api';
import { decryptEncryptedObjectBlob } from '@/utils/encryptedDatasets';

export async function createEncryptedImageFormData({
  datasetName,
  workerID,
  encryptedKey,
  item,
}: {
  datasetName: string;
  workerID: string;
  encryptedKey: CryptoKey;
  item: EncryptedDatasetItem;
}) {
  const encryptedResponse = await apiClient.post(
    '/api/datasets/encrypted/object',
    { datasetName, worker_id: workerID, objectPath: item.objectPath },
    { responseType: 'blob' },
  );
  const decrypted = await decryptEncryptedObjectBlob(encryptedKey, item.objectPath, encryptedResponse.data as Blob);
  const imageBlob = new Blob([decrypted], { type: item.mimeType || 'image/jpeg' });
  const formData = new FormData();
  formData.append('image', imageBlob, item.name || 'encrypted-image');
  formData.append('encryptedConfirmed', 'true');
  return formData;
}

export function appendImageSizeFields(formData: FormData, imageWidth: number | null, imageHeight: number | null) {
  if (imageWidth) formData.append('imageWidth', String(imageWidth));
  if (imageHeight) formData.append('imageHeight', String(imageHeight));
}
