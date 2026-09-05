import type { AxiosProgressEvent } from 'axios';
import { apiClient } from './api';
function encodedHeaderValue(value: string) {
  return encodeURIComponent(value);
}
export async function uploadLoraFile(
  file: File,
  options: {
    triggerWords?: string;
    onUploadProgress?: (event: AxiosProgressEvent) => void;
  } = {},
) {
  return apiClient.post('/api/generate/loras/upload', file, {
    headers: {
      'Content-Type': 'application/octet-stream',
      'X-AITK-File-Name': encodedHeaderValue(file.name),
      ...(options.triggerWords?.trim()
        ? { 'X-AITK-Trigger-Words': encodedHeaderValue(options.triggerWords.trim()) }
        : {}),
    },
    onUploadProgress: options.onUploadProgress,
    timeout: 0,
  });
}
export async function uploadTemporaryMediaFile(
  file: File,
  options:
    | {
        onUploadProgress?: (event: AxiosProgressEvent) => void;
      }
    | ((event: AxiosProgressEvent) => void) = {},
) {
  const normalizedOptions = typeof options === 'function' ? { onUploadProgress: options } : options;
  return apiClient.post('/api/img/upload', file, {
    headers: {
      'Content-Type': 'application/octet-stream',
      'X-AITK-File-Name': encodedHeaderValue(file.name),
    },
    onUploadProgress: normalizedOptions.onUploadProgress,
    timeout: 0,
  });
}
export async function uploadDatasetFile(
  file: File | Blob,
  options: {
    datasetName: string;
    filename?: string;
    workerID?: string;
    relativePath?: string;
    sourceFolderPath?: string;
    failIfDatasetExists?: boolean;
    encryptedObjectPath?: string;
    onUploadProgress?: (event: AxiosProgressEvent) => void;
  },
) {
  const filename = options.filename || (file instanceof File ? file.name : 'upload.bin');
  return apiClient.post('/api/datasets/upload', file, {
    headers: {
      'Content-Type': 'application/octet-stream',
      'X-AITK-Dataset-Name': encodedHeaderValue(options.datasetName),
      'X-AITK-File-Name': encodedHeaderValue(filename),
      ...(options.workerID && options.workerID !== 'local'
        ? { 'X-AITK-Worker-ID': encodedHeaderValue(options.workerID) }
        : {}),

      ...(options.relativePath ? { 'X-AITK-Relative-Path': encodedHeaderValue(options.relativePath) } : {}),
      ...(options.sourceFolderPath ? { 'X-AITK-Source-Folder': encodedHeaderValue(options.sourceFolderPath) } : {}),
      ...(options.failIfDatasetExists ? { 'X-AITK-Fail-If-Dataset-Exists': '1' } : {}),
      ...(options.encryptedObjectPath
        ? { 'X-AITK-Encrypted-Object-Path': encodedHeaderValue(options.encryptedObjectPath) }
        : {}),
    },
    onUploadProgress: options.onUploadProgress,
    timeout: 0,
  });
}
