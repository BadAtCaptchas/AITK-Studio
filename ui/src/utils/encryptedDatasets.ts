import type {
  EncryptedDatasetCatalog,
  EncryptedDatasetItem,
  EncryptedDatasetManifest,
  EncryptedDatasetMediaKind,
} from '@/types';

export const ENCRYPTED_DATASET_FORMAT = 'aitk-encrypted-dataset';
export const ENCRYPTED_DATASET_VERSION = 1;
export const PBKDF2_ITERATIONS = 600_000;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();
const rememberedKeys = new Map<string, string>();

export function normalizeDatasetKey(datasetPathOrName: string) {
  return datasetPathOrName.replace(/[\\/]+$/, '').toLowerCase();
}

export function rememberEncryptedDatasetKey(datasetPathOrName: string, rawKeyB64: string) {
  rememberedKeys.set(normalizeDatasetKey(datasetPathOrName), rawKeyB64);
}

export function getRememberedEncryptedDatasetKey(datasetPathOrName: string) {
  return rememberedKeys.get(normalizeDatasetKey(datasetPathOrName)) || null;
}

export function forgetRememberedEncryptedDatasetKey(datasetPathOrName: string) {
  rememberedKeys.delete(normalizeDatasetKey(datasetPathOrName));
}

export function randomId(bytes = 16) {
  const buf = crypto.getRandomValues(new Uint8Array(bytes));
  return arrayBufferToBase64Url(buf);
}

export function arrayBufferToBase64(buffer: ArrayBuffer | Uint8Array) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  let binary = '';
  bytes.forEach(byte => {
    binary += String.fromCharCode(byte);
  });
  return btoa(binary);
}

export function base64ToArrayBuffer(value: string) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

function arrayBufferToBase64Url(buffer: ArrayBuffer | Uint8Array) {
  return arrayBufferToBase64(buffer).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

export async function importRawAesKey(rawKeyB64: string) {
  return crypto.subtle.importKey('raw', base64ToArrayBuffer(rawKeyB64), { name: 'AES-GCM', length: 256 }, true, [
    'encrypt',
    'decrypt',
  ]);
}

export async function exportRawAesKey(key: CryptoKey) {
  const raw = await crypto.subtle.exportKey('raw', key);
  return arrayBufferToBase64(raw);
}

export async function derivePasswordKey(password: string, manifest: EncryptedDatasetManifest) {
  const kdf = manifest.crypto.kdf;
  if (kdf.type !== 'PBKDF2-SHA256') {
    throw new Error('This dataset requires a key file.');
  }
  const passwordKey = await crypto.subtle.importKey('raw', textEncoder.encode(password), 'PBKDF2', false, [
    'deriveKey',
  ]);
  return crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      hash: 'SHA-256',
      salt: base64ToArrayBuffer(kdf.salt),
      iterations: kdf.iterations,
    },
    passwordKey,
    { name: 'AES-GCM', length: 256 },
    true,
    ['encrypt', 'decrypt'],
  );
}

export async function deriveKeyFileKey(file: File) {
  const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
  return crypto.subtle.importKey('raw', digest, { name: 'AES-GCM', length: 256 }, true, ['encrypt', 'decrypt']);
}

export function objectAad(objectPath: string) {
  return `aitk-encrypted-object:${objectPath}`;
}

export function catalogAad() {
  return 'aitk-encrypted-catalog:v1';
}

export async function encryptBytes(key: CryptoKey, bytes: ArrayBuffer | Uint8Array, aad: string) {
  const nonce = crypto.getRandomValues(new Uint8Array(12));
  const plaintext = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const encrypted = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv: nonce, additionalData: textEncoder.encode(aad), tagLength: 128 },
    key,
    plaintext,
  );
  return { nonce: arrayBufferToBase64(nonce), data: arrayBufferToBase64(encrypted) };
}

export async function decryptBytes(key: CryptoKey, nonceB64: string, dataB64: string | ArrayBuffer, aad: string) {
  const ciphertext = typeof dataB64 === 'string' ? base64ToArrayBuffer(dataB64) : dataB64;
  return crypto.subtle.decrypt(
    {
      name: 'AES-GCM',
      iv: base64ToArrayBuffer(nonceB64),
      additionalData: textEncoder.encode(aad),
      tagLength: 128,
    },
    key,
    ciphertext,
  );
}

export async function createEmptyEncryptedManifest(
  mode: 'password' | 'keyFile',
  secret: string | File,
) {
  let key: CryptoKey;
  const manifest: EncryptedDatasetManifest = {
    format: ENCRYPTED_DATASET_FORMAT,
    version: ENCRYPTED_DATASET_VERSION,
    crypto: {
      algorithm: 'AES-256-GCM',
      kdf:
        mode === 'password'
          ? {
              type: 'PBKDF2-SHA256',
              salt: arrayBufferToBase64(crypto.getRandomValues(new Uint8Array(16))),
              iterations: PBKDF2_ITERATIONS,
              keyLength: 32,
            }
          : {
              type: 'KEYFILE-SHA256',
              keyLength: 32,
            },
    },
    catalog: { nonce: '', data: '' },
  };

  if (mode === 'password') {
    key = await derivePasswordKey(secret as string, manifest);
  } else {
    key = await deriveKeyFileKey(secret as File);
  }

  const { manifest: encryptedManifest } = await encryptCatalog({ version: 1, items: [] }, key, manifest);
  return { manifest: encryptedManifest, key, rawKeyB64: await exportRawAesKey(key) };
}

export async function decryptCatalog(manifest: EncryptedDatasetManifest, key: CryptoKey) {
  if (manifest.format !== ENCRYPTED_DATASET_FORMAT || manifest.version !== ENCRYPTED_DATASET_VERSION) {
    throw new Error('Unsupported encrypted dataset format.');
  }
  const plaintext = await decryptBytes(key, manifest.catalog.nonce, manifest.catalog.data, catalogAad());
  return JSON.parse(textDecoder.decode(plaintext)) as EncryptedDatasetCatalog;
}

export async function encryptCatalog(
  catalog: EncryptedDatasetCatalog,
  key: CryptoKey,
  manifest: EncryptedDatasetManifest,
) {
  const encrypted = await encryptBytes(key, textEncoder.encode(JSON.stringify(catalog)), catalogAad());
  return {
    manifest: {
      ...manifest,
      catalog: encrypted,
    },
  };
}

export function getMediaKind(file: File): EncryptedDatasetMediaKind | null {
  if (file.type.startsWith('image/')) return 'image';
  if (file.type.startsWith('video/')) return 'video';
  if (file.type.startsWith('audio/')) return 'audio';
  const ext = getExtension(file.name).toLowerCase();
  if (['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'].includes(ext)) return 'image';
  if (['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv', '.webm'].includes(ext)) return 'video';
  if (['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'].includes(ext)) return 'audio';
  return null;
}

export function getExtension(fileName: string) {
  const match = fileName.match(/\.[^.]+$/);
  return match ? match[0] : '';
}

export function getBaseName(fileName: string) {
  return fileName.replace(/\.[^.]+$/, '');
}

export function captionObjectPath(itemId = randomId()) {
  return `objects/${itemId}.caption.bin`;
}

export function mediaObjectPath(itemId = randomId()) {
  return `objects/${itemId}.bin`;
}

export async function readTextFile(file: File) {
  return file.text();
}

export async function extractMediaMetadata(file: File, mediaKind: EncryptedDatasetMediaKind) {
  if (mediaKind === 'image') {
    return new Promise<{ width?: number; height?: number; durationMs?: number }>(resolve => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        resolve({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        resolve({});
      };
      img.src = url;
    });
  }

  return new Promise<{ width?: number; height?: number; durationMs?: number }>(resolve => {
    const url = URL.createObjectURL(file);
    const el = document.createElement(mediaKind === 'video' ? 'video' : 'audio');
    el.preload = 'metadata';
    el.onloadedmetadata = () => {
      const video = el as HTMLVideoElement;
      URL.revokeObjectURL(url);
      resolve({
        width: mediaKind === 'video' ? video.videoWidth : undefined,
        height: mediaKind === 'video' ? video.videoHeight : undefined,
        durationMs: Number.isFinite(el.duration) ? Math.round(el.duration * 1000) : undefined,
      });
    };
    el.onerror = () => {
      URL.revokeObjectURL(url);
      resolve({});
    };
    el.src = url;
  });
}

export function pairMediaAndCaptionFiles(files: File[]) {
  const mediaFiles: File[] = [];
  const captionByBaseName = new Map<string, File>();

  files.forEach(file => {
    if (/\.txt$/i.test(file.name)) {
      captionByBaseName.set(getBaseName(file.name).toLowerCase(), file);
      return;
    }
    if (getMediaKind(file)) mediaFiles.push(file);
  });

  return mediaFiles.map(file => ({
    file,
    captionFile: captionByBaseName.get(getBaseName(file.name).toLowerCase()) || null,
  }));
}

export async function buildEncryptedDatasetItem(
  file: File,
  key: CryptoKey,
  caption: string | null,
) {
  const mediaKind = getMediaKind(file);
  if (!mediaKind) throw new Error(`Unsupported file type: ${file.name}`);
  const itemId = randomId();
  const objectPath = mediaObjectPath(itemId);
  const mediaBytes = await file.arrayBuffer();
  const encryptedMedia = await encryptBytes(key, mediaBytes, objectAad(objectPath));
  const now = new Date().toISOString();
  const item: EncryptedDatasetItem = {
    id: itemId,
    name: file.name,
    extension: getExtension(file.name).toLowerCase(),
    mimeType: file.type || 'application/octet-stream',
    mediaKind,
    objectPath,
    size: file.size,
    ...(await extractMediaMetadata(file, mediaKind)),
    createdAt: now,
    updatedAt: now,
  };

  const encryptedObjects: Array<{ objectPath: string; blob: Blob }> = [
    {
      objectPath,
      blob: new Blob([JSON.stringify(encryptedMedia)], { type: 'application/json' }),
    },
  ];

  if (caption != null && caption.trim() !== '') {
    const captionPath = captionObjectPath(randomId());
    const encryptedCaption = await encryptBytes(key, textEncoder.encode(caption), objectAad(captionPath));
    item.captionObjectPath = captionPath;
    encryptedObjects.push({
      objectPath: captionPath,
      blob: new Blob([JSON.stringify(encryptedCaption)], { type: 'application/json' }),
    });
  }

  return { item, encryptedObjects };
}

export async function decryptEncryptedObjectBlob(key: CryptoKey, objectPath: string, blob: Blob) {
  const payload = JSON.parse(await blob.text()) as { nonce: string; data: string };
  return decryptBytes(key, payload.nonce, payload.data, objectAad(objectPath));
}

export async function encryptCaptionObject(key: CryptoKey, objectPath: string, caption: string) {
  return encryptBytes(key, textEncoder.encode(caption), objectAad(objectPath));
}
