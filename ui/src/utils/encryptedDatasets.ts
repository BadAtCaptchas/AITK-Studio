import type {
  EncryptedDatasetCatalog,
  EncryptedDatasetItem,
  EncryptedDatasetManifest,
  EncryptedDatasetMediaKind,
} from '@/types';
import {
  WEBAUTHN_PRF_KDF_TYPE,
  WEBAUTHN_PRF_NATIVE_USB_PROVIDER,
  decryptWithWebAuthnPrfKey,
  encryptWithWebAuthnPrfKey,
  webAuthnPrfWrappedKeyAad,
} from '@/utils/webauthnPrfCrypto';

export const ENCRYPTED_DATASET_FORMAT = 'aitk-encrypted-dataset';
export const ENCRYPTED_DATASET_VERSION = 1;
export const PBKDF2_ITERATIONS = 600_000;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();
const rememberedKeys = new Map<string, string>();
const WEBAUTHN_TIMEOUT_MS = 120_000;

function copyToArrayBuffer(bytes: ArrayBuffer | ArrayBufferView): ArrayBuffer {
  if (bytes instanceof ArrayBuffer) return bytes;
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  return copy.buffer as ArrayBuffer;
}

function encodeUtf8(value: string): ArrayBuffer {
  return copyToArrayBuffer(textEncoder.encode(value));
}

type WebAuthnPrfCredentialDescriptor = Extract<
  EncryptedDatasetManifest['crypto']['kdf'],
  { type: 'WEBAUTHN-PRF' }
>['credentials'][number];

export type DatasetCredentialMode = 'password' | 'keyFile' | 'yubiKey';

export type EncryptedDatasetUnlockRequest =
  | { provider: 'password'; password: string }
  | { provider: 'keyFile'; file: File }
  | { provider: 'webauthnPrf' }
  | { provider: 'nativeUsb' };

export type EncryptedDatasetUnlockResult = {
  provider: Exclude<EncryptedDatasetUnlockRequest['provider'], 'nativeUsb'>;
  key: CryptoKey;
  rawKeyB64: string;
};

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

export function arrayBufferToBase64(buffer: ArrayBuffer | ArrayBufferView) {
  const bytes = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : new Uint8Array(copyToArrayBuffer(buffer));
  let binary = '';
  bytes.forEach(byte => {
    binary += String.fromCharCode(byte);
  });
  return btoa(binary);
}

export function base64ToArrayBuffer(value: string): ArrayBuffer {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer as ArrayBuffer;
}

export function arrayBufferToBase64Url(buffer: ArrayBuffer | ArrayBufferView) {
  return arrayBufferToBase64(buffer).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

export function base64UrlToArrayBuffer(value: string) {
  const padded = value.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(value.length / 4) * 4, '=');
  return base64ToArrayBuffer(padded);
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
  const passwordKey = await crypto.subtle.importKey('raw', encodeUtf8(password), 'PBKDF2', false, ['deriveKey']);
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

function ensureWebAuthnPrfAvailable() {
  if (typeof window === 'undefined' || !window.isSecureContext) {
    throw new Error('YubiKey unlock requires a secure browser context.');
  }
  if (!navigator.credentials || typeof PublicKeyCredential === 'undefined') {
    throw new Error('This browser does not support WebAuthn security keys.');
  }
}

function currentRpId() {
  if (typeof window === 'undefined') return 'localhost';
  return window.location.hostname || 'localhost';
}

function explicitRpId(rpId: string) {
  if (!rpId || /^\d{1,3}(?:\.\d{1,3}){3}$/.test(rpId) || rpId.includes(':')) return undefined;
  return rpId;
}

function requirePublicKeyCredential(value: Credential | null) {
  if (!value || value.type !== 'public-key') {
    throw new Error('A WebAuthn security key response was not returned.');
  }
  return value as PublicKeyCredential;
}

function getPrfFirstResult(credential: PublicKeyCredential) {
  const results = credential.getClientExtensionResults() as any;
  return results?.prf?.results?.first as ArrayBuffer | undefined;
}

function normalizeTransports(value: unknown) {
  if (!Array.isArray(value)) return ['usb'];
  const transports = value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0);
  return transports.length > 0 ? transports : ['usb'];
}

async function evaluateWebAuthnPrf(
  rpId: string,
  credentials: WebAuthnPrfCredentialDescriptor[],
) {
  ensureWebAuthnPrfAvailable();
  const evalByCredential: Record<string, { first: ArrayBuffer }> = {};
  const allowCredentials = credentials.map(credential => {
    evalByCredential[credential.id] = { first: base64ToArrayBuffer(credential.saltB64) };
    return {
      type: 'public-key',
      id: base64UrlToArrayBuffer(credential.id),
      transports: credential.transports,
    };
  });

  const assertion = requirePublicKeyCredential(
    await navigator.credentials.get({
      publicKey: {
        challenge: crypto.getRandomValues(new Uint8Array(32)),
        ...(explicitRpId(rpId) ? { rpId } : {}),
        allowCredentials: allowCredentials as PublicKeyCredentialDescriptor[],
        userVerification: 'preferred',
        timeout: WEBAUTHN_TIMEOUT_MS,
        extensions: {
          prf: { evalByCredential },
        } as any,
      },
    }),
  );
  const credentialId = arrayBufferToBase64Url(assertion.rawId);
  const credential = credentials.find(item => item.id === credentialId || item.id === assertion.id);
  const prfOutput = getPrfFirstResult(assertion);
  if (!credential || !prfOutput) {
    throw new Error('This security key did not return a WebAuthn PRF result for the dataset.');
  }
  return { credential, prfOutput };
}

async function createWebAuthnPrfCredential(rpId: string, saltB64: string) {
  ensureWebAuthnPrfAvailable();
  const salt = base64ToArrayBuffer(saltB64);
  const credential = requirePublicKeyCredential(
    await navigator.credentials.create({
      publicKey: {
        challenge: crypto.getRandomValues(new Uint8Array(32)),
        rp: {
          name: 'AI Toolkit',
          ...(explicitRpId(rpId) ? { id: rpId } : {}),
        },
        user: {
          id: crypto.getRandomValues(new Uint8Array(16)),
          name: 'ai-toolkit-dataset-key',
          displayName: 'AI Toolkit Dataset Key',
        },
        pubKeyCredParams: [
          { type: 'public-key', alg: -7 },
          { type: 'public-key', alg: -257 },
        ],
        authenticatorSelection: {
          authenticatorAttachment: 'cross-platform',
          residentKey: 'discouraged',
          userVerification: 'preferred',
        },
        timeout: WEBAUTHN_TIMEOUT_MS,
        attestation: 'none',
        extensions: {
          prf: { eval: { first: salt } },
        } as any,
      },
    }),
  );
  const credentialId = arrayBufferToBase64Url(credential.rawId);
  const transports = normalizeTransports((credential.response as any).getTransports?.());
  const prfOutput = getPrfFirstResult(credential);
  return { credentialId, transports, prfOutput };
}

async function createWebAuthnPrfDatasetKey(label?: string) {
  const rpId = currentRpId();
  const rawDatasetKey = crypto.getRandomValues(new Uint8Array(32));
  const rawKeyB64 = arrayBufferToBase64(rawDatasetKey);
  const saltB64 = arrayBufferToBase64(crypto.getRandomValues(new Uint8Array(32)));
  const created = await createWebAuthnPrfCredential(rpId, saltB64);
  const unsignedCredential: WebAuthnPrfCredentialDescriptor = {
    id: created.credentialId,
    label: label?.trim() || 'YubiKey / USB Security Key',
    transports: created.transports,
    saltB64,
    createdAt: new Date().toISOString(),
    wrappedKey: {
      algorithm: 'AES-256-GCM',
      nonce: '',
      data: '',
    },
  };
  const prfOutput =
    created.prfOutput || (await evaluateWebAuthnPrf(rpId, [unsignedCredential])).prfOutput;
  const wrappedKey = await encryptWithWebAuthnPrfKey(
    prfOutput,
    rawDatasetKey,
    webAuthnPrfWrappedKeyAad(rpId, unsignedCredential.id, saltB64),
  );

  const kdf: Extract<EncryptedDatasetManifest['crypto']['kdf'], { type: 'WEBAUTHN-PRF' }> = {
    type: 'WEBAUTHN-PRF',
    keyLength: 32,
    rpId,
    credentials: [
      {
        ...unsignedCredential,
        wrappedKey: {
          algorithm: 'AES-256-GCM' as const,
          ...wrappedKey,
        },
      },
    ],
    nativeUsb: {
      provider: WEBAUTHN_PRF_NATIVE_USB_PROVIDER,
      status: 'planned' as const,
    },
  };

  return {
    rawKeyB64,
    kdf,
  };
}

export async function unlockWebAuthnPrfDatasetKey(manifest: EncryptedDatasetManifest) {
  const kdf = manifest.crypto.kdf;
  if (kdf.type !== WEBAUTHN_PRF_KDF_TYPE) {
    throw new Error('This dataset is not protected by a WebAuthn PRF security key.');
  }
  const { credential, prfOutput } = await evaluateWebAuthnPrf(kdf.rpId, kdf.credentials);
  const decrypted = await decryptWithWebAuthnPrfKey(
    prfOutput,
    credential.wrappedKey.nonce,
    credential.wrappedKey.data,
    webAuthnPrfWrappedKeyAad(kdf.rpId, credential.id, credential.saltB64),
  );
  if (decrypted.byteLength !== 32) {
    throw new Error('Invalid WebAuthn PRF wrapped dataset key.');
  }
  const rawKeyB64 = arrayBufferToBase64(decrypted);
  return {
    provider: 'webauthnPrf' as const,
    key: await importRawAesKey(rawKeyB64),
    rawKeyB64,
  };
}

export async function unlockEncryptedDatasetKey(
  manifest: EncryptedDatasetManifest,
  request: EncryptedDatasetUnlockRequest,
): Promise<EncryptedDatasetUnlockResult> {
  if (request.provider === 'password') {
    const key = await derivePasswordKey(request.password, manifest);
    return { provider: 'password', key, rawKeyB64: await exportRawAesKey(key) };
  }
  if (request.provider === 'keyFile') {
    if (manifest.crypto.kdf.type !== 'KEYFILE-SHA256') {
      throw new Error('This dataset does not use a key file.');
    }
    const key = await deriveKeyFileKey(request.file);
    return { provider: 'keyFile', key, rawKeyB64: await exportRawAesKey(key) };
  }
  if (request.provider === 'webauthnPrf') {
    return unlockWebAuthnPrfDatasetKey(manifest);
  }
  throw new Error('Native USB unlock is not implemented yet.');
}

export function objectAad(objectPath: string) {
  return `aitk-encrypted-object:${objectPath}`;
}

export function catalogAad() {
  return 'aitk-encrypted-catalog:v1';
}

export async function encryptBytes(key: CryptoKey, bytes: ArrayBuffer | ArrayBufferView, aad: string) {
  const nonce = crypto.getRandomValues(new Uint8Array(12));
  const encrypted = await crypto.subtle.encrypt(
    { name: 'AES-GCM', iv: nonce, additionalData: encodeUtf8(aad), tagLength: 128 },
    key,
    copyToArrayBuffer(bytes),
  );
  return { nonce: arrayBufferToBase64(nonce), data: arrayBufferToBase64(encrypted) };
}

export async function decryptBytes(key: CryptoKey, nonceB64: string, dataB64: string | ArrayBuffer, aad: string) {
  const ciphertext = typeof dataB64 === 'string' ? base64ToArrayBuffer(dataB64) : dataB64;
  return crypto.subtle.decrypt(
    {
      name: 'AES-GCM',
      iv: base64ToArrayBuffer(nonceB64),
      additionalData: encodeUtf8(aad),
      tagLength: 128,
    },
    key,
    ciphertext,
  );
}

export async function createEmptyEncryptedManifest(
  mode: DatasetCredentialMode,
  secret?: string | File,
) {
  let key: CryptoKey | null = null;
  let rawKeyB64: string | null = null;
  let kdf: EncryptedDatasetManifest['crypto']['kdf'];
  if (mode === 'yubiKey') {
    const result = await createWebAuthnPrfDatasetKey(typeof secret === 'string' ? secret : undefined);
    rawKeyB64 = result.rawKeyB64;
    key = await importRawAesKey(rawKeyB64);
    kdf = result.kdf;
  } else if (mode === 'password') {
    if (typeof secret !== 'string') throw new Error('Password is required.');
    kdf = {
      type: 'PBKDF2-SHA256',
      salt: arrayBufferToBase64(crypto.getRandomValues(new Uint8Array(16))),
      iterations: PBKDF2_ITERATIONS,
      keyLength: 32,
    };
  } else {
    if (!(secret instanceof File)) throw new Error('Key file is required.');
    kdf = {
      type: 'KEYFILE-SHA256',
      keyLength: 32,
    };
  }

  const manifest: EncryptedDatasetManifest = {
    format: ENCRYPTED_DATASET_FORMAT,
    version: ENCRYPTED_DATASET_VERSION,
    crypto: {
      algorithm: 'AES-256-GCM',
      kdf,
    },
    catalog: { nonce: '', data: '' },
  };

  if (mode === 'password') {
    key = await derivePasswordKey(secret as string, manifest);
    rawKeyB64 = await exportRawAesKey(key);
  } else if (mode === 'keyFile') {
    key = await deriveKeyFileKey(secret as File);
    rawKeyB64 = await exportRawAesKey(key);
  }

  if (!key || !rawKeyB64) {
    throw new Error('Encrypted dataset key could not be created.');
  }
  const { manifest: encryptedManifest } = await encryptCatalog({ version: 1, items: [] }, key, manifest);
  return { manifest: encryptedManifest, key, rawKeyB64 };
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
  catalogName = file.name,
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
    name: catalogName,
    extension: getExtension(catalogName).toLowerCase(),
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
