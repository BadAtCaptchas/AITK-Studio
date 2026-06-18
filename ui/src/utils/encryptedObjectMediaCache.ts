import type { EncryptedDatasetItem } from '../types';

export type EncryptedObjectRequestBody = {
  datasetName: string;
  objectPath: string;
  worker_id: string;
  project_id?: string;
};

export type EncryptedObjectMediaStatus = 'locked' | 'loading' | 'ready' | 'error';

export type EncryptedObjectMediaLoadOptions = {
  datasetName: string;
  workerID: string;
  projectID?: string | null;
  cryptoKey: CryptoKey;
  item: Pick<EncryptedDatasetItem, 'objectPath' | 'updatedAt' | 'mimeType'>;
  loadEncryptedObject: (body: EncryptedObjectRequestBody) => Promise<Blob>;
  decryptEncryptedObject: (cryptoKey: CryptoKey, objectPath: string, blob: Blob) => Promise<ArrayBuffer>;
  createObjectUrl?: (blob: Blob) => string;
  revokeObjectUrl?: (url: string) => void;
};

const ENCRYPTED_OBJECT_MEDIA_CACHE_LIMIT = 128;
const ENCRYPTED_OBJECT_MEDIA_LOAD_CONCURRENCY = 4;

type CacheEntry = {
  url: string;
  lastUsed: number;
  revokeObjectUrl: (url: string) => void;
};

type QueueEntry<T> = {
  task: () => Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
};

const cryptoKeyIDs = new WeakMap<object, number>();
const cache = new Map<string, CacheEntry>();
const inFlight = new Map<string, Promise<string>>();
const queue: QueueEntry<unknown>[] = [];

let nextCryptoKeyID = 1;
let accessCounter = 1;
let activeLoads = 0;
let cacheGeneration = 0;

function cryptoKeyID(cryptoKey: CryptoKey) {
  const keyObject = cryptoKey as object;
  let id = cryptoKeyIDs.get(keyObject);
  if (!id) {
    id = nextCryptoKeyID;
    nextCryptoKeyID += 1;
    cryptoKeyIDs.set(keyObject, id);
  }
  return id;
}

function projectScope(workerID: string, projectID?: string | null) {
  return workerID === 'local' && projectID ? projectID : '';
}

function defaultCreateObjectUrl(blob: Blob) {
  if (typeof URL === 'undefined' || typeof URL.createObjectURL !== 'function') {
    throw new Error('Object URLs are not available in this environment.');
  }
  return URL.createObjectURL(blob);
}

function defaultRevokeObjectUrl(url: string) {
  if (typeof URL !== 'undefined' && typeof URL.revokeObjectURL === 'function') {
    URL.revokeObjectURL(url);
  }
}

function cacheKey(options: EncryptedObjectMediaLoadOptions) {
  const workerID = options.workerID || 'local';
  return JSON.stringify([
    workerID,
    projectScope(workerID, options.projectID),
    options.datasetName,
    cryptoKeyID(options.cryptoKey),
    options.item.objectPath,
    options.item.updatedAt || '',
    options.item.mimeType || 'application/octet-stream',
  ]);
}

function touch(entry: CacheEntry) {
  entry.lastUsed = accessCounter;
  accessCounter += 1;
}

function evictLeastRecentlyUsed() {
  while (cache.size > ENCRYPTED_OBJECT_MEDIA_CACHE_LIMIT) {
    let oldestKey = '';
    let oldestUse = Infinity;
    cache.forEach((entry, key) => {
      if (entry.lastUsed < oldestUse) {
        oldestUse = entry.lastUsed;
        oldestKey = key;
      }
    });
    if (!oldestKey) return;
    const oldest = cache.get(oldestKey);
    cache.delete(oldestKey);
    if (oldest) oldest.revokeObjectUrl(oldest.url);
  }
}

function pumpQueue() {
  while (activeLoads < ENCRYPTED_OBJECT_MEDIA_LOAD_CONCURRENCY && queue.length > 0) {
    const entry = queue.shift();
    if (!entry) return;
    activeLoads += 1;
    entry
      .task()
      .then(entry.resolve, entry.reject)
      .finally(() => {
        activeLoads -= 1;
        pumpQueue();
      });
  }
}

function runLimited<T>(task: () => Promise<T>) {
  return new Promise<T>((resolve, reject) => {
    queue.push({ task, resolve: resolve as (value: unknown) => void, reject });
    pumpQueue();
  });
}

export function buildEncryptedObjectRequestBody({
  datasetName,
  workerID,
  projectID,
  objectPath,
}: {
  datasetName: string;
  workerID: string;
  projectID?: string | null;
  objectPath: string;
}): EncryptedObjectRequestBody {
  const body: EncryptedObjectRequestBody = {
    datasetName,
    objectPath,
    worker_id: workerID || 'local',
  };
  if (body.worker_id === 'local' && projectID) {
    body.project_id = projectID;
  }
  return body;
}

export async function loadEncryptedObjectMediaUrl(options: EncryptedObjectMediaLoadOptions) {
  const key = cacheKey(options);
  const cached = cache.get(key);
  if (cached) {
    touch(cached);
    return cached.url;
  }

  const pending = inFlight.get(key);
  if (pending) return pending;

  const generation = cacheGeneration;
  const createObjectUrl = options.createObjectUrl || defaultCreateObjectUrl;
  const revokeObjectUrl = options.revokeObjectUrl || defaultRevokeObjectUrl;
  const promise = runLimited(async () => {
    const body = buildEncryptedObjectRequestBody({
      datasetName: options.datasetName,
      workerID: options.workerID,
      projectID: options.projectID,
      objectPath: options.item.objectPath,
    });
    const encryptedBlob = await options.loadEncryptedObject(body);
    const decrypted = await options.decryptEncryptedObject(options.cryptoKey, options.item.objectPath, encryptedBlob);
    const mediaBlob = new Blob([decrypted], { type: options.item.mimeType || 'application/octet-stream' });
    const url = createObjectUrl(mediaBlob);
    if (generation === cacheGeneration) {
      const entry: CacheEntry = { url, lastUsed: 0, revokeObjectUrl };
      touch(entry);
      cache.set(key, entry);
      evictLeastRecentlyUsed();
    }
    return url;
  }).finally(() => {
    inFlight.delete(key);
  });

  inFlight.set(key, promise);
  return promise;
}

export function clearEncryptedObjectMediaCache() {
  cacheGeneration += 1;
  cache.forEach(entry => entry.revokeObjectUrl(entry.url));
  cache.clear();
}
