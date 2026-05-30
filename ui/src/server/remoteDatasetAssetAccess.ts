import { createHmac, timingSafeEqual } from 'crypto';
import { makeRemoteDatasetAssetRef, type RemoteDatasetAssetType } from '@/utils/remoteDatasetRefs';

const SIGNATURE_TTL_MS = 6 * 60 * 60 * 1000;
const SIGNATURE_CONTEXT = 'remote-dataset-asset-v1';

type RemoteDatasetAssetIdentity = {
  workerID: string;
  remotePath: string;
  expires: number;
};

function authSecret() {
  return process.env.AI_TOOLKIT_AUTH || null;
}

function bearerToken(headers: Headers) {
  const value = headers.get('authorization') || '';
  const match = value.match(/^Bearer\s+(.+)$/i);
  return match?.[1] || null;
}

function payload({ workerID, remotePath, expires }: RemoteDatasetAssetIdentity) {
  return [SIGNATURE_CONTEXT, workerID, remotePath, String(expires)].join('\n');
}

function hmac(secret: string, identity: RemoteDatasetAssetIdentity) {
  return createHmac('sha256', secret).update(payload(identity)).digest('base64url');
}

function safeEqual(left: string, right: string) {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  return leftBuffer.length === rightBuffer.length && timingSafeEqual(leftBuffer, rightBuffer);
}

export function signRemoteDatasetAsset(workerID: string, remotePath: string) {
  const secret = authSecret();
  if (!secret) return null;
  const expires = Date.now() + SIGNATURE_TTL_MS;
  return {
    expires,
    signature: hmac(secret, { workerID, remotePath, expires }),
  };
}

export function makeSignedRemoteDatasetAssetRef(
  workerID: string,
  type: RemoteDatasetAssetType,
  remotePath: string,
  filename?: string,
) {
  return makeRemoteDatasetAssetRef(workerID, type, remotePath, filename, signRemoteDatasetAsset(workerID, remotePath));
}

export function isRemoteDatasetAssetSignatureValid(
  workerID: string,
  remotePath: string,
  expiresValue: string | number | null | undefined,
  signature: string | null | undefined,
) {
  const secret = authSecret();
  if (!secret) return true;
  const expires = Number(expiresValue);
  if (!workerID || !remotePath || !signature || !Number.isSafeInteger(expires) || expires <= Date.now()) {
    return false;
  }
  return safeEqual(signature, hmac(secret, { workerID, remotePath, expires }));
}

export function hasCentralBearerAuth(headers: Headers) {
  const secret = authSecret();
  if (!secret) return true;
  const token = bearerToken(headers);
  return !!token && safeEqual(token, secret);
}

export function isRemoteDatasetAssetRequestAuthorized(
  headers: Headers,
  workerID: string,
  remotePath: string,
  expiresValue: string | number | null | undefined,
  signature: string | null | undefined,
) {
  return (
    hasCentralBearerAuth(headers) ||
    isRemoteDatasetAssetSignatureValid(workerID, remotePath, expiresValue, signature)
  );
}
