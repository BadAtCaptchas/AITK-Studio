import { createHmac, timingSafeEqual } from 'crypto';
import { makeRemoteDatasetAssetRef, type RemoteDatasetAssetType } from '../utils/remoteDatasetRefs';
import { isRequestAuthenticated } from '../utils/authSession';

const SIGNATURE_TTL_MS = 6 * 60 * 60 * 1000;
const SIGNATURE_CONTEXT = 'remote-dataset-asset-v2';

type RemoteDatasetAssetIdentity = {
  workerID: string;
  remotePath: string;
  expires: number;
  type: RemoteDatasetAssetType;
};

function authSecret() {
  return process.env.AI_TOOLKIT_AUTH || null;
}

function bearerToken(headers: Headers) {
  const value = headers.get('authorization') || '';
  const match = value.match(/^Bearer\s+(.+)$/i);
  return match?.[1] || null;
}

function payload({ workerID, remotePath, expires, type }: RemoteDatasetAssetIdentity) {
  return JSON.stringify([SIGNATURE_CONTEXT, 'aitk-studio', 'read', workerID, remotePath, type, expires]);
}

function hmac(secret: string, identity: RemoteDatasetAssetIdentity) {
  const key = process.env.AITK_ASSET_SIGNING_SECRET || createHmac('sha256', secret).update('aitk-asset-signing-key-v2').digest('hex');
  return `v2.${createHmac('sha256', key).update(payload(identity)).digest('base64url')}`;
}

function safeEqual(left: string, right: string) {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  return leftBuffer.length === rightBuffer.length && timingSafeEqual(leftBuffer, rightBuffer);
}

export function signRemoteDatasetAsset(workerID: string, remotePath: string, type: RemoteDatasetAssetType) {
  const secret = authSecret();
  if (!secret) return null;
  const expires = Date.now() + SIGNATURE_TTL_MS;
  return {
    expires,
    signature: hmac(secret, { workerID, remotePath, expires, type }),
  };
}

export function makeSignedRemoteDatasetAssetRef(
  workerID: string,
  type: RemoteDatasetAssetType,
  remotePath: string,
  filename?: string,
) {
  return makeRemoteDatasetAssetRef(workerID, type, remotePath, filename, signRemoteDatasetAsset(workerID, remotePath, type));
}

export function isRemoteDatasetAssetSignatureValid(
  workerID: string,
  remotePath: string,
  expiresValue: string | number | null | undefined,
  signature: string | null | undefined,
  type: RemoteDatasetAssetType,
  method = 'GET',
) {
  if (!['GET', 'HEAD'].includes(method) || !['img', 'file', 'audio-art'].includes(type)) return false;
  const secret = authSecret();
  if (!secret) return true;
  const expires = Number(expiresValue);
  if (!workerID || !remotePath || !signature || !Number.isSafeInteger(expires) || expires <= Date.now()) {
    return false;
  }
  return safeEqual(signature, hmac(secret, { workerID, remotePath, expires, type }));
}

export function hasCentralBearerAuth(headers: Headers) {
  const secret = authSecret();
  if (!secret) return true;
  const token = bearerToken(headers);
  return !!token && safeEqual(token, secret);
}

export async function isRemoteDatasetAssetRequestAuthorized(
  headers: Headers,
  workerID: string,
  remotePath: string,
  expiresValue: string | number | null | undefined,
  signature: string | null | undefined,
  type: RemoteDatasetAssetType,
  method = 'GET',
) {
  return (
    (await isRequestAuthenticated({ headers }, authSecret())) ||
    isRemoteDatasetAssetSignatureValid(workerID, remotePath, expiresValue, signature, type, method)
  );
}
