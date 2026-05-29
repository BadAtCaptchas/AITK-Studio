import crypto from 'crypto';

export type SecureCaptionDirection = 'request' | 'response';

export type SecureCaptionEnvelope = {
  version: 1;
  jobId: string;
  itemId: string;
  salt: string;
  nonce: string;
  data: string;
};

const VERSION = 1;
const NONCE_BYTES = 12;
const SALT_BYTES = 32;
const TAG_BYTES = 16;

function b64(value: Buffer) {
  return value.toString('base64');
}

function fromB64(value: string, field: string) {
  try {
    return Buffer.from(value, 'base64');
  } catch {
    throw new Error(`Invalid secure caption ${field}`);
  }
}

function aad(direction: SecureCaptionDirection, jobId: string, itemId: string) {
  return Buffer.from(`aitk-secure-caption:v${VERSION}:${direction}:${jobId}:${itemId}`, 'utf8');
}

function deriveKey(token: string, salt: Buffer, direction: SecureCaptionDirection) {
  if (!token) {
    throw new Error('Secure caption token is required');
  }
  return Buffer.from(
    crypto.hkdfSync(
      'sha256',
      Buffer.from(token, 'utf8'),
      salt,
      Buffer.from(`aitk-secure-caption:${direction}:v${VERSION}`, 'utf8'),
      32,
    ),
  );
}

export function encryptSecureCaptionJson(
  token: string,
  direction: SecureCaptionDirection,
  jobId: string,
  itemId: string,
  value: unknown,
): SecureCaptionEnvelope {
  const salt = crypto.randomBytes(SALT_BYTES);
  const nonce = crypto.randomBytes(NONCE_BYTES);
  const key = deriveKey(token, salt, direction);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, nonce);
  cipher.setAAD(aad(direction, jobId, itemId));
  const ciphertext = Buffer.concat([
    cipher.update(Buffer.from(JSON.stringify(value), 'utf8')),
    cipher.final(),
    cipher.getAuthTag(),
  ]);

  return {
    version: VERSION,
    jobId,
    itemId,
    salt: b64(salt),
    nonce: b64(nonce),
    data: b64(ciphertext),
  };
}

export function decryptSecureCaptionJson<T>(
  token: string,
  direction: SecureCaptionDirection,
  envelope: SecureCaptionEnvelope,
): T {
  if (!envelope || envelope.version !== VERSION) {
    throw new Error('Unsupported secure caption envelope');
  }
  if (!envelope.jobId || !envelope.itemId) {
    throw new Error('Secure caption envelope is missing context');
  }

  const salt = fromB64(envelope.salt, 'salt');
  const nonce = fromB64(envelope.nonce, 'nonce');
  const payload = fromB64(envelope.data, 'payload');
  if (salt.length !== SALT_BYTES || nonce.length !== NONCE_BYTES || payload.length <= TAG_BYTES) {
    throw new Error('Invalid secure caption envelope');
  }

  const ciphertext = payload.subarray(0, payload.length - TAG_BYTES);
  const tag = payload.subarray(payload.length - TAG_BYTES);
  const key = deriveKey(token, salt, direction);
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, nonce);
  decipher.setAAD(aad(direction, envelope.jobId, envelope.itemId));
  decipher.setAuthTag(tag);

  const plaintext = Buffer.concat([decipher.update(ciphertext), decipher.final()]).toString('utf8');
  return JSON.parse(plaintext) as T;
}

export function getSecureCaptionBearerToken(request: Request) {
  return request.headers.get('authorization')?.split(' ')[1] || '';
}
