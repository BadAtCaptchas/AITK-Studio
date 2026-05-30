export const WEBAUTHN_PRF_KDF_TYPE = 'WEBAUTHN-PRF';
export const WEBAUTHN_PRF_WRAPPING_INFO = 'aitk encrypted dataset WebAuthn PRF DEK wrapping v1';
export const WEBAUTHN_PRF_NATIVE_USB_PROVIDER = 'ctap2-hmac-secret';

const AES_GCM_NONCE_BYTES = 12;
const textEncoder = new TextEncoder();

function getSubtleCrypto() {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) {
    throw new Error('WebCrypto is required for WebAuthn PRF dataset encryption.');
  }
  return subtle as any;
}

function randomBytes(length: number) {
  const bytes = new Uint8Array(length);
  globalThis.crypto.getRandomValues(bytes);
  return bytes;
}

export function bytesToBase64(bytes: ArrayBuffer | Uint8Array) {
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(view).toString('base64');
  }

  let binary = '';
  view.forEach(byte => {
    binary += String.fromCharCode(byte);
  });
  return btoa(binary);
}

export function base64ToBytes(value: string) {
  if (typeof Buffer !== 'undefined') {
    return new Uint8Array(Buffer.from(value, 'base64'));
  }

  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

export function webAuthnPrfWrappedKeyAad(rpId: string, credentialId: string, saltB64: string) {
  return `aitk-webauthn-prf-wrapped-key:v1:${rpId}:${credentialId}:${saltB64}`;
}

export async function deriveWebAuthnPrfWrappingKey(
  prfOutput: ArrayBuffer | Uint8Array,
  usages: string[] = ['encrypt', 'decrypt'],
) {
  const subtle = getSubtleCrypto();
  const prfBytes = prfOutput instanceof Uint8Array ? prfOutput : new Uint8Array(prfOutput);
  const masterKey = await subtle.importKey('raw', prfBytes, 'HKDF', false, ['deriveKey']);
  return subtle.deriveKey(
    {
      name: 'HKDF',
      salt: new Uint8Array(),
      hash: 'SHA-256',
      info: textEncoder.encode(WEBAUTHN_PRF_WRAPPING_INFO),
    },
    masterKey,
    { name: 'AES-GCM', length: 256 },
    false,
    usages,
  );
}

export async function encryptWithWebAuthnPrfKey(
  prfOutput: ArrayBuffer | Uint8Array,
  plaintext: ArrayBuffer | Uint8Array,
  aad: string,
) {
  const subtle = getSubtleCrypto();
  const key = await deriveWebAuthnPrfWrappingKey(prfOutput, ['encrypt']);
  const nonce = randomBytes(AES_GCM_NONCE_BYTES);
  const plaintextBytes = plaintext instanceof Uint8Array ? plaintext : new Uint8Array(plaintext);
  const encrypted = await subtle.encrypt(
    {
      name: 'AES-GCM',
      iv: nonce,
      additionalData: textEncoder.encode(aad),
      tagLength: 128,
    },
    key,
    plaintextBytes,
  );

  return {
    nonce: bytesToBase64(nonce),
    data: bytesToBase64(encrypted),
  };
}

export async function decryptWithWebAuthnPrfKey(
  prfOutput: ArrayBuffer | Uint8Array,
  nonceB64: string,
  dataB64: string,
  aad: string,
) {
  const subtle = getSubtleCrypto();
  const key = await deriveWebAuthnPrfWrappingKey(prfOutput, ['decrypt']);
  return subtle.decrypt(
    {
      name: 'AES-GCM',
      iv: base64ToBytes(nonceB64),
      additionalData: textEncoder.encode(aad),
      tagLength: 128,
    },
    key,
    base64ToBytes(dataB64),
  ) as Promise<ArrayBuffer>;
}
