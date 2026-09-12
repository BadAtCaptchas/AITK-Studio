import { readBoundedResponseText } from './responseBody';
import { timingSafeEqual } from 'crypto';
import type { EncryptedDatasetStartKey } from '../types';
import { getOperation } from './operations';
import { parseJobStartCommand, isRecord } from './commandInput';

declare global {
  var __aitkOperationKeys: Map<string, { keys: EncryptedDatasetStartKey[]; expires: number }> | undefined;
}
const keys = (globalThis.__aitkOperationKeys ??= new Map());
export function internalOperationAuthorized(headers: Headers): boolean {
  const expected = process.env.AITK_INTERNAL_TOKEN;
  const actual = headers.get('authorization')?.replace(/^Bearer /, '');
  if (!expected || !actual) return false;
  const a = Buffer.from(actual),
    b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}
export function rememberOperationKeys(id: string, input: EncryptedDatasetStartKey[]): void {
  for (const [key, value] of keys) if (value.expires < Date.now()) keys.delete(key);
  if (keys.size >= 100 && !keys.has(id)) throw new Error('Too many pending encrypted operations');
  keys.set(id, { keys: input.map(key => ({ ...key })), expires: Date.now() + 60 * 60 * 1000 });
}
export async function readOperationKeys(id: string): Promise<EncryptedDatasetStartKey[] | null> {
  const operation = await getOperation(id),
    stored = keys.get(id);
  if (
    !operation ||
    ['completed', 'failed', 'canceled'].includes(operation.state) ||
    !stored ||
    stored.expires < Date.now()
  ) {
    keys.delete(id);
    return null;
  }
  return stored.keys;
}
export async function fetchOperationKeys(id: string): Promise<EncryptedDatasetStartKey[]> {
  const base = process.env.AITK_INTERNAL_URL;
  if (!base || !process.env.AITK_INTERNAL_TOKEN) throw keysRequired();
  const response = await fetch(`${base}/api/internal/operation-keys/${id}`, {
    headers: { Authorization: `Bearer ${process.env.AITK_INTERNAL_TOKEN}` },
    redirect: 'error',
    cache: 'no-store',
    signal: AbortSignal.timeout(5000),
  });
  if (!response.ok) throw keysRequired();
  const value: unknown = JSON.parse(await readBoundedResponseText(response, 1024 * 1024));
  if (!isRecord(value)) throw keysRequired();
  return parseJobStartCommand(value).encryptedDatasetKeys || [];
}
function keysRequired() {
  return Object.assign(new Error('Unlock datasets again to continue'), { code: 'OPERATION_KEYS_REQUIRED' });
}
