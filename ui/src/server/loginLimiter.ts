import { createHash } from 'crypto';
import { db } from './db';

async function consume(key: string, limit: number): Promise<boolean> {
  const bucket = Math.floor(Date.now() / 60_000);
  for (let retry = 0; retry < 10; retry++) {
    const row = await db.runtime.get(key);
    const value = row?.value;
    const count =
      value &&
      typeof value === 'object' &&
      'bucket' in value &&
      value.bucket === bucket &&
      'count' in value &&
      typeof value.count === 'number'
        ? value.count
        : 0;
    if (count >= limit) return false;
    if (await db.runtime.compareAndSwap(key, row?.version ?? null, { bucket, count: count + 1 })) return true;
  }
  return false;
}
export async function allowLogin(headers: Headers): Promise<boolean> {
  if (!(await consume('login-limit:global', 200))) return false;
  const address = headers.get('x-aitk-client-ip') || 'direct-loopback';
  const hash = createHash('sha256').update(address).digest('hex');
  return consume(`login-limit:${hash}`, 10);
}
