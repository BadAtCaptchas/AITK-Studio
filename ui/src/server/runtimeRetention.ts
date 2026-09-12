import { db } from './db';
import { isRecord } from './commandInput';

/** Bounded rotating cleanup; CAS prevents deleting a concurrently refreshed record. */
export async function cleanupAuthRecords(): Promise<void> {
  for (const prefix of ['session:', 'login-limit:']) {
    const key = 'retention-cursor:' + prefix;
    const cursor = await db.runtime.get(key);
    const rows = await db.runtime.list(prefix, 500, typeof cursor?.value === 'string' ? cursor.value : '');
    const now = Date.now();
    for (const row of rows) {
      const value = row.value;
      if (!isRecord(value)) continue; // Revocation-generation records are retained.
      const expired =
        prefix === 'session:'
          ? typeof value.expiresAt === 'number' &&
            typeof value.idleExpiresAt === 'number' &&
            Math.min(value.expiresAt, value.idleExpiresAt) <= now
          : typeof value.bucket === 'number' && value.bucket < Math.floor(now / 60_000) - 1;
      if (expired) await db.runtime.delete(row.key, row.version);
    }
    await db.runtime.compareAndSwap(key, cursor?.version ?? null, rows.length === 500 ? rows.at(-1)!.key : '');
  }
}
