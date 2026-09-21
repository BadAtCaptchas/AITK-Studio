import os from 'os';
import { randomUUID } from 'crypto';
import { db } from './db';
import { getProcessBirth, processBirthMatches } from './processBirth';

type Owner = { token: string; host: string; pid: number; birth: number; heartbeat: number };
let identity: Promise<number> | undefined;
function isOwner(value: unknown): value is Owner {
  return (
    value !== null &&
    typeof value === 'object' &&
    'token' in value &&
    typeof value.token === 'string' &&
    'host' in value &&
    typeof value.host === 'string' &&
    'pid' in value &&
    typeof value.pid === 'number' &&
    'birth' in value &&
    typeof value.birth === 'number' &&
    'heartbeat' in value &&
    typeof value.heartbeat === 'number'
  );
}
export class LeaseBusyError extends Error {
  readonly status = 409;
  constructor() {
    super('Operation is already owned. Retry or poll its status.');
  }
}

/** Local filesystem work is recovered only after OS identity proves the previous owner is gone. */
export async function acquireProcessLease(key: string) {
  const birth = await (identity ??= getProcessBirth(process.pid)
    .then(value => {
      if (value === null) throw new Error('Process identity unavailable');
      return value;
    })
    .catch(error => {
      identity = undefined;
      throw error;
    }));
  const owner: Owner = { token: randomUUID(), host: os.hostname(), pid: process.pid, birth, heartbeat: Date.now() };
  const recordKey = `lease:${key}`;
  for (let retry = 0; retry < 5; retry++) {
    const row = await db.runtime.get(recordKey);
    if (row) {
      if (!isOwner(row.value) || row.value.host !== owner.host) throw new LeaseBusyError();
      // Expiry is diagnostic only: a slow but live process retains ownership.
      const prior = row.value;
      if (prior.pid === process.pid && prior.birth === birth) throw new LeaseBusyError();
      const actual = await getProcessBirth(prior.pid);
      if (actual !== null && processBirthMatches(actual, prior.birth)) throw new LeaseBusyError();
    }
    if (!(await db.runtime.compareAndSwap(recordKey, row?.version ?? null, owner))) continue;
    let closed = false;
    let heartbeat: Promise<void> = Promise.resolve();
    const touch = async () => {
      const current = await db.runtime.get(recordKey);
      if (!current || !isOwner(current.value) || current.value.token !== owner.token)
        throw new Error('Operation ownership lost');
      if (!(await db.runtime.compareAndSwap(recordKey, current.version, { ...owner, heartbeat: Date.now() })))
        throw new Error('Operation ownership lost');
    };
    const timer = setInterval(() => {
      heartbeat = heartbeat.then(touch).catch(() => undefined);
    }, 10_000);
    timer.unref();
    return {
      token: owner.token,
      async assertOwned() {
        const current = await db.runtime.get(recordKey);
        if (closed || !current || !isOwner(current.value) || current.value.token !== owner.token)
          throw new Error('Operation ownership lost');
      },
      async release() {
        if (closed) return;
        closed = true;
        clearInterval(timer);
        await heartbeat;
        const current = await db.runtime.get(recordKey);
        if (current && isOwner(current.value) && current.value.token === owner.token)
          await db.runtime.delete(recordKey, current.version);
      },
    };
  }
  throw new LeaseBusyError();
}

export async function withProcessLease<T>(
  key: string,
  work: (lease: Awaited<ReturnType<typeof acquireProcessLease>>) => Promise<T>,
): Promise<T> {
  const lease = await acquireProcessLease(key);
  try {
    return await work(lease);
  } finally {
    await lease.release();
  }
}
