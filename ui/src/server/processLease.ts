import os from 'os';
import { randomUUID } from 'crypto';
import { execFile } from 'child_process';
import { promisify } from 'util';
import { db } from './db';
import { getToolkitPythonPath } from './pythonPath';

type Owner = { token: string; host: string; pid: number; birth: number; heartbeat: number };
const execute = promisify(execFile);
let identity: Promise<number> | undefined;
async function processBirth(pid: number): Promise<number | null> {
  const { stdout } = await execute(
    getToolkitPythonPath(),
    [
      '-c',
      'import sys,psutil\ntry: print(psutil.Process(int(sys.argv[1])).create_time())\nexcept psutil.NoSuchProcess: print("absent")',
      String(pid),
    ],
    { timeout: 5_000, maxBuffer: 4096, windowsHide: true },
  );
  if (stdout.trim() === 'absent') return null;
  const birth = Number(stdout.trim());
  if (!Number.isFinite(birth) || birth <= 0) throw new Error('Cannot verify process ownership');
  return birth;
}
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
  const birth = await (identity ??= processBirth(process.pid)
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
      const actual = await processBirth(prior.pid);
      if (actual !== null && Math.abs(actual - prior.birth) < 0.01) throw new LeaseBusyError();
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
