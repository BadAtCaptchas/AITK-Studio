import { db } from './db';
import { isRecord } from './commandInput';

export function executionHealthFromHeartbeat(value: unknown, now = Date.now()) {
  const at = isRecord(value) && typeof value.at === 'number' ? value.at : null;
  const executionReady = at !== null && Number.isFinite(at) && at <= now + 5_000 && now - at < 15_000;
  return { executionReady, executionDetail: executionReady ? 'Queue worker heartbeat is current.' : 'Queue worker has no recent heartbeat. Start the managed app stack and inspect worker logs.' };
}

/** Lightweight readiness for frequent remote health polling; no Python subprocess. */
export async function getExecutionHealth() {
  return executionHealthFromHeartbeat((await db.runtime.get('worker-heartbeat'))?.value);
}
