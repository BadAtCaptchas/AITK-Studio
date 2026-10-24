export type WorkerHealth = {
  ok: boolean;
  app: string;
  timestamp: string;
  cloudflared: unknown;
  ollama?: unknown;
  executionReady: boolean | null;
  executionDetail: string;
  loraContinuationVersion?: number;
};

export function parseWorkerHealth(value: unknown): WorkerHealth {
  if (!value || typeof value !== 'object' || !('ok' in value) || typeof value.ok !== 'boolean')
    throw new Error('Worker returned an invalid health response');
  const continuation = 'loraContinuation' in value ? value.loraContinuation : null;
  const includesAll = (items: unknown, required: string[]) => Array.isArray(items) && required.every(item => items.includes(item));
  const supportsContinuation = continuation !== null && typeof continuation === 'object' &&
    'version' in continuation && continuation.version === 1 &&
    'models' in continuation && includesAll(continuation.models, ['krea2', 'krea2:turbo']) &&
    'modes' in continuation && includesAll(continuation.modes, ['none', 'replay', 'teacher', 'replay_teacher']);
  return {
    ok: value.ok,
    loraContinuationVersion: supportsContinuation ? 1 : undefined,
    app: 'app' in value && typeof value.app === 'string' ? value.app : '',
    timestamp: 'timestamp' in value && typeof value.timestamp === 'string' ? value.timestamp : '',
    cloudflared: 'cloudflared' in value ? value.cloudflared : null,
    ollama: 'ollama' in value ? value.ollama : undefined,
    executionReady: 'executionReady' in value && typeof value.executionReady === 'boolean' ? value.executionReady : null,
    executionDetail: 'executionDetail' in value && typeof value.executionDetail === 'string' ? value.executionDetail : 'Execution readiness is unavailable on this worker version.',
  };
}

export function workerHealthStatus(health: WorkerHealth): string {
  return !health.ok ? 'error' : health.executionReady === true ? 'ready' : health.executionReady === false ? 'degraded' : 'reachable';
}

export function workerHealthLabel(status: string): string {
  switch (status) {
    case 'ready': return 'Ready for jobs';
    case 'degraded': return 'Reachable; queue worker unavailable';
    case 'reachable': return 'Reachable; readiness unknown';
    default: return status;
  }
}
