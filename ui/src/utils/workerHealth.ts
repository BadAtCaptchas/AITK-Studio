export type WorkerHealth = {
  ok: boolean;
  app: string;
  timestamp: string;
  cloudflared: unknown;
  ollama?: unknown;
  executionReady: boolean | null;
  executionDetail: string;
};

export function parseWorkerHealth(value: unknown): WorkerHealth {
  if (!value || typeof value !== 'object' || !('ok' in value) || typeof value.ok !== 'boolean')
    throw new Error('Worker returned an invalid health response');
  return {
    ok: value.ok,
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
