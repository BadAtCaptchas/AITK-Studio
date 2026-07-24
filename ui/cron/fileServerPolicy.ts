import type { Server } from 'http';

export type WorkerCrashDecision = {
  shouldStop: boolean;
  restartDelayMs: number;
};

export function configureFrontServerTimeouts(server: Pick<Server, 'requestTimeout'>) {
  // Model and dataset uploads can be much larger than Node's five-minute
  // request-body window. The proxied request remains streaming and bounded by
  // the client/upstream connection instead.
  server.requestTimeout = 0;
}

export function registerWorkerCrash(
  recentCrashes: number[],
  now: number,
  options: {
    windowMs: number;
    limit: number;
    baseDelayMs?: number;
    maximumDelayMs?: number;
  },
): WorkerCrashDecision {
  recentCrashes.push(now);
  while (
    recentCrashes.length > 0 &&
    recentCrashes[0] < now - options.windowMs
  ) {
    recentCrashes.shift();
  }

  return {
    shouldStop: recentCrashes.length > options.limit,
    restartDelayMs: Math.min(
      (options.baseDelayMs ?? 250) * recentCrashes.length,
      options.maximumDelayMs ?? 2_000,
    ),
  };
}
