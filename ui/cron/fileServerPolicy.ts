import type { Server } from 'http';

export type WorkerCrashDecision = {
  shouldStop: boolean;
  restartDelayMs: number;
};

export type ProxyRetryDecision = {
  retry: boolean;
  delayMs: number;
};

export function proxyRetryDecision(options: {
  method?: string;
  errorCode?: string;
  reusedSocket: boolean;
  attempt: number;
  headersSent: boolean;
  responseDestroyed: boolean;
}): ProxyRetryDecision {
  const bodyless = options.method === 'GET' || options.method === 'HEAD';
  const responseAvailable = !options.headersSent && !options.responseDestroyed;
  if (!bodyless || !responseAvailable || options.attempt >= 120) {
    return { retry: false, delayMs: 0 };
  }
  if (options.errorCode === 'ECONNREFUSED') {
    return { retry: true, delayMs: 250 };
  }
  if (
    options.reusedSocket &&
    (options.errorCode === 'ECONNRESET' || options.errorCode === 'EPIPE')
  ) {
    return { retry: true, delayMs: 0 };
  }
  return { retry: false, delayMs: 0 };
}

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
