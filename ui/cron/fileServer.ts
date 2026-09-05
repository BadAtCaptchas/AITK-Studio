/**
 * Public UI front server.
 *
 * The primary process owns one loopback-only Next.js child. One or more cluster
 * workers share the public port, serve authorized local files with raw Node
 * streams, and proxy every other HTTP/WebSocket request to Next.js.
 */
import { hasObsoleteWorkspaceScope, hasObsoleteWorkspaceHeaders } from '../src/utils/obsoleteWorkspaceGuard';
import cluster, { type Worker } from 'cluster';
import { spawn, type ChildProcess } from 'child_process';
import fs from 'fs';
import http, { type IncomingHttpHeaders, type IncomingMessage, type ServerResponse } from 'http';
import net from 'net';
import os from 'os';
import { pipeline } from 'stream';
import type { FileResponseResolution } from '../src/server/fileServing';
import { configureFrontServerTimeouts, proxyRetryDecision, registerWorkerCrash } from './fileServerPolicy';

const isDev = process.argv.includes('dev');
const UPSTREAM_HOST = '127.0.0.1';
const STARTUP_TIMEOUT_MS = 60_000;

function argValue(name: string, fallback: number) {
  const index = process.argv.indexOf(name);
  if (index < 0) return fallback;
  const value = Number(process.argv[index + 1]);
  return Number.isSafeInteger(value) && value > 0 && value <= 65_535 ? value : fallback;
}

const PUBLIC_PORT = argValue('--port', isDev ? 3000 : 8675);
const numWorkers = (() => {
  const configured = Number.parseInt(process.env.AI_TOOLKIT_FILE_SERVER_WORKERS || '', 10);
  if (Number.isSafeInteger(configured) && configured > 0) return configured;
  return isDev ? 1 : Math.min(4, os.cpus().length);
})();
const WORKER_CRASH_WINDOW_MS = 30_000;
const WORKER_CRASH_LIMIT = Math.max(6, numWorkers * 3);

const hopByHopHeaders = new Set([
  'connection',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
]);

let fileServingModulePromise: Promise<typeof import('../src/server/fileServing')> | null = null;

function getFileServingModule() {
  fileServingModulePromise ??= import('../src/server/fileServing');
  return fileServingModulePromise;
}

function requestHeaders(headers: IncomingHttpHeaders) {
  const result = new Headers();
  for (const [name, rawValue] of Object.entries(headers)) {
    if (rawValue === undefined) continue;
    if (Array.isArray(rawValue)) {
      rawValue.forEach(value => result.append(name, value));
    } else {
      result.set(name, rawValue);
    }
  }
  return result;
}

function proxyHeaders(headers: IncomingHttpHeaders) {
  const result: http.OutgoingHttpHeaders = {};
  const connectionTokens = new Set(
    String(headers.connection || '')
      .split(',')
      .map(value => value.trim().toLowerCase())
      .filter(Boolean),
  );
  for (const [name, value] of Object.entries(headers)) {
    if (hopByHopHeaders.has(name.toLowerCase()) || connectionTokens.has(name.toLowerCase())) continue;
    result[name] = value;
  }
  return result;
}

function sendResolution(
  req: IncomingMessage,
  res: ServerResponse,
  resolution: Exclude<FileResponseResolution, { kind: 'proxy' }>,
  streamBufferBytes: number,
) {
  if (req.destroyed || res.destroyed) return;
  res.writeHead(resolution.status, resolution.headers);
  const bodylessStatus =
    resolution.status === 204 || resolution.status === 304 || (resolution.status >= 100 && resolution.status < 200);
  if (req.method === 'HEAD' || bodylessStatus) {
    res.end();
    return;
  }
  if (!resolution.file) {
    res.end(resolution.body || undefined);
    return;
  }

  const stream = fs.createReadStream(resolution.file.path, {
    ...(resolution.file.start !== undefined ? { start: resolution.file.start } : {}),
    ...(resolution.file.end !== undefined ? { end: resolution.file.end } : {}),
    highWaterMark: streamBufferBytes,
  });
  const abort = () => stream.destroy();
  req.once('aborted', abort);
  res.once('close', abort);
  stream.once('close', () => {
    req.off('aborted', abort);
    res.off('close', abort);
  });
  pipeline(stream, res, error => {
    if (error && !res.destroyed) res.destroy(error);
  });
}

function decodePathSuffix(pathname: string, prefix: string) {
  const encoded = pathname.slice(prefix.length);
  if (!encoded) throw new URIError('Missing file path');
  return decodeURIComponent(encoded);
}

async function tryServeAccelerated(req: IncomingMessage, res: ServerResponse) {
  if (req.method !== 'GET' && req.method !== 'HEAD') return false;

  const requestUrl = new URL(req.url || '/', `http://localhost:${PUBLIC_PORT}`);
  if (hasObsoleteWorkspaceScope(requestUrl.searchParams) || hasObsoleteWorkspaceHeaders(requestHeaders(req.headers))) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Project workspaces have been removed.', code: 'PROJECT_WORKSPACES_REMOVED' }));
    return true;
  }
  const headers = requestHeaders(req.headers);
  const serving = await getFileServingModule();
  let resolution: FileResponseResolution;

  try {
    if (requestUrl.pathname.startsWith('/api/files/')) {
      if (!(await serving.isAcceleratedApiRequestAuthenticated(headers))) {
        resolution = serving.unauthorizedFileResponse();
      } else {
        resolution = await serving.resolveDownloadFileRequest(
          decodePathSuffix(requestUrl.pathname, '/api/files/'),
          headers,
        );
      }
    } else if (requestUrl.pathname.startsWith('/api/img/')) {
      if (!(await serving.isAcceleratedApiRequestAuthenticated(headers))) {
        resolution = serving.unauthorizedFileResponse();
      } else {
        resolution = await serving.resolveLocalMediaRequest(
          decodePathSuffix(requestUrl.pathname, '/api/img/'),
          headers,
        );
      }
    } else {
      return false;
    }
  } catch (error) {
    if (error instanceof URIError) {
      res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Invalid file path');
      return true;
    }
    console.error('Accelerated file request failed:', error);
    res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Internal Server Error');
    return true;
  }

  if (resolution.kind === 'proxy') return false;
  sendResolution(req, res, resolution, serving.FILE_STREAM_BUFFER_BYTES);
  return true;
}

const upstreamAgent = new http.Agent({ keepAlive: true });

function proxy(req: IncomingMessage, res: ServerResponse, upstreamPort: number, attempt = 0) {
  const bodyless = req.method === 'GET' || req.method === 'HEAD';
  const upstreamRequest = http.request(
    {
      host: UPSTREAM_HOST,
      port: upstreamPort,
      path: req.url,
      method: req.method,
      headers: proxyHeaders(req.headers),
      agent: upstreamAgent,
    },
    upstreamResponse => {
      if (res.destroyed) {
        upstreamResponse.resume();
        return;
      }
      res.writeHead(upstreamResponse.statusCode || 502, proxyHeaders(upstreamResponse.headers));
      pipeline(upstreamResponse, res, error => {
        if (error && !res.destroyed) res.destroy(error);
      });
    },
  );

  const abortUpstream = () => upstreamRequest.destroy();
  const abortUpstreamIfIncomplete = () => {
    if (!res.writableEnded) abortUpstream();
  };
  const removeAbortListeners = () => {
    req.off('aborted', abortUpstream);
    res.off('close', abortUpstreamIfIncomplete);
  };
  req.once('aborted', abortUpstream);
  res.once('close', abortUpstreamIfIncomplete);
  upstreamRequest.once('close', removeAbortListeners);
  upstreamRequest.once('error', (error: NodeJS.ErrnoException) => {
    const retryDecision = proxyRetryDecision({
      method: req.method,
      errorCode: error.code,
      reusedSocket: upstreamRequest.reusedSocket,
      attempt,
      headersSent: res.headersSent,
      responseDestroyed: res.destroyed,
    });
    if (retryDecision.retry) {
      setTimeout(() => {
        if (!req.destroyed && !res.destroyed && !res.writableEnded) {
          proxy(req, res, upstreamPort, attempt + 1);
        }
      }, retryDecision.delayMs);
      return;
    }
    if (!res.headersSent && !res.destroyed) {
      res.writeHead(502, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('UI server unavailable');
    } else if (!res.destroyed) {
      res.destroy(error);
    }
  });

  if (bodyless) upstreamRequest.end();
  else req.pipe(upstreamRequest);
}

function proxyUpgrade(req: IncomingMessage, socket: import('stream').Duplex, head: Buffer, upstreamPort: number) {
  const upstream = net.connect(upstreamPort, UPSTREAM_HOST, () => {
    if (socket.destroyed) {
      upstream.destroy();
      return;
    }
    const requestLine = `${req.method || 'GET'} ${req.url || '/'} HTTP/${req.httpVersion}\r\n`;
    let headerBlock = '';
    for (let index = 0; index < req.rawHeaders.length; index += 2) {
      headerBlock += `${req.rawHeaders[index]}: ${req.rawHeaders[index + 1]}\r\n`;
    }
    upstream.write(`${requestLine}${headerBlock}\r\n`);
    if (head.length > 0) upstream.write(head);
    socket.pipe(upstream).pipe(socket);
  });
  const closeBoth = (error?: Error) => {
    if (!upstream.destroyed) upstream.destroy(error);
    if (!socket.destroyed) socket.destroy(error);
  };
  upstream.once('error', closeBoth);
  socket.once('error', closeBoth);
}

function listen(publicPort: number, upstreamPort: number) {
  return new Promise<http.Server>((resolve, reject) => {
    const server = http.createServer((req, res) => {
      void tryServeAccelerated(req, res)
        .then(served => {
          if (!served && !res.headersSent && !res.destroyed) proxy(req, res, upstreamPort);
        })
        .catch(error => {
          console.error('Front server request failed:', error);
          if (!res.headersSent && !res.destroyed) {
            res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
            res.end('Internal Server Error');
          }
        });
    });
    // Model/dataset uploads are deliberately streamed and may take longer
    // than Node's five-minute default on slower links.
    configureFrontServerTimeouts(server);
    server.on('upgrade', (req, socket, head) => proxyUpgrade(req, socket, head, upstreamPort));
    server.once('error', reject);
    server.listen(publicPort, () => {
      server.off('error', reject);
      resolve(server);
    });
  });
}

function getFreePort() {
  return new Promise<number>((resolve, reject) => {
    const probe = net.createServer();
    probe.once('error', reject);
    probe.listen(0, UPSTREAM_HOST, () => {
      const address = probe.address();
      const port = typeof address === 'object' && address ? address.port : 0;
      probe.close(error => {
        if (error) reject(error);
        else resolve(port);
      });
    });
  });
}

function waitForUpstream(port: number, child: ChildProcess) {
  return new Promise<void>((resolve, reject) => {
    const deadline = Date.now() + STARTUP_TIMEOUT_MS;
    let settled = false;
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      child.off('exit', onExit);
      if (error) reject(error);
      else resolve();
    };
    const onExit = (code: number | null) =>
      finish(new Error(`Next.js exited during startup with code ${code ?? 'unknown'}`));
    const probe = () => {
      if (settled) return;
      if (Date.now() >= deadline) {
        finish(new Error('Timed out waiting for Next.js to start'));
        return;
      }
      const socket = net.connect(port, UPSTREAM_HOST);
      socket.once('connect', () => {
        socket.destroy();
        finish();
      });
      socket.once('error', () => {
        socket.destroy();
        setTimeout(probe, 100);
      });
    };
    child.once('exit', onExit);
    probe();
  });
}

function rewriteChildOutput(stream: NodeJS.ReadableStream, destination: NodeJS.WritableStream, upstreamPort: number) {
  const internalAddress = new RegExp(`(https?://)?(127\\.0\\.0\\.1|localhost):${upstreamPort}`, 'g');
  let buffered = '';
  stream.on('data', chunk => {
    buffered += chunk.toString();
    const lines = buffered.split(/\r?\n/);
    buffered = lines.pop() || '';
    for (const line of lines) {
      destination.write(`${line.replace(internalAddress, `http://localhost:${PUBLIC_PORT}`)}\n`);
    }
  });
  stream.on('end', () => {
    if (buffered) {
      destination.write(`${buffered.replace(internalAddress, `http://localhost:${PUBLIC_PORT}`)}\n`);
    }
  });
}

function waitForWorkerReady(worker: Worker) {
  return new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error(`File server worker ${worker.id} did not become ready`));
    }, STARTUP_TIMEOUT_MS);
    const cleanup = () => {
      clearTimeout(timeout);
      worker.off('message', onMessage);
      worker.off('exit', onExit);
    };
    const onMessage = (message: unknown) => {
      if (
        !message ||
        typeof message !== 'object' ||
        !('type' in message) ||
        (message as { type?: unknown }).type !== 'aitk-file-server-ready'
      ) {
        return;
      }
      cleanup();
      resolve();
    };
    const onExit = (code: number | null) => {
      cleanup();
      reject(new Error(`File server worker ${worker.id} exited during startup with code ${code ?? 'unknown'}`));
    };
    worker.on('message', onMessage);
    worker.once('exit', onExit);
  });
}

async function primaryMain() {
  const upstreamPort = await getFreePort();
  const nextBin = require.resolve('next/dist/bin/next');
  const nextArgs = isDev ? ['dev', '--turbopack'] : ['start'];
  nextArgs.push('--port', String(upstreamPort), '--hostname', UPSTREAM_HOST);

  const nextChild = spawn(process.execPath, [nextBin, ...nextArgs], {
    stdio: ['ignore', 'pipe', 'pipe'],
    cwd: process.cwd(),
    env: process.env,
    windowsHide: true,
  });
  rewriteChildOutput(nextChild.stdout!, process.stdout, upstreamPort);
  rewriteChildOutput(nextChild.stderr!, process.stderr, upstreamPort);

  let shuttingDown = false;
  let clusterReady = false;
  let exitCode = 0;
  let localServer: http.Server | null = null;
  const recentWorkerCrashes: number[] = [];
  const forceExit = () => {
    if (nextChild.exitCode === null && nextChild.signalCode === null) {
      try {
        nextChild.kill('SIGKILL');
      } catch {
        // Best effort during shutdown.
      }
    }
    process.exit(exitCode);
  };
  const shutdown = (code: number, signal: NodeJS.Signals = 'SIGTERM') => {
    if (shuttingDown) return;
    shuttingDown = true;
    exitCode = code;
    localServer?.close();
    for (const worker of Object.values(cluster.workers || {})) {
      worker?.kill(signal);
    }
    if (nextChild.exitCode === null && nextChild.signalCode === null) {
      try {
        nextChild.kill(signal);
      } catch {
        forceExit();
        return;
      }
    } else {
      process.exit(exitCode);
      return;
    }
    setTimeout(forceExit, 5_000).unref();
  };

  nextChild.once('error', error => {
    console.error('Next.js failed to start:', error);
    shutdown(1);
  });
  nextChild.once('exit', code => {
    if (!shuttingDown) {
      console.error(`Next.js exited with code ${code ?? 'unknown'}`);
      shutdown(code && code !== 0 ? code : 1);
      return;
    }
    process.exit(exitCode);
  });
  process.once('SIGINT', () => shutdown(0, 'SIGINT'));
  process.once('SIGTERM', () => shutdown(0));
  process.once('disconnect', () => shutdown(0));
  process.on('message', message => {
    if (!message || typeof message !== 'object' || !('type' in message)) return;
    if ((message as { type?: unknown }).type !== 'aitk-shutdown') return;
    const requestedSignal = (message as { signal?: unknown }).signal;
    shutdown(0, requestedSignal === 'SIGINT' ? 'SIGINT' : 'SIGTERM');
  });
  process.once('uncaughtException', error => {
    console.error('File server uncaught exception:', error);
    shutdown(1);
  });
  process.once('unhandledRejection', error => {
    console.error('File server unhandled rejection:', error);
    shutdown(1);
  });

  try {
    await waitForUpstream(upstreamPort, nextChild);

    if (numWorkers === 1) {
      localServer = await listen(PUBLIC_PORT, upstreamPort);
    } else {
      const workerEnvironment = {
        UPSTREAM_PORT: String(upstreamPort),
        PUBLIC_PORT: String(PUBLIC_PORT),
      };
      cluster.on('exit', worker => {
        if (shuttingDown) return;
        if (!clusterReady) {
          console.error(`File server worker ${worker.id} exited before startup completed.`);
          shutdown(1);
          return;
        }

        const crashDecision = registerWorkerCrash(recentWorkerCrashes, Date.now(), {
          windowMs: WORKER_CRASH_WINDOW_MS,
          limit: WORKER_CRASH_LIMIT,
        });
        if (crashDecision.shouldStop) {
          console.error('File server workers are repeatedly crashing; stopping the managed UI.');
          shutdown(1);
          return;
        }

        console.warn(`File server worker ${worker.id} exited; restarting it in ${crashDecision.restartDelayMs}ms.`);
        setTimeout(() => {
          if (!shuttingDown) cluster.fork(workerEnvironment);
        }, crashDecision.restartDelayMs);
      });

      const workers = Array.from({ length: numWorkers }, () => cluster.fork(workerEnvironment));
      await Promise.all(workers.map(waitForWorkerReady));
      clusterReady = true;
    }
  } catch (error) {
    console.error('File server failed to become ready:', error);
    shutdown(1);
    return;
  }

  console.log(
    `AI Toolkit UI: http://localhost:${PUBLIC_PORT} (${numWorkers} file server worker${numWorkers === 1 ? '' : 's'})`,
  );
}

async function workerMain() {
  const publicPort = Number.parseInt(process.env.PUBLIC_PORT || '', 10);
  const upstreamPort = Number.parseInt(process.env.UPSTREAM_PORT || '', 10);
  if (!Number.isSafeInteger(publicPort) || !Number.isSafeInteger(upstreamPort)) {
    throw new Error('Cluster worker is missing its port configuration');
  }
  const server = await listen(publicPort, upstreamPort);
  if (process.send) {
    process.send({ type: 'aitk-file-server-ready' });
  }
  const shutdown = () => {
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 2_000).unref();
  };
  process.once('SIGINT', shutdown);
  process.once('SIGTERM', shutdown);
  process.once('disconnect', shutdown);
}

if (cluster.isPrimary) {
  void primaryMain().catch(error => {
    console.error('File server failed to start:', error);
    process.exit(1);
  });
} else {
  void workerMain().catch(error => {
    console.error('File server worker failed:', error);
    process.exit(1);
  });
}
