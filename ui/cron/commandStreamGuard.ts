import type { ClientRequest, IncomingMessage, ServerResponse } from 'http';
import { Transform } from 'stream';

const binaryUploadPaths = new Set([
  '/api/datasets/upload',
  '/api/img/upload',
  '/api/generate/loras/upload',
  '/api/watermark/check',
  '/api/datasets/layer-caption',
  '/api/datasets/openrouter-layer-caption',
  '/api/datasets/openrouter-boxes',
  '/api/datasets/recaption-single',
  '/api/datasets/auto-boxes',
]);

/** Enforce JSON intake limits before Next can clone a middleware request body. */
export function pipeCommandBody(request: IncomingMessage, response: ServerResponse, upstream: ClientRequest): void {
  const contentType = String(request.headers['content-type'] || '')
    .split(';')[0]
    .trim()
    .toLowerCase();
  const pathname = new URL(request.url || '/', 'http://localhost').pathname;
  const archive = pathname === '/api/jobs/import' || pathname === '/api/datasets/import-archive';
  const binary = [
    'application/octet-stream',
    'application/zip',
    'application/x-zip-compressed',
    'multipart/form-data',
  ].includes(contentType);
  if (!pathname.startsWith('/api/') || (binary && binaryUploadPaths.has(pathname))) {
    request.pipe(upstream);
    return;
  }
  const binaryArchive = binary && archive;
  const limit = binaryArchive ? 64 * 1024 ** 2 : pathname === '/api/auth' ? 4096 : 2 * 1024 ** 2;
  let bytes = 0;
  let timer: ReturnType<typeof setTimeout>;
  let rejected = false;
  const reject = (status: number, message: string) => {
    if (rejected) return;
    rejected = true;
    clearTimeout(timer);
    request.unpipe(guard);
    guard.unpipe(upstream);
    if (!response.headersSent) {
      const body = JSON.stringify({ error: message });
      response.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) });
      response.end(body);
    }
    upstream.destroy();
    // Closing while the client is still writing can replace the HTTP error with
    // ECONNRESET. Discard without buffering so the response can reach the client,
    // but stop a peer that keeps uploading after rejection within five seconds.
    const drainTimer = setTimeout(() => request.destroy(), 5000);
    drainTimer.unref();
    const finishDrain = () => clearTimeout(drainTimer);
    request.once('end', finishDrain);
    request.once('close', finishDrain);
    if (request.readableEnded || request.destroyed) finishDrain();
    request.resume();
  };
  const guard = new Transform({
    transform(chunk: Buffer, _encoding, callback) {
      bytes += chunk.length;
      if (bytes > limit) {
        reject(413, 'Command body is too large');
        callback();
      } else callback(null, chunk);
    },
  });
  timer = setTimeout(() => reject(408, 'Command body timed out'), binaryArchive ? 600000 : 15000);
  const clear = () => clearTimeout(timer);
  request.once('end', clear);
  request.once('aborted', clear);
  response.once('close', clear);
  if (Number(request.headers['content-length'] || 0) > limit) {
    reject(413, 'Command body is too large');
    return;
  }
  request.pipe(guard).pipe(upstream);
}
