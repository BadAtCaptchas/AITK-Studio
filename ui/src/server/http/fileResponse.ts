import fs from 'fs';
import { Readable } from 'stream';
import { FILE_STREAM_BUFFER_BYTES, type FileResponseResolution } from '../fileServing';

export function fileResponse(request: Request, resolution: FileResponseResolution) {
  if (request.signal.aborted) return new Response(null, { status: 499 });
  if (resolution.kind === 'proxy') {
    return new Response('Internal Server Error', { status: 500 });
  }

  if (!resolution.file || request.method === 'HEAD') {
    return new Response(resolution.body, {
      status: resolution.status,
      headers: resolution.headers,
    });
  }

  const stream = fs.createReadStream(resolution.file.path, {
    ...(resolution.file.start !== undefined ? { start: resolution.file.start } : {}),
    ...(resolution.file.end !== undefined ? { end: resolution.file.end } : {}),
    highWaterMark: FILE_STREAM_BUFFER_BYTES,
  });
  const body = Readable.toWeb(stream) as ReadableStream<Uint8Array>;
  const onAbort = () => stream.destroy();
  if (request.signal.aborted) {
    stream.destroy();
  } else {
    request.signal.addEventListener('abort', onAbort, { once: true });
  }
  stream.once('close', () => request.signal.removeEventListener('abort', onAbort));

  return new Response(body, {
    status: resolution.status,
    headers: resolution.headers,
  });
}
