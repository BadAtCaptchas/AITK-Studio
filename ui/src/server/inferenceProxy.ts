import { commandError, readJsonCommand, CommandInputError } from './commandInput';
import { getInferenceEndpoint } from './inferenceEngine';

export async function proxyInferenceRequest(request: Request, segments: string[]): Promise<Response> {
  try {
    const route = segments.join('/');
    const allowed =
      request.method === 'GET'
        ? /^(health|models|queue|stream\/[a-f0-9-]+|outputs\/[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+)$/
        : /^(generate|unload|assets|cancel\/[a-f0-9-]+)$/;
    if (!allowed.test(route) || segments.some(s => s === '.' || s === '..'))
      throw new CommandInputError('Unknown engine route', 404);
    const url = new URL(request.url);
    const endpoint = await getInferenceEndpoint(url.searchParams.get('job_id') || '');
    if (!endpoint) return Response.json({ error: 'Engine is not ready' }, { status: 503 });
    const target = new URL(`${endpoint.url}/${segments.map(encodeURIComponent).join('/')}`);
    const headers = new Headers({ 'X-Engine-Token': endpoint.token });
    let body: BodyInit | undefined;
    if (request.method === 'POST') {
      if (route === 'assets') {
        const maxBytes = 128 * 1024 * 1024;
        const reader = request.body?.getReader();
        const chunks: Uint8Array<ArrayBuffer>[] = [];
        let size = 0;
        try {
          if (reader)
            for (;;) {
              const part = await reader.read();
              if (part.done) break;
              size += part.value.byteLength;
              if (size > maxBytes) throw new CommandInputError('Media upload is too large', 413);
              chunks.push(new Uint8Array(part.value));
            }
        } finally {
          await reader?.cancel().catch(() => undefined);
        }
        body = new Blob(chunks);
        headers.set('Content-Type', 'application/octet-stream');
        target.searchParams.set('name', url.searchParams.get('name') || 'asset.png');
      } else {
        body = JSON.stringify(await readJsonCommand(request, { allowEmpty: true }));
        headers.set('Content-Type', 'application/json');
      }
    }
    const signal = /^(health|models|queue)$/.test(route)
      ? AbortSignal.any([request.signal, AbortSignal.timeout(15000)])
      : request.signal;
    const response = await fetch(target, { method: request.method, headers, body, redirect: 'error', signal });
    const outgoing = new Headers({
      'Cache-Control': 'no-store',
      'Content-Type': response.headers.get('content-type') || 'application/octet-stream',
    });
    const requestID = response.headers.get('x-request-id');
    if (requestID) outgoing.set('X-Request-Id', requestID);
    return new Response(response.body, { status: response.status, headers: outgoing });
  } catch (error) {
    const invalid = commandError(error);
    return Response.json(
      { error: invalid?.error || 'Could not contact the inference engine' },
      { status: invalid?.status || 502 },
    );
  }
}
