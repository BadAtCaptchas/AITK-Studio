import { gzip as gzipCallback } from 'node:zlib';
import { promisify } from 'node:util';

const gzip = promisify(gzipCallback);

function acceptsGzip(header: string | null): boolean {
  const encodings = (header ?? '').split(',').map(part => {
    const [name, ...parameters] = part.trim().toLowerCase().split(';');
    const quality = parameters.find(parameter => parameter.trim().startsWith('q='));
    const q = quality === undefined ? 1 : Number(quality.trim().slice(2));
    return { name: name.trim(), q: Number.isFinite(q) && q >= 0 && q <= 1 ? q : 0 };
  });
  // An explicit refusal takes precedence over a wildcard.
  const explicit = encodings.filter(encoding => encoding.name === 'gzip');
  return (explicit.length ? explicit : encodings.filter(encoding => encoding.name === '*'))
    .some(encoding => encoding.q > 0);
}

/** Compress the existing wire contract, with a limit on decoded bytes, not gzip size. */
export async function metricJsonResponse(request: Request, value: unknown, maxBytes: number): Promise<Response> {
  const body = Buffer.from(JSON.stringify(value));
  const headers = { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', Vary: 'Accept-Encoding' };
  if (body.byteLength > maxBytes) {
    return Response.json({ error: 'Metric response is too large; request fewer points or keys.' }, { status: 413, headers });
  }
  if (body.byteLength >= 1024 && acceptsGzip(request.headers.get('accept-encoding'))) {
    const compressed = await gzip(body, { level: 1 });
    return new Response(new Uint8Array(compressed), { headers: { ...headers, 'Content-Encoding': 'gzip' } });
  }
  return new Response(new Uint8Array(body), { headers });
}
