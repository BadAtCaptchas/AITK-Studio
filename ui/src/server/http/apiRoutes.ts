import type { Context, Hono } from 'hono';

export const HTTP_METHODS = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS'] as const;
export type HttpMethod = (typeof HTTP_METHODS)[number];
type Handler<Params> = (request: Request, context: { params: Promise<Params> }) => Response | Promise<Response>;

export function routeParameter(context: Context, name: string): string {
  const value = context.req.param(name);
  if (value === undefined) throw new URIError(`Missing route parameter: ${name}`);
  return value;
}

/** The generated registry supplies typed parameter names for every existing handler. */
export function registerRoute<Params extends Record<string, string | string[]>>(
  app: Hono,
  pattern: string,
  methods: HttpMethod[],
  load: () => Promise<Partial<Record<HttpMethod, Handler<Params>>>>,
  parameters: (context: Context) => Params,
): void {
  const allowed = new Set(methods);
  if (allowed.has('GET')) allowed.add('HEAD');
  allowed.add('OPTIONS');
  const allow = [...allowed].join(', ');
  app.all(pattern, async context => {
    const method = context.req.method;
    if (method === 'OPTIONS') return new Response(null, { status: 204, headers: { Allow: allow } });
    const selected = HTTP_METHODS.find(candidate => candidate === method);
    if (!selected || !allowed.has(selected))
      return Response.json({ error: 'Method not allowed' }, { status: 405, headers: { Allow: allow } });
    const handlers = await load();
    const handler = handlers[selected] ?? (selected === 'HEAD' ? handlers.GET : undefined);
    if (!handler) return Response.json({ error: 'Method not allowed' }, { status: 405, headers: { Allow: allow } });
    const response = await handler(context.req.raw, { params: Promise.resolve(parameters(context)) });
    const headers = new Headers(response.headers);
    if (!headers.has('Cache-Control')) headers.set('Cache-Control', 'no-store');
    if (method === 'HEAD') await response.body?.cancel();
    return new Response(method === 'HEAD' ? null : response.body, {
      status: response.status, statusText: response.statusText, headers,
    });
  });
}
