import { Hono } from 'hono';
import { registerApiRoutes } from '../../../.runtime/api-routes';
import { proxy as authorizeApiRequest } from '../../proxy';
import { browserRequestError } from '../ingressPolicy';
import { startMonitor } from '../monitor';

export function createApiApp(): Hono {
  startMonitor();
  const app = new Hono({ strict: false });
  app.use('*', async (context, next) => {
    try { decodeURI(new URL(context.req.url).pathname); }
    catch { return context.json({ error: 'Invalid path' }, 400); }
    if (context.req.path.startsWith('/api/')) {
      const rejected = await authorizeApiRequest(context.req.raw);
      if (rejected) return rejected;
    } else {
      const error = browserRequestError(context.req.raw.headers, context.req.url, context.req.method);
      if (error) return context.json({ error }, 403);
    }
    await next();
  });
  registerApiRoutes(app);
  app.notFound(context => context.json({ error: 'Not found' }, 404));
  app.onError((error, context) => {
    if (error instanceof URIError) return context.json({ error: 'Invalid path' }, 400);
    console.error('Request failed:', error);
    return context.json({ error: 'The request could not be completed. Please try again.' }, 500);
  });
  return app;
}
