import http from 'node:http';
import os from 'node:os';
import fs from 'node:fs/promises';
import { createRequire } from 'node:module';
import { getRequestListener } from '@hono/node-server';
import { serveStatic } from '@hono/node-server/serve-static';
import { loadEnvironment } from './load-env.mjs';

const require = createRequire(import.meta.url);
const dev = process.argv.includes('dev');
process.env.NODE_ENV ||= dev ? 'development' : 'production';
loadEnvironment(process.cwd(), dev ? 'development' : 'production');
const port = Number(process.argv[process.argv.indexOf('--port') + 1]);
if (!Number.isSafeInteger(port) || port < 1 || port > 65535) throw new Error('Invalid server port');
if (dev) await (await import('./build-server.mjs')).buildServer();
function loadApplication() {
  const filename = require.resolve('../.runtime/server.cjs');
  delete require.cache[filename];
  const { createApiApp } = require(filename);
  const app = createApiApp();
  app.get('/app-config.js', context => context.body(
    `window.aitkConfig=${JSON.stringify({ authRequired: Boolean(process.env.AI_TOOLKIT_AUTH) })};window.server_platform=${JSON.stringify(os.platform())};`,
    200, { 'Content-Type': 'text/javascript; charset=utf-8', 'Cache-Control': 'no-store' },
  ));
  app.all('/api/*', context => context.json({ error: 'Not found' }, 404));
  return app;
}
const app = loadApplication();

let vite;
let stopWatching;
let listener;
const server = http.createServer((request, response) => listener(request, response));
let apiListener = getRequestListener(app.fetch, { overrideGlobalObjects: false });
if (dev) {
  const { createServer } = await import('vite');
  const { watchApi } = await import('./watch-api.mjs');
  vite = await createServer({
    server: { middlewareMode: true, ws: { server } },
    appType: 'spa',
    plugins: [{
      name: 'studio-api-reload',
      configureServer(developmentServer) {
        stopWatching?.();
        stopWatching = watchApi(developmentServer, () => {
          apiListener = getRequestListener(loadApplication().fetch, { overrideGlobalObjects: false });
        });
      },
    }],
  });
  listener = (request, response) => {
    const pathname = new URL(request.url || '/', 'http://localhost').pathname;
    if (pathname.startsWith('/api/') || pathname === '/app-config.js') return apiListener(request, response);
    vite.middlewares(request, response);
  };
} else {
  const html = await fs.readFile(new URL('../web-dist/index.html', import.meta.url), 'utf8');
  app.use('*', serveStatic({
    root: './web-dist',
    onFound: (filename, context) => {
      context.header('Cache-Control', filename.includes('/assets/') ? 'public, max-age=31536000, immutable' : 'no-cache');
      context.header('X-Content-Type-Options', 'nosniff');
    },
  }));
  app.get('*', context => {
    if (!context.req.header('Accept')?.includes('text/html')) return context.json({ error: 'Not found' }, 404);
    context.header('Cache-Control', 'no-cache');
    return context.html(html);
  });
  listener = apiListener;
}

server.requestTimeout = 0;
server.headersTimeout = 60_000;
server.listen(port, '127.0.0.1', () => console.log(`AITK Studio ready on port ${port}`));
let stopping = false;
async function stop() {
  if (stopping) return;
  stopping = true;
  stopWatching?.();
  const timeout = setTimeout(() => process.exit(0), 4000);
  timeout.unref();
  await vite?.close();
  server.close(() => process.exit(0));
  server.closeIdleConnections();
}
process.once('SIGTERM', stop);
process.once('SIGINT', stop);
process.once('disconnect', stop);
process.on('message', message => { if (message?.type === 'aitk-shutdown') void stop(); });
