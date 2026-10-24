import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { once } from 'node:events';
import http from 'node:http';
import test from 'node:test';
import { Hono } from 'hono';
import { getRequestListener } from '@hono/node-server';
import ts from 'typescript';

const code = ts.transpileModule(readFileSync(new URL('../src/server/http/apiRoutes.ts', import.meta.url), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const module = { exports: {} };
new Function('exports', code)(module.exports);
const { registerRoute, routeParameter } = module.exports;

async function fixture(t, configure) {
  const app = new Hono({ strict: false });
  configure(app);
  const server = http.createServer(getRequestListener(app.fetch, { overrideGlobalObjects: false }));
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  return `http://127.0.0.1:${server.address().port}`;
}

test('HTTP adapter preserves cookies, lazy route loading, method policy and decoded parameters', async t => {
  let loads = 0;
  const origin = await fixture(t, app => {
    registerRoute(app, '/jobs/:jobID', ['GET'], async () => {
      loads++;
      return { GET: async (_request, { params }) => Response.json(await params, {
        headers: { 'Set-Cookie': 'session=example; HttpOnly; SameSite=Strict' },
      }) };
    }, context => ({ jobID: routeParameter(context, 'jobID') }));
  });
  const denied = await fetch(`${origin}/jobs/a`, { method: 'POST', body: 'never parsed' });
  assert.equal(denied.status, 405);
  assert.equal(loads, 0);
  assert.equal(denied.headers.get('allow'), 'GET, HEAD, OPTIONS');
  assert.equal((await fetch(`${origin}/jobs/a`, { method: 'OPTIONS' })).status, 204);
  assert.equal(loads, 0);
  const response = await fetch(`${origin}/jobs/caf%C3%A9`);
  assert.deepEqual(await response.json(), { jobID: 'café' });
  assert.match(response.headers.get('set-cookie'), /HttpOnly; SameSite=Strict/);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  assert.equal(loads, 1);
});

test('HTTP adapter passes streaming request bodies and streaming responses without JSON conversion', async t => {
  const origin = await fixture(t, app => {
    registerRoute(app, '/upload', ['POST'], async () => ({ POST: request =>
      new Response(request.body, { headers: { 'Content-Type': 'application/octet-stream' } }),
    }), () => ({}));
  });
  const source = new Uint8Array(2 * 1024 * 1024).fill(71);
  const response = await fetch(`${origin}/upload`, { method: 'POST', body: source });
  assert.equal(response.status, 200);
  assert.deepEqual(new Uint8Array(await response.arrayBuffer()), source);
});

test('HEAD cancels the unused response stream and keeps its metadata', async t => {
  let canceled = false;
  const origin = await fixture(t, app => {
    registerRoute(app, '/events', ['GET'], async () => ({ GET: () => new Response(new ReadableStream({
      start(controller) { controller.enqueue(new TextEncoder().encode('data: ready\n\n')); },
      cancel() { canceled = true; },
    }), { headers: { 'Content-Type': 'text/event-stream' } }) }), () => ({}));
  });
  const response = await fetch(`${origin}/events`, { method: 'HEAD' });
  assert.equal(await response.text(), '');
  assert.equal(response.headers.get('content-type'), 'text/event-stream');
  assert.equal(canceled, true);
});

test('disconnecting an event stream cancels its producer', async t => {
  let finish;
  const canceled = new Promise(resolve => { finish = resolve; });
  const origin = await fixture(t, app => {
    registerRoute(app, '/events', ['GET'], async () => ({ GET: () => new Response(new ReadableStream({
      start(controller) { controller.enqueue(new TextEncoder().encode('data: ready\n\n')); },
      cancel() { finish(); },
    }), { headers: { 'Content-Type': 'text/event-stream' } }) }), () => ({}));
  });
  const controller = new AbortController();
  const response = await fetch(`${origin}/events`, { signal: controller.signal });
  const reader = response.body.getReader();
  assert.equal((await reader.read()).done, false);
  controller.abort();
  await reader.cancel().catch(() => {});
  let timer;
  try { await Promise.race([canceled, new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Stream was not canceled')), 3000); })]); }
  finally { clearTimeout(timer); }
});
