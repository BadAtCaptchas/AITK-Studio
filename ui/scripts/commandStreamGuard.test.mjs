import assert from 'node:assert/strict';
import http from 'node:http';
import net from 'node:net';
import { once } from 'node:events';
import { setTimeout as delay } from 'node:timers/promises';
import { test } from 'node:test';
import { pipeCommandBody } from '../dist/cron/commandStreamGuard.js';

async function fixture(t) {
  let upstreamBytes = 0;
  const upstream = http.createServer((req, res) => {
    req.on('data', chunk => {
      upstreamBytes += chunk.length;
    });
    req.on('end', () => res.end('accepted'));
  });
  upstream.listen(0, '127.0.0.1');
  await once(upstream, 'listening');
  const front = http.createServer((req, res) => {
    const proxied = http.request(
      { host: '127.0.0.1', port: upstream.address().port, method: req.method, path: req.url, headers: req.headers },
      incoming => {
        if (res.headersSent || res.destroyed) {
          incoming.resume();
          return;
        }
        res.writeHead(incoming.statusCode, incoming.headers);
        incoming.pipe(res);
      },
    );
    proxied.on('error', error => {
      if (!res.headersSent) res.destroy(error);
    });
    pipeCommandBody(req, res, proxied);
  });
  front.listen(0, '127.0.0.1');
  await once(front, 'listening');
  t.after(async () => {
    await Promise.all(
      [front, upstream].map(
        server =>
          new Promise(resolve => {
            server.close(resolve);
            server.closeAllConnections();
          }),
      ),
    );
  });
  return { port: front.address().port, upstreamBytes: () => upstreamBytes };
}

async function upload(
  port,
  { declared = true, path = '/api/jobs', size = 2 * 1024 ** 2 + 16, contentType = 'application/json' } = {},
) {
  const socket = net.createConnection({ host: '127.0.0.1', port });
  try {
    await once(socket, 'connect');
    const response = new Promise((resolve, reject) => {
      let received = '';
      socket.on('error', reject);
      socket.on('data', chunk => {
        received += chunk.toString();
        const boundary = received.indexOf('\r\n\r\n');
        if (boundary < 0) return;
        const header = received.slice(0, boundary);
        const length = Number(header.match(/content-length: (\d+)/i)?.[1]);
        const body = received.slice(boundary + 4);
        if (Number.isFinite(length) && Buffer.byteLength(body) >= length) {
          resolve({ status: Number(header.split(' ')[1]), body });
        }
      });
      socket.on('end', () => reject(new Error('Connection closed before response completed')));
    });
    const send = (async () => {
      socket.write(
        'POST ' +
          path +
          ' HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: ' +
          contentType +
          '\r\n' +
          (declared ? 'Content-Length: ' + size : 'Transfer-Encoding: chunked') +
          '\r\n\r\n',
      );
      for (let sent = 0; sent < size; sent += 64 * 1024) {
        const chunk = Buffer.alloc(Math.min(64 * 1024, size - sent), 'x');
        if (!declared) socket.write(chunk.length.toString(16) + '\r\n');
        if (!socket.write(chunk)) await once(socket, 'drain');
        if (!declared) socket.write('\r\n');
        // Keep writing after the early rejection to exercise the socket-close race.
        await delay(1);
      }
      if (!declared) socket.write('0\r\n\r\n');
    })();
    const [result] = await Promise.all([response, send]);
    return result;
  } finally {
    socket.destroy();
  }
}

test('declared oversized commands return 413 while the client finishes uploading', { timeout: 10_000 }, async t => {
  const server = await fixture(t);
  for (let attempt = 0; attempt < 3; attempt++) {
    const response = await upload(server.port);
    assert.equal(response.status, 413);
    assert.match(response.body, /too large/);
  }
  assert.equal(server.upstreamBytes(), 0);
});

test('chunked command intake stops forwarding at the actual byte limit', { timeout: 10_000 }, async t => {
  const server = await fixture(t);
  assert.equal((await upload(server.port, { declared: false })).status, 413);
  assert.ok(server.upstreamBytes() <= 2 * 1024 ** 2);
});

test('authentication has a smaller cap and valid commands pass through', { timeout: 10_000 }, async t => {
  const server = await fixture(t);
  assert.equal((await upload(server.port, { path: '/api/auth', size: 4097 })).status, 413);
  assert.equal(server.upstreamBytes(), 0);
  assert.deepEqual(await upload(server.port, { size: 100 }), { status: 200, body: 'accepted' });
  assert.equal(server.upstreamBytes(), 100);
});

test(
  'binary content types cannot bypass command caps and real uploads retain their streaming path',
  { timeout: 10_000 },
  async t => {
    const server = await fixture(t);
    const contentType = 'application/octet-stream';
    assert.equal((await upload(server.port, { path: '/api/settings', contentType })).status, 413);
    assert.equal(server.upstreamBytes(), 0);
    assert.deepEqual(await upload(server.port, { path: '/api/img/upload', contentType }), {
      status: 200,
      body: 'accepted',
    });
    assert.equal(server.upstreamBytes(), 2 * 1024 ** 2 + 16);
  },
);
