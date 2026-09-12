// Run after npm prune --omit=dev. Uses only production dependencies and an isolated DB.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { once } from 'node:events';

const server = spawn(process.execPath, ['scripts/serve-e2e.mjs'], {
  stdio: ['ignore', 'inherit', 'inherit', 'ipc'],
  windowsHide: true,
});
const exited = once(server, 'exit');
const origin = 'http://127.0.0.1:15875';
try {
  let ready = false;
  for (let attempt = 0; attempt < 120 && server.exitCode === null; attempt++) {
    try {
      ready = (await fetch(origin, { signal: AbortSignal.timeout(1000) })).ok;
    } catch {
      /* Startup. */
    }
    if (ready) break;
    await delay(500);
  }
  assert.ok(ready, 'Production services did not start');
  const login = await fetch(origin + '/api/auth', {
    method: 'POST',
    headers: { Origin: origin, 'Content-Type': 'application/json' },
    body: JSON.stringify({ token: 'browser-fixture-local-only' }),
  });
  assert.equal(login.status, 200);
  const cookie = login.headers.get('set-cookie')?.split(';')[0];
  assert.ok(cookie);
  const jobs = await fetch(origin + '/api/jobs?limit=1', { headers: { Cookie: cookie } });
  assert.equal(jobs.status, 200);
  assert.deepEqual((await jobs.json()).jobs, []);
  console.log('Production-only stack authentication and database read passed.');
} finally {
  if (server.connected) server.send({ type: 'aitk-test-shutdown' });
  else server.kill('SIGTERM');
  const deadline = new AbortController();
  try {
    await Promise.race([
      exited,
      delay(30000, undefined, { signal: deadline.signal }).then(() => { throw new Error('Test stack failed to shut down'); }),
    ]);
  } finally { deadline.abort(); }
}
await assert.rejects(fetch(origin, { signal: AbortSignal.timeout(1000) }), 'Test port must close');
