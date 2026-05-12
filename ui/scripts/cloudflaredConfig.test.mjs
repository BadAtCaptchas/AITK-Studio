import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const modulePath = require.resolve('../dist/src/server/cloudflared.js');
const envKeys = [
  'AITK_CLOUDFLARED_ENABLED',
  'AITK_CLOUDFLARED_PUBLIC_URL',
  'AITK_CLOUDFLARED_TOKEN_FILE',
  'AITK_CLOUDFLARED_BIN',
  'AITK_CLOUDFLARED_METRICS_ADDR',
  'AITK_CLOUDFLARED_LOG_LEVEL',
  'AI_TOOLKIT_AUTH',
];
const originalEnv = Object.fromEntries(envKeys.map(key => [key, process.env[key]]));

function resetEnv() {
  for (const key of envKeys) {
    if (originalEnv[key] === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = originalEnv[key];
    }
  }
  delete require.cache[modulePath];
}

function loadCloudflared() {
  delete require.cache[modulePath];
  return require(modulePath);
}

afterEach(resetEnv);

test('cloudflared config is disabled by default', () => {
  for (const key of envKeys) delete process.env[key];
  const cloudflared = loadCloudflared();

  const config = cloudflared.getCloudflaredConfig();

  assert.equal(config.enabled, false);
  assert.equal(config.bin, 'cloudflared');
  assert.equal(config.metricsAddr, '127.0.0.1:60123');
});

test('cloudflared config reads managed tunnel env vars without exposing token content', () => {
  process.env.AITK_CLOUDFLARED_ENABLED = '1';
  process.env.AITK_CLOUDFLARED_PUBLIC_URL = 'https://worker.example.com';
  process.env.AITK_CLOUDFLARED_TOKEN_FILE = '/tmp/tunnel-token';
  process.env.AITK_CLOUDFLARED_BIN = '/usr/local/bin/cloudflared';
  process.env.AITK_CLOUDFLARED_METRICS_ADDR = '127.0.0.1:60222';
  process.env.AITK_CLOUDFLARED_LOG_LEVEL = 'warn';

  const cloudflared = loadCloudflared();
  const config = cloudflared.getCloudflaredConfig();

  assert.equal(config.enabled, true);
  assert.equal(config.configured, true);
  assert.equal(config.publicUrl, 'https://worker.example.com');
  assert.equal(config.tokenFile, '/tmp/tunnel-token');
  assert.equal(config.metricsAddr, '127.0.0.1:60222');
  assert.equal(config.logLevel, 'warn');
  assert.equal(JSON.stringify(config).includes('eyJ'), false);
});

test('cloudflared status reports auth requirement when enabled', async () => {
  process.env.AITK_CLOUDFLARED_ENABLED = '1';
  process.env.AITK_CLOUDFLARED_PUBLIC_URL = 'https://worker.example.com';
  process.env.AITK_CLOUDFLARED_TOKEN_FILE = '/tmp/tunnel-token';
  delete process.env.AI_TOOLKIT_AUTH;

  const cloudflared = loadCloudflared();
  const status = await cloudflared.getCloudflaredStatus();

  assert.equal(status.enabled, true);
  assert.equal(status.running, false);
  assert.match(status.error, /AI_TOOLKIT_AUTH/);
});
