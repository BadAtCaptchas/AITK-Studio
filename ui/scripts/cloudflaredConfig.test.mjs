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
  'AITK_CLOUDFLARED_AUTO_DOWNLOAD',
  'AITK_CLOUDFLARED_TARGET_URL',
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
  assert.equal(config.autoDownload, false);
  assert.equal(config.mode, 'quick');
  assert.equal(config.targetUrl, 'http://127.0.0.1:8675');
  assert.equal(config.metricsAddr, '127.0.0.1:60123');
});

test('cloudflared config reads managed tunnel env vars without exposing token content', () => {
  process.env.AITK_CLOUDFLARED_ENABLED = '1';
  process.env.AITK_CLOUDFLARED_PUBLIC_URL = 'https://worker.example.com';
  process.env.AITK_CLOUDFLARED_TOKEN_FILE = '/tmp/tunnel-token';
  process.env.AITK_CLOUDFLARED_BIN = '/usr/local/bin/cloudflared';
  process.env.AITK_CLOUDFLARED_AUTO_DOWNLOAD = '1';
  process.env.AITK_CLOUDFLARED_METRICS_ADDR = '127.0.0.1:60222';
  process.env.AITK_CLOUDFLARED_LOG_LEVEL = 'warn';

  const cloudflared = loadCloudflared();
  const config = cloudflared.getCloudflaredConfig();

  assert.equal(config.enabled, true);
  assert.equal(config.configured, true);
  assert.equal(config.mode, 'named');
  assert.equal(config.publicUrl, 'https://worker.example.com');
  assert.equal(config.tokenFile, '/tmp/tunnel-token');
  assert.equal(config.autoDownload, true);
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

test('cloudflared quick tunnel mode does not require a known public URL', async () => {
  process.env.AITK_CLOUDFLARED_ENABLED = '1';
  process.env.AITK_CLOUDFLARED_BIN = process.execPath;
  process.env.AITK_CLOUDFLARED_TARGET_URL = 'http://127.0.0.1:9999';
  process.env.AI_TOOLKIT_AUTH = 'test-auth';
  delete process.env.AITK_CLOUDFLARED_PUBLIC_URL;
  delete process.env.AITK_CLOUDFLARED_TOKEN_FILE;

  const cloudflared = loadCloudflared();
  const status = await cloudflared.getCloudflaredStatus();
  const args = cloudflared.buildCloudflaredArgs(cloudflared.getCloudflaredConfig());

  assert.equal(status.enabled, true);
  assert.equal(status.mode, 'quick');
  assert.equal(status.error, null);
  assert.equal(status.targetUrl, 'http://127.0.0.1:9999');
  assert.deepEqual(args.slice(-2), ['--url', 'http://127.0.0.1:9999']);
});

test('cloudflared named tunnel mode uses token-file arguments', () => {
  process.env.AITK_CLOUDFLARED_ENABLED = '1';
  process.env.AITK_CLOUDFLARED_TOKEN_FILE = '/tmp/tunnel-token';

  const cloudflared = loadCloudflared();
  const args = cloudflared.buildCloudflaredArgs(cloudflared.getCloudflaredConfig());

  assert.deepEqual(args.slice(-3), ['run', '--token-file', '/tmp/tunnel-token']);
});

test('cloudflared download info uses pinned release assets and checksums for common platforms', () => {
  const cloudflared = loadCloudflared();

  const linux = cloudflared.getCloudflaredDownloadInfoForPlatform('linux', 'x64');
  assert.equal(linux.supported, true);
  assert.equal(linux.assetName, 'cloudflared-linux-amd64');
  assert.equal(
    linux.url,
    'https://github.com/cloudflare/cloudflared/releases/download/2026.5.2/cloudflared-linux-amd64',
  );
  assert.equal(linux.expectedSha256, '5286698547f03df745adb2355f04c12dde52ef425491e81f433642d695521886');

  const windows = cloudflared.getCloudflaredDownloadInfoForPlatform('win32', 'x64');
  assert.equal(windows.supported, true);
  assert.equal(windows.assetName, 'cloudflared-windows-amd64.exe');
  assert.equal(windows.expectedSha256, '20b9638f685333d623798e733effbad2487093f15ba592f6c7752360ff3b7ab7');

  const darwinAmd64 = cloudflared.getCloudflaredDownloadInfoForPlatform('darwin', 'x64');
  assert.equal(darwinAmd64.supported, true);
  assert.equal(darwinAmd64.assetName, 'cloudflared-darwin-amd64.tgz');
  assert.equal(darwinAmd64.archive, 'tgz');
  assert.equal(darwinAmd64.expectedSha256, '7240f709506bc2c1eb9da4d89cf2555499c60280ecb854b7d80e8f17d4b7903d');

  const darwinArm64 = cloudflared.getCloudflaredDownloadInfoForPlatform('darwin', 'arm64');
  assert.equal(darwinArm64.supported, true);
  assert.equal(darwinArm64.assetName, 'cloudflared-darwin-arm64.tgz');
  assert.equal(darwinArm64.archive, 'tgz');
  assert.equal(darwinArm64.expectedSha256, 'ba94054c9fd4297645093d59d51442e5e546d07bb0516120e694a13d5b216d38');
});

test('cloudflared downloader rejects non-HTTPS and untrusted redirect hosts', () => {
  const cloudflared = loadCloudflared();

  assert.throws(
    () => cloudflared.assertCloudflaredDownloadUrlIsTrusted(new URL('http://github.com/cloudflare/cloudflared')),
    /non-HTTPS/,
  );
  assert.throws(
    () => cloudflared.assertCloudflaredDownloadUrlIsTrusted(new URL('https://example.com/cloudflared')),
    /untrusted host/,
  );
  assert.doesNotThrow(() =>
    cloudflared.assertCloudflaredDownloadUrlIsTrusted(
      new URL('https://release-assets.githubusercontent.com/github-production-release-asset/cloudflared'),
    ),
  );
});
