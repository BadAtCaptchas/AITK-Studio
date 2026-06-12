import crypto from 'crypto';
import fs from 'fs';
import fsp from 'fs/promises';
import https from 'https';
import path from 'path';
import { spawn, type ChildProcess } from 'child_process';
import { pipeline } from 'stream/promises';
import zlib from 'zlib';
import { TOOLKIT_ROOT } from '../paths';

export type CloudflaredStatus = {
  configured: boolean;
  enabled: boolean;
  mode: 'named' | 'quick';
  detected: boolean;
  bin: string;
  downloadAvailable: boolean;
  downloadUrl: string | null;
  installPath: string;
  running: boolean;
  pid: number | null;
  publicUrl: string | null;
  targetUrl: string;
  metricsAddr: string;
  message: string;
  error: string | null;
};

const DEFAULT_METRICS_ADDR = '127.0.0.1:60123';
const DEFAULT_TARGET_URL = 'http://127.0.0.1:8675';
const PID_FILE = path.join(TOOLKIT_ROOT, '.cloudflared.pid');
const URL_FILE = path.join(TOOLKIT_ROOT, '.cloudflared.url');
const LOG_FILE = path.join(TOOLKIT_ROOT, '.cloudflared.log');
const LOCAL_BIN_DIR = path.join(TOOLKIT_ROOT, 'bin');
const LOCAL_BIN_PATH = path.join(LOCAL_BIN_DIR, process.platform === 'win32' ? 'cloudflared.exe' : 'cloudflared');
const CLOUDFLARED_RELEASE_VERSION = '2026.5.2';
const CLOUDFLARED_RELEASE_BASE = `https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_RELEASE_VERSION}`;
const CLOUDFLARED_DOWNLOAD_SHA256: Record<string, string> = {
  'cloudflared-darwin-amd64.tgz': '7240f709506bc2c1eb9da4d89cf2555499c60280ecb854b7d80e8f17d4b7903d',
  'cloudflared-darwin-arm64.tgz': 'ba94054c9fd4297645093d59d51442e5e546d07bb0516120e694a13d5b216d38',
  'cloudflared-linux-386': 'ad82d1dbed8bbb9d702807cbd97df932cc774d29e9da5c109b7a3c7f7aee2065',
  'cloudflared-linux-amd64': '5286698547f03df745adb2355f04c12dde52ef425491e81f433642d695521886',
  'cloudflared-linux-arm': '70a4c869a037bd69af6ce2ad0c4da4a7680d94fcfb8d4c70ecddae24d560762f',
  'cloudflared-linux-arm64': '5a4e8ce2701105271412059f44b6a0bf1ae4542b4d98ff3180c0c019443a5815',
  'cloudflared-windows-386.exe': '6736615e8d2b3b61e868e32907e85641b4ec7b2b8c26bd3361ec15e56e53e242',
  'cloudflared-windows-amd64.exe': '20b9638f685333d623798e733effbad2487093f15ba592f6c7752360ff3b7ab7',
};
const TRUSTED_CLOUDFLARED_DOWNLOAD_HOSTS = new Set(['github.com', 'objects.githubusercontent.com', 'release-assets.githubusercontent.com']);
const DOWNLOAD_MAX_REDIRECTS = 5;
const DOWNLOAD_TIMEOUT_MS = 120_000;

export type CloudflaredDownloadInfo = {
  supported: boolean;
  assetName: string | null;
  url: string | null;
  archive: 'none' | 'tgz';
  installPath: string;
  expectedSha256: string | null;
  reason: string | null;
};

export type CloudflaredDownloadResult = CloudflaredDownloadInfo & {
  downloaded: boolean;
  version: string | null;
};

function boolEnv(value: string | undefined) {
  return value === '1' || value?.toLowerCase() === 'true';
}

function hasExplicitCloudflaredBin() {
  return Boolean(process.env.AITK_CLOUDFLARED_BIN?.trim());
}

function getConfiguredCloudflaredBin() {
  const explicitBin = process.env.AITK_CLOUDFLARED_BIN?.trim();
  if (explicitBin) return explicitBin;
  return fs.existsSync(LOCAL_BIN_PATH) ? LOCAL_BIN_PATH : 'cloudflared';
}

export function getCloudflaredConfig() {
  const enabled = boolEnv(process.env.AITK_CLOUDFLARED_ENABLED);
  const bin = getConfiguredCloudflaredBin();
  const autoDownload = boolEnv(process.env.AITK_CLOUDFLARED_AUTO_DOWNLOAD);
  const publicUrl = process.env.AITK_CLOUDFLARED_PUBLIC_URL?.trim() || null;
  const tokenFile = process.env.AITK_CLOUDFLARED_TOKEN_FILE?.trim() || null;
  const mode: 'named' | 'quick' = tokenFile ? 'named' : 'quick';
  const targetUrl = process.env.AITK_CLOUDFLARED_TARGET_URL?.trim() || DEFAULT_TARGET_URL;
  const metricsAddr = process.env.AITK_CLOUDFLARED_METRICS_ADDR?.trim() || DEFAULT_METRICS_ADDR;
  const logLevel = process.env.AITK_CLOUDFLARED_LOG_LEVEL?.trim() || 'info';

  return {
    enabled,
    bin,
    autoDownload,
    mode,
    publicUrl,
    tokenFile,
    targetUrl,
    metricsAddr,
    logLevel,
    configured: enabled && (mode === 'quick' || Boolean(tokenFile)),
  };
}

function archToCloudflaredArch(arch: NodeJS.Architecture) {
  if (arch === 'x64') return 'amd64';
  if (arch === 'ia32') return '386';
  if (arch === 'arm64') return 'arm64';
  if (arch === 'arm') return 'arm';
  return null;
}

export function getCloudflaredDownloadInfoForPlatform(
  platform: NodeJS.Platform = process.platform,
  arch: NodeJS.Architecture = process.arch,
): CloudflaredDownloadInfo {
  const cloudflaredArch = archToCloudflaredArch(arch);
  if (!cloudflaredArch) {
    return {
      supported: false,
      assetName: null,
      url: null,
      archive: 'none',
      installPath: LOCAL_BIN_PATH,
      expectedSha256: null,
      reason: `Unsupported CPU architecture for automatic cloudflared download: ${arch}`,
    };
  }

  let assetName: string | null = null;
  let archive: CloudflaredDownloadInfo['archive'] = 'none';
  if (platform === 'linux') {
    assetName = `cloudflared-linux-${cloudflaredArch}`;
  } else if (platform === 'win32') {
    assetName = `cloudflared-windows-${cloudflaredArch}.exe`;
  } else if (platform === 'darwin') {
    assetName = `cloudflared-darwin-${cloudflaredArch}.tgz`;
    archive = 'tgz';
  }

  if (!assetName) {
    return {
      supported: false,
      assetName: null,
      url: null,
      archive: 'none',
      installPath: LOCAL_BIN_PATH,
      expectedSha256: null,
      reason: `Unsupported operating system for automatic cloudflared download: ${platform}`,
    };
  }

  const expectedSha256 = CLOUDFLARED_DOWNLOAD_SHA256[assetName] || null;
  if (!expectedSha256) {
    return {
      supported: false,
      assetName,
      url: null,
      archive,
      installPath: LOCAL_BIN_PATH,
      expectedSha256: null,
      reason: `Automatic cloudflared download is not supported for ${platform}/${arch} because no pinned checksum is configured.`,
    };
  }

  return {
    supported: true,
    assetName,
    url: `${CLOUDFLARED_RELEASE_BASE}/${assetName}`,
    archive,
    installPath: LOCAL_BIN_PATH,
    expectedSha256,
    reason: null,
  };
}

function isPathLike(value: string) {
  return path.isAbsolute(value) || value.includes('/') || value.includes('\\');
}

async function commandLooksRunnable(bin: string) {
  if (isPathLike(bin)) {
    const stat = await fsp.stat(bin).catch(() => null);
    if (!stat?.isFile()) return false;
  }

  return new Promise<boolean>(resolve => {
    const subprocess = spawn(bin, ['--version'], {
      cwd: TOOLKIT_ROOT,
      stdio: 'ignore',
      windowsHide: true,
    });
    const timer = setTimeout(() => {
      subprocess.kill();
      resolve(false);
    }, 5000);
    subprocess.once('error', () => {
      clearTimeout(timer);
      resolve(false);
    });
    subprocess.once('exit', code => {
      clearTimeout(timer);
      resolve(code === 0);
    });
  });
}

async function getCloudflaredVersion(bin: string) {
  const detected = await commandLooksRunnable(bin);
  if (!detected) return null;

  return new Promise<string | null>(resolve => {
    let output = '';
    const subprocess = spawn(bin, ['--version'], {
      cwd: TOOLKIT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    });
    const timer = setTimeout(() => {
      subprocess.kill();
      resolve(null);
    }, 5000);
    subprocess.stdout.on('data', data => {
      output += data.toString();
    });
    subprocess.stderr.on('data', data => {
      output += data.toString();
    });
    subprocess.once('error', () => {
      clearTimeout(timer);
      resolve(null);
    });
    subprocess.once('exit', code => {
      clearTimeout(timer);
      resolve(code === 0 ? output.trim().slice(0, 200) || null : null);
    });
  });
}

export function assertCloudflaredDownloadUrlIsTrusted(url: URL) {
  if (url.protocol !== 'https:') {
    throw new Error('Refusing to download cloudflared over a non-HTTPS URL');
  }
  if (!TRUSTED_CLOUDFLARED_DOWNLOAD_HOSTS.has(url.hostname)) {
    throw new Error(`Refusing to download cloudflared from untrusted host: ${url.hostname}`);
  }
}

function downloadFile(url: string, destinationPath: string, redirects = 0): Promise<void> {
  return new Promise((resolve, reject) => {
    if (redirects > DOWNLOAD_MAX_REDIRECTS) {
      reject(new Error('Too many redirects while downloading cloudflared'));
      return;
    }

    const parsedUrl = new URL(url);
    try {
      assertCloudflaredDownloadUrlIsTrusted(parsedUrl);
    } catch (error) {
      reject(error);
      return;
    }

    const request = https.get(
      parsedUrl,
      {
        headers: {
          'User-Agent': 'ai-toolkit-cloudflared-downloader',
          Accept: 'application/octet-stream',
        },
        timeout: DOWNLOAD_TIMEOUT_MS,
      },
      async response => {
        const statusCode = response.statusCode || 0;
        const redirectUrl = response.headers.location;
        if (statusCode >= 300 && statusCode < 400 && redirectUrl) {
          response.resume();
          const nextUrl = new URL(redirectUrl, parsedUrl);
          try {
            assertCloudflaredDownloadUrlIsTrusted(nextUrl);
            await downloadFile(nextUrl.toString(), destinationPath, redirects + 1);
            resolve();
          } catch (error) {
            reject(error);
          }
          return;
        }

        if (statusCode < 200 || statusCode >= 300) {
          response.resume();
          reject(new Error(`cloudflared download failed with HTTP ${statusCode}`));
          return;
        }

        try {
          await pipeline(response, fs.createWriteStream(destinationPath, { mode: 0o755 }));
          resolve();
        } catch (error) {
          reject(error);
        }
      },
    );

    request.on('timeout', () => {
      request.destroy(new Error('Timed out downloading cloudflared'));
    });
    request.on('error', reject);
  });
}

async function sha256File(filePath: string) {
  const hash = crypto.createHash('sha256');
  const input = fs.createReadStream(filePath);
  for await (const chunk of input) {
    hash.update(chunk);
  }
  return hash.digest('hex');
}

async function verifyDownloadedCloudflared(filePath: string, expectedSha256: string) {
  const actualSha256 = await sha256File(filePath);
  if (actualSha256 !== expectedSha256) {
    await fsp.rm(filePath, { force: true }).catch(() => undefined);
    throw new Error('Downloaded cloudflared checksum did not match the pinned release checksum');
  }
}

function readTarString(buffer: Buffer, start: number, length: number) {
  const raw = buffer.subarray(start, start + length);
  const nul = raw.indexOf(0);
  return raw.subarray(0, nul >= 0 ? nul : raw.length).toString('utf8').trim();
}

function readTarOctal(buffer: Buffer, start: number, length: number) {
  const text = readTarString(buffer, start, length).replace(/\0/g, '').trim();
  return text ? parseInt(text, 8) : 0;
}

async function extractCloudflaredFromTgz(archivePath: string, destinationPath: string) {
  const archive = await fsp.readFile(archivePath);
  const tar = zlib.gunzipSync(archive);
  let offset = 0;
  while (offset + 512 <= tar.length) {
    const name = readTarString(tar, offset, 100);
    if (!name) break;
    const size = readTarOctal(tar, offset + 124, 12);
    const typeFlag = readTarString(tar, offset + 156, 1);
    const dataStart = offset + 512;
    const dataEnd = dataStart + size;

    if ((typeFlag === '' || typeFlag === '0') && path.basename(name) === 'cloudflared') {
      await fsp.writeFile(destinationPath, tar.subarray(dataStart, dataEnd), { mode: 0o755 });
      return;
    }

    offset = dataStart + Math.ceil(size / 512) * 512;
  }

  throw new Error('Downloaded cloudflared archive did not contain a cloudflared binary');
}

export async function downloadCloudflared(): Promise<CloudflaredDownloadResult> {
  if (hasExplicitCloudflaredBin()) {
    throw new Error('AITK_CLOUDFLARED_BIN is set. Install cloudflared at that path or unset it to use the local downloader.');
  }

  const downloadInfo = getCloudflaredDownloadInfoForPlatform();
  if (!downloadInfo.supported || !downloadInfo.url || !downloadInfo.expectedSha256) {
    throw new Error(downloadInfo.reason || 'Automatic cloudflared download is not supported on this platform');
  }

  await fsp.mkdir(LOCAL_BIN_DIR, { recursive: true });
  const tempPath = `${downloadInfo.installPath}.download`;
  await fsp.rm(tempPath, { force: true }).catch(() => undefined);

  if (downloadInfo.archive === 'tgz') {
    await downloadFile(downloadInfo.url, tempPath);
    await verifyDownloadedCloudflared(tempPath, downloadInfo.expectedSha256);
    await extractCloudflaredFromTgz(tempPath, downloadInfo.installPath);
    await fsp.rm(tempPath, { force: true }).catch(() => undefined);
  } else {
    await downloadFile(downloadInfo.url, tempPath);
    await verifyDownloadedCloudflared(tempPath, downloadInfo.expectedSha256);
    await fsp.rename(tempPath, downloadInfo.installPath);
  }

  if (process.platform !== 'win32') {
    await fsp.chmod(downloadInfo.installPath, 0o755);
  }

  const version = await getCloudflaredVersion(downloadInfo.installPath);
  if (!version) {
    throw new Error('cloudflared downloaded but could not be executed');
  }

  return {
    ...downloadInfo,
    downloaded: true,
    version,
  };
}

async function readPid() {
  try {
    const raw = await fsp.readFile(PID_FILE, 'utf8');
    const pid = Number(raw.trim());
    return Number.isFinite(pid) && pid > 0 ? pid : null;
  } catch {
    return null;
  }
}

async function readGeneratedPublicUrl() {
  const fromFile = await fsp.readFile(URL_FILE, 'utf8').catch(() => '');
  const fromFileUrl = extractPublicUrl(fromFile);
  if (fromFileUrl) return fromFileUrl;

  const fromLog = await fsp.readFile(LOG_FILE, 'utf8').catch(() => '');
  const fromLogUrl = extractPublicUrl(fromLog);
  if (fromLogUrl) {
    await fsp.writeFile(URL_FILE, fromLogUrl, 'utf8').catch(() => undefined);
  }
  return fromLogUrl;
}

function extractPublicUrl(value: string) {
  return value.match(/https:\/\/[a-zA-Z0-9-]+\.trycloudflare\.com\b/)?.[0] || null;
}

async function waitForGeneratedPublicUrl(timeoutMs = 15_000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const publicUrl = await readGeneratedPublicUrl();
    if (publicUrl) return publicUrl;
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  return null;
}

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function isPidRunning(pid: number) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

export async function getCloudflaredStatus(): Promise<CloudflaredStatus> {
  const config = getCloudflaredConfig();
  const downloadInfo = getCloudflaredDownloadInfoForPlatform();
  const detected = await commandLooksRunnable(config.bin);
  const pid = await readPid();
  const running = pid != null && isPidRunning(pid);

  if (pid != null && !running) {
    await fsp.rm(PID_FILE, { force: true }).catch(() => undefined);
    await fsp.rm(URL_FILE, { force: true }).catch(() => undefined);
  }

  let error: string | null = null;
  if (config.enabled && !process.env.AI_TOOLKIT_AUTH) {
    error = 'AI_TOOLKIT_AUTH is required when cloudflared is enabled.';
  } else if (config.enabled && !detected) {
    error = 'cloudflared binary was not found. Download it from Settings or set AITK_CLOUDFLARED_BIN.';
  } else if (config.enabled && config.tokenFile && !fs.existsSync(config.tokenFile)) {
    error = `cloudflared token file does not exist: ${config.tokenFile}`;
  }

  const generatedPublicUrl = running && config.mode === 'quick' ? await readGeneratedPublicUrl() : null;

  return {
    configured: config.configured,
    enabled: config.enabled,
    mode: config.mode,
    detected,
    bin: config.bin,
    downloadAvailable: downloadInfo.supported && !hasExplicitCloudflaredBin(),
    downloadUrl: downloadInfo.url,
    installPath: downloadInfo.installPath,
    running,
    pid: running ? pid : null,
    publicUrl: config.publicUrl || generatedPublicUrl,
    targetUrl: config.targetUrl,
    metricsAddr: config.metricsAddr,
    message: running ? 'cloudflared is running' : config.enabled ? 'cloudflared is not running' : 'cloudflared is disabled',
    error,
  };
}

async function assertStartable(options: { autoDownload?: boolean } = {}) {
  const config = getCloudflaredConfig();
  if (!config.enabled) {
    throw new Error('cloudflared is not enabled. Set AITK_CLOUDFLARED_ENABLED=1.');
  }
  if (!process.env.AI_TOOLKIT_AUTH) {
    throw new Error('AI_TOOLKIT_AUTH is required when cloudflared is enabled.');
  }
  let detected = await commandLooksRunnable(config.bin);
  if (!detected && (options.autoDownload || config.autoDownload)) {
    await downloadCloudflared();
    detected = await commandLooksRunnable(getCloudflaredConfig().bin);
  }
  if (!detected) {
    throw new Error('cloudflared binary was not found. Download it from Settings or set AITK_CLOUDFLARED_BIN.');
  }
  if (config.tokenFile && !fs.existsSync(config.tokenFile)) {
    throw new Error(`cloudflared token file does not exist: ${config.tokenFile}`);
  }
  return getCloudflaredConfig();
}

export function buildCloudflaredArgs(config = getCloudflaredConfig()) {
  const args = ['tunnel', '--metrics', config.metricsAddr, '--loglevel', config.logLevel];
  if (config.mode === 'named') {
    if (!config.tokenFile) {
      throw new Error('AITK_CLOUDFLARED_TOKEN_FILE is required for named cloudflared tunnels.');
    }
    return [...args, 'run', '--token-file', config.tokenFile];
  }
  return [...args, '--url', config.targetUrl];
}

export async function startCloudflared(options: { autoDownload?: boolean } = {}) {
  const current = await getCloudflaredStatus();
  if (current.running) return current;

  const config = await assertStartable(options);
  await fsp.mkdir(path.dirname(PID_FILE), { recursive: true });
  await fsp.rm(URL_FILE, { force: true }).catch(() => undefined);
  await fsp.writeFile(LOG_FILE, '', 'utf8');
  const logOut = fs.openSync(LOG_FILE, 'a');
  const logErr = fs.openSync(LOG_FILE, 'a');

  const subprocess: ChildProcess = spawn(
    config.bin,
    buildCloudflaredArgs(config),
    {
      cwd: TOOLKIT_ROOT,
      stdio: ['ignore', logOut, logErr],
      windowsHide: true,
    },
  );
  fs.closeSync(logOut);
  fs.closeSync(logErr);

  if (subprocess.pid == null) {
    throw new Error('cloudflared did not return a process id.');
  }
  await fsp.writeFile(PID_FILE, String(subprocess.pid), 'utf8');
  if (config.mode === 'quick') {
    await waitForGeneratedPublicUrl();
  }
  return getCloudflaredStatus();
}

async function waitForPidExit(pid: number, timeoutMs: number) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (!isPidRunning(pid)) {
      return true;
    }
    await sleep(200);
  }
  return !isPidRunning(pid);
}

export async function stopCloudflared() {
  const pid = await readPid();
  let stopError: string | null = null;

  if (pid != null && isPidRunning(pid)) {
    try {
      process.kill(pid, 'SIGTERM');
    } catch (error) {
      stopError = error instanceof Error ? error.message : 'cloudflared could not be stopped';
    }

    if (!stopError && !(await waitForPidExit(pid, 4000))) {
      try {
        process.kill(pid, 'SIGKILL');
      } catch (error) {
        stopError = error instanceof Error ? error.message : 'cloudflared could not be force-stopped';
      }
      await waitForPidExit(pid, 1500);
    }
  }

  if (pid == null || !isPidRunning(pid)) {
    await fsp.rm(PID_FILE, { force: true }).catch(() => undefined);
    await fsp.rm(URL_FILE, { force: true }).catch(() => undefined);
  }

  const status = await getCloudflaredStatus();
  if (!stopError) {
    return status;
  }

  return {
    ...status,
    message: `cloudflared could not be stopped: ${stopError}`,
    error: stopError,
  };
}
