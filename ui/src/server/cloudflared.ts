import fs from 'fs';
import fsp from 'fs/promises';
import http from 'http';
import https from 'https';
import path from 'path';
import { spawn, type ChildProcess } from 'child_process';
import { pipeline } from 'stream/promises';
import zlib from 'zlib';
import { TOOLKIT_ROOT } from '../paths';

export type CloudflaredStatus = {
  configured: boolean;
  enabled: boolean;
  detected: boolean;
  bin: string;
  downloadAvailable: boolean;
  downloadUrl: string | null;
  installPath: string;
  running: boolean;
  pid: number | null;
  publicUrl: string | null;
  metricsAddr: string;
  message: string;
  error: string | null;
};

const DEFAULT_METRICS_ADDR = '127.0.0.1:60123';
const PID_FILE = path.join(TOOLKIT_ROOT, '.cloudflared.pid');
const LOCAL_BIN_DIR = path.join(TOOLKIT_ROOT, 'bin');
const LOCAL_BIN_PATH = path.join(LOCAL_BIN_DIR, process.platform === 'win32' ? 'cloudflared.exe' : 'cloudflared');
const CLOUDFLARED_RELEASE_BASE = 'https://github.com/cloudflare/cloudflared/releases/latest/download';
const DOWNLOAD_MAX_REDIRECTS = 5;
const DOWNLOAD_TIMEOUT_MS = 120_000;

export type CloudflaredDownloadInfo = {
  supported: boolean;
  assetName: string | null;
  url: string | null;
  archive: 'none' | 'tgz';
  installPath: string;
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
  const metricsAddr = process.env.AITK_CLOUDFLARED_METRICS_ADDR?.trim() || DEFAULT_METRICS_ADDR;
  const logLevel = process.env.AITK_CLOUDFLARED_LOG_LEVEL?.trim() || 'info';

  return {
    enabled,
    bin,
    autoDownload,
    publicUrl,
    tokenFile,
    metricsAddr,
    logLevel,
    configured: Boolean(publicUrl && tokenFile),
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
      reason: `Unsupported operating system for automatic cloudflared download: ${platform}`,
    };
  }

  return {
    supported: true,
    assetName,
    url: `${CLOUDFLARED_RELEASE_BASE}/${assetName}`,
    archive,
    installPath: LOCAL_BIN_PATH,
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

function downloadFile(url: string, destinationPath: string, redirects = 0): Promise<void> {
  return new Promise((resolve, reject) => {
    if (redirects > DOWNLOAD_MAX_REDIRECTS) {
      reject(new Error('Too many redirects while downloading cloudflared'));
      return;
    }

    const parsedUrl = new URL(url);
    const client = parsedUrl.protocol === 'https:' ? https : http;
    const request = client.get(
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
          try {
            await downloadFile(new URL(redirectUrl, parsedUrl).toString(), destinationPath, redirects + 1);
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
  if (!downloadInfo.supported || !downloadInfo.url) {
    throw new Error(downloadInfo.reason || 'Automatic cloudflared download is not supported on this platform');
  }

  await fsp.mkdir(LOCAL_BIN_DIR, { recursive: true });
  const tempPath = `${downloadInfo.installPath}.download`;
  await fsp.rm(tempPath, { force: true }).catch(() => undefined);

  if (downloadInfo.archive === 'tgz') {
    await downloadFile(downloadInfo.url, tempPath);
    await extractCloudflaredFromTgz(tempPath, downloadInfo.installPath);
    await fsp.rm(tempPath, { force: true }).catch(() => undefined);
  } else {
    await downloadFile(downloadInfo.url, tempPath);
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
  }

  let error: string | null = null;
  if (config.enabled && !process.env.AI_TOOLKIT_AUTH) {
    error = 'AI_TOOLKIT_AUTH is required when cloudflared is enabled.';
  } else if (config.enabled && !detected) {
    error = 'cloudflared binary was not found. Download it from Settings or set AITK_CLOUDFLARED_BIN.';
  } else if (config.enabled && !config.publicUrl) {
    error = 'AITK_CLOUDFLARED_PUBLIC_URL is required when cloudflared is enabled.';
  } else if (config.enabled && !config.tokenFile) {
    error = 'AITK_CLOUDFLARED_TOKEN_FILE is required when cloudflared is enabled.';
  } else if (config.enabled && config.tokenFile && !fs.existsSync(config.tokenFile)) {
    error = `cloudflared token file does not exist: ${config.tokenFile}`;
  }

  return {
    configured: config.configured,
    enabled: config.enabled,
    detected,
    bin: config.bin,
    downloadAvailable: downloadInfo.supported && !hasExplicitCloudflaredBin(),
    downloadUrl: downloadInfo.url,
    installPath: downloadInfo.installPath,
    running,
    pid: running ? pid : null,
    publicUrl: config.publicUrl,
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
  if (!config.publicUrl) {
    throw new Error('AITK_CLOUDFLARED_PUBLIC_URL is required when cloudflared is enabled.');
  }
  if (!config.tokenFile) {
    throw new Error('AITK_CLOUDFLARED_TOKEN_FILE is required when cloudflared is enabled.');
  }
  if (!fs.existsSync(config.tokenFile)) {
    throw new Error(`cloudflared token file does not exist: ${config.tokenFile}`);
  }
  return getCloudflaredConfig();
}

export async function startCloudflared(options: { autoDownload?: boolean } = {}) {
  const current = await getCloudflaredStatus();
  if (current.running) return current;

  const config = await assertStartable(options);
  const tokenFile = config.tokenFile;
  if (!tokenFile) {
    throw new Error('AITK_CLOUDFLARED_TOKEN_FILE is required when cloudflared is enabled.');
  }
  await fsp.mkdir(path.dirname(PID_FILE), { recursive: true });

  const subprocess: ChildProcess = spawn(
    config.bin,
    ['tunnel', '--metrics', config.metricsAddr, '--loglevel', config.logLevel, 'run', '--token-file', tokenFile],
    {
      cwd: TOOLKIT_ROOT,
      detached: true,
      stdio: 'ignore' as const,
      windowsHide: true,
    },
  );

  if (subprocess.pid == null) {
    throw new Error('cloudflared did not return a process id.');
  }
  await fsp.writeFile(PID_FILE, String(subprocess.pid), 'utf8');
  subprocess.unref();
  return getCloudflaredStatus();
}

export async function stopCloudflared() {
  const pid = await readPid();
  if (pid != null && isPidRunning(pid)) {
    try {
      process.kill(pid, 'SIGTERM');
    } catch {
      // Process may exit between the status check and signal.
    }
  }
  await fsp.rm(PID_FILE, { force: true }).catch(() => undefined);
  return getCloudflaredStatus();
}
