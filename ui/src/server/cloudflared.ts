import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { spawn, type ChildProcess } from 'child_process';
import { TOOLKIT_ROOT } from '../paths';

export type CloudflaredStatus = {
  configured: boolean;
  enabled: boolean;
  running: boolean;
  pid: number | null;
  publicUrl: string | null;
  metricsAddr: string;
  message: string;
  error: string | null;
};

const DEFAULT_METRICS_ADDR = '127.0.0.1:60123';
const PID_FILE = path.join(TOOLKIT_ROOT, '.cloudflared.pid');

function boolEnv(value: string | undefined) {
  return value === '1' || value?.toLowerCase() === 'true';
}

export function getCloudflaredConfig() {
  const enabled = boolEnv(process.env.AITK_CLOUDFLARED_ENABLED);
  const bin = process.env.AITK_CLOUDFLARED_BIN?.trim() || 'cloudflared';
  const publicUrl = process.env.AITK_CLOUDFLARED_PUBLIC_URL?.trim() || null;
  const tokenFile = process.env.AITK_CLOUDFLARED_TOKEN_FILE?.trim() || null;
  const metricsAddr = process.env.AITK_CLOUDFLARED_METRICS_ADDR?.trim() || DEFAULT_METRICS_ADDR;
  const logLevel = process.env.AITK_CLOUDFLARED_LOG_LEVEL?.trim() || 'info';

  return {
    enabled,
    bin,
    publicUrl,
    tokenFile,
    metricsAddr,
    logLevel,
    configured: Boolean(publicUrl && tokenFile),
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
  const pid = await readPid();
  const running = pid != null && isPidRunning(pid);

  if (pid != null && !running) {
    await fsp.rm(PID_FILE, { force: true }).catch(() => undefined);
  }

  let error: string | null = null;
  if (config.enabled && !process.env.AI_TOOLKIT_AUTH) {
    error = 'AI_TOOLKIT_AUTH is required when cloudflared is enabled.';
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
    running,
    pid: running ? pid : null,
    publicUrl: config.publicUrl,
    metricsAddr: config.metricsAddr,
    message: running ? 'cloudflared is running' : config.enabled ? 'cloudflared is not running' : 'cloudflared is disabled',
    error,
  };
}

function assertStartable() {
  const config = getCloudflaredConfig();
  if (!config.enabled) {
    throw new Error('cloudflared is not enabled. Set AITK_CLOUDFLARED_ENABLED=1.');
  }
  if (!process.env.AI_TOOLKIT_AUTH) {
    throw new Error('AI_TOOLKIT_AUTH is required when cloudflared is enabled.');
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
  return config;
}

export async function startCloudflared() {
  const current = await getCloudflaredStatus();
  if (current.running) return current;

  const config = assertStartable();
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
