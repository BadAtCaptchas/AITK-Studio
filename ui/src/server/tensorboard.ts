import { spawn, spawnSync, type ChildProcess } from 'child_process';
import fs from 'fs';
import net from 'net';
import path from 'path';

const TOOLKIT_ROOT = path.resolve('@', '..', '..');
const DEFAULT_TENSORBOARD_PORT = 6006;
const DEFAULT_TENSORBOARD_HOST = '127.0.0.1';
const TENSORBOARD_STATUS_RUN_NAME = 'aitk_status';

let managedTensorBoard: ChildProcess | null = null;
let startPromise: Promise<TensorBoardStatus> | null = null;
let cachedTensorBoardInstalled: boolean | null = null;
let autoDisabledReason: string | null = null;

export type TensorBoardStatus = {
  enabled: boolean;
  running: boolean;
  port: number;
  url: string | null;
  logDir: string | null;
  pid: number | null;
  source: 'managed' | 'external' | null;
  error?: string;
};

function getExplicitTensorBoardEnabled() {
  const rawEnv = process.env.AITK_ENABLE_TENSORBOARD;
  if (rawEnv == null || rawEnv.trim() === '') {
    return null;
  }

  const raw = rawEnv.trim().toLowerCase();
  return raw === '1' || raw === 'true' || raw === 'yes' || raw === 'on';
}

function isTensorBoardPackageInstalled() {
  if (cachedTensorBoardInstalled !== null) {
    return cachedTensorBoardInstalled;
  }

  try {
    const result = spawnSync(getToolkitPythonPath(), ['-c', 'import tensorboard'], {
      cwd: TOOLKIT_ROOT,
      stdio: 'ignore',
      timeout: 5000,
      windowsHide: process.platform === 'win32',
    });
    cachedTensorBoardInstalled = result.status === 0;
  } catch {
    cachedTensorBoardInstalled = false;
  }

  return cachedTensorBoardInstalled;
}

export function isTensorBoardEnabled() {
  const explicitEnabled = getExplicitTensorBoardEnabled();
  if (explicitEnabled !== null) {
    return explicitEnabled;
  }

  if (autoDisabledReason !== null) {
    return false;
  }

  isTensorBoardPackageInstalled();
  return false;
}

export function getTensorBoardPort() {
  const port = Number.parseInt(process.env.AITK_TENSORBOARD_PORT || '', 10);
  if (Number.isInteger(port) && port > 0 && port <= 65535) {
    return port;
  }
  return DEFAULT_TENSORBOARD_PORT;
}

export function getTensorBoardHost() {
  return process.env.AITK_TENSORBOARD_HOST?.trim() || DEFAULT_TENSORBOARD_HOST;
}

export function shouldWriteTensorBoardStatusRun() {
  const rawEnv = process.env.AITK_TENSORBOARD_STATUS_RUN;
  if (rawEnv == null || rawEnv.trim() === '') {
    return true;
  }

  const raw = rawEnv.trim().toLowerCase();
  return !(raw === '0' || raw === 'false' || raw === 'no' || raw === 'off');
}

export function getTensorBoardLogDir(trainingRoot: string) {
  const configuredLogDir = process.env.AITK_TENSORBOARD_LOG_DIR?.trim();
  if (configuredLogDir) {
    return path.isAbsolute(configuredLogDir) ? configuredLogDir : path.join(TOOLKIT_ROOT, configuredLogDir);
  }
  return path.join(trainingRoot, '.tensorboard');
}

export function getToolkitPythonPath() {
  const isWindows = process.platform === 'win32';
  const venvDir = fs.existsSync(path.join(TOOLKIT_ROOT, '.venv')) ? '.venv' : fs.existsSync(path.join(TOOLKIT_ROOT, 'venv')) ? 'venv' : null;

  if (!venvDir) {
    return 'python';
  }

  if (isWindows) {
    return path.join(TOOLKIT_ROOT, venvDir, 'Scripts', 'python.exe');
  }

  return path.join(TOOLKIT_ROOT, venvDir, 'bin', 'python');
}

export function getTensorBoardPublicUrl(port = getTensorBoardPort(), requestUrl?: string) {
  const configuredUrl = process.env.AITK_TENSORBOARD_PUBLIC_URL?.trim();
  if (configuredUrl) {
    return configuredUrl;
  }

  if (requestUrl) {
    try {
      const url = new URL(requestUrl);
      const hostname = url.hostname.includes(':') ? `[${url.hostname}]` : url.hostname;
      return `${url.protocol}//${hostname}:${port}`;
    } catch {
      // Fall through to localhost.
    }
  }

  return `http://localhost:${port}`;
}

function isManagedTensorBoardRunning() {
  return managedTensorBoard !== null && !managedTensorBoard.killed && managedTensorBoard.exitCode === null;
}

function isPortListening(port: number) {
  return new Promise<boolean>(resolve => {
    const socket = net.createConnection({ host: '127.0.0.1', port });
    let settled = false;

    const finish = (value: boolean) => {
      if (settled) {
        return;
      }
      settled = true;
      socket.destroy();
      resolve(value);
    };

    socket.setTimeout(500);
    socket.once('connect', () => finish(true));
    socket.once('timeout', () => finish(false));
    socket.once('error', () => finish(false));
  });
}

function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitForPort(port: number, timeoutMs = 5000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (await isPortListening(port)) {
      return true;
    }
    await delay(250);
  }
  return isPortListening(port);
}

function writeTensorBoardStatusEvent(pythonPath: string, logDir: string) {
  if (!shouldWriteTensorBoardStatusRun()) {
    removeTensorBoardStatusRun(logDir);
    return true;
  }

  const script = `
import os
import sys
from torch.utils.tensorboard import SummaryWriter

run_dir = os.path.join(sys.argv[1], "${TENSORBOARD_STATUS_RUN_NAME}")
os.makedirs(run_dir, exist_ok=True)
writer = SummaryWriter(run_dir, flush_secs=1)
writer.add_scalar("aitk/tensorboard_available", 1, 0)
writer.add_text("aitk/status", "TensorBoard started by AI Toolkit UI.", 0)
writer.flush()
writer.close()
`;

  try {
    const result = spawnSync(pythonPath, ['-c', script, logDir], {
      cwd: TOOLKIT_ROOT,
      stdio: 'ignore',
      timeout: 10000,
      windowsHide: process.platform === 'win32',
    });
    return result.status === 0;
  } catch {
    return false;
  }
}

function removeTensorBoardStatusRun(logDir: string) {
  const resolvedLogDir = path.resolve(logDir);
  const statusRunDir = path.resolve(resolvedLogDir, TENSORBOARD_STATUS_RUN_NAME);
  if (statusRunDir !== path.join(resolvedLogDir, TENSORBOARD_STATUS_RUN_NAME)) {
    return;
  }

  try {
    fs.rmSync(statusRunDir, { recursive: true, force: true });
  } catch {
    // Best-effort cleanup for the optional synthetic run.
  }
}

export async function getTensorBoardStatus(trainingRoot: string, requestUrl?: string): Promise<TensorBoardStatus> {
  const enabled = isTensorBoardEnabled();
  const port = getTensorBoardPort();
  const logDir = enabled ? getTensorBoardLogDir(trainingRoot) : null;

  if (!enabled) {
    return {
      enabled,
      running: false,
      port,
      url: null,
      logDir,
      pid: null,
      source: null,
    };
  }

  const managedRunning = isManagedTensorBoardRunning();
  const portListening = managedRunning || (await isPortListening(port));

  return {
    enabled,
    running: portListening,
    port,
    url: getTensorBoardPublicUrl(port, requestUrl),
    logDir,
    pid: managedRunning ? (managedTensorBoard?.pid ?? null) : null,
    source: managedRunning ? 'managed' : portListening ? 'external' : null,
  };
}

export async function startTensorBoard(trainingRoot: string) {
  if (startPromise) {
    return startPromise;
  }

  startPromise = startTensorBoardInternal(trainingRoot).finally(() => {
    startPromise = null;
  });

  return startPromise;
}

async function startTensorBoardInternal(trainingRoot: string): Promise<TensorBoardStatus> {
  if (!isTensorBoardEnabled()) {
    return getTensorBoardStatus(trainingRoot);
  }

  const port = getTensorBoardPort();
  const logDir = getTensorBoardLogDir(trainingRoot);
  const pythonPath = getToolkitPythonPath();
  if (isManagedTensorBoardRunning() || (await isPortListening(port))) {
    try {
      fs.mkdirSync(logDir, { recursive: true });
      writeTensorBoardStatusEvent(pythonPath, logDir);
    } catch {
      // TensorBoard is already available; status event creation is best-effort.
    }
    return getTensorBoardStatus(trainingRoot);
  }

  const args = [
    '-m',
    'tensorboard.main',
    '--logdir',
    logDir,
    '--host',
    getTensorBoardHost(),
    '--port',
    `${port}`,
  ];

  try {
    fs.mkdirSync(logDir, { recursive: true });
    const wroteStatusEvent = writeTensorBoardStatusEvent(pythonPath, logDir);
    if (!wroteStatusEvent && getExplicitTensorBoardEnabled() === null) {
      autoDisabledReason = 'Failed to write TensorBoard status event';
      return getTensorBoardStatus(trainingRoot);
    }

    managedTensorBoard = spawn(pythonPath, args, {
      cwd: TOOLKIT_ROOT,
      detached: true,
      stdio: 'ignore',
      windowsHide: process.platform === 'win32',
      env: {
        ...process.env,
      },
    });

    managedTensorBoard.once('exit', () => {
      managedTensorBoard = null;
    });
    managedTensorBoard.once('error', () => {
      managedTensorBoard = null;
    });

    managedTensorBoard.unref();
  } catch (error: any) {
    if (getExplicitTensorBoardEnabled() === null) {
      autoDisabledReason = error?.message || 'Failed to start TensorBoard';
      return getTensorBoardStatus(trainingRoot);
    }
    return {
      ...(await getTensorBoardStatus(trainingRoot)),
      error: error?.message || 'Failed to start TensorBoard',
    };
  }

  const started = await waitForPort(port, 15000);
  if (!started) {
    if (getExplicitTensorBoardEnabled() === null) {
      autoDisabledReason = 'TensorBoard did not start';
      return getTensorBoardStatus(trainingRoot);
    }
    return {
      ...(await getTensorBoardStatus(trainingRoot)),
      error: 'TensorBoard did not start',
    };
  }

  return getTensorBoardStatus(trainingRoot);
}
