import { spawn, type ChildProcess } from 'child_process';
import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';
import { TOOLKIT_ROOT } from '@/paths';
import { getToolkitPythonPath } from '@/server/pythonPath';
import { getComfyInstallProgressAtPath, type ComfyInstallProgress } from '@/server/comfyInstallProgress';

const MANAGED_COMFY_ROOT = path.resolve(process.env.AITK_COMFY_ROOT || path.join(TOOLKIT_ROOT, '.aitk_comfy', 'ComfyUI'));
const MANAGED_COMFY_STATE_DIR = path.join(TOOLKIT_ROOT, '.aitk_comfy');
const MANAGED_COMFY_PROGRESS_PATH = path.join(MANAGED_COMFY_STATE_DIR, '.comfy_install_progress.json');
const MANAGED_COMFY_LOG_PATH = path.join(MANAGED_COMFY_STATE_DIR, 'install.log');
const MANAGED_COMFY_PID_PATH = path.join(MANAGED_COMFY_STATE_DIR, 'install.pid');

let installProcess: ChildProcess | null = null;

export type ComfyManagedInstallStatus = {
  installed: boolean;
  installing: boolean;
  root: string;
  progressPath: string;
  logPath: string;
  pid: number | null;
  progress: ComfyInstallProgress | null;
  message: string;
  error: string | null;
};

function nowIso() {
  return new Date().toISOString();
}

function isInstalled() {
  return fs.existsSync(path.join(MANAGED_COMFY_ROOT, 'main.py'));
}

function isProcessRunning(pid: number | null) {
  if (!pid || !Number.isFinite(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function readPidFile() {
  try {
    const raw = fs.readFileSync(MANAGED_COMFY_PID_PATH, 'utf-8').trim();
    const pid = Number(raw);
    return Number.isFinite(pid) && pid > 0 ? pid : null;
  } catch {
    return null;
  }
}

function currentPid() {
  if (installProcess?.pid && installProcess.exitCode == null) return installProcess.pid;
  return readPidFile();
}

function writeProgress(
  status: ComfyInstallProgress['status'],
  step: string,
  message: string,
  percent: number | null,
  error: string | null = null,
) {
  const payload = {
    version: 1,
    status,
    step,
    message,
    root: MANAGED_COMFY_ROOT,
    percent,
    error,
    startedAt: nowIso(),
    updatedAt: nowIso(),
  };
  fs.mkdirSync(MANAGED_COMFY_STATE_DIR, { recursive: true });
  fs.writeFileSync(MANAGED_COMFY_PROGRESS_PATH, JSON.stringify(payload), 'utf-8');
}

function clearPidFile() {
  try {
    fs.rmSync(MANAGED_COMFY_PID_PATH, { force: true });
  } catch {
    // Best-effort cleanup only.
  }
}

function getStatusMessage(installed: boolean, installing: boolean, progress: ComfyInstallProgress | null) {
  if (installing) return progress?.message || 'Installing managed ComfyUI';
  if (installed) return 'Managed ComfyUI is installed';
  if (progress?.status === 'failed') return progress.message;
  return 'Managed ComfyUI is not installed';
}

export async function getComfyManagedInstallStatus(): Promise<ComfyManagedInstallStatus> {
  const progress = await getComfyInstallProgressAtPath(MANAGED_COMFY_PROGRESS_PATH);
  const pid = currentPid();
  const processStillRunning = isProcessRunning(pid);
  if (!processStillRunning && pid) {
    clearPidFile();
  }

  const installed = isInstalled();
  const installing = processStillRunning && progress?.status !== 'completed' && progress?.status !== 'failed';

  return {
    installed,
    installing,
    root: MANAGED_COMFY_ROOT,
    progressPath: MANAGED_COMFY_PROGRESS_PATH,
    logPath: MANAGED_COMFY_LOG_PATH,
    pid: installing ? pid : null,
    progress,
    message: getStatusMessage(installed, installing, progress),
    error: progress?.status === 'failed' ? progress.error || progress.message : null,
  };
}

export async function startComfyManagedInstall(): Promise<ComfyManagedInstallStatus> {
  const current = await getComfyManagedInstallStatus();
  if (current.installing) return current;

  await fsp.mkdir(MANAGED_COMFY_STATE_DIR, { recursive: true });
  await fsp.rm(MANAGED_COMFY_PROGRESS_PATH, { force: true }).catch(() => undefined);
  writeProgress('checking', 'start', 'Starting managed ComfyUI install', 0);

  const scriptPath = path.join(TOOLKIT_ROOT, 'scripts', 'install_comfy.py');
  if (!fs.existsSync(scriptPath)) {
    writeProgress('failed', 'start', 'Managed ComfyUI installer script is missing', null, `Missing ${scriptPath}`);
    return getComfyManagedInstallStatus();
  }

  const logFd = fs.openSync(MANAGED_COMFY_LOG_PATH, 'a');
  const args = [
    '-u',
    scriptPath,
    '--root',
    MANAGED_COMFY_ROOT,
    '--progress',
    MANAGED_COMFY_PROGRESS_PATH,
  ];
  const subprocess = spawn(getToolkitPythonPath(), args, {
    cwd: TOOLKIT_ROOT,
    env: {
      ...process.env,
      AITK_COMFY_INSTALL_PROGRESS_PATH: MANAGED_COMFY_PROGRESS_PATH,
      PYTHONUNBUFFERED: '1',
    },
    windowsHide: true,
    stdio: ['ignore', logFd, logFd],
  });

  installProcess = subprocess;
  if (subprocess.pid) {
    fs.writeFileSync(MANAGED_COMFY_PID_PATH, String(subprocess.pid), 'utf-8');
  }

  const closeLog = () => {
    try {
      fs.closeSync(logFd);
    } catch {
      // The descriptor may already be closed.
    }
  };

  subprocess.once('error', error => {
    closeLog();
    clearPidFile();
    if (installProcess === subprocess) installProcess = null;
    writeProgress('failed', 'start', 'Managed ComfyUI install failed to start', null, error.message);
  });

  subprocess.once('exit', (code, signal) => {
    closeLog();
    clearPidFile();
    if (installProcess === subprocess) installProcess = null;
    if (code === 0 && signal == null) {
      return;
    }
    const reason = signal ? `signal ${signal}` : `exit code ${code ?? 'unknown'}`;
    writeProgress('failed', 'install', `Managed ComfyUI install failed with ${reason}`, null, reason);
  });

  subprocess.unref?.();
  return getComfyManagedInstallStatus();
}
