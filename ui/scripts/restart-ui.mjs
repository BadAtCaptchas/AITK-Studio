import { spawn } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const UI_ROOT = path.resolve(path.dirname(__filename), '..');
const TOOLKIT_ROOT = path.resolve(UI_ROOT, '..');
const TMP_ROOT = path.join(TOOLKIT_ROOT, '.tmp');
const STATUS_PATH = path.join(TMP_ROOT, 'repo-update-status.json');
const RUNTIME_PATH = path.join(TMP_ROOT, 'ui-runtime.json');
const RESTART_PID_PATH = path.join(TMP_ROOT, 'ui-restart.pid');
const RESTART_OUT_LOG = path.join(TMP_ROOT, 'ui-restart-out.log');
const RESTART_ERR_LOG = path.join(TMP_ROOT, 'ui-restart-err.log');

const STOP_GRACE_MS = 4000;
const RESPONSE_GRACE_MS = 1500;
const PROJECT_MARKERS = [
  'next start',
  'next dev',
  'node_modules',
  'concurrently',
  'dist/cron/worker.js',
  'cron/worker.ts',
  'ts-node-dev',
  'npm run start',
  'npm run dev',
];

function nowIso() {
  return new Date().toISOString();
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function ensureTmpRoot() {
  await fs.mkdir(TMP_ROOT, { recursive: true });
}

async function readJson(filePath) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch {
    return null;
  }
}

async function writeJsonAtomic(filePath, value) {
  await ensureTmpRoot();
  const tmpPath = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  await fs.writeFile(tmpPath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  await fs.rename(tmpPath, filePath);
}

async function writeStatus(update) {
  const previous = (await readJson(STATUS_PATH)) || {};
  await writeJsonAtomic(STATUS_PATH, {
    schemaVersion: 1,
    ...previous,
    ...update,
    restartPid: process.pid,
    updatedAt: nowIso(),
  });
}

function isPidRunning(pid) {
  if (!Number.isInteger(pid) || pid <= 0 || pid === process.pid) {
    return false;
  }

  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error?.code === 'EPERM';
  }
}

function run(command, args, options = {}) {
  return new Promise(resolve => {
    const child = spawn(command, args, {
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
      ...options,
    });

    let stdout = '';
    let stderr = '';

    child.stdout?.on('data', chunk => {
      stdout += chunk.toString('utf8');
    });

    child.stderr?.on('data', chunk => {
      stderr += chunk.toString('utf8');
    });

    child.on('error', error => {
      resolve({ code: 1, stdout, stderr: error.message });
    });

    child.on('close', code => {
      resolve({ code: code || 0, stdout, stderr });
    });
  });
}

function normalizeCommandLine(value) {
  return String(value || '').replace(/\\/g, '/').toLowerCase();
}

function commandBelongsToRuntime(commandLine) {
  const command = normalizeCommandLine(commandLine);
  const uiRoot = normalizeCommandLine(UI_ROOT);
  const toolkitRoot = normalizeCommandLine(TOOLKIT_ROOT);

  if (!command || command.includes('scripts/restart-ui.mjs') || command.includes('scripts/repo-updater.mjs')) {
    return false;
  }

  if (!command.includes(uiRoot) && !command.includes(toolkitRoot)) {
    return false;
  }

  return PROJECT_MARKERS.some(marker => command.includes(marker));
}

async function listWindowsRuntimePids() {
  const command = [
    '$ErrorActionPreference = "SilentlyContinue";',
    'Get-CimInstance Win32_Process |',
    'Select-Object ProcessId,ParentProcessId,CommandLine |',
    'ConvertTo-Json -Compress',
  ].join(' ');
  const result = await run('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', command]);

  if (result.code !== 0 || !result.stdout.trim()) {
    return [];
  }

  try {
    const parsed = JSON.parse(result.stdout);
    const processes = Array.isArray(parsed) ? parsed : [parsed];
    return processes
      .filter(processInfo => Number(processInfo?.ProcessId) !== process.pid)
      .map(processInfo => ({
        pid: Number(processInfo?.ProcessId),
        ppid: Number(processInfo?.ParentProcessId),
        commandLine: String(processInfo?.CommandLine || ''),
      }))
      .filter(processInfo => Number.isInteger(processInfo.pid) && processInfo.pid > 0 && processInfo.pid !== process.pid);
  } catch {
    return [];
  }
}

async function listUnixRuntimePids() {
  const result = await run('ps', ['-eo', 'pid=,ppid=,args=']);
  if (result.code !== 0) {
    return [];
  }

  return result.stdout
    .split(/\r?\n/)
    .map(line => {
      const match = line.trim().match(/^(\d+)\s+(\d+)\s+(.+)$/);
      return match ? { pid: Number(match[1]), ppid: Number(match[2]), commandLine: match[3] } : null;
    })
    .filter(Boolean)
    .filter(processInfo => processInfo.pid !== process.pid);
}

async function listProcessInfos() {
  return process.platform === 'win32' ? await listWindowsRuntimePids() : await listUnixRuntimePids();
}

function collectDescendantPids(processes, rootPid) {
  if (!Number.isInteger(rootPid) || rootPid <= 0) {
    return new Set();
  }

  const childrenByParent = new Map();
  for (const processInfo of processes) {
    if (!Number.isInteger(processInfo.ppid)) continue;
    const children = childrenByParent.get(processInfo.ppid) || [];
    children.push(processInfo.pid);
    childrenByParent.set(processInfo.ppid, children);
  }

  const descendants = new Set();
  const queue = [rootPid];
  while (queue.length > 0) {
    const pid = queue.shift();
    if (pid !== rootPid) {
      descendants.add(pid);
    }
    for (const childPid of childrenByParent.get(pid) || []) {
      if (!descendants.has(childPid)) {
        queue.push(childPid);
      }
    }
  }
  return descendants;
}

function collectRuntimePids(processes, runtime) {
  const runtimeRootPid = Number(runtime?.rootPid);
  const runtimeLauncherPid = Number(runtime?.launcherPid);
  const pids = new Set();

  if (Number.isInteger(runtimeRootPid) && runtimeRootPid > 0) {
    pids.add(runtimeRootPid);
  }

  if (Number.isInteger(runtimeLauncherPid) && runtimeLauncherPid > 0) {
    pids.add(runtimeLauncherPid);
  }

  for (const pid of collectDescendantPids(processes, runtimeRootPid)) {
    pids.add(pid);
  }

  for (const pid of collectDescendantPids(processes, runtimeLauncherPid)) {
    pids.add(pid);
  }

  for (const processInfo of processes) {
    if (commandBelongsToRuntime(processInfo.commandLine)) {
      pids.add(processInfo.pid);
    }
  }

  pids.delete(process.pid);
  return [...pids].filter(pid => Number.isInteger(pid) && pid > 0 && pid !== process.pid);
}

async function terminatePids(pids) {
  const targets = [...new Set(pids)].filter(pid => isPidRunning(pid));
  if (targets.length === 0) {
    return;
  }

  if (process.platform === 'win32') {
    for (const pid of targets) {
      await run('taskkill.exe', ['/PID', String(pid), '/T', '/F']);
    }
    return;
  }

  for (const pid of targets) {
    try {
      process.kill(pid, 'SIGTERM');
    } catch {
      // It may already be gone by the time we signal it.
    }
  }

  const deadline = Date.now() + STOP_GRACE_MS;
  while (Date.now() < deadline) {
    if (targets.every(pid => !isPidRunning(pid))) {
      return;
    }
    await sleep(200);
  }

  for (const pid of targets) {
    if (!isPidRunning(pid)) continue;
    try {
      process.kill(pid, 'SIGKILL');
    } catch {
      // The process may have already stopped.
    }
  }
}

async function stopRunningServer() {
  await writeStatus({
    state: 'restarting',
    message: 'Stopping UI server',
    restartStep: 'stopping-server',
    canApplyUpdate: false,
  });

  const runtime = await readJson(RUNTIME_PATH);

  for (let attempt = 0; attempt < 4; attempt += 1) {
    const processes = await listProcessInfos();
    const pids = collectRuntimePids(processes, runtime);
    if (pids.length === 0) {
      return;
    }

    await writeStatus({
      state: 'restarting',
      message: `Stopping UI server (${pids.length} processes)`,
      restartStep: 'stopping-server',
      restartTargetPids: pids,
      canApplyUpdate: false,
    });

    await terminatePids(pids);
    await sleep(700);
  }
}

async function appendRestartLogLine(line) {
  await ensureTmpRoot();
  await fs.appendFile(RESTART_OUT_LOG, `${line}\n`, 'utf8');
}

async function startBuildAndStart() {
  await writeStatus({
    state: 'restarting',
    message: 'Starting npm run build_and_start',
    restartStep: 'build-and-start',
    canApplyUpdate: false,
  });

  const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const outHandle = await fs.open(RESTART_OUT_LOG, 'a');
  const errHandle = await fs.open(RESTART_ERR_LOG, 'a');

  try {
    await appendRestartLogLine(`[${nowIso()}] Starting npm run build_and_start in ${UI_ROOT}`);
    const child = spawn(npmCommand, ['run', 'build_and_start'], {
      cwd: UI_ROOT,
      detached: true,
      env: {
        ...process.env,
        AITK_RESTARTED_BY_UI: '1',
      },
      stdio: ['ignore', outHandle.fd, errHandle.fd],
      windowsHide: true,
    });

    await new Promise((resolve, reject) => {
      child.once('spawn', resolve);
      child.once('error', reject);
    });

    child.unref();

    await writeJsonAtomic(RESTART_PID_PATH, {
      pid: child.pid,
      command: 'npm run build_and_start',
      cwd: UI_ROOT,
      startedAt: nowIso(),
    });

    await writeStatus({
      state: 'restarting',
      message: 'Restart launched. The app will come back after build completes.',
      restartStep: 'waiting-for-server',
      restartChildPid: child.pid,
      canApplyUpdate: false,
      nextCheckAt: null,
    });
  } finally {
    await outHandle.close().catch(() => undefined);
    await errHandle.close().catch(() => undefined);
  }
}

async function main() {
  await ensureTmpRoot();
  await writeJsonAtomic(RESTART_PID_PATH, { pid: process.pid, startedAt: nowIso(), phase: 'runner' });
  await writeStatus({
    state: 'restarting',
    message: 'Restart requested',
    restartStep: 'queued',
    restartStartedAt: nowIso(),
    canApplyUpdate: false,
  });

  await sleep(RESPONSE_GRACE_MS);
  await stopRunningServer();
  await sleep(500);
  await startBuildAndStart();
}

main()
  .then(() => {
    process.exit(0);
  })
  .catch(async error => {
    await writeStatus({
      state: 'error',
      message: 'Restart failed',
      restartStep: null,
      restartError: error instanceof Error ? error.message : 'Unknown restart error',
      error: error instanceof Error ? error.message : 'Unknown restart error',
      canApplyUpdate: false,
    }).catch(() => undefined);
    process.exit(1);
  });
