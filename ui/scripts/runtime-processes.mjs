import { spawn } from 'node:child_process';
import { promises as fs } from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);

export const UI_ROOT = path.resolve(path.dirname(__filename), '..');
export const TOOLKIT_ROOT = path.resolve(UI_ROOT, '..');
export const TMP_ROOT = path.join(TOOLKIT_ROOT, '.tmp');
export const RUNTIME_PATH = path.join(TMP_ROOT, 'ui-runtime.json');
export const REPO_UPDATER_PID_PATH = path.join(TMP_ROOT, 'repo-updater.pid');
export const TENSORBOARD_PID_PATH = path.join(TMP_ROOT, 'tensorboard.pid');
export const CLOUDFLARED_PID_PATH = path.join(TOOLKIT_ROOT, '.cloudflared.pid');

export const DEFAULT_START_PORT = 8675;
export const DEFAULT_DEV_PORT = 3000;

const RUNTIME_MARKERS = [
  'scripts/run-app.mjs',
  'node_modules/next/dist/bin/next',
  'next start',
  'next dev',
  'dist/cron/worker.js',
  'cron/worker.ts',
  'node_modules/ts-node-dev',
  'ts-node-dev',
  'tensorboard.main',
  'cloudflared tunnel',
  'scripts/repo-updater.mjs',
  'scripts/start-updater.mjs',
  'concurrently',
];

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
    child.once('error', error => {
      resolve({ code: 1, stdout, stderr: error.message });
    });
    child.once('close', code => {
      resolve({ code: code ?? 0, stdout, stderr });
    });
  });
}

export function normalizeCommandLine(value) {
  return String(value || '').replace(/\\/g, '/').toLowerCase();
}

function normalizedRoot(value) {
  return normalizeCommandLine(value).replace(/\/+$/, '');
}

export function commandBelongsToRuntime(commandLine) {
  const command = normalizeCommandLine(commandLine);
  const uiRoot = normalizedRoot(UI_ROOT);
  const toolkitRoot = normalizedRoot(TOOLKIT_ROOT);

  if (!command) {
    return false;
  }

  if (command.includes('scripts/restart-ui.mjs')) {
    return false;
  }

  if (command.includes(`${toolkitRoot}/run.py`) || /\brun\.py\b/.test(command)) {
    return false;
  }

  if (!command.includes(uiRoot) && !command.includes(toolkitRoot)) {
    return false;
  }

  return RUNTIME_MARKERS.some(marker => command.includes(marker));
}

function commandLooksLikeManagedService(commandLine) {
  const command = normalizeCommandLine(commandLine);
  return command.includes('tensorboard.main') || (command.includes('cloudflared') && command.includes(' tunnel'));
}

function processBelongsToRuntime(processInfo) {
  return Boolean(
    processInfo &&
      (commandBelongsToRuntime(processInfo.commandLine) || commandLooksLikeManagedService(processInfo.commandLine)),
  );
}

export async function ensureTmpRoot() {
  await fs.mkdir(TMP_ROOT, { recursive: true });
}

export async function readJson(filePath) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch {
    return null;
  }
}

export async function writeJsonAtomic(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tmpPath = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  await fs.writeFile(tmpPath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  await fs.rename(tmpPath, filePath);
}

export function nowIso() {
  return new Date().toISOString();
}

export function isPidRunning(pid) {
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

export async function listWindowsProcessInfos() {
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

export async function listUnixProcessInfos() {
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

export async function listProcessInfos() {
  return process.platform === 'win32' ? await listWindowsProcessInfos() : await listUnixProcessInfos();
}

export function collectDescendantPids(processes, rootPid) {
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

async function readPidFile(filePath) {
  const raw = await fs.readFile(filePath, 'utf8').catch(() => '');
  if (!raw.trim()) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw);
    const pid = Number(parsed?.pid);
    return Number.isInteger(pid) && pid > 0 ? pid : null;
  } catch {
    const pid = Number(raw.trim());
    return Number.isInteger(pid) && pid > 0 ? pid : null;
  }
}

export async function readKnownRuntimePids() {
  const pids = await Promise.all([
    readPidFile(REPO_UPDATER_PID_PATH),
    readPidFile(TENSORBOARD_PID_PATH),
    readPidFile(CLOUDFLARED_PID_PATH),
  ]);
  return pids.filter(pid => Number.isInteger(pid) && pid > 0);
}

export function collectRuntimePids(processes, runtime = {}, extraPids = []) {
  const pids = new Set();
  const byPid = new Map(processes.map(processInfo => [processInfo.pid, processInfo]));
  const roots = [
    runtime?.rootPid,
    runtime?.launcherPid,
    runtime?.supervisorPid,
    runtime?.uiPid,
    runtime?.workerPid,
    runtime?.updaterPid,
    ...(Array.isArray(runtime?.managedPids) ? runtime.managedPids : []),
  ]
    .map(Number)
    .filter(pid => Number.isInteger(pid) && pid > 0 && pid !== process.pid);

  for (const pid of roots) {
    const processInfo = byPid.get(pid);
    if (!processInfo || !commandBelongsToRuntime(processInfo.commandLine)) {
      continue;
    }
    pids.add(pid);
    for (const descendantPid of collectDescendantPids(processes, pid)) {
      const descendant = byPid.get(descendantPid);
      if (processBelongsToRuntime(descendant)) {
        pids.add(descendantPid);
      }
    }
  }

  for (const pid of extraPids) {
    const processInfo = byPid.get(pid);
    if (!processBelongsToRuntime(processInfo)) {
      continue;
    }
    pids.add(pid);
    for (const descendantPid of collectDescendantPids(processes, pid)) {
      const descendant = byPid.get(descendantPid);
      if (processBelongsToRuntime(descendant)) {
        pids.add(descendantPid);
      }
    }
  }

  for (const processInfo of processes) {
    if (commandBelongsToRuntime(processInfo.commandLine)) {
      pids.add(processInfo.pid);
    }
  }

  pids.delete(process.pid);
  return [...pids].filter(pid => Number.isInteger(pid) && pid > 0 && pid !== process.pid);
}

export function describeStopFailure(result) {
  const denied = result.denied || [];
  const remaining = result.remaining || [];
  const parts = [];

  if (denied.length > 0) {
    parts.push(`permission was denied for PID(s): ${denied.join(', ')}`);
  }
  if (remaining.length > 0) {
    parts.push(`PID(s) still running after shutdown: ${remaining.join(', ')}`);
  }

  return parts.join('; ');
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export async function terminatePidsGracefully(pids, options = {}) {
  const timeoutMs = options.timeoutMs ?? 4000;
  const forceTimeoutMs = options.forceTimeoutMs ?? 1500;
  const targets = [...new Set(pids)].filter(pid => isPidRunning(pid));
  const denied = [];

  if (targets.length === 0) {
    return { targeted: [], denied, remaining: [] };
  }

  for (const pid of targets) {
    try {
      process.kill(pid, options.signal || 'SIGTERM');
    } catch (error) {
      if (error?.code === 'EPERM') {
        denied.push(pid);
      }
    }
  }

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (targets.every(pid => !isPidRunning(pid))) {
      return { targeted: targets, denied, remaining: [] };
    }
    await sleep(200);
  }

  for (const pid of targets) {
    if (!isPidRunning(pid)) continue;
    try {
      process.kill(pid, options.forceSignal || 'SIGKILL');
    } catch (error) {
      if (error?.code === 'EPERM' && !denied.includes(pid)) {
        denied.push(pid);
      }
    }
  }

  const forceDeadline = Date.now() + forceTimeoutMs;
  while (Date.now() < forceDeadline) {
    if (targets.every(pid => !isPidRunning(pid))) {
      return { targeted: targets, denied, remaining: [] };
    }
    await sleep(200);
  }

  return {
    targeted: targets,
    denied,
    remaining: targets.filter(pid => isPidRunning(pid)),
  };
}

export function isPortAvailable(port) {
  return new Promise(resolve => {
    const server = net.createServer();
    let settled = false;
    let listening = false;

    const finish = value => {
      if (settled) return;
      settled = true;
      if (listening) {
        server.close(() => resolve(value));
      } else {
        resolve(value);
      }
    };

    server.once('error', () => finish(false));
    server.listen(port, '::', () => {
      listening = true;
      finish(true);
    });
  });
}

export async function getPortOwnerPids(port) {
  if (process.platform === 'win32') {
    const result = await run('netstat.exe', ['-ano', '-p', 'tcp']);
    if (result.code !== 0) return [];

    const pids = new Set();
    for (const line of result.stdout.split(/\r?\n/)) {
      if (!/\bLISTENING\b/i.test(line)) continue;
      const columns = line.trim().split(/\s+/);
      const localAddress = columns[1] || '';
      const pid = Number(columns[columns.length - 1]);
      if (localAddress.endsWith(`:${port}`) && Number.isInteger(pid) && pid > 0) {
        pids.add(pid);
      }
    }
    return [...pids];
  }

  const lsof = await run('lsof', ['-nP', `-iTCP:${port}`, '-sTCP:LISTEN', '-t']);
  if (lsof.code === 0 && lsof.stdout.trim()) {
    return [
      ...new Set(
        lsof.stdout
          .split(/\s+/)
          .map(Number)
          .filter(pid => Number.isInteger(pid) && pid > 0),
      ),
    ];
  }

  const ss = await run('ss', ['-ltnp']);
  if (ss.code !== 0) return [];

  const pids = new Set();
  for (const line of ss.stdout.split(/\r?\n/)) {
    if (!line.includes(`:${port}`)) continue;
    for (const match of line.matchAll(/pid=(\d+)/g)) {
      pids.add(Number(match[1]));
    }
  }
  return [...pids].filter(pid => Number.isInteger(pid) && pid > 0);
}

export function formatPidList(pids, processes = []) {
  const byPid = new Map(processes.map(processInfo => [processInfo.pid, processInfo]));
  return pids
    .map(pid => {
      const commandLine = byPid.get(pid)?.commandLine;
      return commandLine ? `${pid} (${commandLine})` : `${pid}`;
    })
    .join(', ');
}

export async function stopStaleRuntime(options = {}) {
  const logger = options.logger || console;
  const runtime = (await readJson(options.runtimePath || RUNTIME_PATH)) || {};
  const knownPids = await readKnownRuntimePids();
  const processes = await listProcessInfos();
  const pids = collectRuntimePids(processes, runtime, knownPids);

  if (pids.length > 0) {
    logger.log?.(`Stopping stale AI Toolkit runtime process(es): ${formatPidList(pids, processes)}`);
    const result = await terminatePidsGracefully(pids, {
      timeoutMs: options.timeoutMs ?? 4000,
      forceTimeoutMs: options.forceTimeoutMs ?? 1500,
    });
    const failure = describeStopFailure(result);
    if (failure) {
      throw new Error(`Could not stop stale AI Toolkit runtime without elevated permissions: ${failure}`);
    }
  }

  const port = options.port;
  if (Number.isInteger(port) && port > 0 && !(await isPortAvailable(port))) {
    const ownerPids = await getPortOwnerPids(port);
    const refreshedProcesses = await listProcessInfos();
    const ownedPortPids = ownerPids.filter(pid => {
      const processInfo = refreshedProcesses.find(candidate => candidate.pid === pid);
      return commandBelongsToRuntime(processInfo?.commandLine || '');
    });

    if (ownedPortPids.length > 0) {
      logger.log?.(`Stopping stale AI Toolkit process(es) holding port ${port}: ${formatPidList(ownedPortPids, refreshedProcesses)}`);
      const result = await terminatePidsGracefully(ownedPortPids, {
        timeoutMs: options.timeoutMs ?? 4000,
        forceTimeoutMs: options.forceTimeoutMs ?? 1500,
      });
      const failure = describeStopFailure(result);
      if (failure) {
        throw new Error(`Port ${port} is still held by an AI Toolkit process, but it could not be stopped: ${failure}`);
      }
    }

    if (!(await isPortAvailable(port))) {
      const stillOwnerPids = await getPortOwnerPids(port);
      const details = stillOwnerPids.length > 0 ? ` PID(s): ${stillOwnerPids.join(', ')}` : '';
      throw new Error(
        `Port ${port} is already in use by a process this app does not own.${details} Stop that process and start AI Toolkit again.`,
      );
    }
  }

  return { stoppedPids: pids };
}

export function getUiPort(mode) {
  return mode === 'dev' ? DEFAULT_DEV_PORT : DEFAULT_START_PORT;
}

export function buildAppCommands(mode, port = getUiPort(mode)) {
  const nextBin = path.join(UI_ROOT, 'node_modules', 'next', 'dist', 'bin', 'next');
  const tsNodeDevBin = path.join(UI_ROOT, 'node_modules', 'ts-node-dev', 'lib', 'bin.js');
  const updaterScript = path.join(UI_ROOT, 'scripts', 'repo-updater.mjs');

  if (mode === 'dev') {
    return [
      {
        label: 'WORKER',
        critical: true,
        command: process.execPath,
        args: [
          tsNodeDevBin,
          '--project',
          'tsconfig.worker.json',
          '--respawn',
          '--watch',
          'cron',
          '--transpile-only',
          '--exit-child',
          'cron/worker.ts',
        ],
      },
      {
        label: 'UI',
        critical: true,
        command: process.execPath,
        args: [nextBin, 'dev', '--turbopack'],
      },
      {
        label: 'UPDATER',
        critical: false,
        command: process.execPath,
        args: [updaterScript],
      },
    ];
  }

  return [
    {
      label: 'WORKER',
      critical: true,
      command: process.execPath,
      args: [path.join(UI_ROOT, 'dist', 'cron', 'worker.js')],
    },
    {
      label: 'UI',
      critical: true,
      command: process.execPath,
      args: [nextBin, 'start', '--port', String(port)],
    },
    {
      label: 'UPDATER',
      critical: false,
      command: process.execPath,
      args: [updaterScript],
    },
  ];
}
