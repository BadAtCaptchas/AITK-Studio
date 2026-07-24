import { spawn } from 'node:child_process';
import {
  RUNTIME_PATH,
  TOOLKIT_ROOT,
  UI_ROOT,
  buildAppCommands,
  ensureTmpRoot,
  getUiPort,
  nowIso,
  stopStaleRuntime,
  writeJsonAtomic,
} from './runtime-processes.mjs';
import {
  runStartupWithCleanup,
  stopManagedChildren,
} from './app-supervisor-lifecycle.mjs';

const SHUTDOWN_GRACE_MS = 8000;
const FORCE_GRACE_MS = 2000;

function parseMode() {
  const modeArgIndex = process.argv.indexOf('--mode');
  const rawMode = modeArgIndex >= 0 ? process.argv[modeArgIndex + 1] : 'start';
  if (rawMode === 'dev' || rawMode === 'start') {
    return rawMode;
  }
  throw new Error(`Unknown app mode: ${rawMode || '(empty)'}`);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function isChildRunning(child) {
  return child.exitCode == null && child.signalCode == null;
}

function prefixStream(stream, label, write) {
  let buffer = '';
  stream.on('data', chunk => {
    buffer += chunk.toString('utf8');
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || '';
    for (const line of lines) {
      write(`[${label}] ${line}\n`);
    }
  });
  stream.on('end', () => {
    if (buffer) {
      write(`[${label}] ${buffer}\n`);
      buffer = '';
    }
  });
}

function waitForChildExit(child, timeoutMs) {
  if (!isChildRunning(child)) {
    return Promise.resolve(true);
  }

  return new Promise(resolve => {
    const timeout = setTimeout(() => {
      child.off('exit', handleExit);
      resolve(false);
    }, timeoutMs);

    function handleExit() {
      clearTimeout(timeout);
      resolve(true);
    }

    child.once('exit', handleExit);
  });
}

async function stopChild(child, label, signal) {
  if (!child || !isChildRunning(child)) {
    return;
  }

  let requestedViaIpc = false;
  if (child.connected && typeof child.send === 'function') {
    try {
      child.send({ type: 'aitk-shutdown', signal });
      requestedViaIpc = true;
    } catch {
      requestedViaIpc = false;
    }
  }

  if (!requestedViaIpc) {
    try {
      child.kill(signal);
    } catch (error) {
      if (error?.code === 'EPERM') {
        console.error(`[APP] Permission denied while stopping ${label} process ${child.pid}.`);
        return;
      }
    }
  }

  const stopped = await waitForChildExit(child, SHUTDOWN_GRACE_MS);
  if (stopped || !isChildRunning(child)) {
    return;
  }

  if (requestedViaIpc) {
    try {
      child.kill(signal);
    } catch (error) {
      if (error?.code === 'EPERM') {
        console.error(`[APP] Permission denied while stopping ${label} process ${child.pid}.`);
        return;
      }
    }
    const stoppedAfterSignal = await waitForChildExit(child, FORCE_GRACE_MS);
    if (stoppedAfterSignal || !isChildRunning(child)) {
      return;
    }
  }

  try {
    child.kill('SIGKILL');
  } catch (error) {
    if (error?.code === 'EPERM') {
      console.error(`[APP] Permission denied while force-stopping ${label} process ${child.pid}.`);
      return;
    }
  }

  await waitForChildExit(child, FORCE_GRACE_MS);
}

async function writeRuntime(children, mode) {
  const byLabel = new Map(children.map(entry => [entry.spec.label, entry.child.pid ?? null]));
  const managedPids = children.map(entry => entry.child.pid).filter(pid => Number.isInteger(pid) && pid > 0);
  await writeJsonAtomic(RUNTIME_PATH, {
    schemaVersion: 1,
    rootPid: process.pid,
    launcherPid: process.pid,
    supervisorPid: process.pid,
    uiPid: byLabel.get('UI') || null,
    fileServerPid: byLabel.get('UI') || null,
    workerPid: byLabel.get('WORKER') || null,
    updaterPid: byLabel.get('UPDATER') || null,
    managedPids,
    lifecycleEvent: mode,
    uiRoot: UI_ROOT,
    toolkitRoot: TOOLKIT_ROOT,
    startedAt: nowIso(),
  });
}

async function markRuntimeStopped(mode, reason) {
  await writeJsonAtomic(RUNTIME_PATH, {
    schemaVersion: 1,
    rootPid: process.pid,
    launcherPid: process.pid,
    supervisorPid: process.pid,
    lifecycleEvent: mode,
    uiRoot: UI_ROOT,
    toolkitRoot: TOOLKIT_ROOT,
    stoppedAt: nowIso(),
    stopReason: reason,
  }).catch(() => undefined);
}

function spawnManaged(spec) {
  const child = spawn(spec.command, spec.args, {
    cwd: UI_ROOT,
    env: {
      ...process.env,
      AITK_APP_SUPERVISOR_PID: String(process.pid),
    },
    stdio: spec.ipc ? ['ignore', 'pipe', 'pipe', 'ipc'] : ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });

  prefixStream(child.stdout, spec.label, chunk => process.stdout.write(chunk));
  prefixStream(child.stderr, spec.label, chunk => process.stderr.write(chunk));
  return child;
}

async function main() {
  const mode = parseMode();
  const port = getUiPort(mode);
  const children = [];
  let shuttingDown = false;
  let exitCode = 0;

  async function shutdown(reason, signal = 'SIGTERM') {
    if (shuttingDown) {
      return;
    }
    shuttingDown = true;
    console.log(`[APP] Shutting down (${reason})...`);

    await stopManagedChildren(
      children,
      entry => stopChild(entry.child, entry.spec.label, signal),
      (error, entry) => {
        console.error(
          `[APP] Failed while stopping ${entry.spec.label}: ${error instanceof Error ? error.message : error}`,
        );
      },
    );

    await markRuntimeStopped(mode, reason);
    await sleep(50);
    process.exit(exitCode);
  }

  process.once('SIGINT', () => {
    exitCode = 0;
    void shutdown('SIGINT', 'SIGINT');
  });
  process.once('SIGTERM', () => {
    exitCode = 0;
    void shutdown('SIGTERM', 'SIGTERM');
  });

  await runStartupWithCleanup(
    async () => {
      await ensureTmpRoot();
      await stopStaleRuntime({
        port,
        logger: {
          log: message => console.log(`[APP] ${message}`),
        },
      });

      const specs = buildAppCommands(mode, port);
      for (const spec of specs) {
        const child = spawnManaged(spec);
        children.push({ spec, child });
        console.log(`[APP] ${spec.label} started with pid ${child.pid}.`);

        child.once('error', error => {
          console.error(`[APP] ${spec.label} failed to start: ${error.message}`);
          if (spec.critical && !shuttingDown) {
            exitCode = 1;
            void shutdown(`${spec.label} failed to start`);
          }
        });

        child.once('exit', (code, signal) => {
          if (shuttingDown) {
            return;
          }

          const reason = signal ? `signal ${signal}` : `code ${code ?? 'unknown'}`;
          console.log(`[APP] ${spec.label} exited with ${reason}.`);
          if (spec.critical) {
            exitCode = code && code !== 0 ? code : 1;
            void shutdown(`${spec.label} exited`);
          }
        });
      }

      await writeRuntime(children, mode);
    },
    async error => {
      console.error(`[APP] ${error instanceof Error ? error.message : error}`);
      exitCode = 1;
      await shutdown('startup failed');
    },
  );
}

main().catch(error => {
  console.error(`[APP] ${error instanceof Error ? error.message : error}`);
  process.exit(1);
});
