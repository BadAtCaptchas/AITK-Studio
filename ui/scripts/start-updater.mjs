import { spawn } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const UI_ROOT = path.resolve(path.dirname(__filename), '..');
const TOOLKIT_ROOT = path.resolve(UI_ROOT, '..');
const TMP_ROOT = path.join(TOOLKIT_ROOT, '.tmp');
const PID_PATH = path.join(TMP_ROOT, 'repo-updater.pid');
const STATUS_PATH = path.join(TMP_ROOT, 'repo-update-status.json');
const RUNTIME_PATH = path.join(TMP_ROOT, 'ui-runtime.json');
const UPDATER_SCRIPT = path.join(UI_ROOT, 'scripts', 'repo-updater.mjs');
const DEFAULT_REPO_OWNER = 'BadAtCaptchas';
const DEFAULT_REPO_NAME = 'AITK-Studio';
const DESIRED_REPO_FULL_NAME = `${process.env.AITK_UPDATE_REPO_OWNER || DEFAULT_REPO_OWNER}/${
  process.env.AITK_UPDATE_REPO_NAME || DEFAULT_REPO_NAME
}`;
const UPDATER_GENERATION = 3;

function isPidRunning(pid) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false;
  }

  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error?.code === 'EPERM';
  }
}

async function readExistingPid() {
  try {
    const payload = JSON.parse(await fs.readFile(PID_PATH, 'utf8'));
    return Number(payload?.pid);
  } catch {
    return null;
  }
}

async function readExistingStatus() {
  try {
    return JSON.parse(await fs.readFile(STATUS_PATH, 'utf8'));
  } catch {
    return null;
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function stopExistingUpdater(pid) {
  try {
    process.kill(pid, 'SIGTERM');
  } catch {
    return;
  }

  for (let i = 0; i < 10; i += 1) {
    await sleep(200);
    if (!isPidRunning(pid)) {
      return;
    }
  }
}

async function writeStartupFailure(error) {
  try {
    await fs.mkdir(TMP_ROOT, { recursive: true });
    const statusPath = path.join(TMP_ROOT, 'repo-update-status.json');
    const tmpPath = `${statusPath}.${process.pid}.${Date.now()}.tmp`;
    await fs.writeFile(
      tmpPath,
      `${JSON.stringify(
        {
          schemaVersion: 1,
          state: 'error',
          message: 'Update checker could not start',
          error: error instanceof Error ? error.message : 'Unknown launcher error',
          checkedAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          nextCheckAt: null,
        },
        null,
        2,
      )}\n`,
      'utf8',
    );
    await fs.rename(tmpPath, statusPath);
  } catch {
    // Startup should never block the main app.
  }
}

async function writeRuntimeLaunchInfo() {
  if (!process.env.npm_lifecycle_event) {
    return;
  }

  try {
    await fs.mkdir(TMP_ROOT, { recursive: true });
    const tmpPath = `${RUNTIME_PATH}.${process.pid}.${Date.now()}.tmp`;
    await fs.writeFile(
      tmpPath,
      `${JSON.stringify(
        {
          schemaVersion: 1,
          rootPid: process.ppid,
          launcherPid: process.pid,
          lifecycleEvent: process.env.npm_lifecycle_event,
          uiRoot: UI_ROOT,
          toolkitRoot: TOOLKIT_ROOT,
          startedAt: new Date().toISOString(),
        },
        null,
        2,
      )}\n`,
      'utf8',
    );
    await fs.rename(tmpPath, RUNTIME_PATH);
  } catch {
    // Restart support should not prevent the UI from starting.
  }
}

async function startUpdater() {
  await writeRuntimeLaunchInfo();

  const existingPid = await readExistingPid();
  if (isPidRunning(existingPid)) {
    const existingStatus = await readExistingStatus();
    if (
      !existingStatus ||
      (existingStatus.repoFullName === DESIRED_REPO_FULL_NAME && existingStatus.updaterGeneration === UPDATER_GENERATION)
    ) {
      console.log(`Repo updater already running with pid ${existingPid}.`);
      return;
    }

    console.log(`Restarting repo updater for ${DESIRED_REPO_FULL_NAME}.`);
    await stopExistingUpdater(existingPid);
    if (isPidRunning(existingPid)) {
      console.warn(`Repo updater pid ${existingPid} is still running; not starting a duplicate.`);
      return;
    }
  }

  const child = spawn(process.execPath, [UPDATER_SCRIPT], {
    cwd: UI_ROOT,
    detached: true,
    env: process.env,
    stdio: 'ignore',
    windowsHide: true,
  });

  child.unref();
  console.log(`Repo updater started with pid ${child.pid}.`);
}

startUpdater().catch(async error => {
  await writeStartupFailure(error);
  console.error(error instanceof Error ? error.message : error);
  process.exit(0);
});
