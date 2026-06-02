import { spawn } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const UI_ROOT = path.resolve(path.dirname(__filename), '..');
const TOOLKIT_ROOT = path.resolve(UI_ROOT, '..');
const TMP_ROOT = path.join(TOOLKIT_ROOT, '.tmp');
const PID_PATH = path.join(TMP_ROOT, 'repo-updater.pid');
const UPDATER_SCRIPT = path.join(UI_ROOT, 'scripts', 'repo-updater.mjs');

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

async function startUpdater() {
  const existingPid = await readExistingPid();
  if (isPidRunning(existingPid)) {
    console.log(`Repo updater already running with pid ${existingPid}.`);
    return;
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
