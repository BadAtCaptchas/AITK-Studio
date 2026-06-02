import { spawn } from 'node:child_process';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const UI_ROOT = path.resolve(path.dirname(__filename), '..');
const TOOLKIT_ROOT = path.resolve(UI_ROOT, '..');
const TMP_ROOT = path.join(TOOLKIT_ROOT, '.tmp');
const STATUS_PATH = path.join(TMP_ROOT, 'repo-update-status.json');
const REQUEST_PATH = path.join(TMP_ROOT, 'repo-update-request.json');
const PID_PATH = path.join(TMP_ROOT, 'repo-updater.pid');

const DEFAULT_INTERVAL_MINUTES = 360;
const REQUEST_POLL_MS = 5000;
const GIT_TIMEOUT_MS = 20000;
const GIT_FETCH_TIMEOUT_MS = 60000;
const MAX_OUTPUT_LENGTH = 1024 * 1024;

let nextCheckTimer = null;
let requestPollTimer = null;
let isChecking = false;
let isStopping = false;
let handledRequestMtime = 0;

function nowIso() {
  return new Date().toISOString();
}

function truthyEnv(value) {
  if (!value) return false;
  return ['1', 'true', 'yes', 'on'].includes(String(value).trim().toLowerCase());
}

function getIntervalMinutes() {
  const raw = Number(process.env.AITK_UPDATE_CHECK_INTERVAL_MINUTES);
  if (!Number.isFinite(raw) || raw <= 0) {
    return DEFAULT_INTERVAL_MINUTES;
  }
  return Math.max(15, Math.round(raw));
}

const intervalMinutes = getIntervalMinutes();
const intervalMs = intervalMinutes * 60 * 1000;

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
  const status = {
    schemaVersion: 1,
    ...previous,
    ...update,
    updaterPid: process.pid,
    intervalMinutes,
    updatedAt: nowIso(),
  };
  await writeJsonAtomic(STATUS_PATH, status);
  return status;
}

function capAppend(current, chunk) {
  if (current.length >= MAX_OUTPUT_LENGTH) {
    return current;
  }
  const next = current + chunk.toString('utf8');
  return next.length > MAX_OUTPUT_LENGTH ? next.slice(0, MAX_OUTPUT_LENGTH) : next;
}

function runGit(args, options = {}) {
  const timeoutMs = options.timeoutMs || GIT_TIMEOUT_MS;
  return new Promise((resolve, reject) => {
    const child = spawn('git', args, {
      cwd: TOOLKIT_ROOT,
      env: {
        ...process.env,
        GIT_TERMINAL_PROMPT: '0',
      },
      windowsHide: true,
    });

    let stdout = '';
    let stderr = '';
    let timedOut = false;

    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
    }, timeoutMs);

    child.stdout.on('data', chunk => {
      stdout = capAppend(stdout, chunk);
    });

    child.stderr.on('data', chunk => {
      stderr = capAppend(stderr, chunk);
    });

    child.on('error', error => {
      clearTimeout(timer);
      reject(error);
    });

    child.on('close', code => {
      clearTimeout(timer);
      if (timedOut) {
        reject(new Error(`git ${args.join(' ')} timed out`));
        return;
      }

      if (code !== 0) {
        const detail = (stderr || stdout).trim();
        reject(new Error(detail || `git ${args.join(' ')} exited with code ${code}`));
        return;
      }

      resolve({ stdout, stderr });
    });
  });
}

async function safeGit(args, options) {
  try {
    return await runGit(args, options);
  } catch {
    return null;
  }
}

function trimOutput(result) {
  return result?.stdout?.trim() || '';
}

function shortSha(sha) {
  return sha ? sha.slice(0, 12) : '';
}

function plural(value, singular, pluralLabel = `${singular}s`) {
  return `${value} ${value === 1 ? singular : pluralLabel}`;
}

function normalizeRemoteWebUrl(remoteUrl) {
  const raw = (remoteUrl || '').trim();
  if (!raw) return null;

  const sshMatch = raw.match(/^git@([^:]+):(.+?)(?:\.git)?$/);
  if (sshMatch) {
    return `https://${sshMatch[1]}/${sshMatch[2]}`;
  }

  const httpMatch = raw.match(/^(https?:\/\/.+?)(?:\.git)?$/);
  if (httpMatch) {
    return httpMatch[1];
  }

  return null;
}

function buildCompareUrl(remoteWebUrl, localCommit, remoteCommit) {
  if (!remoteWebUrl || !remoteWebUrl.includes('github.com') || !localCommit || !remoteCommit) {
    return null;
  }
  return `${remoteWebUrl}/compare/${shortSha(localCommit)}...${shortSha(remoteCommit)}`;
}

async function resolveUpstreamRef(branch) {
  const configured = trimOutput(await safeGit(['rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}']));
  if (configured) {
    return configured;
  }

  if (branch && branch !== 'HEAD') {
    const candidate = `origin/${branch}`;
    const verified = await safeGit(['rev-parse', '--verify', candidate]);
    if (verified) {
      return candidate;
    }
  }

  const originHead = trimOutput(await safeGit(['rev-parse', '--abbrev-ref', 'origin/HEAD']));
  return originHead || '';
}

async function checkForUpdates(trigger) {
  if (isChecking || isStopping) {
    return;
  }

  isChecking = true;
  const startedAt = nowIso();
  const previous = (await readJson(STATUS_PATH)) || {};

  await writeStatus({
    state: 'checking',
    message: 'Checking for repository updates',
    trigger,
    startedAt,
    error: null,
    lastSuccessfulCheckAt: previous.lastSuccessfulCheckAt || null,
  });

  try {
    const insideWorkTree = trimOutput(await runGit(['rev-parse', '--is-inside-work-tree']));
    if (insideWorkTree !== 'true') {
      await writeStatus({
        state: 'unsupported',
        message: 'Update checks need a git checkout',
        checkedAt: nowIso(),
        lastSuccessfulCheckAt: previous.lastSuccessfulCheckAt || null,
      });
      return;
    }

    const remoteUrl = trimOutput(await safeGit(['remote', 'get-url', 'origin']));
    if (!remoteUrl) {
      await writeStatus({
        state: 'unsupported',
        message: 'No origin remote is configured',
        checkedAt: nowIso(),
        lastSuccessfulCheckAt: previous.lastSuccessfulCheckAt || null,
      });
      return;
    }

    const branch = trimOutput(await safeGit(['rev-parse', '--abbrev-ref', 'HEAD'])) || 'HEAD';
    const localCommit = trimOutput(await runGit(['rev-parse', 'HEAD']));

    await runGit(['fetch', '--quiet', '--prune', 'origin'], { timeoutMs: GIT_FETCH_TIMEOUT_MS });

    const upstream = await resolveUpstreamRef(branch);
    if (!upstream) {
      await writeStatus({
        state: 'unsupported',
        message: 'No upstream branch is available to compare against',
        checkedAt: nowIso(),
        branch,
        remote: remoteUrl,
        remoteWebUrl: normalizeRemoteWebUrl(remoteUrl),
        localCommit,
        localShortCommit: shortSha(localCommit),
        lastSuccessfulCheckAt: previous.lastSuccessfulCheckAt || null,
      });
      return;
    }

    const remoteCommit = trimOutput(await runGit(['rev-parse', upstream]));
    const counts = trimOutput(await runGit(['rev-list', '--left-right', '--count', `HEAD...${upstream}`]));
    const [aheadRaw, behindRaw] = counts.split(/\s+/);
    const ahead = Number.parseInt(aheadRaw, 10) || 0;
    const behind = Number.parseInt(behindRaw, 10) || 0;
    const remoteWebUrl = normalizeRemoteWebUrl(remoteUrl);
    const checkedAt = nowIso();

    let state = 'up_to_date';
    let message = 'Repository is up to date';
    if (behind > 0) {
      state = 'update_available';
      message = `Update available: ${plural(behind, 'commit')} behind ${upstream}`;
      if (ahead > 0) {
        message = `Update available: ${plural(behind, 'commit')} behind and ${plural(ahead, 'commit')} ahead of ${upstream}`;
      }
    } else if (ahead > 0) {
      message = `Local branch is ${plural(ahead, 'commit')} ahead of ${upstream}`;
    }

    await writeStatus({
      state,
      message,
      checkedAt,
      lastSuccessfulCheckAt: checkedAt,
      trigger,
      startedAt,
      branch,
      upstream,
      remote: remoteUrl,
      remoteWebUrl,
      compareUrl: buildCompareUrl(remoteWebUrl, localCommit, remoteCommit),
      localCommit,
      localShortCommit: shortSha(localCommit),
      remoteCommit,
      remoteShortCommit: shortSha(remoteCommit),
      ahead,
      behind,
      error: null,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown update check error';
    await writeStatus({
      state: 'error',
      message: 'Update check failed',
      error: message,
      checkedAt: nowIso(),
      trigger,
      startedAt,
      lastSuccessfulCheckAt: previous.lastSuccessfulCheckAt || null,
    });
  } finally {
    isChecking = false;
  }
}

async function checkAndSchedule(trigger) {
  if (nextCheckTimer) {
    clearTimeout(nextCheckTimer);
    nextCheckTimer = null;
  }

  await checkForUpdates(trigger);

  if (!isStopping) {
    const nextCheckAt = new Date(Date.now() + intervalMs).toISOString();
    await writeStatus({ nextCheckAt });
    nextCheckTimer = setTimeout(() => {
      void checkAndSchedule('schedule');
    }, intervalMs);
  }
}

async function pollForRequests() {
  if (isChecking || isStopping) {
    return;
  }

  try {
    const stat = await fs.stat(REQUEST_PATH);
    if (stat.mtimeMs <= handledRequestMtime) {
      return;
    }
    handledRequestMtime = stat.mtimeMs;
    await checkAndSchedule('manual');
  } catch {
    // The request file only exists when the UI asks for an immediate check.
  }
}

async function writePid() {
  await writeJsonAtomic(PID_PATH, { pid: process.pid, startedAt: nowIso() });
}

async function start() {
  await ensureTmpRoot();
  await writePid();

  if (truthyEnv(process.env.AITK_UPDATE_CHECK_DISABLED)) {
    await writeStatus({
      state: 'disabled',
      message: 'Update checks are disabled',
      checkedAt: nowIso(),
      lastSuccessfulCheckAt: null,
      nextCheckAt: null,
    });
    return;
  }

  requestPollTimer = setInterval(() => {
    void pollForRequests();
  }, REQUEST_POLL_MS);

  await checkAndSchedule('startup');
}

async function stop(signal) {
  if (isStopping) {
    return;
  }
  isStopping = true;
  if (nextCheckTimer) {
    clearTimeout(nextCheckTimer);
  }
  if (requestPollTimer) {
    clearInterval(requestPollTimer);
  }
  await writeStatus({
    state: 'stopped',
    message: `Update checker stopped by ${signal}`,
    nextCheckAt: null,
  }).catch(() => undefined);
  process.exit(0);
}

process.on('SIGINT', signal => {
  void stop(signal);
});
process.on('SIGTERM', signal => {
  void stop(signal);
});

void start().catch(async error => {
  const message = error instanceof Error ? error.message : 'Unable to start update checker';
  await writeStatus({
    state: 'error',
    message: 'Update checker could not start',
    error: message,
    checkedAt: nowIso(),
    nextCheckAt: null,
  }).catch(() => undefined);
  process.exit(1);
});
