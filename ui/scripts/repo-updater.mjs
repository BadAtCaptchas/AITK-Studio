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
const VERSION_PATH = path.join(TOOLKIT_ROOT, 'version.py');
const RESTART_SCRIPT = path.join(UI_ROOT, 'scripts', 'restart-ui.mjs');

const DEFAULT_REPO_OWNER = 'rmcc3';
const DEFAULT_REPO_NAME = 'ai-toolkit-revamped';
const REPO_OWNER = (process.env.AITK_UPDATE_REPO_OWNER || DEFAULT_REPO_OWNER).trim();
const REPO_NAME = (process.env.AITK_UPDATE_REPO_NAME || DEFAULT_REPO_NAME).trim();
const REPO_BRANCH = (process.env.AITK_UPDATE_REPO_BRANCH || '').trim();
const REPO_FULL_NAME = `${REPO_OWNER}/${REPO_NAME}`;
const REPO_WEB_URL = `https://github.com/${REPO_FULL_NAME}`;
const REPO_API_BASE = `https://api.github.com/repos/${encodeURIComponent(REPO_OWNER)}/${encodeURIComponent(REPO_NAME)}`;
const UPDATER_GENERATION = 3;

const DEFAULT_INTERVAL_MINUTES = 360;
const REQUEST_POLL_MS = 5000;
const GIT_TIMEOUT_MS = 10000;
const GIT_UPDATE_TIMEOUT_MS = 120000;
const HTTP_TIMEOUT_MS = 30000;
const MAX_OUTPUT_LENGTH = 1024 * 1024;
const RECENT_COMMIT_COUNT = 30;

let nextCheckTimer = null;
let requestPollTimer = null;
let isChecking = false;
let isUpdating = false;
let isStopping = false;
let handledRequestMtime = 0;

class GitHubHttpError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = 'GitHubHttpError';
    this.status = status;
    this.data = data;
  }
}

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
    repoFullName: REPO_FULL_NAME,
    repoWebUrl: REPO_WEB_URL,
    updaterGeneration: UPDATER_GENERATION,
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

function redactRemoteCredentials(remoteUrl) {
  const raw = (remoteUrl || '').trim();
  if (!raw) return null;

  return raw.replace(/^(https?:\/\/)(?:[^/?#]*@)/i, '$1');
}

function normalizeRemoteWebUrl(remoteUrl) {
  const raw = redactRemoteCredentials(remoteUrl);
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

function normalizeRepoUrlForCompare(url) {
  return (url || '').replace(/\.git$/, '').replace(/\/$/, '').toLowerCase();
}

function encodeBranchPath(branch) {
  return branch.split('/').map(encodeURIComponent).join('/');
}

function buildRepoArchiveUrl(branch) {
  return `${REPO_WEB_URL}/archive/refs/heads/${encodeBranchPath(branch)}.zip`;
}

function buildCompareUrl(localCommit, remoteCommit, comparable) {
  if (!comparable || !localCommit || !remoteCommit) {
    return null;
  }
  return `${REPO_WEB_URL}/compare/${shortSha(localCommit)}...${shortSha(remoteCommit)}`;
}

function manualUpdateReason(localInfo, result) {
  if (result?.state !== 'update_available') {
    return null;
  }
  if (!localInfo || localInfo.installKind !== 'git') {
    return `This install is not a git checkout. Download updates from ${REPO_WEB_URL}.`;
  }
  if (!localInfo.localCommit) {
    return 'The local git commit could not be detected.';
  }
  if (localInfo.branch === 'HEAD') {
    return 'Detached HEAD checkouts need to be updated manually.';
  }
  if (!result?.comparable) {
    return `The local checkout cannot be compared with ${REPO_FULL_NAME}.`;
  }
  if (result.compareMode !== 'commit' || Number(result.behind || 0) <= 0) {
    return 'This update cannot be fast-forwarded automatically from the local checkout.';
  }
  if (Number(result.ahead || 0) > 0) {
    return 'This branch has local commits, so the updater will not rewrite history automatically.';
  }
  return null;
}

function parseVersionParts(value) {
  if (!value) return null;
  const normalized = String(value).trim().replace(/^v/i, '');
  const parts = normalized.split(/[.-]/).filter(Boolean);
  if (parts.length === 0) return null;
  return parts.map(part => (/^\d+$/.test(part) ? Number(part) : part.toLowerCase()));
}

function compareVersions(a, b) {
  const left = parseVersionParts(a);
  const right = parseVersionParts(b);
  if (!left || !right) return null;

  const length = Math.max(left.length, right.length);
  for (let i = 0; i < length; i += 1) {
    const l = left[i] ?? 0;
    const r = right[i] ?? 0;
    if (typeof l === 'number' && typeof r === 'number') {
      if (l !== r) return l > r ? 1 : -1;
      continue;
    }
    const ls = String(l);
    const rs = String(r);
    if (ls !== rs) return ls > rs ? 1 : -1;
  }
  return 0;
}

async function readLocalVersion() {
  try {
    const versionFile = await fs.readFile(VERSION_PATH, 'utf8');
    const match = versionFile.match(/VERSION\s*=\s*["']([^"']+)["']/);
    return match?.[1] || null;
  } catch {
    return null;
  }
}

async function githubGet(pathname, options = {}) {
  if (typeof fetch !== 'function') {
    throw new Error('This Node.js runtime does not provide fetch for GitHub update checks');
  }

  const timeoutMs = options.timeoutMs || HTTP_TIMEOUT_MS;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const headers = {
    Accept: 'application/vnd.github+json',
    'User-Agent': 'ai-toolkit-revamped-updater',
    'X-GitHub-Api-Version': '2022-11-28',
  };
  const token = process.env.AITK_UPDATE_GITHUB_TOKEN || process.env.GITHUB_TOKEN;
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  try {
    const response = await fetch(`${REPO_API_BASE}${pathname}`, {
      headers,
      signal: controller.signal,
    });
    const text = await response.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = { message: text };
      }
    }

    if (!response.ok) {
      throw new GitHubHttpError(data?.message || `GitHub returned ${response.status}`, response.status, data);
    }

    return data;
  } finally {
    clearTimeout(timer);
  }
}

async function safeGithubGet(pathname, options = {}) {
  try {
    return await githubGet(pathname, options);
  } catch (error) {
    if (error instanceof GitHubHttpError && error.status === 404 && options.allow404) {
      return null;
    }
    throw error;
  }
}

async function getLocalInstallInfo() {
  const localVersion = await readLocalVersion();
  const insideWorkTree = trimOutput(await safeGit(['rev-parse', '--is-inside-work-tree'])) === 'true';

  if (!insideWorkTree) {
    return {
      installKind: 'archive',
      localVersion,
      branch: null,
      localCommit: null,
      localShortCommit: null,
      sourceRemote: null,
      sourceRemoteWebUrl: null,
      sourceRemoteMatchesCanonical: null,
    };
  }

  const branch = trimOutput(await safeGit(['rev-parse', '--abbrev-ref', 'HEAD'])) || 'HEAD';
  const localCommit = trimOutput(await safeGit(['rev-parse', 'HEAD'])) || null;
  const sourceRemote = redactRemoteCredentials(trimOutput(await safeGit(['remote', 'get-url', 'origin'])));
  const sourceRemoteWebUrl = normalizeRemoteWebUrl(sourceRemote);

  return {
    installKind: 'git',
    localVersion,
    branch,
    localCommit,
    localShortCommit: shortSha(localCommit),
    sourceRemote,
    sourceRemoteWebUrl,
    sourceRemoteMatchesCanonical: sourceRemoteWebUrl
      ? normalizeRepoUrlForCompare(sourceRemoteWebUrl) === normalizeRepoUrlForCompare(REPO_WEB_URL)
      : null,
  };
}

function normalizeCommit(commit) {
  const sha = commit?.sha || '';
  const message = commit?.commit?.message || '';
  const [headline, ...bodyLines] = message.split(/\r?\n/);
  const author = commit?.commit?.author || {};
  const committer = commit?.commit?.committer || {};

  return {
    sha,
    shortSha: shortSha(sha),
    message: headline || sha,
    body: bodyLines.join('\n').trim() || null,
    authorName: author.name || commit?.author?.login || null,
    authorDate: author.date || null,
    committerName: committer.name || commit?.committer?.login || null,
    committerDate: committer.date || null,
    url: commit?.html_url || (sha ? `${REPO_WEB_URL}/commit/${sha}` : null),
  };
}

async function getRecentCommits(branch) {
  const commits = await githubGet(`/commits?sha=${encodeURIComponent(branch)}&per_page=${RECENT_COMMIT_COUNT}`);
  if (!Array.isArray(commits)) {
    return [];
  }
  return commits.map(normalizeCommit).filter(commit => commit.sha);
}

async function getRemoteRepoInfo() {
  const repo = await githubGet('');
  const branch = REPO_BRANCH || repo.default_branch || 'main';
  const branchInfo = await githubGet(`/branches/${encodeURIComponent(branch)}`);
  const latestRelease = await safeGithubGet('/releases/latest', { allow404: true });
  const recentCommits = await getRecentCommits(branch);
  const remoteCommit = branchInfo?.commit?.sha || null;
  const remoteShortCommit = shortSha(remoteCommit);
  const latestVersion = latestRelease?.tag_name || null;
  const downloadUrl = latestRelease?.zipball_url || buildRepoArchiveUrl(branch);

  return {
    branch,
    remoteCommit,
    remoteShortCommit,
    remoteCommitDate:
      branchInfo?.commit?.commit?.committer?.date || branchInfo?.commit?.commit?.author?.date || null,
    remoteMessage: branchInfo?.commit?.commit?.message || null,
    latestVersion,
    latestReleaseUrl: latestRelease?.html_url || null,
    latestReleasePublishedAt: latestRelease?.published_at || null,
    cloneUrl: repo.clone_url || `${REPO_WEB_URL}.git`,
    webUrl: repo.html_url || REPO_WEB_URL,
    downloadUrl,
    recentCommits,
  };
}

async function compareByCommit(localInfo, remoteInfo) {
  if (!localInfo.localCommit || !remoteInfo.remoteCommit) {
    return null;
  }

  if (localInfo.localCommit === remoteInfo.remoteCommit) {
    return {
      comparable: true,
      ahead: 0,
      behind: 0,
      state: 'up_to_date',
      message: 'Repository is up to date',
      compareMode: 'commit',
    };
  }

  try {
    const compare = await githubGet(`/compare/${localInfo.localCommit}...${remoteInfo.remoteCommit}`);
    const behind = Number(compare?.ahead_by || 0);
    const ahead = Number(compare?.behind_by || 0);

    if (behind > 0) {
      return {
        comparable: true,
        ahead,
        behind,
        state: 'update_available',
        message:
          ahead > 0
            ? `Update available: ${plural(behind, 'commit')} behind and ${plural(ahead, 'commit')} ahead of ${remoteInfo.branch}`
            : `Update available: ${plural(behind, 'commit')} behind ${remoteInfo.branch}`,
        compareMode: 'commit',
      };
    }

    return {
      comparable: true,
      ahead,
      behind,
      state: 'up_to_date',
      message: ahead > 0 ? `Local checkout is ${plural(ahead, 'commit')} ahead of ${remoteInfo.branch}` : 'Repository is up to date',
      compareMode: 'commit',
    };
  } catch (error) {
    if (error instanceof GitHubHttpError && error.status === 404) {
      return {
        comparable: false,
        ahead: null,
        behind: null,
        state: null,
        message: null,
        compareMode: 'commit',
      };
    }
    throw error;
  }
}

function compareByVersion(localInfo, remoteInfo) {
  if (!localInfo.localVersion || !remoteInfo.latestVersion) {
    return null;
  }

  const comparison = compareVersions(localInfo.localVersion, remoteInfo.latestVersion);
  if (comparison == null) {
    return null;
  }

  if (comparison < 0) {
    return {
      comparable: true,
      ahead: null,
      behind: null,
      state: 'update_available',
      message: `Update available: ${localInfo.localVersion} -> ${remoteInfo.latestVersion}`,
      compareMode: 'version',
    };
  }

  return {
    comparable: true,
    ahead: null,
    behind: null,
    state: 'up_to_date',
    message:
      comparison === 0
        ? `App version ${localInfo.localVersion} matches the latest release`
        : `App version ${localInfo.localVersion} is newer than the latest release`,
    compareMode: 'version',
  };
}

async function resolveUpdateResult(localInfo, remoteInfo) {
  const commitResult = await compareByCommit(localInfo, remoteInfo);
  if (commitResult?.state) {
    return commitResult;
  }

  const versionResult = compareByVersion(localInfo, remoteInfo);
  if (versionResult?.state) {
    return versionResult;
  }

  return {
    comparable: false,
    ahead: null,
    behind: null,
    state: 'unknown_current',
    message:
      localInfo.installKind === 'archive'
        ? `Latest ${REPO_FULL_NAME} build found on GitHub`
        : `Latest ${REPO_FULL_NAME} commit found; local checkout could not be compared`,
    compareMode: localInfo.localCommit ? 'commit' : localInfo.localVersion ? 'version' : 'none',
  };
}

async function checkForUpdates(trigger) {
  if (isChecking || isUpdating || isStopping) {
    return;
  }

  isChecking = true;
  const startedAt = nowIso();
  const previous = (await readJson(STATUS_PATH)) || {};

  await writeStatus({
    state: 'checking',
    message: `Checking ${REPO_FULL_NAME} for updates`,
    trigger,
    startedAt,
    error: null,
    lastSuccessfulCheckAt: previous.lastSuccessfulCheckAt || null,
  });

  try {
    const localInfo = await getLocalInstallInfo();
    const remoteInfo = await getRemoteRepoInfo();
    const result = await resolveUpdateResult(localInfo, remoteInfo);
    const applyUnavailableReason = manualUpdateReason(localInfo, result);
    const canApplyUpdate =
      result.state === 'update_available' &&
      localInfo.installKind === 'git' &&
      !applyUnavailableReason &&
      Number(result.behind || 0) > 0 &&
      Number(result.ahead || 0) === 0;
    const checkedAt = nowIso();

    await writeStatus({
      state: result.state,
      message: result.message,
      checkedAt,
      lastSuccessfulCheckAt: checkedAt,
      trigger,
      startedAt,
      installKind: localInfo.installKind,
      branch: localInfo.branch,
      upstream: remoteInfo.branch,
      remote: remoteInfo.cloneUrl,
      remoteWebUrl: remoteInfo.webUrl,
      repoFullName: REPO_FULL_NAME,
      repoWebUrl: remoteInfo.webUrl,
      downloadUrl: remoteInfo.downloadUrl,
      latestVersion: remoteInfo.latestVersion,
      latestReleaseUrl: remoteInfo.latestReleaseUrl,
      latestReleasePublishedAt: remoteInfo.latestReleasePublishedAt,
      remoteCommitDate: remoteInfo.remoteCommitDate,
      recentCommits: remoteInfo.recentCommits || [],
      sourceRemote: localInfo.sourceRemote,
      sourceRemoteWebUrl: localInfo.sourceRemoteWebUrl,
      sourceRemoteMatchesCanonical: localInfo.sourceRemoteMatchesCanonical,
      compareUrl: buildCompareUrl(localInfo.localCommit, remoteInfo.remoteCommit, result.comparable),
      compareMode: result.compareMode,
      localVersion: localInfo.localVersion,
      localCommit: localInfo.localCommit,
      localShortCommit: localInfo.localShortCommit,
      remoteCommit: remoteInfo.remoteCommit,
      remoteShortCommit: remoteInfo.remoteShortCommit,
      ahead: result.ahead,
      behind: result.behind,
      canApplyUpdate,
      applyUpdateUnavailableReason: applyUnavailableReason,
      updateStep: null,
      updateError: null,
      needsRestart: false,
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
      repoFullName: REPO_FULL_NAME,
      repoWebUrl: REPO_WEB_URL,
      lastSuccessfulCheckAt: previous.lastSuccessfulCheckAt || null,
    });
  } finally {
    isChecking = false;
  }
}

async function getWorkingTreeStatus() {
  const status = trimOutput(await runGit(['status', '--porcelain=v1', '--untracked-files=all']));
  return {
    dirty: status.length > 0,
    status,
  };
}

async function gitPathExists(name) {
  const gitPath = trimOutput(await runGit(['rev-parse', '--git-path', name]));
  if (!gitPath) {
    return false;
  }
  const resolved = path.isAbsolute(gitPath) ? gitPath : path.join(TOOLKIT_ROOT, gitPath);
  try {
    await fs.stat(resolved);
    return true;
  } catch {
    return false;
  }
}

async function getActiveGitOperation() {
  const checks = [
    ['MERGE_HEAD', 'merge'],
    ['rebase-merge', 'rebase'],
    ['rebase-apply', 'rebase'],
    ['CHERRY_PICK_HEAD', 'cherry-pick'],
    ['REVERT_HEAD', 'revert'],
  ];

  for (const [gitPath, label] of checks) {
    if (await gitPathExists(gitPath)) {
      return label;
    }
  }

  return null;
}

async function stashLocalChanges() {
  const worktree = await getWorkingTreeStatus();
  if (!worktree.dirty) {
    return null;
  }

  const message = `AITK updater auto-stash ${nowIso()}`;
  await runGit(['stash', 'push', '--include-untracked', '-m', message], { timeoutMs: GIT_UPDATE_TIMEOUT_MS });
  const stashRef = trimOutput(await runGit(['stash', 'list', '--format=%gd', '-n', '1']));

  return {
    ref: stashRef || 'stash@{0}',
    message,
    status: worktree.status,
  };
}

async function restoreStash(stashInfo) {
  if (!stashInfo?.ref) {
    return { restored: false, dropped: false };
  }

  await runGit(['stash', 'apply', '--index', stashInfo.ref], { timeoutMs: GIT_UPDATE_TIMEOUT_MS });
  await runGit(['stash', 'drop', stashInfo.ref], { timeoutMs: GIT_TIMEOUT_MS });

  return { restored: true, dropped: true };
}

function buildStatusFields(localInfo, remoteInfo, result) {
  const applyUnavailableReason = manualUpdateReason(localInfo, result);
  const canApplyUpdate =
    result.state === 'update_available' &&
    localInfo.installKind === 'git' &&
    !applyUnavailableReason &&
    Number(result.behind || 0) > 0 &&
    Number(result.ahead || 0) === 0;

  return {
    installKind: localInfo.installKind,
    branch: localInfo.branch,
    upstream: remoteInfo.branch,
    remote: remoteInfo.cloneUrl,
    remoteWebUrl: remoteInfo.webUrl,
    repoFullName: REPO_FULL_NAME,
    repoWebUrl: remoteInfo.webUrl,
    downloadUrl: remoteInfo.downloadUrl,
    latestVersion: remoteInfo.latestVersion,
    latestReleaseUrl: remoteInfo.latestReleaseUrl,
    latestReleasePublishedAt: remoteInfo.latestReleasePublishedAt,
    remoteCommitDate: remoteInfo.remoteCommitDate,
    recentCommits: remoteInfo.recentCommits || [],
    sourceRemote: localInfo.sourceRemote,
    sourceRemoteWebUrl: localInfo.sourceRemoteWebUrl,
    sourceRemoteMatchesCanonical: localInfo.sourceRemoteMatchesCanonical,
    compareUrl: buildCompareUrl(localInfo.localCommit, remoteInfo.remoteCommit, result.comparable),
    compareMode: result.compareMode,
    localVersion: localInfo.localVersion,
    localCommit: localInfo.localCommit,
    localShortCommit: localInfo.localShortCommit,
    remoteCommit: remoteInfo.remoteCommit,
    remoteShortCommit: remoteInfo.remoteShortCommit,
    ahead: result.ahead,
    behind: result.behind,
    canApplyUpdate,
    applyUpdateUnavailableReason: applyUnavailableReason,
  };
}

async function writeUpdateBlocked(localInfo, remoteInfo, result, reason, startedAt, trigger) {
  await writeStatus({
    state: 'update_blocked',
    message: reason,
    checkedAt: nowIso(),
    trigger,
    startedAt,
    ...buildStatusFields(localInfo, remoteInfo, result),
    canApplyUpdate: false,
    applyUpdateUnavailableReason: reason,
    updateStep: null,
    updateError: null,
    needsRestart: false,
    error: null,
  });
}

async function applyGitUpdate(trigger) {
  if (isChecking || isUpdating || isStopping) {
    return;
  }

  isUpdating = true;
  const startedAt = nowIso();
  const previous = (await readJson(STATUS_PATH)) || {};
  let stashInfo = null;

  await writeStatus({
    state: 'updating',
    message: `Applying update from ${REPO_FULL_NAME}`,
    trigger,
    startedAt,
    updateStartedAt: startedAt,
    updateStep: 'checking',
    canApplyUpdate: false,
    updateError: null,
    error: null,
    lastSuccessfulCheckAt: previous.lastSuccessfulCheckAt || null,
  });

  try {
    const localInfo = await getLocalInstallInfo();
    const remoteInfo = await getRemoteRepoInfo();
    const result = await resolveUpdateResult(localInfo, remoteInfo);
    const unavailableReason = manualUpdateReason(localInfo, result);

    if (unavailableReason) {
      await writeUpdateBlocked(localInfo, remoteInfo, result, unavailableReason, startedAt, trigger);
      return;
    }

    if (result.state !== 'update_available' || Number(result.behind || 0) <= 0) {
      await writeStatus({
        state: result.state,
        message: result.message,
        checkedAt: nowIso(),
        trigger,
        startedAt,
        ...buildStatusFields(localInfo, remoteInfo, result),
        updateStep: null,
        updateError: null,
        needsRestart: false,
        error: null,
      });
      return;
    }

    const activeOperation = await getActiveGitOperation();
    if (activeOperation) {
      await writeUpdateBlocked(
        localInfo,
        remoteInfo,
        result,
        `Finish the in-progress git ${activeOperation} before updating.`,
        startedAt,
        trigger,
      );
      return;
    }

    await writeStatus({
      state: 'updating',
      message: 'Saving local changes before update',
      updateStep: 'stashing',
      ...buildStatusFields(localInfo, remoteInfo, result),
      canApplyUpdate: false,
    });
    stashInfo = await stashLocalChanges();

    await writeStatus({
      state: 'updating',
      message: `Fetching ${REPO_FULL_NAME}`,
      updateStep: 'fetching',
      stashCreated: Boolean(stashInfo),
      stashRef: stashInfo?.ref || null,
      canApplyUpdate: false,
    });
    await runGit(['fetch', '--quiet', remoteInfo.cloneUrl, remoteInfo.branch], { timeoutMs: GIT_UPDATE_TIMEOUT_MS });

    await writeStatus({
      state: 'updating',
      message: 'Applying fast-forward update',
      updateStep: 'fast-forwarding',
      canApplyUpdate: false,
    });
    await runGit(['merge', '--ff-only', 'FETCH_HEAD'], { timeoutMs: GIT_UPDATE_TIMEOUT_MS });

    const updatedCommit = trimOutput(await runGit(['rev-parse', 'HEAD']));

    if (stashInfo) {
      await writeStatus({
        state: 'updating',
        message: 'Restoring local changes',
        updateStep: 'restoring-local-changes',
        canApplyUpdate: false,
      });
      await restoreStash(stashInfo);
    }

    await writeStatus({
      state: 'updated',
      message: 'Update applied. Restart the app to use the new version.',
      checkedAt: nowIso(),
      updateCompletedAt: nowIso(),
      trigger,
      startedAt,
      ...buildStatusFields(
        {
          ...localInfo,
          localCommit: updatedCommit,
          localShortCommit: shortSha(updatedCommit),
        },
        remoteInfo,
        {
          comparable: true,
          ahead: 0,
          behind: 0,
          state: 'up_to_date',
          message: 'Repository is up to date',
          compareMode: 'commit',
        },
      ),
      previousLocalCommit: localInfo.localCommit,
      updateStep: null,
      updateError: null,
      stashCreated: Boolean(stashInfo),
      stashRef: null,
      localChangesRestored: Boolean(stashInfo),
      canApplyUpdate: false,
      needsRestart: true,
      error: null,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown update error';

    if (stashInfo) {
      try {
        await restoreStash(stashInfo);
        await writeStatus({
          state: 'update_failed',
          message: 'Update failed; local changes were restored.',
          checkedAt: nowIso(),
          updateCompletedAt: nowIso(),
          trigger,
          startedAt,
          updateStep: null,
          updateError: message,
          stashCreated: true,
          stashRef: null,
          localChangesRestored: true,
          canApplyUpdate: false,
          needsRestart: false,
          error: message,
        });
      } catch (restoreError) {
        const restoreMessage = restoreError instanceof Error ? restoreError.message : 'Could not restore local changes';
        await writeStatus({
          state: 'update_conflict',
          message: `Update stopped. Local changes are preserved in ${stashInfo.ref}.`,
          checkedAt: nowIso(),
          updateCompletedAt: nowIso(),
          trigger,
          startedAt,
          updateStep: null,
          updateError: `${message}\n${restoreMessage}`,
          stashCreated: true,
          stashRef: stashInfo.ref,
          localChangesRestored: false,
          canApplyUpdate: false,
          needsRestart: false,
          error: restoreMessage,
        });
      }
    } else {
      await writeStatus({
        state: 'update_failed',
        message: 'Update failed.',
        checkedAt: nowIso(),
        updateCompletedAt: nowIso(),
        trigger,
        startedAt,
        updateStep: null,
        updateError: message,
        canApplyUpdate: false,
        needsRestart: false,
        error: message,
      });
    }
  } finally {
    isUpdating = false;
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

async function applyAndSchedule(trigger) {
  if (nextCheckTimer) {
    clearTimeout(nextCheckTimer);
    nextCheckTimer = null;
  }

  await applyGitUpdate(trigger);

  if (!isStopping) {
    const nextCheckAt = new Date(Date.now() + intervalMs).toISOString();
    await writeStatus({ nextCheckAt });
    nextCheckTimer = setTimeout(() => {
      void checkAndSchedule('schedule');
    }, intervalMs);
  }
}

async function restartUi(trigger) {
  if (isStopping) {
    return;
  }

  isStopping = true;
  if (nextCheckTimer) {
    clearTimeout(nextCheckTimer);
    nextCheckTimer = null;
  }
  if (requestPollTimer) {
    clearInterval(requestPollTimer);
    requestPollTimer = null;
  }

  const startedAt = nowIso();
  await writeStatus({
    state: 'restarting',
    message: 'Restart requested',
    trigger,
    startedAt,
    restartStartedAt: startedAt,
    restartStep: 'handoff',
    canApplyUpdate: false,
    nextCheckAt: null,
    updateStep: null,
    updateError: null,
    error: null,
  });

  const child = spawn(process.execPath, [RESTART_SCRIPT], {
    cwd: UI_ROOT,
    detached: true,
    env: process.env,
    stdio: 'ignore',
    windowsHide: true,
  });

  child.unref();

  await writeStatus({
    state: 'restarting',
    message: 'Restarting UI',
    trigger,
    startedAt,
    restartStartedAt: startedAt,
    restartStep: 'runner-started',
    restartPid: child.pid,
    canApplyUpdate: false,
    nextCheckAt: null,
  });

  setTimeout(() => {
    process.exit(0);
  }, 250).unref();
}

async function pollForRequests() {
  if (isChecking || isUpdating || isStopping) {
    return;
  }

  try {
    const stat = await fs.stat(REQUEST_PATH);
    if (stat.mtimeMs <= handledRequestMtime) {
      return;
    }
    handledRequestMtime = stat.mtimeMs;
    const request = (await readJson(REQUEST_PATH)) || {};
    await fs.unlink(REQUEST_PATH).catch(() => undefined);
    if (request.action === 'apply') {
      await applyAndSchedule('manual');
    } else if (request.action === 'restart') {
      await restartUi('manual');
    } else {
      await checkAndSchedule('manual');
    }
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
